# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Public MegaMoe experts forward/backward system-test workers."""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, fields
from pathlib import Path

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# The launcher activates CANN and the multicore payload before importing this
# worker. Framework imports stay out of the ``test_mega_moe.py`` launcher.
# pylint: disable=wrong-import-position
import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_moe import MegaMoeExperts
from hyper_parallel.core.multicore.platform.torch.mega_moe.native import (
    create_native_moe,
)

# pylint: enable=wrong-import-position

_RTOL = 2e-2
_ATOL = 2e-3


def _move_to_test_device(module: torch.nn.Module) -> torch.nn.Module:
    """Move a runtime Torch module to the worker device and dtype."""
    return module.to(device=_DEVICE, dtype=torch.bfloat16)


@dataclass(frozen=True)
class _MoeShape:
    """Minimal public-layer test shape."""

    local_num_tokens: int = 128
    hidden_size: int = 512
    intermediate_size: int = 128
    num_experts: int = 4
    top_k: int = 2
    ep_size: int = 2

    @property
    def local_experts(self) -> int:
        """Return expert weights owned by each EP rank."""
        return self.num_experts // self.ep_size


@dataclass(frozen=True)
class _RunResult:
    """Forward and backward tensors compared across the two backends."""

    output: torch.Tensor
    input_grad: torch.Tensor
    route_weight_grad: torch.Tensor
    gate_up_weight_grad: torch.Tensor
    down_weight_grad: torch.Tensor


def _init_dist() -> tuple[int, torch.device]:
    """Bind the local NPU and initialize the EP process group."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    world_size = dist.get_world_size()
    expected_world_size = int(os.getenv("HP_MEGA_MOE_WORLD_SIZE", "2"))
    if world_size != expected_world_size:
        raise ValueError(
            f"worker requires {expected_world_size} ranks, got {world_size}."
        )
    return dist.get_rank(), torch.device("npu", local_rank)


_RANK, _DEVICE = _init_dist()


def _make_balanced_route(
    shape: _MoeShape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the balanced token-major route and its exact local histogram."""
    positions = torch.arange(
        shape.local_num_tokens * shape.top_k,
        dtype=torch.int64,
        device=_DEVICE,
    ).view(shape.local_num_tokens, shape.top_k)
    topk_ids = (positions + _RANK * shape.top_k).remainder(shape.num_experts)
    base_weights = torch.arange(
        shape.top_k,
        0,
        -1,
        dtype=torch.float32,
        device=_DEVICE,
    )
    topk_weights = (
        (base_weights / base_weights.sum())
        .expand(
            shape.local_num_tokens,
            -1,
        )
        .clone()
    )
    routed_slots = shape.local_num_tokens * shape.top_k
    if routed_slots % shape.num_experts:
        raise ValueError(
            "balanced route requires routed slots divisible by num_experts, "
            f"got routed_slots={routed_slots}, num_experts={shape.num_experts}."
        )
    tokens_per_expert = torch.full(
        (shape.num_experts,),
        routed_slots // shape.num_experts,
        dtype=torch.int32,
        device=_DEVICE,
    )
    return topk_ids.to(torch.int32), topk_weights, tokens_per_expert


def _make_overflow_route(
    shape: _MoeShape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a valid route whose maximum destination load exceeds the mean."""
    if shape.ep_size != 2 or shape.top_k > shape.local_experts:
        raise ValueError("overflow route requires EP=2 and top_k <= local experts.")
    topk_ids = (
        torch.arange(
            shape.top_k,
            dtype=torch.int64,
            device=_DEVICE,
        )
        .expand(shape.local_num_tokens, -1)
        .clone()
    )
    if _RANK == 1:
        topk_ids[shape.local_num_tokens // 2 :] += shape.local_experts
    topk_weights = torch.full(
        (shape.local_num_tokens, shape.top_k),
        1.0 / shape.top_k,
        dtype=torch.float32,
        device=_DEVICE,
    )
    tokens_per_expert = torch.zeros(
        shape.num_experts,
        dtype=torch.int32,
        device=_DEVICE,
    )
    if _RANK == 0:
        tokens_per_expert[: shape.top_k] = shape.local_num_tokens
    else:
        local_split = shape.local_num_tokens // 2
        tokens_per_expert[: shape.top_k] = local_split
        tokens_per_expert[shape.local_experts : shape.local_experts + shape.top_k] = (
            shape.local_num_tokens - local_split
        )
    return topk_ids.to(torch.int32), topk_weights, tokens_per_expert


def _new_layers(
    shape: _MoeShape,
    *,
    expert_capacity_factor: float = 1.5,
) -> tuple[MegaMoeExperts, torch.nn.Module]:
    """Construct parameter-aligned public and native expert layers."""
    torch.manual_seed(2026 + _RANK)
    native = create_native_moe(
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        ep_size=shape.ep_size,
        ep_group=dist.group.WORLD,
    )
    native = _move_to_test_device(native)
    mega = MegaMoeExperts(
        local_num_tokens=shape.local_num_tokens,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        expert_capacity_factor=expert_capacity_factor,
        ep_size=shape.ep_size,
        ep_group=dist.group.WORLD,
    ).to(device=_DEVICE, dtype=torch.bfloat16)
    mega.load_state_dict(native.state_dict())
    return mega, native


def _make_data(shape: _MoeShape) -> tuple[torch.Tensor, torch.Tensor]:
    """Create rank-specific input and upstream gradient tensors."""
    torch.manual_seed(30_000 + _RANK)
    torch.npu.manual_seed(30_000 + _RANK)
    hidden_states = torch.randn(
        shape.local_num_tokens,
        shape.hidden_size,
        dtype=torch.float32,
        device=_DEVICE,
    ).to(torch.bfloat16)
    grad_output = torch.randn(
        shape.local_num_tokens,
        shape.hidden_size,
        dtype=torch.float32,
        device=_DEVICE,
    ).to(torch.bfloat16)
    return hidden_states, grad_output


def _run_layer(
    layer: MegaMoeExperts | torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor | None,
    grad_output: torch.Tensor,
) -> _RunResult:
    """Run one layer F+B and detach all compared tensors."""
    layer.zero_grad(set_to_none=True)
    hidden = hidden_states.detach().clone().requires_grad_(True)
    route_weights = topk_weights.detach().clone().requires_grad_(True)
    output = layer(
        hidden,
        topk_ids,
        route_weights,
        tokens_per_expert=tokens_per_expert,
    )
    output.backward(grad_output)
    torch.npu.synchronize(_DEVICE)
    gradients = {
        "input": hidden.grad,
        "route weight": route_weights.grad,
        "gate/up weight": layer.gate_up_weight.grad,
        "down weight": layer.down_weight.grad,
    }
    missing = [name for name, gradient in gradients.items() if gradient is None]
    if missing:
        raise AssertionError(f"rank={_RANK}: missing gradients {missing}.")
    return _RunResult(
        output=output.detach().clone(),
        input_grad=hidden.grad.detach().clone(),
        route_weight_grad=route_weights.grad.detach().clone(),
        gate_up_weight_grad=layer.gate_up_weight.grad.detach().clone(),
        down_weight_grad=layer.down_weight.grad.detach().clone(),
    )


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Require every compared result to be finite."""
    finite = bool(torch.isfinite(tensor).all().item())
    assert finite, (
        f"rank={_RANK}: {name} contains non-finite values, shape={tuple(tensor.shape)}."
    )


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Compare BF16 native and MegaMoe results with diagnostic maxima."""
    if torch.allclose(actual, expected, rtol=_RTOL, atol=_ATOL):
        return
    difference = (actual.float() - expected.float()).abs()
    maximum_absolute = float(difference.max().item())
    maximum_relative = float(
        (difference / expected.float().abs().clamp_min(1e-12)).max().item()
    )
    raise AssertionError(
        f"rank={_RANK}: {name} differs from native baseline at rtol={_RTOL}, "
        f"atol={_ATOL}, max_abs={maximum_absolute}, max_rel={maximum_relative}."
    )


def _assert_active_expert_gradients(
    shape: _MoeShape,
    local_counts: torch.Tensor,
    result: _RunResult,
    backend: str,
) -> None:
    """Require useful dW for every expert in the balanced route."""
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    expert_start = _RANK * shape.local_experts
    for local_idx in range(shape.local_experts):
        global_idx = expert_start + local_idx
        count = int(global_counts[global_idx].item())
        assert count > 0, (
            f"rank={_RANK}: balanced route did not activate expert {global_idx}."
        )
        for name, gradient in (
            ("gate_up_weight_grad", result.gate_up_weight_grad[local_idx]),
            ("down_weight_grad", result.down_weight_grad[local_idx]),
        ):
            nonzero = int(torch.count_nonzero(gradient).item())
            assert nonzero > 0, (
                f"rank={_RANK}: {backend} active expert {global_idx} {name} is all zero, "
                f"route_count={count}."
            )


def _assert_results_close(native_result: _RunResult, mega_result: _RunResult) -> None:
    """Compare every forward and backward result from two expert backends."""
    for result_field in fields(_RunResult):
        field_name = result_field.name
        native_tensor = getattr(native_result, field_name)
        mega_tensor = getattr(mega_result, field_name)
        assert mega_tensor.shape == native_tensor.shape, (
            f"rank={_RANK}: {field_name} shape mismatch: "
            f"mega={tuple(mega_tensor.shape)}, native={tuple(native_tensor.shape)}."
        )
        _assert_finite(f"native {field_name}", native_tensor)
        _assert_finite(f"MegaMoe {field_name}", mega_tensor)
        _assert_close(field_name, mega_tensor, native_tensor)


def _run_precision_case() -> None:
    """Compare all public F+B outputs against the native oracle."""
    shape = _MoeShape()
    topk_ids, topk_weights, tokens_per_expert = _make_balanced_route(shape)
    hidden_states, grad_output = _make_data(shape)
    mega, native = _new_layers(shape)
    native_result = _run_layer(
        native,
        hidden_states,
        topk_ids,
        topk_weights,
        None,
        grad_output,
    )
    mega_result = _run_layer(
        mega,
        hidden_states,
        topk_ids,
        topk_weights,
        None,
        grad_output,
    )
    _assert_results_close(native_result, mega_result)
    _assert_active_expert_gradients(shape, tokens_per_expert, native_result, "native")
    _assert_active_expert_gradients(shape, tokens_per_expert, mega_result, "MegaMoe")


def test_mega_moe_level0_balanced() -> None:
    """Compare fixed-mode MegaMoe and native F+B on the essential balanced route."""
    _run_precision_case()


def test_mega_moe_level0_capacity_fallback() -> None:
    """Compare overflowing native fallback F+B with the native oracle."""
    shape = _MoeShape()
    topk_ids, topk_weights, tokens_per_expert = _make_overflow_route(shape)
    expert_capacity_factor = 1.0
    receive_capacity = shape.local_num_tokens * shape.top_k
    destination_counts = tokens_per_expert.reshape(
        shape.ep_size,
        shape.local_experts,
    ).sum(dim=1)
    dist.all_reduce(destination_counts, op=dist.ReduceOp.SUM)
    actual_capacity = int(destination_counts.max().item())
    assert actual_capacity > receive_capacity, (
        f"overflow route must exceed its bound: actual={actual_capacity}, "
        f"receive_capacity={receive_capacity}."
    )
    hidden_states, grad_output = _make_data(shape)
    mega, native = _new_layers(
        shape,
        expert_capacity_factor=expert_capacity_factor,
    )
    native_result = _run_layer(
        native,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
    )
    mega_result = _run_layer(
        mega,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
    )
    _assert_results_close(native_result, mega_result)


def _performance_shape() -> _MoeShape:
    """Return the fixed representative four-card performance shape."""
    return _MoeShape(
        local_num_tokens=4096,
        hidden_size=5120,
        intermediate_size=1792,
        num_experts=32,
        top_k=8,
        ep_size=4,
    )


def _new_performance_layer(
    shape: _MoeShape,
    backend: str,
) -> MegaMoeExperts | torch.nn.Module:
    """Construct one seeded backend without retaining its comparison peer."""
    torch.manual_seed(2026 + _RANK)
    if backend == "native":
        layer = create_native_moe(
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            ep_size=shape.ep_size,
            ep_group=dist.group.WORLD,
        )
        layer = _move_to_test_device(layer)
        return layer
    if backend != "mega_moe":
        raise ValueError(
            f"HP_MEGA_MOE_PERF_BACKEND must be native or mega_moe, got {backend!r}."
        )
    return MegaMoeExperts(
        local_num_tokens=shape.local_num_tokens,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        ep_size=shape.ep_size,
        ep_group=dist.group.WORLD,
    ).to(
        device=_DEVICE,
        dtype=torch.bfloat16,
    )


def _rank_max_latency(elapsed_ms: float) -> float:
    """Return one F+B latency reduced by maximum across all ranks."""
    latency = torch.tensor(elapsed_ms, dtype=torch.float32, device=_DEVICE)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return float(latency.cpu().item())


def _timed_fwd_bwd(
    layer: MegaMoeExperts | torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    validate_gradients: bool = False,
) -> float:
    """Time one synchronized layer forward and backward operation."""
    layer.zero_grad(set_to_none=True)
    hidden = hidden_states.detach().requires_grad_(True)
    route_weights = topk_weights.detach().requires_grad_(True)
    dist.barrier()
    torch.npu.synchronize(_DEVICE)
    start = time.perf_counter()
    output = layer(
        hidden,
        topk_ids,
        route_weights,
        tokens_per_expert=tokens_per_expert,
    )
    output.backward(grad_output)
    torch.npu.synchronize(_DEVICE)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    gradients = (
        ("input", hidden.grad),
        ("route weight", route_weights.grad),
        ("gate/up weight", layer.gate_up_weight.grad),
        ("down weight", layer.down_weight.grad),
    )
    missing = [name for name, gradient in gradients if gradient is None]
    if missing:
        raise AssertionError(
            f"rank={_RANK}: performance F+B missed gradients {missing}."
        )
    if validate_gradients:
        finite = torch.ones((), dtype=torch.int32, device=_DEVICE)
        for _, gradient in gradients:
            finite = torch.minimum(
                finite,
                torch.isfinite(gradient).all().to(torch.int32),
            )
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite.cpu().item()):
            raise AssertionError(
                f"rank={_RANK}: performance F+B produced non-finite gradients."
            )
    return _rank_max_latency(elapsed_ms)


def _write_performance_result(
    backend: str,
    shape: _MoeShape,
    warmup_steps: int,
    latencies: list[float],
) -> None:
    """Write one fresh-process performance sample from rank zero."""
    if _RANK != 0:
        return
    result_path = os.getenv("HP_MEGA_MOE_PERF_RESULT")
    if not result_path:
        raise ValueError("HP_MEGA_MOE_PERF_RESULT must name the output JSON path.")
    result = {
        "backend": backend,
        "topology": {
            "world_size": dist.get_world_size(),
            "ep": shape.ep_size,
        },
        "shape": {
            "local_num_tokens": shape.local_num_tokens,
            "hidden_size": shape.hidden_size,
            "intermediate_size": shape.intermediate_size,
            "num_experts": shape.num_experts,
            "top_k": shape.top_k,
            "dtype": "bfloat16",
        },
        "warmup_steps": warmup_steps,
        "measured_steps": len(latencies),
        "validation": {"gradients_present": True, "gradients_finite": True},
        "rank_max_fwd_bwd_ms": {
            "median": statistics.median(latencies),
            "minimum": min(latencies),
            "maximum": max(latencies),
            "samples": latencies,
        },
        "device_healthy": True,
    }
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PERF_RESULT_JSON={path}")
    print(json.dumps(result, sort_keys=True))


def test_mega_moe_fwd_bwd_performance() -> None:
    """Measure one fresh-process representative backend F+B sequence."""
    backend = os.getenv("HP_MEGA_MOE_PERF_BACKEND", "")
    warmup_steps = int(os.getenv("HP_MEGA_MOE_PERF_WARMUP", "20"))
    measured_steps = int(os.getenv("HP_MEGA_MOE_PERF_MEASURED", "50"))
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError(
            f"invalid performance steps: warmup={warmup_steps}, measured={measured_steps}."
        )

    shape = _performance_shape()
    topk_ids, topk_weights, tokens_per_expert = _make_balanced_route(shape)
    hidden_states, grad_output = _make_data(shape)
    layer = _new_performance_layer(shape, backend)
    latencies = []
    _timed_fwd_bwd(
        layer,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
        validate_gradients=True,
    )
    for step in range(warmup_steps + measured_steps):
        latency = _timed_fwd_bwd(
            layer,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        )
        if step >= warmup_steps:
            latencies.append(latency)
    _write_performance_result(
        backend,
        shape,
        warmup_steps,
        latencies,
    )
