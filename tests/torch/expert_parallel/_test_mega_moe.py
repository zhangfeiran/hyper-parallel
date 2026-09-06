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
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, fields
from pathlib import Path

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# The launcher activates the Ascend toolkit and multicore payload before importing this
# worker. Framework imports stay out of the ``test_mega_moe.py`` launcher.
# pylint: disable=wrong-import-position
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.modules.mega_moe import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.placement import (
    build_constrained_expert_placement,
)
from tests.torch.expert_parallel._test_mega_moe_common import (
    DEVICE as _DEVICE,
    RANK as _RANK,
    expert_weight_gradients,
    forward_layer,
    new_common_moe,
    new_layers as new_common_layers,
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


@dataclass(frozen=True)
class _PerformanceRouteSpec:
    """Reproducible route distribution selected by the performance worker."""

    profile: str
    dirichlet_alpha: float = 10000.0
    dirichlet_seed: int = 42
    clustered_experts: int = 12


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


def _dirichlet_expert_counts(
    shape: _MoeShape,
    alpha: float,
    seed: int,
) -> torch.Tensor:
    """Apportion one reproducible Dirichlet profile into exact Top-K slots."""
    if not math.isfinite(alpha) or alpha < 0.001:
        raise ValueError(
            "Dirichlet alpha must be finite and at least 0.001, got "
            f"{alpha}."
        )
    total_slots = shape.local_num_tokens * shape.top_k
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        concentration = torch.full(
            (shape.num_experts,),
            alpha,
            dtype=torch.float64,
        )
        shares = torch.distributions.Dirichlet(concentration).sample()

    # One token cannot select an expert twice. Water-fill any expert whose
    # sampled share exceeds one choice per token before integer apportionment.
    target = torch.zeros(shape.num_experts, dtype=torch.float64)
    remaining = torch.ones(shape.num_experts, dtype=torch.bool)
    remaining_slots = total_slots
    while remaining_slots:
        active = remaining.nonzero(as_tuple=False).flatten()
        active_shares = shares[active]
        share_sum = active_shares.sum()
        if not bool(share_sum > 0):
            active_shares = torch.ones_like(active_shares)
            share_sum = active_shares.sum()
        proposed = active_shares / share_sum * remaining_slots
        saturated = proposed > shape.local_num_tokens
        if not bool(saturated.any()):
            target[active] = proposed
            break
        saturated_experts = active[saturated]
        target[saturated_experts] = shape.local_num_tokens
        remaining[saturated_experts] = False
        remaining_slots -= shape.local_num_tokens * saturated_experts.numel()

    counts = target.floor().to(torch.int64)
    residual = total_slots - int(counts.sum())
    if residual:
        fractions = target - counts
        fractions[counts >= shape.local_num_tokens] = -1.0
        order = torch.argsort(fractions, descending=True, stable=True)
        counts[order[:residual]] += 1
    if int(counts.sum()) != total_slots or bool(
        (counts > shape.local_num_tokens).any()
    ):
        raise RuntimeError(
            "Dirichlet route apportionment violated its slot constraints."
        )
    return counts.to(device=_DEVICE)


def _make_dirichlet_route(
    shape: _MoeShape,
    alpha: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a token-major Top-K route with a controlled expert skew."""
    counts = _dirichlet_expert_counts(shape, alpha, seed)
    expert_ids = torch.repeat_interleave(
        torch.arange(
            shape.num_experts,
            dtype=torch.int64,
            device=_DEVICE,
        ),
        counts,
    )
    topk_ids = expert_ids.view(
        shape.top_k,
        shape.local_num_tokens,
    ).transpose(0, 1).contiguous()
    sorted_ids = topk_ids.sort(dim=-1).values
    if shape.top_k > 1 and not bool(
        torch.all(sorted_ids[:, 1:] != sorted_ids[:, :-1]).item()
    ):
        raise AssertionError("Dirichlet route selected one expert twice for a token.")
    base_weights = torch.arange(
        shape.top_k,
        0,
        -1,
        dtype=torch.float32,
        device=_DEVICE,
    )
    topk_weights = (
        (base_weights / base_weights.sum())
        .expand(shape.local_num_tokens, -1)
        .clone()
    )
    return topk_ids.to(torch.int32), topk_weights, counts.to(torch.int32)


def _make_clustered_route(
    shape: _MoeShape,
    active_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a valid route concentrated in the leading expert home ranks."""
    if not shape.top_k <= active_experts <= shape.num_experts:
        raise ValueError(
            f"clustered_experts must be in [{shape.top_k}, {shape.num_experts}], "
            f"got {active_experts}."
        )
    token_positions = torch.arange(shape.local_num_tokens).reshape(-1, 1)
    choice_offsets = torch.arange(shape.top_k).reshape(1, -1)
    topk_ids = (token_positions + choice_offsets).remainder(active_experts)
    tokens_per_expert = torch.bincount(
        topk_ids.reshape(-1),
        minlength=shape.num_experts,
    ).to(torch.int32)
    base_weights = torch.arange(
        shape.top_k,
        0,
        -1,
        dtype=torch.float32,
        device=_DEVICE,
    )
    topk_weights = (
        (base_weights / base_weights.sum())
        .expand(shape.local_num_tokens, -1)
        .clone()
    )
    return (
        topk_ids.to(device=_DEVICE, dtype=torch.int32),
        topk_weights,
        tokens_per_expert.to(device=_DEVICE),
    )


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


def _make_destination_skew_route(
    shape: _MoeShape,
    concentrated_ranks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route leading ranks to destination zero and other ranks locally."""
    if shape.ep_size != dist.get_world_size() or shape.top_k > shape.local_experts:
        raise ValueError(
            "destination-skew route requires one EP rank per process and "
            "top_k <= local experts."
        )
    if not 1 <= concentrated_ranks <= shape.ep_size:
        raise ValueError(
            f"concentrated_ranks must be in [1, {shape.ep_size}], got "
            f"{concentrated_ranks}."
        )
    destination = 0 if _RANK < concentrated_ranks else _RANK
    expert_start = destination * shape.local_experts
    topk_ids = (
        torch.arange(
            expert_start,
            expert_start + shape.top_k,
            dtype=torch.int64,
            device=_DEVICE,
        )
        .expand(shape.local_num_tokens, -1)
        .clone()
    )
    base_weights = torch.arange(
        shape.top_k,
        0,
        -1,
        dtype=torch.float32,
        device=_DEVICE,
    )
    topk_weights = (
        (base_weights / base_weights.sum())
        .expand(shape.local_num_tokens, -1)
        .clone()
    )
    tokens_per_expert = torch.zeros(
        shape.num_experts,
        dtype=torch.int32,
        device=_DEVICE,
    )
    tokens_per_expert[expert_start : expert_start + shape.top_k] = (
        shape.local_num_tokens
    )
    return topk_ids.to(torch.int32), topk_weights, tokens_per_expert


def _new_layers(
    shape: _MoeShape,
    *,
    expert_capacity_factor: float = 1.5,
    replica_slots_per_rank: int = 0,
) -> tuple[MegaMoeExperts, torch.nn.Module]:
    """Construct MegaMoe and the standard EP reference with identical weights."""
    return new_common_layers(
        shape,
        expert_capacity_factor=expert_capacity_factor,
        replica_slots_per_rank=replica_slots_per_rank,
    )


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
    output = forward_layer(
        layer,
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
        "gate/up weight": expert_weight_gradients(layer)[0],
        "down weight": expert_weight_gradients(layer)[1],
    }
    missing = [name for name, gradient in gradients.items() if gradient is None]
    if missing:
        raise AssertionError(f"rank={_RANK}: missing gradients {missing}.")
    return _RunResult(
        output=output.detach().clone(),
        input_grad=hidden.grad.detach().clone(),
        route_weight_grad=route_weights.grad.detach().clone(),
        gate_up_weight_grad=expert_weight_gradients(layer)[0].detach().clone(),
        down_weight_grad=expert_weight_gradients(layer)[1].detach().clone(),
    )


def _run_checkpointed_layer(
    layer: MegaMoeExperts,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    use_reentrant: bool,
) -> _RunResult:
    """Run one expert layer through an activation-checkpoint replay."""
    layer.zero_grad(set_to_none=True)
    hidden = hidden_states.detach().clone().requires_grad_(True)
    route_weights = topk_weights.detach().clone().requires_grad_(True)

    def _expert_forward(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return layer(
            states,
            topk_ids,
            weights,
            tokens_per_expert=tokens_per_expert,
        )

    output = checkpoint(
        _expert_forward,
        hidden,
        route_weights,
        use_reentrant=use_reentrant,
    )
    output.backward(grad_output)
    torch.npu.synchronize(_DEVICE)
    gradients = {
        "input": hidden.grad,
        "route weight": route_weights.grad,
        "gate/up weight": expert_weight_gradients(layer)[0],
        "down weight": expert_weight_gradients(layer)[1],
    }
    missing = [name for name, gradient in gradients.items() if gradient is None]
    if missing:
        raise AssertionError(
            f"rank={_RANK}: checkpoint replay missed gradients {missing}."
        )
    return _RunResult(
        output=output.detach().clone(),
        input_grad=hidden.grad.detach().clone(),
        route_weight_grad=route_weights.grad.detach().clone(),
        gate_up_weight_grad=expert_weight_gradients(layer)[0].detach().clone(),
        down_weight_grad=expert_weight_gradients(layer)[1].detach().clone(),
    )


def _run_retained_graph_layer(
    layer: MegaMoeExperts,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[_RunResult, _RunResult]:
    """Run backward twice against one retained multi-wave forward graph."""
    layer.zero_grad(set_to_none=True)
    hidden = hidden_states.detach().clone().requires_grad_(True)
    route_weights = topk_weights.detach().clone().requires_grad_(True)
    output = forward_layer(
        layer,
        hidden,
        topk_ids,
        route_weights,
        tokens_per_expert=tokens_per_expert,
    )

    def _capture_result(label: str) -> _RunResult:
        torch.npu.synchronize(_DEVICE)
        gradients = {
            "input": hidden.grad,
            "route weight": route_weights.grad,
            "gate/up weight": expert_weight_gradients(layer)[0],
            "down weight": expert_weight_gradients(layer)[1],
        }
        missing = [name for name, gradient in gradients.items() if gradient is None]
        if missing:
            raise AssertionError(
                f"rank={_RANK}: {label} retained-graph backward missed {missing}."
            )
        return _RunResult(
            output=output.detach().clone(),
            input_grad=hidden.grad.detach().clone(),
            route_weight_grad=route_weights.grad.detach().clone(),
            gate_up_weight_grad=expert_weight_gradients(layer)[0].detach().clone(),
            down_weight_grad=expert_weight_gradients(layer)[1].detach().clone(),
        )

    output.backward(grad_output, retain_graph=True)
    first = _capture_result("first")
    layer.zero_grad(set_to_none=True)
    hidden.grad = None
    route_weights.grad = None
    output.backward(grad_output)
    second = _capture_result("second")
    return first, second


def _run_deferred_backward_pair(
    layer: MegaMoeExperts,
    hidden_states: tuple[torch.Tensor, torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_outputs: tuple[torch.Tensor, torch.Tensor],
    *,
    second_route: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    reverse_backward: bool = False,
) -> tuple[_RunResult, _RunResult]:
    """Run two forwards before consuming either shared-workspace graph."""
    layer.zero_grad(set_to_none=True)
    inputs = tuple(
        value.detach().clone().requires_grad_(True) for value in hidden_states
    )
    first_route = (topk_ids, topk_weights, tokens_per_expert)
    routes = (first_route, first_route if second_route is None else second_route)
    route_weights = tuple(
        route[1].detach().clone().requires_grad_(True) for route in routes
    )
    outputs = tuple(
        layer(
            value,
            route[0],
            weights,
            tokens_per_expert=route[2],
        )
        for value, weights, route in zip(inputs, route_weights, routes)
    )

    results = {}
    backward_order = (1, 0) if reverse_backward else (0, 1)
    for index in backward_order:
        output = outputs[index]
        output.backward(grad_outputs[index])
        torch.npu.synchronize(_DEVICE)
        gradients = {
            "input": inputs[index].grad,
            "route weight": route_weights[index].grad,
            "gate/up weight": expert_weight_gradients(layer)[0],
            "down weight": expert_weight_gradients(layer)[1],
        }
        missing = [name for name, gradient in gradients.items() if gradient is None]
        if missing:
            raise AssertionError(
                f"rank={_RANK}: deferred backward {index} missed {missing}."
            )
        results[index] = _RunResult(
            output=output.detach().clone(),
            input_grad=inputs[index].grad.detach().clone(),
            route_weight_grad=route_weights[index].grad.detach().clone(),
            gate_up_weight_grad=expert_weight_gradients(layer)[0].detach().clone(),
            down_weight_grad=expert_weight_gradients(layer)[1].detach().clone(),
        )
        layer.zero_grad(set_to_none=True)
    return results[0], results[1]


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
    *,
    atol: float = _ATOL,
) -> None:
    """Compare BF16 native and MegaMoe results with diagnostic maxima."""
    if torch.allclose(actual, expected, rtol=_RTOL, atol=atol):
        return
    difference = (actual.float() - expected.float()).abs()
    maximum_absolute = float(difference.max().item())
    maximum_relative = float(
        (difference / expected.float().abs().clamp_min(1e-12)).max().item()
    )
    raise AssertionError(
        f"rank={_RANK}: {name} differs from native baseline at rtol={_RTOL}, "
        f"atol={atol}, max_abs={maximum_absolute}, max_rel={maximum_relative}."
    )


def _assert_chunked_weight_grad_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    wave_count: int,
) -> None:
    """Allow BF16 reduction-order error while still rejecting a missing wave."""
    epsilon = torch.finfo(actual.dtype).eps
    for expert, (actual_grad, expected_grad) in enumerate(zip(actual, expected)):
        scale = max(
            float(actual_grad.float().abs().max().item()),
            float(expected_grad.float().abs().max().item()),
        )
        accumulation_atol = max(_ATOL, 2 * wave_count * epsilon * scale)
        _assert_close(
            f"{name}[{expert}]",
            actual_grad,
            expected_grad,
            atol=accumulation_atol,
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


def _assert_results_close(
    native_result: _RunResult,
    mega_result: _RunResult,
    *,
    weight_grad_wave_count: int = 1,
) -> None:
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
        if weight_grad_wave_count > 1 and field_name in {
            "gate_up_weight_grad",
            "down_weight_grad",
        }:
            _assert_chunked_weight_grad_close(
                field_name,
                mega_tensor,
                native_tensor,
                weight_grad_wave_count,
            )
        else:
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
    wave_count = (actual_capacity + receive_capacity - 1) // receive_capacity
    _assert_results_close(
        native_result,
        mega_result,
        weight_grad_wave_count=wave_count,
    )


def test_mega_moe_non_power_of_two_topk_multi_wave() -> None:
    """Compare a two-wave Top-K=3 route with the native MoE oracle."""
    shape = _MoeShape(num_experts=6, top_k=3)
    topk_ids, topk_weights, tokens_per_expert = _make_destination_skew_route(
        shape,
        concentrated_ranks=shape.ep_size,
    )
    hidden_states, grad_output = _make_data(shape)
    mega, native = _new_layers(shape, expert_capacity_factor=1.0)
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
    _assert_results_close(
        native_result,
        mega_result,
        weight_grad_wave_count=2,
    )


def test_mega_moe_hot_expert_replication() -> None:
    """Compare one-wave replicated execution and gradients with native MoE."""
    shape = _MoeShape()
    topk_ids, topk_weights, tokens_per_expert = _make_overflow_route(shape)
    hidden_states, grad_output = _make_data(shape)
    mega, native = _new_layers(
        shape,
        expert_capacity_factor=1.0,
        replica_slots_per_rank=1,
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

    # The hot expert's dW is produced by two physical GMM groups and then
    # owner-reduced, so it has the same BF16 reduction-order error as two waves.
    _assert_results_close(
        native_result,
        mega_result,
        weight_grad_wave_count=2,
    )


def test_mega_moe_deferred_backward_hot_expert_replication() -> None:
    """Compare shared and owner-invalidated persistent pools with native."""
    shape = _MoeShape()
    topk_ids, topk_weights, tokens_per_expert = _make_overflow_route(shape)
    hidden_states, grad_output = _make_data(shape)
    second_hidden = hidden_states.flip(0).contiguous()
    second_grad_output = grad_output.roll(1, dims=0).contiguous()
    mega, native = _new_layers(
        shape,
        expert_capacity_factor=1.0,
        replica_slots_per_rank=1,
    )
    native_results = (
        _run_layer(
            native,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        ),
        _run_layer(
            native,
            second_hidden,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            second_grad_output,
        ),
    )
    mega_results = _run_deferred_backward_pair(
        mega,
        (hidden_states, second_hidden),
        topk_ids,
        topk_weights,
        tokens_per_expert,
        (grad_output, second_grad_output),
    )
    for native_result, mega_result in zip(native_results, mega_results):
        _assert_results_close(
            native_result,
            mega_result,
            weight_grad_wave_count=2,
        )

    # Only the owner changes its local version counters. The cache decision
    # must still invalidate collectively so every matching P2P peer refreshes.
    if _RANK == 0:
        with torch.no_grad():
            native.gate_up_weight.add_(0.125)
            mega.gate_up_weight.add_(0.125)
            native.down_weight.sub_(0.0625)
            mega.down_weight.sub_(0.0625)
    refreshed_native = _run_layer(
        native,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
    )
    refreshed_mega = _run_layer(
        mega,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
    )
    _assert_results_close(
        refreshed_native,
        refreshed_mega,
        weight_grad_wave_count=2,
    )


def test_mega_moe_replication_with_residual_multi_wave() -> None:
    """Compare bounded replication plus residual multi-wave with native MoE."""
    shape = _MoeShape(num_experts=8, top_k=4)
    topk_ids, topk_weights, tokens_per_expert = _make_destination_skew_route(
        shape,
        concentrated_ranks=shape.ep_size,
    )
    hidden_states, grad_output = _make_data(shape)
    mega, native = _new_layers(
        shape,
        expert_capacity_factor=1.0,
        replica_slots_per_rank=1,
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

    # Four equally hot home experts start on rank zero. One replica can move
    # only one expert, leaving 768 rows on rank zero for a 512-row window.
    _assert_results_close(
        native_result,
        mega_result,
        weight_grad_wave_count=2,
    )


def test_mega_moe_checkpointed_capacity_fallback() -> None:
    """Compare W1 and W2 checkpoint replays with the native F+B oracle."""
    shape = _MoeShape()
    topk_ids, topk_weights, tokens_per_expert = _make_overflow_route(shape)
    hidden_states, grad_output = _make_data(shape)
    mega, native = _new_layers(shape, expert_capacity_factor=1.0)
    native_result = _run_layer(
        native,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
    )
    for use_reentrant in (True, False):
        mega_result = _run_checkpointed_layer(
            mega,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
            use_reentrant=use_reentrant,
        )
        _assert_results_close(
            native_result,
            mega_result,
            weight_grad_wave_count=2,
        )

    topk_ids, topk_weights, tokens_per_expert = _make_balanced_route(shape)
    mega, native = _new_layers(shape)
    native_result = _run_layer(
        native,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
    )
    for use_reentrant in (True, False):
        mega_result = _run_checkpointed_layer(
            mega,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
            use_reentrant=use_reentrant,
        )
        _assert_results_close(native_result, mega_result)


def test_mega_moe_retained_graph_capacity_fallback() -> None:
    """Compare both backwards from one retained W2 graph with native."""
    shape = _MoeShape()
    topk_ids, topk_weights, tokens_per_expert = _make_overflow_route(shape)
    hidden_states, grad_output = _make_data(shape)
    mega, native = _new_layers(shape, expert_capacity_factor=1.0)
    native_result = _run_layer(
        native,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
    )
    retained_results = _run_retained_graph_layer(
        mega,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
    )
    for mega_result in retained_results:
        _assert_results_close(
            native_result,
            mega_result,
            weight_grad_wave_count=2,
        )


def test_mega_moe_retained_graph_one_wave() -> None:
    """Compare both backwards from one retained W1 graph with native."""
    shape = _MoeShape()
    topk_ids, topk_weights, tokens_per_expert = _make_balanced_route(shape)
    hidden_states, grad_output = _make_data(shape)
    mega, native = _new_layers(shape)
    native_result = _run_layer(
        native,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
    )
    retained_results = _run_retained_graph_layer(
        mega,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
    )
    for mega_result in retained_results:
        _assert_results_close(native_result, mega_result)


def test_mega_moe_deferred_backward_capacity_fallback() -> None:
    """Compare outstanding W1 and W2 forwards with sequential native calls."""
    shape = _MoeShape()
    topk_ids, topk_weights, tokens_per_expert = _make_overflow_route(shape)
    hidden_states, grad_output = _make_data(shape)
    second_hidden = hidden_states.flip(0).contiguous()
    second_grad_output = grad_output.roll(1, dims=0).contiguous()
    mega, native = _new_layers(shape, expert_capacity_factor=1.0)
    native_results = (
        _run_layer(
            native,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        ),
        _run_layer(
            native,
            second_hidden,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            second_grad_output,
        ),
    )
    mega_results = _run_deferred_backward_pair(
        mega,
        (hidden_states, second_hidden),
        topk_ids,
        topk_weights,
        tokens_per_expert,
        (grad_output, second_grad_output),
    )
    for native_result, mega_result in zip(native_results, mega_results):
        _assert_results_close(
            native_result,
            mega_result,
            weight_grad_wave_count=2,
        )

    topk_ids, topk_weights, tokens_per_expert = _make_balanced_route(shape)
    mega, native = _new_layers(shape)
    native_results = (
        _run_layer(
            native,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        ),
        _run_layer(
            native,
            second_hidden,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            second_grad_output,
        ),
    )
    mega_results = _run_deferred_backward_pair(
        mega,
        (hidden_states, second_hidden),
        topk_ids,
        topk_weights,
        tokens_per_expert,
        (grad_output, second_grad_output),
    )
    for native_result, mega_result in zip(native_results, mega_results):
        _assert_results_close(native_result, mega_result)


def test_mega_moe_mixed_deferred_backward_capacity() -> None:
    """Keep both W1 and W2 graphs live and compare both backward orders."""
    shape = _MoeShape()
    balanced = _make_balanced_route(shape)
    overflow = _make_overflow_route(shape)
    hidden, grad_output = _make_data(shape)
    hidden_states = (hidden, hidden.flip(0).contiguous())
    grad_outputs = (grad_output, grad_output.roll(1, dims=0).contiguous())
    cases = (
        ((overflow, balanced), (2, 1), True),
        ((balanced, overflow), (1, 2), True),
        ((overflow, balanced), (2, 1), False),
    )
    for routes, wave_counts, reverse_backward in cases:
        mega, native = _new_layers(shape, expert_capacity_factor=1.0)
        native_results = tuple(
            _run_layer(native, value, *route, gradient)
            for value, route, gradient in zip(hidden_states, routes, grad_outputs)
        )
        mega_results = _run_deferred_backward_pair(
            mega,
            hidden_states,
            *routes[0],
            grad_outputs,
            second_route=routes[1],
            reverse_backward=reverse_backward,
        )
        for reference, actual, waves in zip(native_results, mega_results, wave_counts):
            _assert_results_close(reference, actual, weight_grad_wave_count=waves)


def test_mega_moe_mixed_wave_replay() -> None:
    """Reuse one workspace across repeated W1, W2, and W4 calls."""
    shape = _MoeShape(
        local_num_tokens=128,
        hidden_size=512,
        intermediate_size=128,
        num_experts=16,
        top_k=2,
        ep_size=8,
    )
    repetitions = int(os.getenv("HP_MEGA_MOE_MIXED_WAVE_REPETITIONS", "100"))
    if repetitions <= 0:
        raise ValueError(f"mixed-wave repetitions must be positive, got {repetitions}.")
    routes = tuple(
        (expected_waves, _make_destination_skew_route(shape, expected_waves))
        for expected_waves in (1, 2, 1, 4)
    )
    hidden_states, grad_output = _make_data(shape)
    mega, native = _new_layers(shape, expert_capacity_factor=1.0)
    receive_capacity = shape.local_num_tokens * shape.top_k
    for repetition in range(repetitions):
        for expected_waves, route in routes:
            topk_ids, topk_weights, tokens_per_expert = route
            destination_counts = tokens_per_expert.reshape(
                shape.ep_size,
                shape.local_experts,
            ).sum(dim=1)
            dist.all_reduce(destination_counts, op=dist.ReduceOp.SUM)
            wave_count = math.ceil(
                int(destination_counts.max().item()) / receive_capacity
            )
            assert wave_count == expected_waves, (
                f"rank={_RANK}: mixed-wave route mismatch at repetition "
                f"{repetition}: expected={expected_waves}, actual={wave_count}."
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
            _assert_results_close(
                native_result,
                mega_result,
                weight_grad_wave_count=wave_count,
            )


def _performance_shape() -> _MoeShape:
    """Return the requested eight-card E48 production-shape benchmark."""
    return _MoeShape(
        local_num_tokens=4096,
        hidden_size=7168,
        intermediate_size=2048,
        num_experts=48,
        top_k=8,
        ep_size=8,
    )


def _new_performance_layer(
    shape: _MoeShape,
    backend: str,
    replica_slots_per_rank: int,
) -> MegaMoeExperts | torch.nn.Module:
    """Construct one seeded backend without retaining its comparison peer."""
    torch.manual_seed(2026 + _RANK)
    if backend == "native":
        if replica_slots_per_rank:
            raise ValueError("native performance backend does not support replicas.")
        return new_common_moe(shape)
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
        expert_capacity_factor=float(
            os.getenv("HP_MEGA_MOE_CAPACITY_FACTOR", "1.5")
        ),
        replica_slots_per_rank=replica_slots_per_rank,
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


def _percentile(samples: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile for benchmark reporting."""
    if not samples:
        raise ValueError("performance samples must not be empty.")
    index = max(0, math.ceil(len(samples) * fraction) - 1)
    return sorted(samples)[index]


def _workspace_tensor_storage(layer: object) -> tuple[int, dict[str, int]]:
    """Return unique MegaMoe workspace storage and its field-level ledger."""
    resource_group = getattr(layer, "_resource_group", None)
    backend = getattr(resource_group, "resources", None)
    workspace = getattr(backend, "workspace", None)
    if workspace is None:
        return 0, {}

    unique_storages: dict[tuple[str, int], int] = {}
    by_field: dict[str, int] = {}
    for field_name, value in vars(workspace).items():
        if not isinstance(value, torch.Tensor):
            continue
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr())
        storage_bytes = storage.nbytes()
        unique_storages[key] = max(unique_storages.get(key, 0), storage_bytes)
        by_field[field_name] = storage_bytes
    return sum(unique_storages.values()), by_field


def _device_memory_snapshot() -> dict[str, int]:
    """Return allocator and runtime HBM counters for the current rank."""
    free_bytes, total_bytes = torch.npu.mem_get_info(_DEVICE)
    return {
        "allocated_bytes": int(torch.npu.memory_allocated(_DEVICE)),
        "reserved_bytes": int(torch.npu.memory_reserved(_DEVICE)),
        "runtime_free_bytes": int(free_bytes),
        "runtime_total_bytes": int(total_bytes),
        "runtime_used_bytes": int(total_bytes - free_bytes),
    }


def _collect_memory_result(layer: object, baseline: dict[str, int]) -> dict:
    """Gather allocator peaks and a non-overlapping SHMEM upper bound."""
    final = _device_memory_snapshot()
    workspace_bytes, workspace_by_field = _workspace_tensor_storage(layer)
    symmetric_heap_bytes = int(os.getenv("SYMMETRIC_MEMORY_HEAP_SIZE", "0"))
    local = {
        "baseline_allocated_bytes": baseline["allocated_bytes"],
        "baseline_reserved_bytes": baseline["reserved_bytes"],
        "baseline_runtime_free_bytes": baseline["runtime_free_bytes"],
        "baseline_runtime_used_bytes": baseline["runtime_used_bytes"],
        "final_allocated_bytes": final["allocated_bytes"],
        "final_reserved_bytes": final["reserved_bytes"],
        "final_runtime_free_bytes": final["runtime_free_bytes"],
        "final_runtime_used_bytes": final["runtime_used_bytes"],
        "runtime_total_bytes": final["runtime_total_bytes"],
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated(_DEVICE)),
        "peak_reserved_bytes": int(torch.npu.max_memory_reserved(_DEVICE)),
        "workspace_tensor_storage_bytes": workspace_bytes,
        "symmetric_heap_bytes": symmetric_heap_bytes,
    }
    local["peak_allocated_delta_bytes"] = max(
        0,
        local["peak_allocated_bytes"] - local["baseline_allocated_bytes"],
    )
    local["peak_reserved_delta_bytes"] = max(
        0,
        local["peak_reserved_bytes"] - local["baseline_reserved_bytes"],
    )
    # SHMEM is allocated by aclshmem's external manager and therefore is not
    # registered in the PyTorch NPU allocator peak. Workspace tensor storage
    # is only a view ledger and must not be added a second time.
    local["conservative_total_peak_bytes"] = (
        local["peak_allocated_bytes"] + symmetric_heap_bytes
    )

    scalar_keys = tuple(local)
    local_tensor = torch.tensor(
        [local[key] for key in scalar_keys],
        dtype=torch.int64,
        device=_DEVICE,
    )
    gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_tensor)
    per_rank = [
        {
            key: int(value)
            for key, value in zip(scalar_keys, rank_tensor.cpu().tolist())
        }
        for rank_tensor in gathered
    ]
    rank_max = {
        key: max(rank_stats[key] for rank_stats in per_rank)
        for key in scalar_keys
        if "runtime_free" not in key
    }
    rank_min = {
        key: min(rank_stats[key] for rank_stats in per_rank)
        for key in scalar_keys
        if "runtime_free" in key
    }
    return {
        "per_rank": per_rank,
        "rank_max": rank_max,
        "rank_min": rank_min,
        "workspace_by_field_rank0": workspace_by_field if _RANK == 0 else {},
        "accounting": {
            "plotted_total": (
                "rank-max PyTorch peak_allocated_bytes plus the configured "
                "aclshmem heap"
            ),
            "workspace_ledger": (
                "diagnostic tensor views only; already represented by either "
                "the allocator peak or SHMEM heap"
            ),
        },
    }


def _collect_kernel_cycle_profile(layer: object) -> dict | None:
    """Read and gather the last backward profile from every EP rank."""
    if os.getenv("HP_MEGA_MOE_KERNEL_CYCLE_PROFILE", "0") != "1":
        return None
    resource_group = getattr(layer, "_resource_group", None)
    backend = getattr(resource_group, "resources", None)
    workspace = getattr(backend, "workspace", None)
    config = getattr(backend, "spec", None)
    if workspace is None or config is None:
        raise RuntimeError("kernel cycle profiling requires a MegaMoe backend.")
    local_profile = workspace.collect_cycle_profile(config.num_cube_cores)
    local_rows = (
        local_profile["cube"]["per_core"]
        + local_profile["vector"]["per_core"]
    )
    local_tensor = torch.tensor(
        local_rows,
        dtype=torch.int64,
        device=_DEVICE,
    )
    gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_tensor)
    categories = local_profile["categories"]
    num_cube_cores = config.num_cube_cores

    def _summarize(rows: list[list[int]]) -> dict:
        return {
            "per_core": rows,
            "sum": {
                category: sum(row[index] for row in rows)
                for index, category in enumerate(categories)
            },
            "max": {
                category: max(row[index] for row in rows)
                for index, category in enumerate(categories)
            },
        }

    per_rank = []
    for rank, rank_tensor in enumerate(gathered):
        rows = rank_tensor.cpu().tolist()
        per_rank.append(
            {
                "rank": rank,
                "cube": _summarize(rows[:num_cube_cores]),
                "vector": _summarize(rows[num_cube_cores:]),
            }
        )

    def _rank_max(core_type: str) -> dict[str, int]:
        return {
            category: max(
                rank_profile[core_type]["max"][category]
                for rank_profile in per_rank
            )
            for category in categories
        }

    return {
        "unit": local_profile["unit"],
        "categories": categories,
        "per_rank": per_rank,
        "rank_max": {
            "cube": _rank_max("cube"),
            "vector": _rank_max("vector"),
        },
        "accounting": local_profile["accounting"],
    }


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
    output = forward_layer(
        layer,
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
        ("gate/up weight", expert_weight_gradients(layer)[0]),
        ("down weight", expert_weight_gradients(layer)[1]),
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


def _write_performance_result(result: dict) -> None:
    """Write one fresh-process performance sample from rank zero."""
    if _RANK != 0:
        return
    result_path = os.getenv("HP_MEGA_MOE_PERF_RESULT")
    if not result_path:
        raise ValueError("HP_MEGA_MOE_PERF_RESULT must name the output JSON path.")
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PERF_RESULT_JSON={path}")
    print(json.dumps(result, sort_keys=True))


def _performance_receive_capacity(shape: _MoeShape) -> int:
    """Return the configured fixed receive window in token rows."""
    capacity_factor = float(os.getenv("HP_MEGA_MOE_CAPACITY_FACTOR", "1.5"))
    granularity = 128
    requested = math.ceil(shape.local_num_tokens * shape.top_k * capacity_factor)
    return (requested + granularity - 1) // granularity * granularity


def _performance_route_spec_from_environment() -> _PerformanceRouteSpec:
    """Return the validated route selector configured by the launcher."""
    profile = os.getenv("HP_MEGA_MOE_PERF_ROUTE_PROFILE", "dirichlet")
    if profile not in {"dirichlet", "clustered"}:
        raise ValueError(
            "HP_MEGA_MOE_PERF_ROUTE_PROFILE must be dirichlet or clustered, "
            f"got {profile!r}."
        )
    return _PerformanceRouteSpec(
        profile=profile,
        dirichlet_alpha=float(
            os.getenv("HP_MEGA_MOE_PERF_DIRICHLET_ALPHA", "10000")
        ),
        dirichlet_seed=int(os.getenv("HP_MEGA_MOE_PERF_DIRICHLET_SEED", "42")),
        clustered_experts=int(
            os.getenv("HP_MEGA_MOE_PERF_CLUSTERED_EXPERTS", "12")
        ),
    )


def _make_performance_route(
    shape: _MoeShape,
    route_spec: _PerformanceRouteSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Materialize one configured route and its reproducibility metadata."""
    if route_spec.profile == "dirichlet":
        topk_ids, topk_weights, tokens_per_expert = _make_dirichlet_route(
            shape,
            route_spec.dirichlet_alpha,
            route_spec.dirichlet_seed,
        )
        descriptor = {
            "profile": route_spec.profile,
            "dirichlet_alpha": route_spec.dirichlet_alpha,
            "dirichlet_seed": route_spec.dirichlet_seed,
        }
    else:
        topk_ids, topk_weights, tokens_per_expert = _make_clustered_route(
            shape,
            route_spec.clustered_experts,
        )
        descriptor = {
            "profile": route_spec.profile,
            "clustered_experts": route_spec.clustered_experts,
        }
    return topk_ids, topk_weights, tokens_per_expert, descriptor


def _classify_performance_route(
    replica_slots_per_rank: int,
    baseline_wave_count: int,
    planned_wave_count: int,
) -> str:
    """Name the home, replica, and residual multi-wave execution regime."""
    if replica_slots_per_rank == 0:
        return "home-only" if planned_wave_count == 1 else "multi-wave-only"
    if planned_wave_count > 1:
        return "replica+residual-multi-wave"
    if baseline_wave_count > 1:
        return "replica-only"
    return "replica-on-one-wave"


def _summarize_performance_route(
    shape: _MoeShape,
    tokens_per_expert: torch.Tensor,
    replica_slots_per_rank: int,
    descriptor: dict,
) -> dict:
    """Gather rank counts and predict the exact physical placement regime."""
    gathered_counts = torch.empty(
        shape.ep_size * shape.num_experts,
        dtype=torch.int32,
        device=_DEVICE,
    )
    dist.all_gather_into_tensor(gathered_counts, tokens_per_expert)
    logical_counts = tuple(
        tuple(int(count) for count in source_counts)
        for source_counts in gathered_counts.reshape(
            shape.ep_size,
            shape.num_experts,
        )
        .cpu()
        .tolist()
    )
    planner_capacity = shape.local_num_tokens * shape.top_k
    plan = build_constrained_expert_placement(
        logical_counts,
        replica_slots_per_rank,
        receive_capacity=planner_capacity,
    )
    receive_capacity = _performance_receive_capacity(shape)
    baseline_wave_count = plan.baseline_wave_count(receive_capacity)
    planned_wave_count = plan.wave_count(receive_capacity)
    expert_loads = [
        sum(source_counts[expert] for source_counts in logical_counts)
        for expert in range(shape.num_experts)
    ]
    baseline_loads = plan.baseline_destination_loads
    planned_loads = plan.destination_loads
    mean_destination_load = sum(baseline_loads) / len(baseline_loads)
    return {
        **descriptor,
        "expert_loads": expert_loads,
        "destination_loads": list(baseline_loads),
        "physical_destination_loads": list(planned_loads),
        "realized_destination_skew": max(baseline_loads) / mean_destination_load,
        "physical_destination_skew": max(planned_loads) / mean_destination_load,
        "receive_capacity": receive_capacity,
        "planner_target_capacity": planner_capacity,
        "baseline_wave_count": baseline_wave_count,
        "expected_wave_count": planned_wave_count,
        "replica_slots_per_rank": replica_slots_per_rank,
        "replica_slots_used": len(plan.replica_transfers),
        "replica_slot_bindings": [list(row) for row in plan.replica_expert_ids],
        "execution_mode": _classify_performance_route(
            replica_slots_per_rank,
            baseline_wave_count,
            planned_wave_count,
        ),
    }


def _build_performance_result(
    backend: str,
    shape: _MoeShape,
    route: dict,
    warmup_steps: int,
    latencies: list[float],
    memory: dict,
) -> dict:
    """Build one route-matched latency and HBM record."""
    return {
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
        "route": route,
        "warmup_steps": warmup_steps,
        "measured_steps": len(latencies),
        "validation": {"gradients_present": True, "gradients_finite": True},
        "rank_max_fwd_bwd_ms": {
            "median": statistics.median(latencies),
            "p90": _percentile(latencies, 0.9),
            "minimum": min(latencies),
            "maximum": max(latencies),
            "samples": latencies,
        },
        "memory": memory,
        "device_healthy": True,
    }


def _measure_performance_route(
    layer: MegaMoeExperts | torch.nn.Module,
    backend: str,
    shape: _MoeShape,
    route_spec: _PerformanceRouteSpec,
    replica_slots_per_rank: int,
    hidden_states: torch.Tensor,
    grad_output: torch.Tensor,
    warmup_steps: int,
    measured_steps: int,
) -> dict:
    """Measure one route profile while reusing a backend instance."""
    topk_ids, topk_weights, tokens_per_expert, route_descriptor = _make_performance_route(
        shape,
        route_spec,
    )
    route_summary = _summarize_performance_route(
        shape,
        tokens_per_expert,
        replica_slots_per_rank,
        route_descriptor,
    )

    _timed_fwd_bwd(
        layer,
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        grad_output,
        validate_gradients=True,
    )
    for _ in range(warmup_steps):
        _timed_fwd_bwd(
            layer,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        )
    layer.zero_grad(set_to_none=True)
    dist.barrier()
    torch.npu.synchronize(_DEVICE)
    baseline_memory = _device_memory_snapshot()
    torch.npu.reset_peak_memory_stats(_DEVICE)
    latencies = [
        _timed_fwd_bwd(
            layer,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        )
        for _ in range(measured_steps)
    ]
    memory = _collect_memory_result(layer, baseline_memory)
    result = _build_performance_result(
        backend,
        shape,
        route_summary,
        warmup_steps,
        latencies,
        memory,
    )
    cycle_profile = _collect_kernel_cycle_profile(layer)
    if cycle_profile is not None:
        result["kernel_cycle_profile"] = cycle_profile
    return result


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
    hidden_states, grad_output = _make_data(shape)
    replica_slots_per_rank = int(
        os.getenv("HP_MEGA_MOE_PERF_REPLICA_SLOTS", "0")
    )
    layer = _new_performance_layer(shape, backend, replica_slots_per_rank)
    route_spec = _performance_route_spec_from_environment()
    result = _measure_performance_route(
        layer,
        backend,
        shape,
        route_spec,
        replica_slots_per_rank,
        hidden_states,
        grad_output,
        warmup_steps,
        measured_steps,
    )
    _write_performance_result(result)


def test_mega_moe_dirichlet_performance_sweep() -> None:
    """Measure an alpha-by-seed sweep in one fresh backend process."""
    backend = os.getenv("HP_MEGA_MOE_PERF_BACKEND", "")
    warmup_steps = int(os.getenv("HP_MEGA_MOE_PERF_WARMUP", "2"))
    measured_steps = int(os.getenv("HP_MEGA_MOE_PERF_MEASURED", "8"))
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError(
            f"invalid performance steps: warmup={warmup_steps}, measured={measured_steps}."
        )
    alphas = [
        float(value)
        for value in os.getenv(
            "HP_MEGA_MOE_PERF_DIRICHLET_ALPHAS",
            "10000,3000,1000,300,100,30,10,3,1,0.5,0.2,0.1,0.05,0.02,0.01",
        ).split(",")
    ]
    seeds = [
        int(value)
        for value in os.getenv(
            "HP_MEGA_MOE_PERF_DIRICHLET_SEEDS",
            "7,42,83",
        ).split(",")
    ]
    points = [(alpha, seed) for alpha in alphas for seed in seeds]
    random.Random(20260903).shuffle(points)

    shape = _performance_shape()
    hidden_states, grad_output = _make_data(shape)
    replica_slots_per_rank = int(
        os.getenv("HP_MEGA_MOE_PERF_REPLICA_SLOTS", "0")
    )
    layer = _new_performance_layer(shape, backend, replica_slots_per_rank)
    results = []
    for alpha, seed in points:
        route_spec = _PerformanceRouteSpec(
            profile="dirichlet",
            dirichlet_alpha=alpha,
            dirichlet_seed=seed,
        )
        result = _measure_performance_route(
            layer,
            backend,
            shape,
            route_spec,
            replica_slots_per_rank,
            hidden_states,
            grad_output,
            warmup_steps,
            measured_steps,
        )
        if _RANK == 0:
            results.append(result)
            print(
                "DIRICHLET_SWEEP_POINT=" + json.dumps(result, sort_keys=True),
                flush=True,
            )

    if _RANK != 0:
        return
    result_path = os.getenv("HP_MEGA_MOE_PERF_RESULT")
    if not result_path:
        raise ValueError("HP_MEGA_MOE_PERF_RESULT must name the output JSON path.")
    sweep_result = {
        "backend": backend,
        "alphas": alphas,
        "seeds": seeds,
        "shuffle_seed": 20260903,
        "points": results,
    }
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sweep_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"DIRICHLET_SWEEP_RESULT_JSON={path}", flush=True)
