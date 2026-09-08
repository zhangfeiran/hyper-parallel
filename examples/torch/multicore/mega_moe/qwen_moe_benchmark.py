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

"""Minimal eight-NPU Qwen MoE optimizer-step benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from collections.abc import Iterator
from pathlib import Path
from typing import Any

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# The launcher activates CANN and the native payload before framework import.
# pylint: disable=wrong-import-position
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import torch_npu
import hyper_parallel
from examples.torch.multicore.mega_moe.qwen_moe_config import (
    QwenMoeConfig, make_config, optimizer_contract, parse_args,
)
from examples.torch.multicore.mega_moe.qwen_moe_model import QwenMoeModel, build_qwen_moe_model
from examples.torch.multicore.mega_moe.qwen_moe_snapshots import compare_snapshot, save_snapshot

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.components.optim import (
    Float16OptimizerWithFloat16Params,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.platform.torch.symmetric_memory import lifecycle

# pylint: enable=wrong-import-position

_WORLD_SIZE = 8
_BATCH_SIZE = 1
_RTOL = 2e-2
_ATOL = 2e-3
_SNAPSHOT_PHASES = ("inputs", "initial_parameters", "forward", "gradients", "updated_parameters")


@dataclass(frozen=True)
class _Workload:
    """Objects needed by one complete optimizer step."""

    model: QwenMoeModel
    optimizer: Float16OptimizerWithFloat16Params
    input_ids: torch.Tensor
    labels: torch.Tensor
    world_size: int
    device: torch.device


class _AccuracySnapshots:
    """Coordinate rank-local CPU references without retaining a second device model."""

    def __init__(self, args: argparse.Namespace, config: QwenMoeConfig, rank: int, world_size: int) -> None:
        """Bind snapshot ownership to the complete workload and global rank."""
        self.reference = args.reference_dir is not None
        self.directory = Path(args.reference_dir or args.snapshot_dir) / f"rank{rank}"
        self.contract = {
            "configuration": asdict(config), "rank": rank, "world_size": world_size, "seed": args.seed,
            "optimizer": optimizer_contract(args),
        }
        self.world_size = world_size
        self.results: dict[str, Any] = {}

    def phase(self, name: str, tensors: Iterator[tuple[str, torch.Tensor]]) -> None:
        """Save or compare one phase, propagating local I/O errors to every rank."""
        local = {}
        try:
            if name not in _SNAPSHOT_PHASES or name in self.results:
                raise ValueError(f"unexpected or repeated Qwen snapshot phase: {name}")
            contract = {**self.contract, "phase": name}
            path = self.directory / f"{name}.pt"
            if self.reference:
                manifest = json.loads((self.directory / "complete.json").read_text(encoding="utf-8"))
                if manifest != {"contract": self.contract, "phases": list(_SNAPSHOT_PHASES)}:
                    raise ValueError("Qwen reference is incomplete or belongs to a different workload.")
                exact = name in ("inputs", "initial_parameters")
                local["result"] = compare_snapshot(path, contract, tensors,
                                                   rtol=0.0 if exact else _RTOL, atol=0.0 if exact else _ATOL)
            else:
                local["result"] = save_snapshot(path, contract, tensors)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            local["error"] = f"{type(error).__name__}: {error}"
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local)
        errors = {rank: value["error"] for rank, value in enumerate(gathered) if "error" in value}
        if errors:
            raise RuntimeError(f"Qwen {name} snapshot errors by rank: {errors}")
        results = [value["result"] for value in gathered]
        result = dict(results[0])
        if any(value["tensor_count"] != result["tensor_count"] for value in results):
            raise RuntimeError(f"Qwen {name} tensor counts differ across ranks: {results}")
        if self.reference:
            result["passed"] = all(value["passed"] for value in results)
            for metric in ("absolute", "normalized"):
                worst = max(results, key=lambda value: value[f"max_{metric}_error"])
                for key in (f"max_{metric}_error", f"max_{metric}_error_tensor"):
                    result[key] = worst[key]
            if not result["passed"]:
                raise RuntimeError(f"Qwen isolated {name} comparison failed: {result}")
        self.results[name] = result

    def finish(self) -> None:
        """Publish completion only after every snapshot and optimizer step succeeded."""
        local_error = None
        try:
            if set(self.results) != set(_SNAPSHOT_PHASES):
                raise ValueError(f"Qwen snapshot phases incomplete: {sorted(self.results)}")
            if not self.reference:
                with (self.directory / "complete.json").open("x", encoding="utf-8") as stream:
                    json.dump({"contract": self.contract, "phases": list(_SNAPSHOT_PHASES)}, stream, sort_keys=True)
        except (OSError, ValueError) as error:
            local_error = str(error)
        errors = [None] * self.world_size
        dist.all_gather_object(errors, local_error)
        if any(error is not None for error in errors):
            raise RuntimeError(f"Qwen snapshot completion failed: {errors}")


def _init_runtime() -> tuple[int, int, torch.device]:
    """Bind the local NPU and initialize the fixed EP world."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != _WORLD_SIZE:
        raise ValueError(
            f"Qwen benchmark requires {_WORLD_SIZE} ranks, got {world_size}."
        )
    return rank, world_size, torch.device("npu", local_rank)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local value of a plain tensor or DTensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _reset_seed(seed: int) -> None:
    """Reset CPU and NPU initializers before constructing one backend."""
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)


def _build_backend_model(
    config: QwenMoeConfig,
    device: torch.device,
    seed: int,
    backend: str,
) -> QwenMoeModel:
    """Select reproducible initialization, leaving backend assembly to the model."""
    _reset_seed(seed)
    return build_qwen_moe_model(config, device, backend=backend, ep_group=dist.group.WORLD)


def _build_workload(
    model: QwenMoeModel,
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    world_size: int,
    device: torch.device,
) -> _Workload:
    """Attach an independent FP32-main-parameter AdamW optimizer."""
    chained_optimizer = get_hyper_optimizer(
        model=model,
        muon_params=[],
        adamw_params=[
            {
                "params": list(model.parameters()),
                "weight_decay": args.weight_decay,
            }
        ],
        adamw_kwargs={
            "lr": args.learning_rate,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
    )
    return _Workload(
        model=model,
        optimizer=Float16OptimizerWithFloat16Params(
            chained_optimizer,
            model,
        ),
        input_ids=input_ids,
        labels=labels,
        world_size=world_size,
        device=device,
    )


def _build_batch(
    config: QwenMoeConfig,
    rank: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic rank-local token IDs and labels."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10_000 + rank)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (_BATCH_SIZE, config.local_num_tokens // _BATCH_SIZE),
        generator=generator,
        dtype=torch.int64,
    ).to(device)
    return input_ids, input_ids.clone()


def _dense_gradients(model: QwenMoeModel) -> list[torch.Tensor]:
    """Return gradients of replicated parameters, excluding expert shards."""
    expert_parameters = {id(parameter) for parameter in model.expert_parameters()}
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if id(parameter) not in expert_parameters and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("Qwen optimizer step produced no dense gradients.")
    return gradients


def _synchronize_dense_gradients(model: QwenMoeModel, world_size: int) -> None:
    """Average replicated dense gradients before the optimizer step."""
    gradients = _dense_gradients(model)
    dist.all_reduce_coalesced(gradients, op=dist.ReduceOp.SUM)
    torch._foreach_div_(gradients, world_size)  # pylint: disable=protected-access


def _optimizer_step(
    workload: _Workload,
    *,
    capture_gradients: bool = False,
    gradient_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    """Run one AdamW step and optionally retain first-step gradients."""
    with SkipDTensorDispatch():
        workload.optimizer.zero_grad(set_to_none=True)
    result = workload.model(workload.input_ids, workload.labels)
    loss = result["loss"]
    logits = result["logits"]
    if loss is None:
        raise RuntimeError("Qwen model did not return a training loss.")
    if logits is None:
        raise RuntimeError("Qwen model did not return logits.")
    loss.backward()
    _synchronize_dense_gradients(workload.model, workload.world_size)
    gradients = None
    if capture_gradients:
        gradients = {
            name: tensor.detach().to(device=gradient_device, copy=True) if gradient_device is not None
            else tensor.detach().clone()
            for name, tensor in _iter_canonical_tensors(
                workload.model,
                gradients=True,
            )
        }
    with SkipDTensorDispatch():
        workload.optimizer.step()
    return loss.detach(), logits.detach(), gradients


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    """Reduce a host predicate across all benchmark ranks."""
    status = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    return bool(status.cpu().item())


def _all_gradients_finite(
    gradients: dict[str, torch.Tensor],
    device: torch.device,
) -> bool:
    """Return whether every captured model gradient is finite."""
    finite = torch.ones((), dtype=torch.int32, device=device)
    for gradient in gradients.values():
        finite = torch.minimum(
            finite,
            torch.isfinite(gradient).all().to(device=device, dtype=torch.int32),
        )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.cpu().item())


def _rank_max(value: float, device: torch.device) -> float:
    """Return a floating-point measurement reduced by maximum across ranks."""
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu().item())


def _timed_step(workload: _Workload) -> tuple[torch.Tensor, float]:
    """Run one optimizer step from a synchronized, untimed boundary."""
    dist.barrier()
    torch.npu.synchronize(workload.device)
    start = time.perf_counter()
    loss, _, _ = _optimizer_step(workload)
    torch.npu.synchronize(workload.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return loss, _rank_max(elapsed_ms, workload.device)


def _expert_update_sample(parameter: torch.Tensor) -> torch.Tensor:
    """Sample every local expert, not only the possibly cold first expert."""
    local = _local_tensor(parameter).detach()
    return local.reshape(local.shape[0], -1)[:, :4096]


def _expert_parameters_updated(
    probes: list[tuple[torch.Tensor, torch.Tensor]],
    gradients: dict[str, torch.Tensor],
    device: torch.device,
) -> bool:
    """Require updates on active owners and at least one update globally.

    This bounded first-step smoke check supplements the full tensor accuracy
    comparison. Cold owners may stay unchanged under skewed routing.
    """
    updated = any(
        not torch.equal(before, _expert_update_sample(parameter))
        for parameter, before in probes
    )
    active = any(
        bool(torch.count_nonzero(gradient).cpu().item())
        for name, gradient in gradients.items() if name.startswith("experts.")
    )
    active_owners_updated = _all_ranks_true(not active or updated, device)
    any_owner_updated = _rank_max(float(updated), device) > 0
    return active_owners_updated and any_owner_updated


def _validate_first_step(
    workload: _Workload,
    *,
    gradient_device: torch.device | None = None,
) -> tuple[
    dict[str, Any],
    float,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Validate finite work and dense/expert updates on the first step."""
    dense_parameter = workload.model.lm_head.weight
    dense_before = dense_parameter.detach().reshape(-1)[:4096].clone()
    expert_probes = [
        (parameter, _expert_update_sample(parameter).clone())
        for parameter in workload.model.expert_parameters()
    ]
    dist.barrier()
    torch.npu.synchronize(workload.device)
    start = time.perf_counter()
    loss, logits, gradients = _optimizer_step(
        workload,
        capture_gradients=True,
        gradient_device=gradient_device,
    )
    torch.npu.synchronize(workload.device)
    if gradients is None:
        raise RuntimeError("Qwen first optimizer step did not capture gradients.")
    first_step_ms = _rank_max(
        (time.perf_counter() - start) * 1000.0,
        workload.device,
    )
    validation = {
        "loss": float(loss.float().cpu().item()),
        "loss_finite": _all_ranks_true(
            bool(torch.isfinite(loss).cpu().item()),
            workload.device,
        ),
        "gradients_finite": _all_gradients_finite(gradients, workload.device),
        "dense_parameter_updated": _all_ranks_true(
            not torch.equal(dense_before, dense_parameter.detach().reshape(-1)[:4096]),
            workload.device,
        ),
        "expert_parameter_updated": _expert_parameters_updated(
            expert_probes,
            gradients,
            workload.device,
        ),
    }
    if not all(value for name, value in validation.items() if name != "loss"):
        raise RuntimeError(f"optimizer-step validation failed: {validation}.")
    return validation, first_step_ms, loss, logits, gradients


def _iter_canonical_tensors(
    model: QwenMoeModel,
    *,
    gradients: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Read the model-owned canonical interface without knowing backend layouts."""
    return model.named_canonical_tensors(gradients=gradients)


def _canonical_tensors(model: QwenMoeModel, *, gradients: bool) -> dict[str, torch.Tensor]:
    """Retain the original paired-backend oracle's canonical tensor interface."""
    return dict(_iter_canonical_tensors(model, gradients=gradients))


def _compare_tensor_maps(
    name: str,
    common_tensors: dict[str, torch.Tensor],
    mega_tensors: dict[str, torch.Tensor],
    device: torch.device,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare tensor maps on device and reduce diagnostics across ranks."""
    if common_tensors.keys() != mega_tensors.keys():
        raise RuntimeError(
            f"{name} tensor names differ: common={sorted(common_tensors)}, "
            f"mega={sorted(mega_tensors)}."
        )
    passed = torch.ones((), dtype=torch.int32, device=device)
    tensor_names = []
    absolute_errors = []
    normalized_errors = []
    for tensor_name, common_tensor in common_tensors.items():
        tensor_names.append(tensor_name)
        mega_tensor = mega_tensors[tensor_name]
        if mega_tensor.shape != common_tensor.shape:
            raise RuntimeError(
                f"{name}.{tensor_name} shape differs: common={tuple(common_tensor.shape)}, "
                f"mega={tuple(mega_tensor.shape)}."
            )
        common_float = common_tensor.detach().float()
        mega_float = mega_tensor.detach().float()
        difference = (mega_float - common_float).abs()
        passed = torch.minimum(
            passed,
            torch.isclose(
                mega_float,
                common_float,
                rtol=rtol,
                atol=atol,
            )
            .all()
            .to(torch.int32),
        )
        absolute_errors.append(difference.max())
        tolerance = atol + rtol * common_float.abs()
        normalized_errors.append(
            (difference / tolerance.clamp_min(torch.finfo(torch.float32).tiny)).max()
        )
    absolute_by_tensor = torch.stack(absolute_errors)
    normalized_by_tensor = torch.stack(normalized_errors)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    dist.all_reduce(absolute_by_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(normalized_by_tensor, op=dist.ReduceOp.MAX)
    maximum_absolute, absolute_index = absolute_by_tensor.max(dim=0)
    maximum_normalized, normalized_index = normalized_by_tensor.max(dim=0)
    result = {
        "passed": bool(passed.cpu().item()),
        "tensor_count": len(common_tensors),
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": float(maximum_absolute.cpu().item()),
        "max_absolute_error_tensor": tensor_names[int(absolute_index.cpu().item())],
        "max_normalized_error": float(maximum_normalized.cpu().item()),
        "max_normalized_error_tensor": tensor_names[int(normalized_index.cpu().item())],
    }
    if not result["passed"]:
        raise RuntimeError(f"Qwen common/MegaMoe {name} comparison failed: {result}.")
    return result


def _symmetric_heap_bytes() -> int:
    """Read the external SHMEM allocation, which Torch allocator statistics exclude."""
    return lifecycle._PROCESS_STATE.heap_size or 0  # pylint: disable=protected-access


def _measure_backend(
    workload: _Workload,
    validation: dict[str, Any],
    first_step_ms: float,
    args: argparse.Namespace,
    *,
    collect_memory: bool = False,
) -> dict[str, Any]:
    """Finish warmup and measure one backend after its compared first step."""
    for _ in range(args.warmup_steps - 1):
        _timed_step(workload)
    if collect_memory:
        torch.npu.synchronize(workload.device)
        torch.npu.reset_peak_memory_stats(workload.device)
    latencies = []
    losses = []
    for _ in range(args.measured_steps):
        loss, latency = _timed_step(workload)
        latencies.append(latency)
        losses.append(float(loss.float().cpu().item()))
    result = {
        "validation": validation,
        "first_optimizer_step_ms": first_step_ms,
        "steady_state_optimizer_step_ms": {
            "median": statistics.median(latencies),
            "minimum": min(latencies),
            "maximum": max(latencies),
            "samples": latencies,
        },
        "measured_losses": losses,
    }
    if collect_memory:
        torch.npu.synchronize(workload.device)
        free_bytes, total_bytes = torch.npu.mem_get_info(workload.device)
        local_memory = {
            "peak_allocated_bytes": torch.npu.max_memory_allocated(workload.device),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(workload.device),
            "final_allocated_bytes": torch.npu.memory_allocated(workload.device),
            "final_reserved_bytes": torch.npu.memory_reserved(workload.device),
            "symmetric_heap_bytes": _symmetric_heap_bytes(),
            "final_device_used_bytes": total_bytes - free_bytes,
            "final_device_total_bytes": total_bytes,
        }
        # B pools are already in the allocator, and workspace views are already
        # in SHMEM. Do not count either twice. This excludes opaque driver/HCCL
        # allocations; the device-wide final sample is not a runtime peak.
        local_memory["conservative_total_peak_bytes"] = (
            local_memory["peak_allocated_bytes"] + local_memory["symmetric_heap_bytes"]
        )
        per_rank = [None] * workload.world_size
        dist.all_gather_object(per_rank, local_memory)
        result["steady_state_memory_per_rank"] = per_rank
    return result


def _shared_pool_ledger(model: QwenMoeModel) -> dict[str, int]:
    """Inspect the actual shared pool after all accuracy and timing windows."""
    # This diagnostic is confined to the example, not the library hot path.
    # pylint: disable=protected-access
    groups = [layer.mlp.experts._resource_group for layer in model.layers]
    if any(group is not groups[0] for group in groups):
        raise RuntimeError("Qwen decoder layers did not share their execution resources.")
    resources = groups[0].resources
    manager = resources.replica_manager if resources is not None else None
    weights = manager._shared_weights if manager is not None else None
    gradients = manager._shared_gradients if manager is not None else None
    return {
        "shared_weight_bytes": sum(tensor.numel() * tensor.element_size() for tensor in (weights or ())),
        "shared_gradient_bytes": sum(tensor.numel() * tensor.element_size() for tensor in (gradients or ())),
        "resource_group_count": 1,
    }


def _write_result(
    args: argparse.Namespace,
    config: QwenMoeConfig,
    rank: int,
    accuracy: dict[str, Any],
    backends: dict[str, dict[str, Any]],
    shared_pools: list[dict[str, int]],
) -> None:
    """Write common-versus-MegaMoe accuracy and performance from rank zero."""
    common_median = backends["common"]["steady_state_optimizer_step_ms"]["median"]
    mega_median = backends["mega_moe"]["steady_state_optimizer_step_ms"]["median"]
    ratio = mega_median / common_median
    if rank == 0:
        result = {
            "configuration": asdict(config),
            "initialization_seed": args.seed,
            "shared_pool_per_rank": shared_pools,
            "topology": {"world_size": _WORLD_SIZE, "tp": 1, "ep": config.ep_size},
            "shape": {
                "batch_size": _BATCH_SIZE,
                "sequence_length": config.local_num_tokens // _BATCH_SIZE,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "num_experts": config.num_experts,
                "top_k": config.top_k,
                "expert_capacity_factor": config.expert_capacity_factor,
                "dtype": "bfloat16",
            },
            "optimizer": {
                "name": "HyperParallel AdamW",
                "wrapper": "Float16OptimizerWithFloat16Params",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "model_parameter_dtype": "bfloat16",
                "main_parameter_dtype": "float32",
            },
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "comparison": {
                "accuracy": accuracy,
                "performance": {
                    "order": ["common", "mega_moe"],
                    "metric": "rank-max complete optimizer-step milliseconds",
                    "common_median_ms": common_median,
                    "mega_moe_median_ms": mega_median,
                    "mega_over_common": ratio,
                    "mega_reduction_percent": (1.0 - ratio) * 100.0,
                },
            },
            "backends": backends,
            "runtime": {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
            },
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"RESULT_JSON={path}")
        print(json.dumps(result, sort_keys=True))
    dist.barrier()


def _compare_first_step_accuracy(
    models: dict[str, QwenMoeModel],
    workloads: dict[str, _Workload],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], float]]]:
    """Compare initial state, first forward/backward, and optimizer updates."""
    accuracy = {
        "initial_parameters": _compare_tensor_maps(
            "initial parameters",
            _canonical_tensors(models["common"], gradients=False),
            _canonical_tensors(models["mega_moe"], gradients=False),
            device,
            rtol=0.0,
            atol=0.0,
        )
    }
    (
        common_validation,
        common_first_ms,
        common_loss,
        common_logits,
        common_gradients,
    ) = _validate_first_step(workloads["common"])
    (
        mega_validation,
        mega_first_ms,
        mega_loss,
        mega_logits,
        mega_gradients,
    ) = _validate_first_step(workloads["mega_moe"])
    accuracy.update(
        {
            "forward": _compare_tensor_maps(
                "forward",
                {"loss": common_loss, "logits": common_logits},
                {"loss": mega_loss, "logits": mega_logits},
                device,
                rtol=_RTOL,
                atol=_ATOL,
            ),
            "gradients": _compare_tensor_maps(
                "gradients",
                common_gradients,
                mega_gradients,
                device,
                rtol=_RTOL,
                atol=_ATOL,
            ),
            "updated_parameters": _compare_tensor_maps(
                "updated parameters",
                _canonical_tensors(models["common"], gradients=False),
                _canonical_tensors(models["mega_moe"], gradients=False),
                device,
                rtol=_RTOL,
                atol=_ATOL,
            ),
        }
    )
    del common_gradients, mega_gradients
    del common_loss, mega_loss, common_logits, mega_logits
    torch.npu.empty_cache()
    first_steps = {
        "common": (common_validation, common_first_ms),
        "mega_moe": (mega_validation, mega_first_ms),
    }
    return accuracy, first_steps


def _runtime_identity() -> dict[str, Any]:
    """Record local source/artifact identity outside every timing window."""
    repo = Path(__file__).resolve().parents[4]
    if Path(hyper_parallel.__file__).resolve() != repo / "hyper_parallel/__init__.py":
        raise RuntimeError("Qwen worker must import hyper_parallel from this checkout; reinstall with pip -e .")
    sources = list(Path(__file__).parent.glob("*.py"))
    sources.extend((repo / "hyper_parallel").rglob("*.py"))
    artifacts = [path for path in (repo / "build/native/payload/hyper_parallel").rglob("*")
                 if path.is_file() and path.suffix in (".so", ".o")]
    return {
        "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "source_scope": "qwen-example-and-hyper_parallel-python-v1",
        "native_scope": "build/native/payload/hyper_parallel/**/*.{so,o}",
        "source_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(sources)},
        "native_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(artifacts)},
    }


def _resident_training_ledger(workload: _Workload) -> dict[str, int]:
    """Count the one resident backend's local model and optimizer tensors after timing."""
    def tensor_bytes(tensor: torch.Tensor) -> int:
        """Count local storage elements without allocating diagnostic tensors."""
        value = _local_tensor(tensor)
        return value.numel() * value.element_size()

    parameters = list(workload.model.parameters())
    main_parameters = [parameter for group in workload.optimizer.param_groups for parameter in group["params"]]
    state = [value for optimizer in workload.optimizer.chained_optimizers
             for values in optimizer.state.values() for value in values.values() if isinstance(value, torch.Tensor)]
    return {
        "resident_backend_count": 1,
        "local_parameter_count": sum(_local_tensor(parameter).numel() for parameter in parameters),
        "model_bytes": sum(map(tensor_bytes, parameters)),
        "main_parameter_bytes": sum(map(tensor_bytes, main_parameters)),
        "main_gradient_bytes": sum(tensor_bytes(parameter.grad) for parameter in main_parameters
                                   if parameter.grad is not None),
        "optimizer_state_bytes": sum(map(tensor_bytes, state)),
    }


def _run_single_backend(
    args: argparse.Namespace, config: QwenMoeConfig, rank: int, world_size: int, device: torch.device,
) -> int:
    """Validate and time exactly one backend, keeping accuracy snapshots on CPU."""
    identity = _runtime_identity()
    snapshots = _AccuracySnapshots(args, config, rank, world_size)
    input_ids, labels = _build_batch(config, rank, args.seed, device)
    snapshots.phase("inputs", iter((("input_ids", input_ids), ("labels", labels))))
    model = _build_backend_model(config, device, args.seed, args.backend)
    try:
        snapshots.phase("initial_parameters", _iter_canonical_tensors(model, gradients=False))
        workload = _build_workload(model, args, input_ids, labels, world_size, device)
        validation, first_ms, loss, logits, gradients = _validate_first_step(
            workload, gradient_device=torch.device("cpu"),
        )
        snapshots.phase("forward", iter((("loss", loss), ("logits", logits))))
        snapshots.phase("gradients", iter(gradients.items()))
        snapshots.phase("updated_parameters", _iter_canonical_tensors(model, gradients=False))
        del gradients, loss, logits
        torch.npu.synchronize(device)
        torch.npu.empty_cache()
        measurement = _measure_backend(workload, validation, first_ms, args, collect_memory=True)
        snapshots.finish()
        ledger = [None] * world_size
        dist.all_gather_object(ledger, _resident_training_ledger(workload))
        pools = [None] * world_size
        dist.all_gather_object(pools, _shared_pool_ledger(model) if args.backend == "mega_moe" else None)
        if not _all_ranks_true(identity == _runtime_identity(), device):
            raise RuntimeError("Qwen source/artifact identity changed during the run.")
        identities = [None] * world_size
        dist.all_gather_object(identities, identity)
        if any(value != identity for value in identities):
            raise RuntimeError("Qwen source/artifact identity differs across ranks.")
        # Every node's framework-free parent validates its own local result file.
        if int(os.environ.get("LOCAL_RANK", str(rank))) == 0:
            result = {
                "mode": "isolated_backend", "backend": args.backend, "configuration": asdict(config),
                "initialization_seed": args.seed, "world_size": world_size,
                "warmup_steps": args.warmup_steps, "measured_steps": args.measured_steps,
                "optimizer": snapshots.contract["optimizer"],
                "accuracy": {"mode": "compare" if snapshots.reference else "export",
                             "phases": snapshots.results, "reference_dir": args.reference_dir,
                             "snapshot_dir": args.snapshot_dir},
                "measurement": measurement, "resident_training_per_rank": ledger,
                "shared_pool_per_rank": pools, "runtime": identity,
            }
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"RESULT_JSON={path}", flush=True)
        dist.barrier()
        return 0
    finally:
        model.close()


def main(argv: list[str] | None = None) -> int:
    """Compare common and MegaMoe Qwen accuracy, then time A and B.

    Args:
        argv: Optional command-line arguments for tests or direct invocation.

    Returns:
        Zero after the distributed comparison completes successfully.
    """
    args = parse_args(argv)
    rank, world_size, device = _init_runtime()
    config = make_config(args)
    models: dict[str, QwenMoeModel] = {}
    try:
        if args.backend != "both":
            return _run_single_backend(args, config, rank, world_size, device)
        input_ids, labels = _build_batch(config, rank, args.seed, device)
        for backend in ("common", "mega_moe"):
            models[backend] = _build_backend_model(
                config,
                device,
                args.seed,
                backend,
            )
        workloads = {
            backend: _build_workload(
                model,
                args,
                input_ids,
                labels,
                world_size,
                device,
            )
            for backend, model in models.items()
        }
        accuracy, first_steps = _compare_first_step_accuracy(
            models,
            workloads,
            device,
        )
        common_validation, common_first_ms = first_steps["common"]
        mega_validation, mega_first_ms = first_steps["mega_moe"]
        backends = {
            "common": _measure_backend(
                workloads["common"],
                common_validation,
                common_first_ms,
                args,
            ),
            "mega_moe": _measure_backend(
                workloads["mega_moe"],
                mega_validation,
                mega_first_ms,
                args,
            ),
        }
        shared_pools = [None] * world_size
        dist.all_gather_object(shared_pools, _shared_pool_ledger(models["mega_moe"]))
        _write_result(
            args,
            config,
            rank,
            accuracy,
            backends,
            shared_pools,
        )
        return 0
    finally:
        for model in models.values():
            model.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
