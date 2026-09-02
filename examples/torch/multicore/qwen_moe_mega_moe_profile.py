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

"""Eight-NPU Qwen MoE optimizer-step validation and benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# The launcher activates CANN and the native payload before framework import.
# pylint: disable=wrong-import-position
# This is a Torch-only executable example, not a platform-agnostic library module.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import torch_npu

from qwen_moe_mega_moe_model import MoeBackend, QwenMoeConfig, QwenMoeModel

# pylint: enable=wrong-import-position

_WORLD_SIZE = 8
_BATCH_SIZE = 1
_SEQUENCE_LENGTH = 1024
_DTYPE = torch.bfloat16
_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.01


@dataclass(frozen=True)
class _OptimizerWorkload:
    """Objects required to execute one distributed optimizer step."""

    model: QwenMoeModel
    optimizer: torch.optim.Optimizer
    input_ids: torch.Tensor
    labels: torch.Tensor
    world_size: int
    device: torch.device


@dataclass(frozen=True)
class _BenchmarkOutcome:
    """Validated timing and memory evidence written by the benchmark."""

    validation: dict[str, Any]
    latencies: list[float]
    local_memory: dict[str, dict[str, int]]
    rank_max_memory: dict[str, dict[str, int]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the fixed optimizer-benchmark interface.

    Args:
        argv: Optional argument list for host-side validation.

    Returns:
        Parsed benchmark arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run one fixed eight-NPU, eight-layer Qwen MoE optimizer workload.",
    )
    parser.add_argument(
        "--backend",
        choices=("mega_moe", "native"),
        required=True,
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/qwen_moe_optimizer")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)
    if args.warmup_steps < 2:
        raise ValueError(
            "warmup_steps must be at least two so first-step and stable-step "
            f"memory can be sampled, got {args.warmup_steps}."
        )
    if args.measured_steps <= 0:
        raise ValueError(f"measured_steps must be positive, got {args.measured_steps}.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be finite and positive, got {args.learning_rate}."
        )
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError(
            f"weight_decay must be finite and nonnegative, got {args.weight_decay}."
        )
    return args


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
            f"Qwen optimizer benchmark requires {_WORLD_SIZE} ranks, got {world_size}."
        )
    return rank, world_size, torch.device("npu", local_rank)


def _set_model_seed(seed: int) -> None:
    """Initialize identical replicated parameters on every rank."""
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)


def _build_model(
    config: QwenMoeConfig,
    backend: MoeBackend,
    device: torch.device,
) -> QwenMoeModel:
    """Build BF16 weights while preserving FP32 rotary-frequency buffers."""
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(_DTYPE)
        model = QwenMoeModel(
            config,
            backend=backend,
            ep_group=None,
        )
    finally:
        torch.set_default_dtype(default_dtype)
    return model.to(device=device)


def _build_batch(
    config: QwenMoeConfig,
    rank: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one deterministic rank-local token batch."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10_000 + rank)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (_BATCH_SIZE, _SEQUENCE_LENGTH),
        generator=generator,
        dtype=torch.int64,
    ).to(device)
    return input_ids, input_ids.clone()


def _dense_gradients(model: QwenMoeModel) -> list[torch.Tensor]:
    """Return replicated dense gradients, excluding local expert weights."""
    expert_parameters = {id(parameter) for parameter in model.expert_parameters()}
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if id(parameter) not in expert_parameters and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("Qwen optimizer step produced no dense gradients.")
    return gradients


def _synchronize_dense_gradients(
    model: QwenMoeModel,
    world_size: int,
) -> None:
    """Average all replicated dense gradients in one coalesced collective."""
    gradients = _dense_gradients(model)
    dist.all_reduce_coalesced(gradients, op=dist.ReduceOp.SUM)
    # PyTorch exposes foreach arithmetic with leading-underscore names.
    torch._foreach_div_(gradients, world_size)  # pylint: disable=protected-access


def _optimizer_step(
    workload: _OptimizerWorkload,
) -> torch.Tensor:
    """Run the complete operation measured by this benchmark."""
    workload.optimizer.zero_grad(set_to_none=True)
    result = workload.model(workload.input_ids, workload.labels)
    loss = result["loss"]
    if loss is None:
        raise RuntimeError("Qwen model did not return a training loss.")
    loss.backward()
    _synchronize_dense_gradients(workload.model, workload.world_size)
    workload.optimizer.step()
    return loss.detach()


def _all_gradients_finite(model: QwenMoeModel, device: torch.device) -> bool:
    """Return a world-wide finite and present gradient predicate."""
    finite = torch.ones((), dtype=torch.int32, device=device)
    for parameter in model.parameters():
        if parameter.grad is None:
            finite.zero_()
            break
        finite = torch.minimum(
            finite,
            torch.isfinite(parameter.grad).all().to(torch.int32),
        )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.cpu().item())


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    """Reduce a host predicate across the benchmark world."""
    status = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    return bool(status.cpu().item())


def _validation_parameters(
    model: QwenMoeModel,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Return dense and expert parameters with BF16-observable updates."""
    return model.lm_head.weight, model.expert_parameters()[0]


def _validate_first_step(
    workload: _OptimizerWorkload,
) -> dict[str, Any]:
    """Run the first warmup and validate loss, gradients, and updates."""
    dense_parameter, expert_parameter = _validation_parameters(workload.model)
    dense_before = dense_parameter.detach().reshape(-1)[:256].clone()
    expert_before = expert_parameter.detach().reshape(-1)[:256].clone()
    loss = _optimizer_step(workload)
    torch.npu.synchronize(workload.device)
    loss_finite = _all_ranks_true(
        bool(torch.isfinite(loss).cpu().item()),
        workload.device,
    )
    gradients_finite = _all_gradients_finite(workload.model, workload.device)
    dense_updated = _all_ranks_true(
        not torch.equal(
            dense_before,
            dense_parameter.detach().reshape(-1)[:256],
        ),
        workload.device,
    )
    expert_updated = _all_ranks_true(
        not torch.equal(
            expert_before,
            expert_parameter.detach().reshape(-1)[:256],
        ),
        workload.device,
    )
    validation = {
        "loss": float(loss.float().cpu().item()),
        "loss_finite": loss_finite,
        "gradients_finite": gradients_finite,
        "dense_parameter_updated": dense_updated,
        "expert_parameter_updated": expert_updated,
    }
    required = (
        "loss_finite",
        "gradients_finite",
        "dense_parameter_updated",
        "expert_parameter_updated",
    )
    if not all(validation[name] for name in required):
        raise RuntimeError(f"optimizer-step validation failed: {validation}.")
    return validation


class _DeviceHbmSampler:  # pylint: disable=too-few-public-methods
    """Poll total-minus-free device HBM without using allocator counters."""

    def __init__(self, device: torch.device) -> None:
        """Initialize sampling state for one local NPU."""
        self._device = device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_used_bytes = 0
        self._error: Exception | None = None

    @staticmethod
    def _used_bytes(device: torch.device) -> int:
        """Read process-visible device HBM usage from the NPU runtime."""
        free_bytes, total_bytes = torch.npu.mem_get_info(device)
        return int(total_bytes - free_bytes)

    def _sample(self) -> None:
        """Update the sampled peak once."""
        self._peak_used_bytes = max(
            self._peak_used_bytes,
            self._used_bytes(self._device),
        )

    def _run(self) -> None:
        """Poll until the owner stops the measurement."""
        try:
            torch.npu.set_device(self._device)
            while not self._stop.is_set():
                self._sample()
                self._stop.wait(_MEMORY_SAMPLE_INTERVAL_SECONDS)
        except Exception as error:  # pylint: disable=broad-exception-caught
            self._error = error
            self._stop.set()

    def start(self) -> None:
        """Start sampling after capturing the phase boundary."""
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="qwen-device-hbm-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> int:
        """Stop sampling and return peak total-minus-free HBM bytes."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        if self._error is not None:
            raise RuntimeError("device HBM sampler failed") from self._error
        return self._peak_used_bytes


def _start_memory_measurement(device: torch.device) -> _DeviceHbmSampler:
    """Reset allocator peaks and start device-side HBM polling."""
    torch.npu.reset_peak_memory_stats(device)
    sampler = _DeviceHbmSampler(device)
    sampler.start()
    return sampler


def _finish_memory_measurement(
    sampler: _DeviceHbmSampler,
    device: torch.device,
) -> dict[str, int]:
    """Synchronize and collect allocator plus total device HBM metrics."""
    torch.npu.synchronize(device)
    peak_hbm_used = sampler.stop()
    free_bytes, total_bytes = torch.npu.mem_get_info(device)
    return {
        "allocator_allocated": int(torch.npu.memory_allocated(device)),
        "allocator_reserved": int(torch.npu.memory_reserved(device)),
        "allocator_peak_allocated": int(torch.npu.max_memory_allocated(device)),
        "allocator_peak_reserved": int(torch.npu.max_memory_reserved(device)),
        "device_hbm_used": int(total_bytes - free_bytes),
        "device_hbm_peak_used": peak_hbm_used,
        "device_hbm_total": int(total_bytes),
    }


def _run_synchronized_step(
    workload: _OptimizerWorkload,
) -> torch.Tensor:
    """Run one untimed optimizer step from a synchronized boundary."""
    dist.barrier()
    torch.npu.synchronize(workload.device)
    loss = _optimizer_step(workload)
    torch.npu.synchronize(workload.device)
    return loss


def _rank_max_latency(elapsed_ms: float, device: torch.device) -> float:
    """Return the slowest rank's elapsed time."""
    latency = torch.tensor(elapsed_ms, dtype=torch.float32, device=device)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return float(latency.cpu().item())


def _run_timed_steps(
    workload: _OptimizerWorkload,
    measured_steps: int,
) -> list[float]:
    """Run measured optimizer steps and return cross-rank maximum latency."""
    latencies = []
    for _ in range(measured_steps):
        dist.barrier()
        torch.npu.synchronize(workload.device)
        start = time.perf_counter()
        _optimizer_step(workload)
        torch.npu.synchronize(workload.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(_rank_max_latency(elapsed_ms, workload.device))
    return latencies


def _percentile(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile for a nonempty sample."""
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _rank_max_memory(
    local_memory: dict[str, dict[str, int]],
    device: torch.device,
) -> dict[str, dict[str, int]]:
    """Reduce every memory metric to its maximum across ranks."""
    reduced = {}
    for phase, metrics in local_memory.items():
        names = tuple(metrics)
        values = torch.tensor(
            [metrics[name] for name in names],
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        reduced[phase] = {
            name: int(value)
            for name, value in zip(names, values.cpu().tolist())
        }
    return reduced


def _sha256(path: Path) -> str:
    """Return the SHA256 digest of one activated native artifact."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _artifact_identity(path: Path) -> dict[str, Any]:
    """Return a stable path, size, and digest for one native artifact."""
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _native_build_fingerprint() -> dict[str, Any] | None:
    """Fingerprint the exact activated vendor and framework adapter."""
    for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
        vendor_root = Path(value)
        if vendor_root.name != "hyper_parallel_multicore_nn":
            continue
        component_root = vendor_root.parent.parent
        op_api = vendor_root / "op_api" / "lib" / "libcust_opapi.so"
        adapters = sorted((component_root / "framework" / "torch").glob("*.so"))
        version_info = vendor_root / "version.info"
        fingerprint = {
            "vendor_root": str(vendor_root.resolve()),
            "op_api": _artifact_identity(op_api) if op_api.is_file() else None,
            "framework_adapters": [_artifact_identity(path) for path in adapters],
            "vendor_version": (
                version_info.read_text(encoding="utf-8").strip()
                if version_info.is_file()
                else None
            ),
        }
        manifest_path = component_root / "native-build.json"
        if manifest_path.is_file():
            fingerprint["manifest"] = {
                **_artifact_identity(manifest_path),
                "content": json.loads(manifest_path.read_text(encoding="utf-8")),
            }
        return fingerprint
    return None


def _result_path(args: argparse.Namespace) -> Path:
    """Resolve one backend-tagged aggregate result path."""
    tag = args.tag or args.backend
    return Path(args.output_dir) / f"optimizer_step_{tag}.json"


def _rank_result_path(path: Path, rank: int) -> Path:
    """Resolve the raw per-rank result path."""
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


def _result_metadata(
    args: argparse.Namespace,
    config: QwenMoeConfig,
    validation: dict[str, Any],
    latencies: list[float],
) -> dict[str, Any]:
    """Build metadata shared by aggregate and per-rank results."""
    return {
        "backend": args.backend,
        "world_size": _WORLD_SIZE,
        "topology": {"tp": 1, "ep": config.ep_size},
        "shape": {
            "batch_size": _BATCH_SIZE,
            "sequence_length": _SEQUENCE_LENGTH,
            "local_num_tokens": config.local_num_tokens,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_layers": config.num_layers,
            "num_experts": config.num_experts,
            "top_k": config.top_k,
            "expert_capacity_factor": config.expert_capacity_factor,
            "dtype": "bfloat16",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "validation": validation,
        "rank_max_step_ms": {
            "p50": statistics.median(latencies),
            "p90": _percentile(latencies, 0.90),
            "minimum": min(latencies),
            "maximum": max(latencies),
            "samples": latencies,
        },
        "runtime": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
            "native_build_fingerprint": _native_build_fingerprint(),
        },
    }


def _write_results(
    args: argparse.Namespace,
    config: QwenMoeConfig,
    outcome: _BenchmarkOutcome,
    rank: int,
) -> Path:
    """Persist raw per-rank data and the rank-max aggregate."""
    path = _result_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    common = _result_metadata(
        args,
        config,
        outcome.validation,
        outcome.latencies,
    )
    local_result = {
        **common,
        "rank": rank,
        "memory_bytes": outcome.local_memory,
    }
    _rank_result_path(path, rank).write_text(
        json.dumps(local_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dist.barrier()
    if rank == 0:
        aggregate = {
            **common,
            "memory_bytes_rank_max": outcome.rank_max_memory,
            "raw_rank_files": f"{path.stem}.rank<0-{_WORLD_SIZE - 1}>{path.suffix}",
            "device_healthy": True,
        }
        path.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"RESULT_JSON={path}")
        print(json.dumps(aggregate, sort_keys=True))
    dist.barrier()
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the fixed optimizer-step validation and benchmark."""
    args = parse_args(argv)
    rank, world_size, device = _init_runtime()
    config = QwenMoeConfig()
    succeeded = False
    try:
        _set_model_seed(args.seed)
        initialization_sampler = _start_memory_measurement(device)
        model = _build_model(config, args.backend, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        input_ids, labels = _build_batch(config, rank, args.seed, device)
        workload = _OptimizerWorkload(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            labels=labels,
            world_size=world_size,
            device=device,
        )
        initialization_memory = _finish_memory_measurement(
            initialization_sampler,
            device,
        )

        first_step_sampler = _start_memory_measurement(device)
        validation = _validate_first_step(workload)
        first_step_memory = _finish_memory_measurement(first_step_sampler, device)

        for _ in range(args.warmup_steps - 2):
            _run_synchronized_step(workload)
        stable_step_sampler = _start_memory_measurement(device)
        _run_synchronized_step(workload)
        stable_step_memory = _finish_memory_measurement(stable_step_sampler, device)

        latencies = _run_timed_steps(
            workload,
            args.measured_steps,
        )
        local_memory = {
            "initialization": initialization_memory,
            "first_optimizer_step": first_step_memory,
            "stable_optimizer_step": stable_step_memory,
        }
        outcome = _BenchmarkOutcome(
            validation=validation,
            latencies=latencies,
            local_memory=local_memory,
            rank_max_memory=_rank_max_memory(local_memory, device),
        )
        _write_results(
            args,
            config,
            outcome,
            rank,
        )
        succeeded = True
        return 0
    finally:
        if dist.is_initialized():
            if succeeded:
                torch.npu.synchronize(device)
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
