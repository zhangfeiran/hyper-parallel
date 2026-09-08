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
"""Measure route-matched checkpointed MoE layers with native, B=0 or shared B=1.

Run this worker with torchrun from an editable repository checkout. Inputs and
parameters are independent per layer; all forwards precede reverse backwards.
This is not a connected transformer model. Both F+B and F+B + AdamW step are
timed directly; route sampling is excluded. Updates invalidate replica versions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any

from examples.torch.multicore.mega_moe_layer_profile import (
    profile_layer_cycles,
    profile_layer_step,
    profile_trace_paths,
)
from examples.torch.multicore.mega_moe_p2p_profile import profile_layer_p2p

ALPHAS = (10000, 3000, 1000, 300, 100, 30, 10, 3, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
EXTRA_PAIRS = ((0.01, 1942), (0.01, 1708), (0.01, 0), (0.005, 943), (0.01, 2104),
               (0.01, 4050), (0.01, 2486), (0.01, 1164), (0.01, 29), (0.02, 1942))
SCOPE = "independent MoE layers; all forwards then reverse backwards; no attention or router network"
ADAMW_OPTIONS = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8,
                 "weight_decay": 0.01, "foreach": False, "fused": False}
_PYTHON_SOURCE_ROOTS = (
    "hyper_parallel/core/multicore",
    "hyper_parallel/core/symmetric_memory",
    "hyper_parallel/platform/torch/symmetric_memory",
    "hyper_parallel/core/expert_parallel",
    "hyper_parallel/platform/torch/common",
)

# Reuse the ST's route generator, native implementation and non-overlapping HBM
# ledger so the multi-layer example and single-layer curves compare the same work.
# pylint: disable=protected-access


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("native", "b0", "b1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=7)
    parser.add_argument("--pairs", help="Override the 55-route sweep, e.g. 10000:42,0.005:943.")
    parser.add_argument("--shuffle-seed", type=int, default=20260903)
    parser.add_argument("--limit", type=int, help="Only run this many routes from the shuffled sweep prefix.")
    parser.add_argument("--profile-output", type=Path,
                        help="After all measurements, trace one extra step of the first route on rank zero.")
    parser.add_argument("--profile-all-ranks", action="store_true",
                        help="With --profile-output, write one .rankN.json trace per rank instead of rank zero only.")
    parser.add_argument("--profile-kernel-cycles", action="store_true",
                        help="After timing, gather last forward/backward cycles on all ranks (first route, MegaMoe).")
    parser.add_argument("--profile-p2p", action="store_true",
                        help="After timing, collect all-rank P2P payload and readiness events (first route, MegaMoe).")
    args = parser.parse_args(argv)
    if args.layers <= 0 or args.warmup < 0 or args.measured <= 0:
        parser.error("layers and measured must be positive; warmup must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.profile_kernel_cycles and args.variant == "native":
        parser.error("profile-kernel-cycles requires a MegaMoe variant")
    if args.profile_p2p and args.variant == "native":
        parser.error("profile-p2p requires a MegaMoe variant")
    if os.getenv("HP_MEGA_MOE_KERNEL_CYCLE_PROFILE", "0") == "1":
        parser.error("unset HP_MEGA_MOE_KERNEL_CYCLE_PROFILE for timing; use --profile-kernel-cycles instead")
    if args.profile_all_ranks and args.profile_output is None:
        parser.error("profile-all-ranks requires profile-output")
    if args.profile_output is not None:
        if args.profile_output.exists() or args.profile_output.resolve() == args.output.resolve():
            parser.error("profile-output must be a new path distinct from output")
        if args.profile_output.suffix != ".json":
            parser.error("profile-output must have a .json suffix")
        try:
            paths = profile_trace_paths(args.profile_output, int(os.getenv("WORLD_SIZE", "8")),
                                        all_ranks=args.profile_all_ranks)
        except ValueError as exc:
            parser.error(str(exc))
        for path in paths:
            if path.exists() or path.resolve() == args.output.resolve():
                parser.error(f"trace destination must be new and distinct from output: {path}")
    return args


def _route_pairs(args: argparse.Namespace) -> list[tuple[float, int]]:
    points = [(float(alpha), seed) for alpha in ALPHAS for seed in (7, 42, 83)] + list(EXTRA_PAIRS)
    if args.pairs:
        points = [(float(alpha), int(seed)) for alpha, seed in
                  (pair.split(":", maxsplit=1) for pair in args.pairs.split(","))]
    if len(points) != len(set(points)):
        raise ValueError("duplicate alpha/seed pairs")
    if any(not math.isfinite(alpha) or alpha <= 0 or seed < 0 for alpha, seed in points):
        raise ValueError("alpha must be positive and finite; seed must be nonnegative")
    random.Random(args.shuffle_seed).shuffle(points)
    return points[:args.limit]


def _make_optimizer(torch: Any, parameters: Any) -> Any:
    return torch.optim.AdamW(parameters, **ADAMW_OPTIONS)


def _execution_route_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Report the home bypass instead of the ST helper's mean-target proposal."""
    slots = summary["replica_slots_per_rank"]
    if not slots or summary["baseline_wave_count"] > 1:
        return summary
    return {
        **summary,
        "physical_destination_loads": list(summary["destination_loads"]),
        "physical_destination_skew": summary["realized_destination_skew"],
        "planner_target_capacity": summary["receive_capacity"],
        "expected_wave_count": 1,
        "replica_slots_used": 0,
        "replica_slot_bindings": [[-1] * slots for _ in summary["destination_loads"]],
        "execution_mode": "home-only",
    }


def _optimizer_state_ledger(torch: Any, optimizer: Any) -> dict[str, Any]:
    # These tensors already belong to the allocator peak, not an extra HBM term.
    by_device, by_dtype = {}, {}
    for state in optimizer.state.values():
        for name, tensor in state.items():
            if not torch.is_tensor(tensor):
                continue
            tensor = _local_tensor(tensor)
            if not bool(torch.isfinite(tensor).all().item()):
                raise AssertionError(f"nonfinite AdamW state: {name}")
            size = tensor.numel() * tensor.element_size()
            device, dtype = tensor.device.type, str(tensor.dtype)
            by_device[device] = by_device.get(device, 0) + size
            by_dtype[dtype] = by_dtype.get(dtype, 0) + size
    return {"parameter_state_count": len(optimizer.state), "tensor_bytes": sum(by_device.values()),
            "tensor_bytes_by_device": by_device, "tensor_bytes_by_dtype": by_dtype}


def _local_tensor(tensor: Any) -> Any:
    """Use the rank-local shard for native DTensor validation and HBM accounting."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _python_source_identity(repo: Path) -> dict[str, Any]:
    """Hash dirty runtime sources, excluding generated native build trees."""
    paths = sorted(
        path
        for relative_root in _PYTHON_SOURCE_ROOTS
        for path in (repo / relative_root).rglob("*.py")
        if not {"mindspore", "build", "prebuild"}.intersection(path.relative_to(repo).parts)
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repo).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {"scope": "torch_runtime_with_standard_ep_without_generated_build_trees_v3",
            "file_count": len(paths), "sha256": digest.hexdigest()}


def _source_identity() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    roots = (repo / "build/native/payload/hyper_parallel/core", repo / "hyper_parallel/core/symmetric_memory/lib")
    artifacts = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".so", ".o"}
    }
    return {
        "repo": str(repo),
        "git_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "worker_sha256": hashlib.sha256(
            (repo / "tests/torch/expert_parallel/_test_mega_moe.py").read_bytes()
        ).hexdigest(),
        "common_worker_sha256": hashlib.sha256(
            (repo / "tests/torch/expert_parallel/_test_mega_moe_common.py").read_bytes()
        ).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("mega_moe_layer_profile.py").read_bytes()
        ).hexdigest(),
        "p2p_profile_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("mega_moe_p2p_profile.py").read_bytes()
        ).hexdigest(),
        "python_implementation": _python_source_identity(repo),
        "native_artifacts": artifacts,
    }


class _LayerSweep:
    """Own one persistent set of layer weights and shared execution resources."""

    def __init__(self, worker: Any, args: argparse.Namespace) -> None:
        """Construct fixed production-shape layers and independent inputs."""
        self.worker = worker
        self.args = args
        self.checkpoint = importlib.import_module("torch.utils.checkpoint").checkpoint
        self.shape = worker._MoeShape(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                                      num_experts=48, top_k=8, ep_size=8)
        self.slots = int(args.variant == "b1")
        self.layers = self._make_layers()
        hidden, self.grad_output = worker._make_data(self.shape)
        self.inputs = [hidden.clone() for _ in self.layers]
        self.optimizer = _make_optimizer(worker.torch, self.layers.parameters())

    def _make_layers(self) -> Any:
        worker, shape = self.worker, self.shape
        layers = []
        for index in range(self.args.layers):
            seed = 2026 + index * 100
            if self.args.variant == "native":
                layer = worker.new_common_moe(shape, seed=seed)
            else:
                layer, common = worker.new_common_layers(
                    shape, seed=seed, expert_capacity_factor=1.5,
                    replica_slots_per_rank=self.slots, shared_physical_weights=bool(self.slots),
                )
                del common
            layers.append(layer)
        if self.args.variant != "native":
            worker.MegaMoeExperts.share_execution_resources(layers)
        return worker.torch.nn.ModuleList(layers)

    def _train_step(self, route: tuple, *, validate: bool = False) -> None:
        worker = self.worker
        self.optimizer.zero_grad(set_to_none=True)
        outputs, leaves = [], []
        for layer, data in zip(self.layers, self.inputs):
            hidden = data.detach().requires_grad_(True)
            weights = route[1].detach().requires_grad_(True)
            function = partial(worker.forward_layer, layer, tokens_per_expert=route[2])
            output = self.checkpoint(function, hidden, route[0], weights, use_reentrant=True)
            outputs.append(output)
            if validate:
                if not bool(worker.torch.isfinite(output).all().item()):
                    raise AssertionError("nonfinite checkpoint forward output")
                leaves.extend((hidden, weights))
        del output, hidden, weights
        while outputs:
            output = outputs.pop()
            output.backward(self.grad_output)
            del output
        if validate:
            for tensor in [*leaves, *self.layers.parameters()]:
                gradient = _local_tensor(tensor.grad)
                if gradient is None or not bool(worker.torch.isfinite(gradient).all().item()):
                    raise AssertionError("missing or nonfinite input/router/parameter gradient")

    def _pool_ledger(self) -> dict[str, Any]:
        if not self.slots:
            return {"shared_weight_bytes": 0, "shared_gradient_bytes": 0}
        resources = self.layers[0]._get_execution_resources(self.inputs[0])
        if any(layer._get_execution_resources(self.inputs[0]) is not resources for layer in self.layers):
            raise AssertionError("layers do not share one execution resource group")
        manager = resources.replica_manager
        ledger = {
            "shared_weight_bytes": sum(tensor.numel() * tensor.element_size()
                                       for tensor in (manager._shared_weights or ())),
            "shared_gradient_bytes": sum(tensor.numel() * tensor.element_size()
                                         for tensor in (manager._shared_gradients or ())),
        }
        if any(parameter.untyped_storage().nbytes() != parameter.numel() * parameter.element_size()
               for parameter in self.layers.parameters()):
            raise AssertionError("shared B changed the home parameter storage size")
        return ledger

    def _measure_steps(self, route: tuple) -> tuple[list[float], list[float]]:
        worker = self.worker
        latencies, step_latencies = [], []
        for _ in range(self.args.measured):
            worker.dist.barrier()
            worker.torch.npu.synchronize(worker._DEVICE)
            started = time.perf_counter()
            self._train_step(route)
            worker.torch.npu.synchronize(worker._DEVICE)
            elapsed = (time.perf_counter() - started) * 1000
            self.optimizer.step()
            worker.torch.npu.synchronize(worker._DEVICE)
            step_elapsed = (time.perf_counter() - started) * 1000
            # Aggregate only after both timers stop: diagnostics are not training.
            latencies.append(worker._rank_max_latency(elapsed))
            step_latencies.append(worker._rank_max_latency(step_elapsed))
        return latencies, step_latencies

    def measure(self, alpha: float, seed: int) -> dict[str, Any]:
        """Measure one route, validating before warmup and after all updates."""
        worker = self.worker
        route = worker._make_dirichlet_route(self.shape, alpha, seed)
        summary = _execution_route_summary(worker._summarize_performance_route(
            self.shape, route[2], self.slots,
            {"profile": "dirichlet", "dirichlet_alpha": alpha, "dirichlet_seed": seed},
        ))
        self._train_step(route, validate=True)
        self.optimizer.step()
        for _ in range(self.args.warmup):
            self._train_step(route)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        worker.dist.barrier()
        worker.torch.npu.synchronize(worker._DEVICE)
        baseline = worker._device_memory_snapshot()
        worker.torch.npu.reset_peak_memory_stats(worker._DEVICE)
        latencies, step_latencies = self._measure_steps(route)
        memory = worker._collect_memory_result(self.layers[0], baseline)
        # Capture memory first: finite checks allocate untimed diagnostic tensors.
        self._train_step(route, validate=True)
        for parameter in self.layers.parameters():
            if not bool(worker.torch.isfinite(_local_tensor(parameter)).all().item()):
                raise AssertionError("nonfinite parameter after AdamW updates")
        local_ledger = {"shared_pool": self._pool_ledger(),
                        "optimizer_state": _optimizer_state_ledger(worker.torch, self.optimizer)}
        ledgers = [None] * self.shape.ep_size
        worker.dist.all_gather_object(ledgers, local_ledger)
        result = worker._build_performance_result(
            "native" if self.args.variant == "native" else "mega_moe",
            self.shape, summary, self.args.warmup, latencies, memory,
        )
        result["rank_max_train_step_ms"] = {
            "samples": step_latencies, "median": statistics.median(step_latencies),
            "p90": worker._percentile(step_latencies, 0.9),
            "minimum": min(step_latencies), "maximum": max(step_latencies),
        }
        result["shared_pool_per_rank"] = [ledger["shared_pool"] for ledger in ledgers]
        result["optimizer_state_per_rank"] = [ledger["optimizer_state"] for ledger in ledgers]
        result["validation"].update(post_measurement_finite=True, optimizer_states_finite=True)
        result["layers"] = self.args.layers
        result["checkpoint"] = "reentrant"
        self.optimizer.zero_grad(set_to_none=True)
        return result


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "capacity_factor": 1.5, "replica_slots_per_rank": int(args.variant == "b1"),
        "compact_physical_weights": False, "shared_physical_weights": args.variant == "b1",
        "layers": args.layers, "checkpoint": "reentrant", "scope": SCOPE,
        "warmup_steps": args.warmup, "measured_steps": args.measured, "shuffle_seed": args.shuffle_seed,
        "optimizer_step": True, "optimizer": {"name": "AdamW", **ADAMW_OPTIONS},
        "latency_metric": "train_step",
        "optimizer_timing": "direct F+B and full AdamW-step wall timers; states/updates inside HBM peak window",
        "validation_protocol": "finite forward and gradients before warmup and after measurement; finite AdamW state",
        "weight_refresh": "parameters updated every iteration; replica refresh included in next F+B",
        "workspace_lifetime": "one shared execution group across layers and routes; high-watermarks retained",
        "runtime_environment": {key: value for key, value in os.environ.items()
                                if key.startswith(("HP_MEGA_MOE_", "HYPER_PARALLEL_MEGA_MOE_"))},
    }


def main() -> None:
    """Run one variant; compatible JSON can be passed directly to the plotter."""
    args = _parse_args()
    points = _route_pairs(args)
    os.environ["HP_MEGA_MOE_CAPACITY_FACTOR"] = "1.5"
    os.environ["HP_MEGA_MOE_WORLD_SIZE"] = "8"
    worker = importlib.import_module("tests.torch.expert_parallel._test_mega_moe")
    try:
        source = _source_identity()
        sweep = _LayerSweep(worker, args)
        results = []
        for alpha, seed in points:
            result = sweep.measure(alpha, seed)
            if worker._RANK == 0:
                results.append(result)
                print("LAYER_SWEEP_POINT=" + json.dumps(result, sort_keys=True), flush=True)
        trace_paths = ()
        if args.profile_output is not None:
            trace_paths = profile_trace_paths(args.profile_output, worker.dist.get_world_size(),
                                              all_ranks=args.profile_all_ranks)
            profile_layer_step(sweep, worker._make_dirichlet_route(sweep.shape, *points[0]), args.profile_output,
                               all_ranks=args.profile_all_ranks)
        kernel_cycles = None
        if args.profile_kernel_cycles:
            kernel_cycles = profile_layer_cycles(sweep, worker._make_dirichlet_route(sweep.shape, *points[0]))
        p2p_profile = None
        if args.profile_p2p:
            p2p_profile = profile_layer_p2p(sweep, worker._make_dirichlet_route(sweep.shape, *points[0]))
        if source != _source_identity():
            raise RuntimeError("source or native artifacts changed during the sweep")
        if worker._RANK == 0:
            payload = {"variant": args.variant, "kind": "performance", "points": results,
                       "source": source, "configuration": _configuration(args)}
            if args.profile_output is not None:
                payload["diagnostic_trace"] = {
                    "path": str(trace_paths[0]), "rank": 0, "alpha": points[0][0], "seed": points[0][1],
                    "timing": "one extra post-update step after all performance and memory measurements",
                }
                if args.profile_all_ranks:
                    payload["diagnostic_trace"]["per_rank"] = [
                        {"rank": rank, "path": str(path)} for rank, path in enumerate(trace_paths)
                    ]
            if kernel_cycles is not None:
                payload["diagnostic_kernel_cycles"] = {"alpha": points[0][0], "seed": points[0][1], **kernel_cycles}
            if p2p_profile is not None:
                payload["diagnostic_p2p"] = {"alpha": points[0][0], "seed": points[0][1], **p2p_profile}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"LAYER_SWEEP_RESULT={args.output}", flush=True)
        worker.dist.barrier()
    finally:
        if worker.dist.is_initialized():
            worker.dist.destroy_process_group()


if __name__ == "__main__":
    main()
