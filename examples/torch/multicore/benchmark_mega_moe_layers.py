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
This is not a connected transformer model. Route sampling and SGD updates are
outside F+B timing, but every update invalidates replica weight versions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any

ALPHAS = (10000, 3000, 1000, 300, 100, 30, 10, 3, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
EXTRA_PAIRS = ((0.01, 1942), (0.01, 1708), (0.01, 0), (0.005, 943), (0.01, 2104),
               (0.01, 4050), (0.01, 2486), (0.01, 1164), (0.01, 29), (0.02, 1942))
SCOPE = "independent MoE layers; all forwards then reverse backwards; no attention or router network"

# Reuse the ST's route generator, native implementation and non-overlapping HBM
# ledger so the multi-layer example and single-layer curves compare the same work.
# pylint: disable=protected-access


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("native", "b0", "b1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=7)
    parser.add_argument("--pairs", help="Override the 55-route sweep, e.g. 10000:42,0.005:943.")
    parser.add_argument("--shuffle-seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.layers <= 0 or args.warmup < 0 or args.measured <= 0:
        parser.error("layers and measured must be positive; warmup must be nonnegative")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
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
    return points


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
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
        self.optimizer = worker.torch.optim.SGD(self.layers.parameters(), lr=1e-5, foreach=False)

    def _make_layers(self) -> Any:
        worker, shape = self.worker, self.shape
        layers = []
        for index in range(self.args.layers):
            worker.torch.manual_seed(2026 + worker._RANK + index * 100)
            kwargs = {
                "hidden_size": shape.hidden_size, "intermediate_size": shape.intermediate_size,
                "num_experts": shape.num_experts, "top_k": shape.top_k,
                "ep_size": shape.ep_size, "ep_group": worker.dist.group.WORLD,
            }
            if self.args.variant == "native":
                layer = worker.create_native_moe(**kwargs)
            else:
                layer = worker.MegaMoeExperts(
                    **kwargs, local_num_tokens=shape.local_num_tokens, expert_capacity_factor=1.5,
                    replica_slots_per_rank=self.slots, shared_physical_weights=bool(self.slots),
                )
            layers.append(layer.to(device=worker._DEVICE, dtype=worker.torch.bfloat16))
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
            function = partial(layer, tokens_per_expert=route[2])
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
                if tensor.grad is None or not bool(worker.torch.isfinite(tensor.grad).all().item()):
                    raise AssertionError("missing or nonfinite input/router/parameter gradient")

    def _pool_ledger(self) -> dict[str, Any]:
        if not self.slots:
            return {"shared_weight_bytes": 0, "shared_gradient_bytes": 0}
        backend = self.layers[0]._get_execution_backend(self.inputs[0])
        if any(layer._get_execution_backend(self.inputs[0]) is not backend for layer in self.layers):
            raise AssertionError("layers do not share one execution backend")
        manager = backend.replica_manager
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

    def measure(self, alpha: float, seed: int) -> dict[str, Any]:
        """Validate gradients, then measure one route with fresh weights each step."""
        worker = self.worker
        route = worker._make_dirichlet_route(self.shape, alpha, seed)
        summary = worker._summarize_performance_route(
            self.shape, route[2], self.slots,
            {"profile": "dirichlet", "dirichlet_alpha": alpha, "dirichlet_seed": seed},
        )
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
        latencies = []
        for _ in range(self.args.measured):
            worker.dist.barrier()
            worker.torch.npu.synchronize(worker._DEVICE)
            started = time.perf_counter()
            self._train_step(route)
            worker.torch.npu.synchronize(worker._DEVICE)
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(worker._rank_max_latency(elapsed))
            # Refresh parameter versions between measurements without timing SGD.
            self.optimizer.step()
            worker.torch.npu.synchronize(worker._DEVICE)
        memory = worker._collect_memory_result(self.layers[0], baseline)
        result = worker._build_performance_result(
            "native" if self.args.variant == "native" else "mega_moe",
            self.shape, summary, self.args.warmup, latencies, memory,
        )
        ledgers = [None] * self.shape.ep_size
        worker.dist.all_gather_object(ledgers, self._pool_ledger())
        result["shared_pool_per_rank"] = ledgers
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
        "optimizer_step": True, "optimizer": "SGD lr=1e-5 foreach=False; outside timing, inside HBM peak window",
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
        if source != _source_identity():
            raise RuntimeError("source or native artifacts changed during the sweep")
        if worker._RANK == 0:
            payload = {"variant": args.variant, "kind": "performance", "points": results,
                       "source": source, "configuration": _configuration(args)}
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
