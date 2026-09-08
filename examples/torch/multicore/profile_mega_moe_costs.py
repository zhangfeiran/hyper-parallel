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
"""Collect untimed task diagnostics beside separate profile-off MegaMoe measurements.

Launch with torchrun from the editable checkout after loading its matching CANN
and native payload. This worker reuses the independent native ST precision oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path

from examples.torch.multicore.mega_moe_layer_profile import profile_scoped_events
from examples.torch.multicore.mega_moe_profile_context import (
    ProfileContext,
    new_capture_id,
)

# Framework initialization belongs in the distributed worker, not its launcher.
from tests.torch.expert_parallel import _test_mega_moe as worker

# The coupling to the precision harness is confined to this diagnostic example.
# pylint: disable=protected-access


def _identity() -> dict:
    root = Path(__file__).resolve().parents[3]
    layout = importlib.import_module("hyper_parallel.core.multicore.scheduler.config")
    paths = list((root / "hyper_parallel/core/multicore").rglob("*.py"))
    paths += list((root / "hyper_parallel/core/multicore/ops").rglob("*.h"))
    paths += list((root / "hyper_parallel/core/multicore/ops").rglob("*.hpp"))
    paths += list((root / "build/native/payload").rglob("*.so"))
    paths += list((root / "build/native/payload").rglob("*.o"))
    paths += [Path(__file__), Path(__file__).with_name("mega_moe_layer_profile.py"),
              Path(__file__).with_name("mega_moe_profile_context.py")]
    return {
        "root": str(root),
        "torch": worker.torch.__version__,
        "runtime_environment": {
            name: value for name, value in sorted(os.environ.items()) if name.startswith("HP_MEGA_MOE_")
        } | {"HP_MEGA_MOE_COMPACT_RUNTIME": str(int(layout.compact_runtime_enabled()))},
        "sources_and_payload": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)
        },
    }


def _counts(route: tuple) -> list[list[int]]:
    local = route[2]
    gathered = [worker.torch.empty_like(local) for _ in range(worker.dist.get_world_size())]
    worker.dist.all_gather(gathered, local)
    return [tensor.cpu().tolist() for tensor in gathered]


def main() -> None:
    """Measure profile-off first, then validate and collect profiled kernels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=1792)
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--pairs", default="10000:42,1:42,0.1:7,0.01:29,0.005:943")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=10)
    parser.add_argument("--profile-scopes", action="store_true", help="Capture extra CPU and device scope intervals")
    parser.add_argument(
        "--skip-task-profile",
        action="store_true",
        help="Collect only profile-off measurements",
    )
    parser.add_argument(
        "--baseline-copy-runs",
        action="store_true",
        help="Use the original quadratic CPU quota scan",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"rank{worker._RANK}.json"
    if output.exists():
        raise FileExistsError(output)
    route_module = importlib.import_module("hyper_parallel.core.multicore.modules.mega_moe.route")
    if args.baseline_copy_runs:

        def original_runs(plan: object, source_rank: int) -> tuple:
            """Reproduce the original quota scan for a fresh-process control."""
            physical_ids, counts = [], []
            mapping, row = (
                plan.physical_to_logical_expert,
                plan.dispatch_counts[source_rank],
            )
            for logical in range(plan.logical_num_experts):
                for physical, mapped in enumerate(mapping):
                    count = row[physical]
                    if mapped == logical and count:
                        physical_ids.append(physical)
                        counts.append(count)
            return tuple(physical_ids), tuple(counts)

        route_module._physical_copy_runs = original_runs
    shape = worker._MoeShape(
        local_num_tokens=args.tokens,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_experts=args.experts,
        top_k=8,
        ep_size=worker.dist.get_world_size(),
    )
    hidden, grad_output = worker._make_data(shape)
    layer, native = worker._new_layers(
        shape,
        replica_slots_per_rank=args.slots,
        compact_physical_weights=bool(args.slots),
    )
    results = []
    os.environ["HP_MEGA_MOE_KERNEL_CYCLE_PROFILE"] = "0"
    os.environ["HP_MEGA_MOE_TASK_PROFILE"] = "0"
    identity = _identity()
    for pair in args.pairs.split(","):
        alpha, seed = pair.split(":")
        route = worker._make_dirichlet_route(shape, float(alpha), int(seed))
        counts = _counts(route)
        route_summary = worker._summarize_performance_route(
            shape,
            route[2],
            args.slots,
            {
                "profile": "dirichlet",
                "dirichlet_alpha": float(alpha),
                "dirichlet_seed": int(seed),
            },
        )
        waves = route_summary["expected_wave_count"]
        reference = worker._run_layer(native, hidden, *route, grad_output)
        actual = worker._run_layer(layer, hidden, *route, grad_output)
        worker._assert_results_close(reference, actual, weight_grad_wave_count=waves)
        del reference, actual
        layer.zero_grad(set_to_none=True)
        native.zero_grad(set_to_none=True)
        for _ in range(args.warmup):
            worker._timed_fwd_bwd(layer, hidden, *route, grad_output)
        layer.zero_grad(set_to_none=True)
        worker.torch.npu.synchronize()
        baseline = worker._device_memory_snapshot()
        worker.torch.npu.reset_peak_memory_stats()
        samples = [worker._timed_fwd_bwd(layer, hidden, *route, grad_output) for _ in range(args.measured)]
        memory = worker._collect_memory_result(layer, baseline)
        scoped, captures = [], {}
        if args.profile_scopes:
            context = ProfileContext(rank=worker._RANK, world_size=shape.ep_size, source=identity,
                                     capture_id=new_capture_id(worker.dist, "scopes"))
            with context.observe(worker.torch, [layer]):
                scoped = profile_scoped_events(
                    worker.torch, lambda route=route: worker._timed_fwd_bwd(layer, hidden, *route, grad_output),
                    context=context,
                )
            captures["scopes"] = context.manifest()
        profiles, cycles = {}, {}
        if not args.skip_task_profile:
            os.environ["HP_MEGA_MOE_KERNEL_CYCLE_PROFILE"] = "1"
            os.environ["HP_MEGA_MOE_TASK_PROFILE"] = "1"
            context = ProfileContext(rank=worker._RANK, world_size=shape.ep_size, source=identity,
                                     capture_id=new_capture_id(worker.dist, "tasks"))
            with context.observe(worker.torch, [layer], checkpoint="reentrant"):
                actual = worker._run_checkpointed_layer(layer, hidden, *route, grad_output, use_reentrant=True)
            captures["tasks"] = context.manifest()
            reference = worker._run_layer(native, hidden, *route, grad_output)
            worker._assert_results_close(reference, actual, weight_grad_wave_count=waves)
            del reference, actual
            resources = layer._resource_group.resources
            profiles = {
                direction: resources.workspace.collect_task_profile(
                    resources.spec.num_cube_cores, forward=direction == "forward"
                )
                for direction in ("forward", "backward")
            }
            cycles = {
                direction: resources.workspace.collect_cycle_profile(
                    resources.spec.num_cube_cores, forward=direction == "forward"
                )
                for direction in ("forward", "backward")
            }
            os.environ["HP_MEGA_MOE_KERNEL_CYCLE_PROFILE"] = "0"
            os.environ["HP_MEGA_MOE_TASK_PROFILE"] = "0"
        results.append(
            {
                "pair": pair,
                "shape": asdict(shape),
                "counts": counts,
                "route": route_summary,
                "profile_off_median_ms": statistics.median(samples),
                "profile_off_samples_ms": samples,
                "memory": memory,
                "task_profiles": profiles,
                "scoped_events": scoped,
                "cycle_profiles": cycles,
                "captures": captures,
                "task_contexts": {direction: context.last_launch[(0, direction)] for direction in profiles},
                "precision": "native passed; detailed checkpoint checked when task profiling is enabled",
            }
        )
        if worker._RANK == 0:
            print(
                f"COST_PROFILE_POINT={pair} median_ms={statistics.median(samples):.4f}",
                flush=True,
            )
    if identity != _identity():
        raise RuntimeError("source or native artifacts changed during profiling")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "rank": worker._RANK,
                "profile_schema_version": 2,
                "identity": identity,
                "arguments": vars(args) | {"output": str(args.output)},
                "results": results,
            },
            stream,
        )
    worker.dist.barrier()
    worker.dist.destroy_process_group()
    print(f"COST_PROFILE_COMPLETE={output}", flush=True)


if __name__ == "__main__":
    main()
