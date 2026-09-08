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
"""Simulate the current hot-expert placement at 16/32/64 ranks on a CPU.

Run with ``python -m examples.torch.multicore.simulate_mega_moe --help``.
Reports distinguish virtual network/compute time from measured CPU planning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from examples.torch.multicore.megamoe_simulator.model import (
    Costs,
    Training,
    TrainingGraph,
)
from examples.torch.multicore.megamoe_simulator.network import EventSimulator, Fabric
from examples.torch.multicore.megamoe_simulator.workload import (
    PLANNER_PATH,
    REPOSITORY,
    Shape,
    build_plan,
    count_digest,
    make_counts,
    validate_counts,
)

LIMITATIONS = [
    "Simulation estimates, not multi-node NPU measurements or numerical kernel validation.",
    "Exact production placement and payload bytes; token messages are aggregated per rank pair and wave.",
    "Rank-stage compute waits for local wave inputs; AIV/Cube tile pipelining and tile ACKs are not replayed.",
    "Collectives use recursive doubling, not a reconstruction of HCCL's selected algorithm.",
    "No packet-level RDMA, switch routing/PFC, retries, NIC queue limits or HBM/communication contention.",
    "Effective compute rate and local costs require calibration; zero optimizer cost excludes optimizer execution.",
    "CPU planner measurements run without rank-process contention; copying count matrices to the host is in route_us.",
    "Only shared B-slot or per-layer B-slot lifetimes are modeled; compact/dense physical-weight caches are excluded.",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse simulation scenarios; bandwidth is aggregate GB/s per node/direction."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--node-map", type=Path, help="JSON array mapping ranks to node IDs; requires one rank count")
    parser.add_argument(
        "--slots", type=int, nargs="+", default=[0, 1], help="Replica slots per rank; zero is MegaMoe B=0"
    )
    parser.add_argument(
        "--experts", type=int, default=384, help="Fixed global experts; must divide every selected rank count"
    )
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=1792)
    parser.add_argument("--capacity-factor", type=float, nargs="+", default=[1.5])
    parser.add_argument(
        "--inter-gbps", type=float, nargs="+", default=[25.0], help="Aggregate effective GB/s per node/direction"
    )
    parser.add_argument("--inter-latency-us", type=float, nargs="+", default=[10.0])
    parser.add_argument("--intra-gbps", type=float, default=150.0, help="Effective per-rank intranode GB/s")
    parser.add_argument("--intra-node-gbps", type=float, default=600.0)
    parser.add_argument("--intra-latency-us", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--checkpoint", choices=["reentrant", "none"], default="reentrant")
    parser.add_argument("--pool", choices=["shared", "per-layer"], default="shared")
    parser.add_argument(
        "--update-every", type=int, default=1, help="Parameter update interval; zero means unchanged weights"
    )
    parser.add_argument("--routing", choices=["balanced", "hotspot", "dirichlet"], default="hotspot")
    parser.add_argument("--hot-fraction", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--counts", type=Path, help="JSON count matrix, or {counts: [...]} for one rank count")
    parser.add_argument("--rotate-experts", type=int, default=0, help="Rotate count columns each forward layer/step")
    parser.add_argument(
        "--compute-tflops", type=float, default=100.0, help="Assumed effective GEMM TFLOP/s per virtual rank"
    )
    parser.add_argument("--costs", type=Path, help="JSON {costs: Costs fields, provenance: {...}} from calibration")
    parser.add_argument(
        "--planner-us", type=float, help="Override measured planner median with an explicit per-call cost"
    )
    parser.add_argument("--output", type=Path, default=Path("output/megamoe_simulation"))
    parser.add_argument("--trace", action="store_true", help="Write a Chrome/Perfetto event trace for each scenario")
    return parser.parse_args(argv)


def _source_identity() -> dict[str, Any]:
    roots = [PLANNER_PATH.parent, REPOSITORY / "examples/torch/multicore/megamoe_simulator"]
    paths = [path for root in roots for path in sorted(root.rglob("*.py"))]
    paths.append(Path(__file__).resolve())
    paths.extend(sorted((REPOSITORY / "hyper_parallel/core/multicore/ops").rglob("worker_kernel.*")))
    return {
        "repository": str(REPOSITORY),
        "git_head": subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip(),
        "planner_path": str(PLANNER_PATH),
        "sha256": {str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _trace(path: Path, graph: TrainingGraph, result: Any, fabric: Fabric) -> None:
    events = []
    for task_id, task in enumerate(graph.tasks):
        if result.finishes_us[task_id] == result.starts_us[task_id]:
            continue
        events.append(
            {
                "name": task.name,
                "cat": task.phase,
                "ph": "X",
                "pid": fabric.node_by_rank[task.rank],
                "tid": f"rank{task.rank}/{task.phase}",
                "ts": result.starts_us[task_id],
                "dur": result.finishes_us[task_id] - result.starts_us[task_id],
                "args": {
                    "bytes": task.size_bytes,
                    "target": task.target,
                    "serialization_end_us": result.serialization_ends_us[task_id],
                },
            }
        )
    _write_json(path, {"displayTimeUnit": "ms", "traceEvents": events})


def _plan_summary(plan: Any, capacity: int) -> dict[str, Any]:
    return {
        "counts_sha256": count_digest(plan.logical_counts_by_source),
        "baseline_destination_loads": plan.baseline_destination_loads,
        "destination_loads": plan.destination_loads,
        "wave_count": plan.wave_count(capacity),
        "replica_transfers": [asdict(item) for item in plan.replica_transfers],
        "slot_to_logical_expert": plan.slot_to_logical_expert,
    }


def _prepare_plans(counts: Any, shape: Shape, slots: int, args: argparse.Namespace) -> tuple[list[Any], list[float]]:
    plans, timings = [], []
    seen = {}
    for call_index in range(args.layers * args.steps):
        shift = (call_index * args.rotate_experts) % shape.experts
        if shift not in seen:
            shifted = tuple(row[-shift:] + row[:-shift] if shift else row for row in counts)
            samples = []
            for _ in range(3):
                start = time.perf_counter_ns()
                plan = build_plan(shifted, slots, shape.capacity)
                samples.append((time.perf_counter_ns() - start) / 1000)
            seen[shift] = plan
            timings.append(statistics.median(samples))
        plans.append(seen[shift])
    return plans, timings


def _scenario(
    shape: Shape, plans: list[Any], costs: Costs, fabric: Fabric, args: argparse.Namespace, name: str
) -> dict[str, Any]:
    training = Training(args.layers, args.steps, args.checkpoint == "reentrant", args.pool, args.update_every)
    graph = TrainingGraph(shape, plans, costs, training)
    start = time.perf_counter()
    tasks = graph.build()
    result = EventSimulator(fabric, tasks).run()
    wall_seconds = time.perf_counter() - start
    ends = [max(result.finishes_us[task] for task in step) for step in graph.step_ends]
    step_ms = [(end - previous) / 1000 for end, previous in zip(ends, [0.0] + ends[:-1])]
    per_rank_compute = [0.0] * len(fabric.node_by_rank)
    for task in tasks:
        if task.phase in {"forward_compute", "backward_compute"}:
            per_rank_compute[task.rank] += task.duration_us
    if args.trace:
        _trace(args.output / f"{name}.trace.json", graph, result, fabric)
    pool_count = args.layers if args.pool == "per-layer" else 1
    slots = plans[0].replica_slots_per_rank
    report = {
        "name": name,
        "evidence_kind": "simulation",
        "limitations": LIMITATIONS,
        "shape": asdict(shape),
        "fabric": asdict(fabric),
        "costs": asdict(costs),
        "training": asdict(training),
        "replica_slots_per_rank": slots,
        "plans": [_plan_summary(plan, shape.capacity) for plan in dict.fromkeys(plans)],
        "step_ms": step_ms,
        "mean_step_ms": statistics.mean(step_ms),
        "traffic_bytes_by_phase": result.traffic,
        "resource_bytes": result.resource_bytes,
        "cache_events": graph.cache_events,
        "task_count": len(tasks),
        "simulator_wall_seconds": wall_seconds,
        "compute_busy_fraction_by_rank": [value / result.makespan_us for value in per_rank_compute],
        "replica_pool_bytes_per_rank": sum(shape.weight_elements) * slots * (shape.element_bytes + 4) * pool_count,
        "replica_pool_memory_scope": "only B-slot weights plus FP32 gradients; excludes all other HBM allocations",
    }
    _write_json(args.output / f"{name}.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run matched B-slot and network sweeps and save reviewable input/output artifacts."""
    if args.ranks_per_node < 1 or any(rank < 1 or rank & (rank - 1) for rank in args.ranks):
        raise ValueError("ranks must be powers of two and ranks_per_node must be positive.")
    if (args.node_map or args.counts) and len(args.ranks) != 1:
        raise ValueError("--node-map and --counts require exactly one --ranks value.")
    if any(slot < 0 for slot in args.slots):
        raise ValueError("Replica slots must be nonnegative.")
    args.output.mkdir(parents=True, exist_ok=True)
    cost_document = _read_json(args.costs) if args.costs else {}
    costs = Costs(**cost_document.get("costs", {"effective_tflops": args.compute_tflops}))
    metadata = {
        "schema_version": 1,
        "evidence_kind": "simulation",
        "limitations": LIMITATIONS,
        "source": _source_identity(),
        "cost_provenance": cost_document.get("provenance", "uncalibrated assumptions"),
        "scaling": "fixed global experts and tokens per rank; global token count grows with ranks",
        "command": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scenarios": [],
    }
    for ranks in args.ranks:
        shape = Shape(args.experts, args.tokens_per_rank, args.top_k, args.hidden, args.intermediate)
        node_map = (
            tuple(_read_json(args.node_map))
            if args.node_map
            else tuple(rank // args.ranks_per_node for rank in range(ranks))
        )
        if len(node_map) != ranks:
            raise ValueError("node-map length must match ranks.")
        if args.counts:
            document = _read_json(args.counts)
            counts = validate_counts(document["counts"] if isinstance(document, dict) else document, shape, ranks)
        else:
            counts = make_counts(
                shape, ranks, routing=args.routing, seed=args.seed, hot_fraction=args.hot_fraction, alpha=args.alpha
            )
        _write_json(args.output / f"rank{ranks}.counts.json", {"counts": counts, "shape": asdict(shape)})
        for factor in args.capacity_factor:
            shape = replace(shape, capacity_factor=factor)
            for slots in args.slots:
                plans, timings = _prepare_plans(counts, shape, slots, args)
                if args.planner_us is not None:
                    planner_us, planner_kind = args.planner_us, "override"
                elif "planner_us" in cost_document.get("costs", {}):
                    planner_us, planner_kind = costs.planner_us, "cost_file"
                else:
                    planner_us, planner_kind = statistics.median(timings), "measured_cpu"
                for bandwidth in args.inter_gbps:
                    for latency in args.inter_latency_us:
                        fabric = Fabric(
                            node_map, bandwidth, latency, args.intra_gbps, args.intra_latency_us, args.intra_node_gbps
                        )
                        name = f"r{ranks}-b{slots}-c{factor:g}-bw{bandwidth:g}-lat{latency:g}"
                        report = _scenario(shape, plans, replace(costs, planner_us=planner_us), fabric, args, name)
                        summary = {
                            "name": name,
                            "ranks": ranks,
                            "slots": slots,
                            "capacity_factor": factor,
                            "inter_gbps": bandwidth,
                            "latency_us": latency,
                            "mean_step_ms": report["mean_step_ms"],
                            "weight_inter_bytes": report["traffic_bytes_by_phase"]
                            .get("weight_prefetch", {})
                            .get("inter", 0),
                            "gradient_inter_bytes": report["traffic_bytes_by_phase"]
                            .get("replica_gradient", {})
                            .get("inter", 0),
                            "planner_measured_us": timings,
                            "planner_cost_kind": planner_kind,
                        }
                        metadata["scenarios"].append(summary)
                        print(f"{name}: simulated mean step {report['mean_step_ms']:.3f} ms", flush=True)
    _write_json(args.output / "summary.json", metadata)
    columns = (
        "name",
        "ranks",
        "slots",
        "capacity_factor",
        "inter_gbps",
        "latency_us",
        "mean_step_ms",
        "weight_inter_bytes",
        "gradient_inter_bytes",
    )
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata["scenarios"])
    return metadata


if __name__ == "__main__":
    run(parse_args())
