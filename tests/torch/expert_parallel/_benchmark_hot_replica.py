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
"""Fresh-process replica ablation with separate cold, warm and diagnostic steps."""

import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaCostModel
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore._loader import get_multicore_paths
from hyper_parallel.core.multicore.profiler import mega_kernel_profile
from tests.common.port_utils import allocate_port
from tests.torch.expert_parallel.hot_replica_checks import execution_identity, file_identity, saved_route
from tests.torch.expert_parallel.hot_replica_measurements import HostMeasurements, memory_snapshot


def _routes(tokens, top_k, experts, device):
    positions = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k)
    hot = positions.remainder(top_k).long()
    return {"balanced": [positions.remainder(experts).long()], "home_hot": [hot],
            "rotating_hot": [(hot + offset).remainder(experts) for offset in range(0, experts, 6)]}


def _plan(output, planner, budget, cost_model):
    if not budget:
        return {"planner": "disabled", "dw_dtype": "bfloat16"}
    route = saved_route(output)
    actual = "device" if route.device_plan is not None else "cpu"
    if actual != planner:
        raise AssertionError(f"Expected {planner}, executed {actual}")
    return {"planner": actual, "dw_dtype": "float32", "slots": route.plan.slot_to_logical,
            "counts": route.plan.destination_counts, "transfers": len(route.plan.transfers),
            "score_ms": None if cost_model is None else cost_model.estimate_ms(route.plan)}


def _measure(module, optimizer, values, ids, probabilities, gradient, *, inspect=False, cost_model=None):
    optimizer.zero_grad(set_to_none=True)
    values.grad = None
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    output = module(values, ids, probabilities)
    plan = _plan(output, module.replica_planner, module.replica_slots_per_rank, cost_model) if inspect else None
    memory = memory_snapshot(module, values, output) if inspect else None
    output.backward(gradient)
    optimizer.step()
    torch.npu.synchronize()
    local_ms = (time.perf_counter() - start) * 1000
    maximum = torch.tensor([local_ms], device=values.device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return {"rank_max_step_ms": float(maximum.cpu()[0]), "local_step_ms": local_ms,
            "plan": plan, "forward_memory": memory}


def _diagnose(module, optimizer, values, ids, probabilities, gradient, directory, audit):
    optimizer.zero_grad(set_to_none=True)
    values.grad = None
    dist.barrier()
    torch.npu.synchronize()
    plan, forward_memory = audit["plan"], audit["forward_memory"]
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                                torch_npu.profiler.ProfilerActivity.NPU]) as outer:
        with mega_kernel_profile(detailed_task_names=True) as internal, HostMeasurements() as host:
            output = module(values, ids, probabilities)
            output.backward(gradient)
            optimizer.step()
            torch.npu.synchronize()
    outer.export_chrome_trace(str(directory / "framework.json"))
    trace = internal.export_chrome_trace(directory / "internal.json")
    if trace["megaKernelCycleTrace"]["droppedRecordCount"]:
        raise AssertionError("Diagnostic trace dropped records")
    wait_records = 0
    for event in trace["traceEvents"]:
        args = event.get("args", {})
        if args.get("task_stage") == "ReplicaWeightReadyWait":
            wait_records += 1
            expert = plan["slots"][dist.get_rank()][6 + args["replica_slot"]]
            args.update(logical_expert=expert, owner_peer=expert // 6, layer="benchmark_experts",
                        direction="forward" if args["consumer_stage"] in ("GMM1", "GMM2") else "backward")
    if module.replica_slots_per_rank and module._get_execution_resources(values).spec.replica_transport in (
            "shmem_signal_sdma_overlap", "shmem_signal_sdma_projection", "shmem_signal_kernel_gradient"):
        if any(plan["counts"][dist.get_rank()][6:]) and not wait_records:
            raise AssertionError("Guest compute has no weight-ready trace; verify the loaded kernel payload")
    (directory / "internal.json").write_text(json.dumps(trace) + "\n", encoding="utf-8")
    return {"host_spans": host.records, "plan": plan, "forward_memory": forward_memory,
            "scope": "Inclusive host/stream diagnostic spans can overlap and include queue gaps; "
                     "use framework kernel records for device execution, not their sum as step latency."}


def _host_breakdown(module, optimizer, values, ids, probabilities, gradient):
    records = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        values.grad = None
        dist.barrier()
        torch.npu.synchronize()
        with HostMeasurements(device_intervals=False) as host:
            output = module(values, ids, probabilities)
            output.backward(gradient)
            optimizer.step()
        torch.npu.synchronize()
        records.append(host.records)
    return {"samples": records, "scope": "Untimed host-only inclusive spans; no Torch/internal profiler or "
                                         "per-stage device events. Spans nest and must not be summed as step latency."}


def run(args: argparse.Namespace) -> None:
    """Run one shape/variant per process, keeping profiling out of timed samples.

    Args:
        args: Benchmark shape, transport, planner and diagnostic options.
    """
    rank, size = dist.get_rank(), dist.get_world_size()
    endpoint = [f"tcp://127.0.0.1:{allocate_port()}" if rank == 0 else None]
    dist.broadcast_object_list(endpoint, src=0)
    os.environ["HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT"] = endpoint[0]
    directory = Path(args.result_dir) / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    calibration = None
    if args.cost_model:
        calibration = ExpertReplicaCostModel(**json.loads(Path(args.cost_model).read_text(encoding="utf-8"))["model"])
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(917 + rank)
    module = MegaMoeExperts(local_num_tokens=args.tokens, hidden_size=args.hidden,
                            intermediate_size=args.intermediate, num_experts=size * 6, top_k=args.top_k,
                            ep_size=size, ep_group=dist.group.WORLD, dispatch_mode=args.backend,
                            replica_slots_per_rank=args.budget, replica_transport=args.replica_transport,
                            replica_planner=args.replica_planner, replica_cost_model=calibration).to(
                                device=device, dtype=torch.bfloat16)
    values = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16).requires_grad_()
    probabilities = torch.full((args.tokens, args.top_k), 1 / args.top_k, device=device)
    gradient = torch.randn_like(values) / args.tokens
    optimizer = torch.optim.SGD(module.parameters(), lr=0.001, momentum=0.9)
    routes = _routes(args.tokens, args.top_k, size * 6, device)
    try:
        cold = []
        for pattern in ("balanced", "home_hot"):
            cold.append({"pattern": pattern, **_measure(module, optimizer, values, routes[pattern][0],
                                                        probabilities, gradient),
                         "memory": memory_snapshot(module, values)})
        results = {}
        for pattern, ids_list in routes.items():
            for index in range(args.warmup):
                _measure(module, optimizer, values, ids_list[index % len(ids_list)], probabilities, gradient)
            torch.npu.reset_peak_memory_stats()
            before = memory_snapshot(module, values)
            samples = [_measure(module, optimizer, values, ids_list[index % len(ids_list)], probabilities, gradient)
                       for index in range(args.iterations)]
            after = memory_snapshot(module, values)
            if before["capacity"] != after["capacity"] or before["heap_epoch"] != after["heap_epoch"]:
                raise AssertionError("Warm timing included a heap growth")
            plans = [_measure(module, optimizer, values, ids, probabilities, gradient,
                               inspect=True, cost_model=calibration) for ids in ids_list]
            results[pattern] = {"samples": samples, "median_ms": statistics.median(
                row["rank_max_step_ms"] for row in samples), "memory_before": before, "memory_after": after,
                "torch_peak_plus_constant_heap": after["torch_peak_allocated"] + after["shmem_heap_reserved"],
                "torch_reserved_peak_plus_constant_heap": after["torch_peak_reserved"] + after["shmem_heap_reserved"],
                "plans": plans}
        diagnostic = None
        if args.diagnose:
            diagnostic = _diagnose(module, optimizer, values, routes[args.diagnose_pattern][0], probabilities, gradient,
                                   directory, results[args.diagnose_pattern]["plans"][0])
        host_breakdown = None
        if args.host_breakdown:
            host_breakdown = {pattern: _host_breakdown(module, optimizer, values, routes[pattern][0],
                                                       probabilities, gradient)
                              for pattern in ("balanced", "home_hot")}
        vendor, adapter = get_multicore_paths()
        identity = execution_identity()
        identity.update(torch_npu=torch_npu.__version__, configuration=vars(args),
                        ep_members=dist.get_process_group_ranks(dist.group.WORLD),
                        adapter=file_identity(adapter), kernels=[file_identity(p) for p in sorted(vendor.rglob("*.o"))],
                        transport_abi=torch.ops.hyper_parallel.mega_moe_transport_version())
        (directory / "result.json").write_text(json.dumps({"identity": identity, "cold": cold, "warm": results,
                                                          "diagnostic": diagnostic, "host_breakdown": host_breakdown,
                                                          "calibration":
                                                          None if calibration is None else asdict(calibration)},
                                                         indent=2) + "\n", encoding="utf-8")
        if rank == 0:
            print(json.dumps({pattern: value["median_ms"] for pattern, value in results.items()}), flush=True)
    finally:
        module.close()


def main() -> None:
    """Select the production variant without modifying constructors or schedules."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("push", "pull"), default="push")
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--replica-transport", default="p2p")
    parser.add_argument("--replica-planner", choices=("cpu", "device"), default="cpu")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=1792)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--cost-model")
    parser.add_argument("--host-breakdown", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--diagnose-pattern", choices=("balanced", "home_hot"), default="home_hot")
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    if args.iterations < 2 or args.warmup < 1:
        raise ValueError("Benchmark requires at least two measured and one warmup iteration")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
