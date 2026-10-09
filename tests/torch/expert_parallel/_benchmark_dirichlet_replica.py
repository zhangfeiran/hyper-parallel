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
"""Route-matched native/MegaMoe B=0/1 checkpointed MoE training sweep."""

import argparse
from datetime import timedelta
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu
from torch.utils.checkpoint import checkpoint

from hyper_parallel import DTensor, Shard, init_device_mesh
from hyper_parallel.components.modules.moe import GroupedExperts
from hyper_parallel.core.expert_parallel import ExpertParallel
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore._loader import get_multicore_paths
from hyper_parallel.core.multicore.profiler import mega_kernel_profile
from tests.common.port_utils import allocate_port
from tests.torch.expert_parallel.dirichlet_routes import RouteShape, SOURCE_REVISION, make_dirichlet_route, route_pairs
from tests.torch.expert_parallel.hot_replica_checks import execution_identity, file_identity, saved_route
from tests.torch.expert_parallel.hot_replica_measurements import HostMeasurements

ADAMW_OPTIONS = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8,
                 "weight_decay": 0.01, "foreach": False, "fused": False}


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def _native_forward(module, values, ids, scores, counts):
    routed, mapping = torch_npu.npu_moe_token_permute(values, ids)
    output = module(routed, counts)
    return torch_npu.npu_moe_token_unpermute(output, mapping, probs=scores.float().contiguous())


def _forward(module, values, ids, scores, counts):
    if isinstance(module, MegaMoeExperts):
        return module(values, ids, scores)
    return _native_forward(module, values, ids, scores, counts)


def _make_layer(args, mesh, index, backend=None, budget=None):
    backend = backend or args.backend
    budget = args.budget if budget is None else budget
    rank, size = dist.get_rank(), dist.get_world_size()
    home = args.experts // size
    # Local seeded shards avoid materializing E full matrices on every device.
    generator = torch.Generator().manual_seed(2026 + index * 100 + rank * 100000)
    bound = 1 / math.sqrt(args.hidden * args.intermediate)
    weights = {name: torch.empty(shape).uniform_(-bound, bound, generator=generator).to(torch.bfloat16)
               for name, shape in (("w1", (home, args.intermediate, args.hidden)),
                                   ("w2", (home, args.hidden, args.intermediate)),
                                   ("w3", (home, args.intermediate, args.hidden)))}
    identity = {name: _digest(value) for name, value in weights.items()}
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    if backend == "native":
        with torch.device("meta"):
            module = GroupedExperts(args.hidden, args.intermediate, args.experts, use_grouped_mm=True)
        ExpertParallel(replica_slots_per_rank=budget, replica_planner=args.replica_planner).apply(module, mesh)
        for name, value in weights.items():
            local = value.to(device)
            global_shape = (args.experts, *value.shape[1:])
            setattr(module, name, torch.nn.Parameter(DTensor.from_local(local, mesh, [Shard(0)], shape=global_shape)))
    else:
        module = MegaMoeExperts(local_num_tokens=args.tokens, hidden_size=args.hidden,
                                intermediate_size=args.intermediate, num_experts=args.experts, top_k=args.top_k,
                                ep_size=size, ep_group=dist.group.WORLD, dispatch_mode="push",
                                initial_capacity_factor=1.5, replica_slots_per_rank=budget,
                                replica_transport=args.replica_transport, replica_planner=args.replica_planner,
                                replica_target_load=args.replica_target_load).to(
                                    device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            module.gate_up_weight.copy_(torch.cat((weights["w1"].transpose(1, 2),
                                                  weights["w3"].transpose(1, 2)), dim=-1).to(device))
            module.down_weight.copy_(weights["w2"].transpose(1, 2).to(device))
    return module, identity


def _parameters(module):
    return [_local(value) for value in module.parameters()]


def _transport_evidence(module, values, expected):
    if not isinstance(module, MegaMoeExperts):
        return {"actual_transport": "p2p"}
    resources = module._get_execution_resources(values)  # pylint: disable=protected-access
    provider = resources.workspace.replica_provider
    evidence = {"actual_transport": resources.spec.replica_transport,
                "provider": type(provider).__name__,
                "kernel_gradients": getattr(provider, "kernel_gradients", False),
                "adaptive_gradients": getattr(provider, "adaptive_gradients", False),
                "last_kernel_gradient_matrices": list(getattr(provider, "last_kernel_gradient_matrices", ())),
                "overlap_home": getattr(provider, "overlap_home", False),
                "planner_target_load": module.replica_target_load,
                "w2_ready_events": list(resources.plan.replica_w2_events),
                "w13_ready_events": list(resources.plan.replica_w13_events)}
    matrices = evidence["last_kernel_gradient_matrices"]
    evidence["effective_gradient_transport"] = (
        "none" if not module.replica_slots_per_rank else
        "p2p" if expected == "p2p" else
        "kernel_gradient" if matrices == [1, 0] else
        "kernel_w2_sdma_w13" if matrices == [1] else "sdma_parallel")
    if evidence["actual_transport"] != expected:
        raise AssertionError("Requested replica transport did not execute")
    if module.replica_slots_per_rank and expected in (
            "shmem_signal_kernel_gradient", "shmem_signal_kernel_gradient_adaptive"):
        if not evidence["kernel_gradients"] or not evidence["w2_ready_events"]:
            raise AssertionError("Kernel-gradient provider and W2 schedule were not enabled")
    return evidence


def _finite(tensors):
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    flags = [torch.isfinite(value).all().to(device) for value in tensors if value is not None]
    valid = torch.stack(flags).all().to(torch.int32)
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not bool(valid.cpu()):
        raise AssertionError("Nonfinite output, gradient, parameter or optimizer state")


def _memory(layers, values, backend):
    heap, capacity, epoch = 0, None, None
    if backend == "megamoe":
        resources = layers[0]._get_execution_resources(values)  # pylint: disable=protected-access
        heap = resources.heap_manager.heap_bytes
        capacity, epoch = resources.workspace.capacity_floor, resources.heap_manager.epoch
    return {"allocated": torch.npu.memory_allocated(), "reserved": torch.npu.memory_reserved(),
            "peak_allocated": torch.npu.max_memory_allocated(), "peak_reserved": torch.npu.max_memory_reserved(),
            "shmem_heap": heap, "total_peak_bytes": torch.npu.max_memory_allocated() + heap,
            "total_reserved_peak_bytes": torch.npu.max_memory_reserved() + heap,
            "capacity": capacity, "heap_epoch": epoch}


def _step(layers, inputs, gradient, route, optimizer, validate=False):
    optimizer.zero_grad(set_to_none=True)
    outputs, leaves = [], []
    for layer, data in zip(layers, inputs):
        values = data.detach().requires_grad_()
        scores = route[1].detach().requires_grad_()
        output = checkpoint(partial(_forward, layer, counts=route[2]), values, route[0], scores, use_reentrant=True)
        outputs.append(output)
        if validate:
            _finite([output])
            leaves.extend((values, scores))
    del output, values, scores
    while outputs:
        outputs.pop().backward(gradient)
    if validate:
        parameters = [value for layer in layers for value in _parameters(layer)]
        if any(value.grad is None for value in [*leaves, *parameters]):
            raise AssertionError("Missing input, router or parameter gradient")
        _finite([value.grad for value in [*leaves, *parameters]])


def _statistics(samples):
    return {"samples": samples, "median": statistics.median(samples), "minimum": min(samples),
            "maximum": max(samples)}


def _point(args, layers, inputs, gradient, optimizer, alpha, seed, directory):
    shape = RouteShape(args.tokens, args.experts, args.top_k)
    cpu_route = make_dirichlet_route(shape, alpha, seed)
    route = tuple(value.to(inputs[0].device) for value in cpu_route)
    counts = cpu_route[2].long() * dist.get_world_size()
    loads = counts.reshape(dist.get_world_size(), -1).sum(1).tolist()
    route_hash = _digest(cpu_route[0])
    _step(layers, inputs, gradient, route, optimizer, validate=True)
    optimizer.step()
    for _ in range(args.warmup):
        _step(layers, inputs, gradient, route, optimizer)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.npu.synchronize()
    before = _memory(layers, inputs[0], args.backend)
    parameter_versions = [value._version for layer in layers for value in _parameters(layer)]
    torch.npu.reset_peak_memory_stats()
    samples, windows = [], []
    for index in range(args.measured):
        dist.barrier()
        torch.npu.synchronize()
        started = time.monotonic_ns()
        _step(layers, inputs, gradient, route, optimizer)
        torch.npu.synchronize()
        backward_end = time.monotonic_ns()
        optimizer.step()
        torch.npu.synchronize()
        ended = time.monotonic_ns()
        durations = torch.tensor([(backward_end - started) / 1e6, (ended - started) / 1e6],
                                 device=inputs[0].device)
        dist.all_reduce(durations, op=dist.ReduceOp.MAX)
        samples.append(durations.cpu().tolist())
        windows.append({"sample": index, "start_ns": started,
                        "fwd_bwd_end_ns": backward_end, "end_ns": ended})
    after = _memory(layers, inputs[0], args.backend)
    updated_versions = [value._version for layer in layers for value in _parameters(layer)]
    if any(after_version <= before_version for before_version, after_version in
           zip(parameter_versions, updated_versions)):
        raise AssertionError("AdamW did not update every original parameter during measured steps")
    if (before["capacity"], before["heap_epoch"]) != (after["capacity"], after["heap_epoch"]):
        raise AssertionError("Measured steps included SHMEM growth")
    _step(layers, inputs, gradient, route, optimizer, validate=True)
    optimizer_tensors = [value for state in optimizer.state.values()
                         for value in state.values() if isinstance(value, torch.Tensor)]
    _finite([value for layer in layers for value in _parameters(layer)] + optimizer_tensors)
    evidence = None
    if args.budget:
        output = _forward(layers[0], inputs[0].detach().requires_grad_(), *route[:2], route[2])
        plan = saved_route(output).plan
        evidence = {"destination_loads": plan.destination_loads, "slots": plan.slot_to_logical,
                    "transfers": len(plan.transfers)}
        del output
    result = {"alpha": alpha, "seed": seed, "route_sha256": route_hash, "expert_counts": counts.tolist(),
              "home_destination_loads": loads, "skew": max(loads) / statistics.mean(loads),
              "fwd_bwd_ms": _statistics([value[0] for value in samples]),
              "step_ms": _statistics([value[1] for value in samples]), "memory_before": before,
              "memory": after, "windows": windows, "replica_plan": evidence,
              "transport": _transport_evidence(layers[0], inputs[0], args.replica_transport),
              "finite_validation": True, "parameter_versions_before": parameter_versions,
              "parameter_versions_after": updated_versions, "pid": os.getpid()}
    path = directory / f"alpha{alpha:g}-seed{seed}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if dist.get_rank() == 0:
        print(json.dumps({"alpha": alpha, "seed": seed, "skew": result["skew"],
                          "step_ms": result["step_ms"]["median"], "hbm_gib": after["total_peak_bytes"] / 2**30}),
              flush=True)
    optimizer.zero_grad(set_to_none=True)
    return result


def _compare(actual, expected, label):
    difference = actual.detach().float() - expected.detach().float()
    relative = difference.norm() / expected.detach().float().norm().clamp_min(1e-12)
    maximum = difference.abs().max()
    metrics = torch.stack((relative, maximum))
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    result = metrics.cpu().tolist()
    if not math.isfinite(result[0]) or result[0] > 0.01:
        raise AssertionError(f"{label}: relative_l2={result[0]}, max_abs={result[1]}")
    return {"relative_l2_rank_max": result[0], "max_abs_rank_max": result[1]}


def _diagnose(args, layer, inputs, gradient, optimizer, alpha, seed, directory):
    """Capture one checkpointed MoE layer separately from formal timing."""
    route = tuple(value.to(inputs.device) for value in
                  make_dirichlet_route(RouteShape(args.tokens, args.experts, args.top_k), alpha, seed))
    _step([layer], [inputs], gradient, route, optimizer)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.npu.synchronize()
    output = _forward(layer, inputs.detach().requires_grad_(), *route[:2], route[2])
    replica_route = saved_route(output)
    plan = replica_route.plan
    evidence = {"destination_loads": plan.destination_loads, "slots": plan.slot_to_logical,
                "destination_counts": plan.destination_counts, "transfers": len(plan.transfers)}
    del output
    torch.npu.synchronize()
    dist.barrier()
    prefix = directory / f"diagnostic-alpha{alpha:g}-seed{seed}"
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                                torch_npu.profiler.ProfilerActivity.NPU]) as outer:
        with mega_kernel_profile(detailed_task_names=True) as internal, HostMeasurements() as host:
            with torch.profiler.record_function("checkpointed_moe_fwd_bwd"):
                _step([layer], [inputs], gradient, route, optimizer)
            with torch.profiler.record_function("adamw"):
                optimizer.step()
            torch.npu.synchronize()
    outer.export_chrome_trace(str(prefix) + "-framework.json")
    trace = internal.export_chrome_trace(str(prefix) + "-internal.json")
    if trace["megaKernelCycleTrace"]["droppedRecordCount"]:
        raise AssertionError("Diagnostic internal trace dropped records")
    waits = [event for event in trace["traceEvents"]
             if event.get("args", {}).get("task_stage") == "ReplicaWeightReadyWait"]
    home = args.experts // dist.get_world_size()
    if args.replica_transport != "p2p" and any(plan.destination_counts[dist.get_rank()][home:]) and not waits:
        raise AssertionError("Active guest has no weight-ready wait trace")
    metadata = {"alpha": alpha, "seed": seed, "plan": evidence, "host_spans": host.records,
                "transport": _transport_evidence(layer, inputs, args.replica_transport),
                "scope": "Profiler-on single MoE layer: original forward, checkpoint recompute, backward and AdamW. "
                         "Inclusive host and stream spans overlap; not formal 16-layer step timing."}
    Path(str(prefix) + "-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _logical_gradients(module):
    if isinstance(module, GroupedExperts):
        return {name: _local(getattr(module, name).grad) for name in ("w1", "w2", "w3")}
    gate, up = module.gate_up_weight.grad.chunk(2, dim=-1)
    return {"w1": gate.transpose(1, 2), "w2": module.down_weight.grad.transpose(1, 2), "w3": up.transpose(1, 2)}


def _accept(args, mesh, module, values, gradient, points):
    reference, _ = _make_layer(args, mesh, 0, backend="native", budget=0)
    results = []
    try:
        for alpha, seed in points:
            reference.zero_grad(set_to_none=True)
            module.zero_grad(set_to_none=True)
            ids, scores, counts = (value.to(values.device) for value in
                                   make_dirichlet_route(RouteShape(args.tokens, args.experts, args.top_k), alpha, seed))
            actual_input, expected_input = [values.detach().clone().requires_grad_() for _ in range(2)]
            actual_scores, expected_scores = [scores.detach().clone().requires_grad_() for _ in range(2)]
            expected = _forward(reference, expected_input, ids, expected_scores, counts)
            expected.backward(gradient)
            actual = _forward(module, actual_input, ids, actual_scores, counts)
            actual.backward(gradient)
            metrics = {"output": _compare(actual, expected, "output"),
                       "input_gradient": _compare(actual_input.grad, expected_input.grad, "input gradient"),
                       "router_gradient": _compare(actual_scores.grad, expected_scores.grad, "router gradient")}
            expected_gradients = _logical_gradients(reference)
            for name, value in _logical_gradients(module).items():
                metrics[name] = _compare(value, expected_gradients[name], name)
            results.append({"alpha": alpha, "seed": seed, "checks": metrics,
                            "transport": _transport_evidence(module, values, args.replica_transport)})
    finally:
        del reference
    return results


def run(args: argparse.Namespace) -> None:
    """Run a persistent layer sweep, keeping diagnostics outside timing."""
    rank, size = dist.get_rank(), dist.get_world_size()
    directory = Path(args.result_dir) / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    mesh = init_device_mesh("npu", (size,), mesh_dim_names=("ep",))
    if args.backend == "megamoe":
        endpoint = [f"tcp://127.0.0.1:{allocate_port()}" if rank == 0 else None]
        dist.broadcast_object_list(endpoint, src=0)
        os.environ["HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT"] = endpoint[0]
    points = route_pairs()
    if args.pairs:
        points = [(float(alpha), int(seed)) for alpha, seed in (value.split(":") for value in args.pairs.split(","))]
    layers, identities = [], []
    try:
        for index in range(1 if args.accept else args.layers):
            layer, identity = _make_layer(args, mesh, index)
            layers.append(layer)
            identities.append(identity)
        if args.backend == "megamoe":
            MegaMoeExperts.share_execution_resources(layers)
        torch.manual_seed(30000 + rank)
        torch.npu.manual_seed(30000 + rank)
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        values = torch.randn(args.tokens, args.hidden, device=device).to(torch.bfloat16)
        gradient = torch.randn(args.tokens, args.hidden, device=device).to(torch.bfloat16)
        inputs = [values.clone() for _ in layers]
        identity = execution_identity()
        root = Path(identity["root"])
        identity.update(configuration=vars(args), torch_npu=torch_npu.__version__,
                        initial_weights=identities, input_sha256=_digest(values), gradient_sha256=_digest(gradient),
                        route_source_revision=SOURCE_REVISION, pid=os.getpid(),
                        process_start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8").split()[21])
        identity["measurement_scripts"] = {name: file_identity(root / "scripts" / name)
                                           for name in ("run_dirichlet_replica_sweep.py",
                                                        "plot_dirichlet_replica_sweep.py")}
        if args.backend == "megamoe":
            vendor, adapter = get_multicore_paths()
            identity.update(adapter=file_identity(adapter), kernels=[file_identity(path) for path in
                                                                     sorted(vendor.rglob("*.o"))])
        (directory / "identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        if args.accept:
            results = _accept(args, mesh, layers[0], values, gradient, points)
        else:
            optimizer = torch.optim.AdamW([value for layer in layers for value in _parameters(layer)], **ADAMW_OPTIONS)
            results = [_point(args, layers, inputs, gradient, optimizer, alpha, seed, directory)
                       for alpha, seed in points]
            if args.diagnose:
                for alpha, seed in points:
                    _diagnose(args, layers[0], inputs[0], gradient, optimizer, alpha, seed, directory)
        identity["transport"] = _transport_evidence(layers[0], values, args.replica_transport)
        (directory / "identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        (directory / "complete.json").write_text(json.dumps({"points": len(results), "results": results}, indent=2)
                                                + "\n", encoding="utf-8")
        if rank == 0:
            print(f"COMPLETE {args.backend} B={args.budget}: {len(results)} routes", flush=True)
    finally:
        for layer in layers:
            if isinstance(layer, MegaMoeExperts):
                layer.close()


def main() -> None:
    """Configure one fresh-process variant; initialize devices only in workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "megamoe"), required=True)
    parser.add_argument("--budget", type=int, choices=(0, 1), required=True)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=1792)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=7)
    parser.add_argument("--replica-planner", choices=("cpu", "device"), default="cpu")
    parser.add_argument("--replica-transport", choices=("p2p", "shmem_signal_kernel_gradient",
                                                     "shmem_signal_kernel_gradient_adaptive"), default="p2p")
    parser.add_argument("--replica-target-load", type=int)
    parser.add_argument("--diagnose", action="store_true", help="Capture separate single-layer diagnostic traces.")
    parser.add_argument("--pairs")
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    if args.backend == "native" and args.replica_transport != "p2p":
        parser.error("native supports only p2p replica transport")
    if args.replica_target_load is not None and (args.backend != "megamoe" or args.replica_target_load <= 0):
        parser.error("replica-target-load must be positive and is supported only for megamoe")
    if args.diagnose and (args.backend != "megamoe" or args.accept or not args.budget):
        parser.error("diagnose requires a megamoe B=1 timing run")
    if args.experts % int(os.environ["WORLD_SIZE"]) or min(args.layers, args.measured, args.hidden,
                                                           args.intermediate) <= 0 or args.warmup < 0:
        parser.error("experts must divide WORLD_SIZE; dimensions/measurements must be positive")
    RouteShape(args.tokens, args.experts, args.top_k)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=300))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
