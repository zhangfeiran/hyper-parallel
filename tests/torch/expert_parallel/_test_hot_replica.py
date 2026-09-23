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
"""Native and multicore expert-replica distributed precision acceptance."""

import argparse
import importlib
import sys
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import init_device_mesh
from hyper_parallel.components.modules.moe import GroupedExperts
from hyper_parallel.core.expert_parallel import ExpertParallel
from hyper_parallel.core.expert_parallel.hot_replica import build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute
from hyper_parallel.core.expert_parallel.hot_replica.signal_transport import (
    SIGNAL_TRANSPORT_MODES, signal_transport_options,
)
from hyper_parallel.core.expert_parallel.hot_replica.transport import prefetch_weights, return_gradients
from tests.common.port_utils import allocate_port


def _local(tensor):
    """Read the local EP shard when a parameter or gradient is a DTensor."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _native_forward(module, values, ids, probabilities, experts):
    """Use the real GroupedExperts entry with EP pre/post forward hooks."""
    order = torch.argsort(ids.flatten(), stable=True)
    expanded = values[:, None, :].expand(-1, ids.shape[1], -1).reshape(-1, values.shape[-1])
    counts = torch.bincount(ids.flatten(), minlength=experts)
    output = module(expanded[order], counts)
    restored = torch.empty_like(output)
    restored[order] = output
    return (restored.reshape(values.shape[0], ids.shape[1], -1) * probabilities.unsqueeze(-1)).sum(1)


def _expert_forward(module, executor, values, ids, probabilities, experts):
    """Select the controlled reference executor without changing parameters."""
    if executor is None:
        return _native_forward(module, values, ids, probabilities, experts)
    packed = torch.cat((_local(module.w1), _local(module.w3)), dim=1).transpose(1, 2).contiguous()
    down = _local(module.w2).transpose(1, 2).contiguous()
    return executor(values, ids, probabilities, expert_weights=(packed, down))


def _check(actual, expected, label, *, elementwise=True, rtol=2e-2, atol=2e-3):
    """Check precision collectively so failure cannot strand peers in cleanup."""
    actual, expected = actual.detach().float(), expected.detach().float()
    difference = actual - expected
    relative = float(difference.square().sum().sqrt() / expected.square().sum().sqrt().clamp_min(1e-8))
    maximum = float(difference.abs().max())
    error = None
    try:
        if elementwise:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        if not torch.isfinite(actual).all() or relative > 0.01:
            raise AssertionError(f"relative_l2={relative}, max_abs={maximum}")
    except AssertionError as failure:
        error = f"rank={dist.get_rank()} {label}: {failure}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    failures = [error for error in errors if error is not None]
    if failures:
        raise AssertionError("\n".join(failures))
    return {"relative_l2": relative, "max_abs": maximum}


def _deferred_backward(base, candidate, executor, experts, rank, device, tokens, hidden, top_k, reference):
    """Keep different plans live, then run backward in reverse invocation order."""
    # Isolate saved-plan lifetime from tolerated optimizer-rounding drift.
    with torch.no_grad():
        for name in ("w1", "w2", "w3"):
            _local(getattr(candidate, name)).copy_(_local(getattr(base, name)))
    base.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    partials = ({}, {})
    accumulated = ({}, {})
    hooks = []
    for index, module in enumerate((base, candidate)):
        for name in ("w1", "w2", "w3"):
            def _capture(gradient, destination=partials[index], key=name):
                destination[key] = _local(gradient).detach().clone()
            hooks.append(getattr(module, name).register_hook(_capture))
    pending = []
    for modulus in (max(6, top_k), 2 if top_k <= 6 else experts):
        torch.manual_seed(833 + rank + modulus)
        values = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device).requires_grad_()
        other = values.detach().clone().requires_grad_()
        ids = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k).remainder(modulus).long()
        probabilities = torch.full((tokens, top_k), 1.0 / top_k, device=device, requires_grad=True)
        other_probabilities = probabilities.detach().clone().requires_grad_()
        expected = _expert_forward(base, reference, values, ids, probabilities, experts)
        actual = _expert_forward(candidate, executor, other, ids, other_probabilities, experts)
        pending.append((expected, actual, values, other, probabilities, other_probabilities))
    checks = []
    for expected, actual, values, other, probabilities, other_probabilities in reversed(pending):
        gradient = torch.randn_like(expected)
        expected.backward(gradient)
        actual.backward(gradient)
        for name in ("w1", "w2", "w3"):
            _check(partials[1][name], partials[0][name], "deferred partial " + name)
            for index, module in enumerate((base, candidate)):
                if name not in accumulated[index]:
                    accumulated[index][name] = partials[index][name].clone()
                else:
                    accumulated[index][name].add_(partials[index][name])
                _check(_local(getattr(module, name).grad), accumulated[index][name],
                       "exact accumulated " + name, rtol=0, atol=0)
        checks.append({"output": _check(actual, expected, "deferred output"),
                       "dx": _check(other.grad, values.grad, "deferred dx"),
                       "dprob": _check(other_probabilities.grad, probabilities.grad, "deferred dprob")})
    for hook in hooks:
        hook.remove()
    # Elementwise tolerances apply to each incoming partial above. BF16 summation
    # can amplify relative error at cancellation; exact accumulation is checked
    # against the captured partials, with a separate normwise comparison here.
    accumulated_error = {name: _check(_local(getattr(candidate, name).grad), _local(getattr(base, name).grad),
                                      "accumulated " + name, elementwise=False)
                         for name in ("w1", "w2", "w3")}
    return {"invocations": checks, "accumulated": accumulated_error}


def _cross_layer_pool(backend, budget, mesh, provider, executor, reference, tokens, hidden, intermediate, top_k):
    """Two independent parameter owners share storage through reversed backward."""
    experts = mesh.size() * 6
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    pairs = []
    for seed in (731, 951):
        modules = []
        for hot in (False, True):
            torch.manual_seed(seed)
            module = GroupedExperts(hidden, intermediate, experts, use_grouped_mm=True).to(
                device=device, dtype=torch.bfloat16)
            ExpertParallel(replica_slots_per_rank=budget if hot and backend == "native" else 0,
                           replica_transport=provider if hot else None).apply(module, mesh)
            modules.append(module)
        pairs.append(modules)
    pending = []
    for index, (base, candidate) in enumerate(pairs):
        torch.manual_seed(581 + index + dist.get_rank())
        values = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16).requires_grad_()
        other = values.detach().clone().requires_grad_()
        ids = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k).remainder(max(top_k, 2)).long()
        probs = torch.full((tokens, top_k), 1 / top_k, device=device)
        expected = _expert_forward(base, reference, values, ids, probs, experts)
        actual = _expert_forward(candidate, executor, other, ids, probs, experts)
        pending.append((base, candidate, values, other, expected, actual))
    checks = []
    for base, candidate, values, other, expected, actual in reversed(pending):
        grad = torch.randn_like(expected)
        expected.backward(grad)
        actual.backward(grad)
        checks.append({"output": _check(actual, expected, "cross-layer output"),
                       "dx": _check(other.grad, values.grad, "cross-layer dx"),
                       **{name: _check(_local(getattr(candidate, name).grad), _local(getattr(base, name).grad),
                                       "cross-layer " + name) for name in ("w1", "w2", "w3")}})
    return checks


def _signal_stress(provider, mesh, hidden, intermediate, budget):
    """Queue rotating owners on alternating streams before checking any result."""
    rank, size = dist.get_rank(), mesh.size()
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    snapshots = []
    for step in range(12):
        owner = step % size
        counts = [[0] * (size * 6) for _ in range(size)]
        for row in counts:
            row[owner * 6] = 120
        plan = build_expert_replica_plan(counts, budget)
        route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), rank, mesh.get_group(), provider)
        weights = (torch.full((6, hidden, 2 * intermediate), step * 10 + rank,
                              device=device, dtype=torch.bfloat16),
                   torch.full((6, intermediate, hidden), step * 10 + rank,
                              device=device, dtype=torch.bfloat16))
        stream = streams[step % 2]
        stream.wait_stream(torch.npu.current_stream())
        if rank == step % size:
            time.sleep(0.003)
        with torch.npu.stream(stream):
            with prefetch_weights(weights, route, backward=True, overlap=True) as pool:
                pool.wait_weights()
                copies = [(value[slot - 6].clone(), step * 10 + owner)
                          for slot, expert in enumerate(plan.slot_to_logical[rank]) if slot >= 6 and expert >= 0
                          for value in pool.weights]
                for gradient in pool.gradients:
                    gradient.fill_(rank + 1)
                home = tuple(torch.ones_like(weight, dtype=torch.float32) for weight in weights)
                gradients = return_gradients(home, route, pool.gradients)
                expected = [torch.ones_like(value) for value in gradients]
                if rank == owner:
                    for value in expected:
                        value[0].add_(sum(item.target_rank + 1 for item in plan.transfers))
                snapshots.append((copies, gradients, expected))
            for weight in weights:
                weight.record_stream(stream)
    torch.npu.synchronize()
    for copies, gradients, expected in snapshots:
        for actual, value in copies:
            torch.testing.assert_close(actual, torch.full_like(actual, value), rtol=0, atol=0)
        for actual, value in zip(gradients, expected):
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
    return {"iterations": 12, "streams": 2, "owners": size}


def _benchmark(candidate, executor, experts, tokens, hidden, top_k, iterations):
    """Measure fresh-process warmed forward/backward plus SGD on a fixed hot route."""
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(712 + dist.get_rank())
    values = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device).requires_grad_()
    ids = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k).remainder(max(6, top_k)).long()
    probabilities = torch.full((tokens, top_k), 1 / top_k, device=device)
    gradient = torch.randn_like(values) / tokens
    optimizer = torch.optim.SGD(candidate.parameters(), lr=0.001, momentum=0.9)
    measured = []
    for iteration in range(iterations + 3):
        optimizer.zero_grad(set_to_none=True)
        values.grad = None
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        output = _expert_forward(candidate, executor, values, ids, probabilities, experts)
        output.backward(gradient)
        optimizer.step()
        torch.npu.synchronize()
        elapsed = torch.tensor([(time.perf_counter() - start) * 1000], device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if iteration >= 3:
            measured.append(float(elapsed.cpu()[0]))
    return {"step_ms": measured, "iterations": iterations, "warmup": 3}


def run(backend: str, budget: int, result_dir: str, *, tokens: int = 128,
        hidden: int = 128, intermediate: int = 128, top_k: int = 2,
        same_backend_reference: bool = False, replica_transport: str = "p2p",
        benchmark_iterations: int = 0) -> None:
    """Compare full forward/backward with native B=0, preserving EP ownership."""
    rank, size = dist.get_rank(), dist.get_world_size()
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    home = 6
    experts = size * home
    mesh = init_device_mesh(device_type="npu", mesh_shape=(size,), mesh_dim_names=("ep",))
    torch.manual_seed(371)
    base = GroupedExperts(hidden, intermediate, experts, use_grouped_mm=True).to(device=device, dtype=torch.bfloat16)
    torch.manual_seed(371)
    candidate = GroupedExperts(hidden, intermediate, experts, use_grouped_mm=True).to(
        device=device, dtype=torch.bfloat16)
    provider = None
    symmetric = None
    shmem_api = None
    if backend != "native" or replica_transport != "p2p":
        endpoint = [f"tcp://127.0.0.1:{allocate_port()}" if rank == 0 else None]
        dist.broadcast_object_list(endpoint, src=0)
        os.environ["HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT"] = endpoint[0]
    if backend == "native" and replica_transport != "p2p":
        # Explicitly inject a test runtime; the native adapter has no multicore import.
        shmem_api = importlib.import_module("hyper_parallel.core.multicore.shmem")
        provider_type = importlib.import_module(
            "hyper_parallel.core.expert_parallel.hot_replica.one_sided").OneSidedReplicaTransport
        needed = budget * hidden * intermediate * 2 * 4
        if replica_transport in SIGNAL_TRANSPORT_MODES:
            signal_module = importlib.import_module("hyper_parallel.core.expert_parallel.hot_replica.signal_transport")
            provider_type = signal_module.SignalReplicaTransport
            needed = signal_module.signal_storage_bytes(
                ((hidden, 2 * intermediate), (intermediate, hidden)), budget, size, 2,
                projection_ready=signal_transport_options(replica_transport)["projection_ready"])
        heap = ((needed + 511 + 2**21 - 1) // 2**21) * 2**21
        shmem_api.acquire(mesh.get_group(), heap_size_bytes=heap)
        symmetric = shmem_api.empty((needed,), dtype=torch.uint8, alignment=512)
        if replica_transport in SIGNAL_TRANSPORT_MODES:
            provider = provider_type(shmem_api, symmetric, budget, size,
                                     **signal_transport_options(replica_transport))
        else:
            provider = provider_type(shmem_api, symmetric)
    ExpertParallel().apply(base, mesh)
    ExpertParallel(replica_slots_per_rank=budget if backend == "native" else 0,
                   replica_transport=provider).apply(candidate, mesh)
    executor = None
    reference = None
    if backend != "native":
        mega_moe = importlib.import_module("hyper_parallel.core.multicore").MegaMoeExperts
        options = {"initial_capacity_factor": 1.0} if backend == "push" else {}
        executor = mega_moe(local_num_tokens=tokens, hidden_size=hidden, intermediate_size=intermediate,
                                 num_experts=experts, top_k=top_k, ep_size=size, ep_group=mesh.get_group(),
                                 create_parameters=False, dispatch_mode=backend, replica_slots_per_rank=budget,
                                 replica_transport=replica_transport,
                                 **options)
        if same_backend_reference:
            reference = mega_moe(local_num_tokens=tokens, hidden_size=hidden, intermediate_size=intermediate,
                                 num_experts=experts, top_k=top_k, ep_size=size, ep_group=mesh.get_group(),
                                 create_parameters=False, dispatch_mode=backend, replica_slots_per_rank=0,
                                 **options)
    results = []
    optimizer_base = torch.optim.SGD(base.parameters(), lr=0.01, momentum=0.9)
    optimizer_candidate = torch.optim.SGD(candidate.parameters(), lr=0.01, momentum=0.9)
    try:
        for pattern in ("balanced", "home_hot", "one_hot_pair", "balanced"):
            torch.manual_seed(719 + rank)
            x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16).requires_grad_()
            x_candidate = x.detach().clone().requires_grad_()
            positions = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k)
            modulus = experts if pattern == "balanced" else (max(home, top_k) if pattern == "home_hot" else top_k)
            ids = positions.remainder(modulus).long()
            probs = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1).requires_grad_()
            probs_candidate = probs.detach().clone().requires_grad_()
            base.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            expected = _expert_forward(base, reference, x, ids, probs, experts)
            actual = _expert_forward(candidate, executor, x_candidate, ids, probs_candidate, experts)
            gradient = torch.randn_like(expected)
            expected.backward(gradient)
            actual.backward(gradient)
            evidence = {"pattern": pattern, "output": _check(actual, expected, "output"),
                        "dx": _check(x_candidate.grad, x.grad, "dx"),
                        "dprob": _check(probs_candidate.grad, probs.grad, "dprob")}
            for name in ("w1", "w2", "w3"):
                evidence[name] = _check(_local(getattr(candidate, name).grad), _local(getattr(base, name).grad), name)
            if executor is not None:
                resources = executor._get_execution_resources(x_candidate)
                evidence["capacity"] = resources.workspace.capacity_floor
                evidence["maximum_capacity"] = resources.spec.maximum_receive_capacity
                evidence["heap_epoch"] = resources.heap_manager.epoch
                if evidence["capacity"] > evidence["maximum_capacity"]:
                    raise AssertionError("dynamic push capacity exceeded theoretical bound")
            if replica_transport in SIGNAL_TRANSPORT_MODES:
                active_provider = provider if executor is None else resources.workspace.replica_provider
                evidence["gradient_scratch_bytes"] = active_provider.gradient_scratch_bytes
                evidence["gradient_scratch_limit_bytes"] = hidden * intermediate * 3 * 4
                if evidence["gradient_scratch_bytes"] > evidence["gradient_scratch_limit_bytes"]:
                    raise AssertionError("gradient scratch exceeded one FP32 expert")
            optimizer_base.step()
            optimizer_candidate.step()
            for name in ("w1", "w2", "w3"):
                expected_parameter, actual_parameter = getattr(base, name), getattr(candidate, name)
                evidence[name + "_updated"] = _check(_local(actual_parameter), _local(expected_parameter), name)
                evidence[name + "_momentum"] = _check(
                    _local(optimizer_candidate.state[actual_parameter]["momentum_buffer"]),
                    _local(optimizer_base.state[expected_parameter]["momentum_buffer"]), name + " momentum")
            results.append(evidence)
            if rank == 0:
                print(json.dumps({"backend": backend, "B": budget, **evidence}), flush=True)
        if backend == "push" and budget == 1 and size >= 4:
            if not any(row["heap_epoch"] > 0 for row in results):
                raise AssertionError("expected a real dynamic push growth event")
        deferred = _deferred_backward(
            base, candidate, executor, experts, rank, device, tokens, hidden, top_k, reference)
        if rank == 0:
            print(json.dumps({"backend": backend, "B": budget, "deferred": deferred}), flush=True)
        if backend == "native" and replica_transport == "p2p" and any(
                name.startswith("hyper_parallel.core.multicore") for name in sys.modules):
            raise AssertionError("native execution imported multicore")
        cross_layer = []
        if hidden <= 128:
            cross_layer = _cross_layer_pool(backend, budget, mesh, provider, executor, reference,
                                           tokens, hidden, intermediate, top_k)
        signal_stress = None
        if replica_transport in SIGNAL_TRANSPORT_MODES and hidden <= 128:
            active_provider = provider if executor is None else resources.workspace.replica_provider
            signal_stress = _signal_stress(active_provider, mesh, hidden, intermediate, budget)
        timing = None if not benchmark_iterations else _benchmark(
            candidate, executor, experts, tokens, hidden, top_k, benchmark_iterations)
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        Path(result_dir, f"{backend}-b{budget}-rank{rank}.json").write_text(
            json.dumps({"steps": results, "deferred": deferred, "cross_layer": cross_layer,
                        "replica_transport": replica_transport, "signal_stress": signal_stress, "timing": timing},
                       indent=2) + "\n", encoding="utf-8")
    finally:
        if executor is not None:
            executor.close()
        if reference is not None:
            reference.close()
        if symmetric is not None:
            torch.npu.synchronize()
            shmem_api.free(symmetric)
            shmem_api.release()


def main() -> None:
    """Initialize one worker group and run the selected executor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "push", "pull"), default="native")
    parser.add_argument("--replica-transport",
                        choices=("p2p", "shmem", *SIGNAL_TRANSPORT_MODES),
                        default="p2p")
    parser.add_argument("--benchmark-iterations", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--same-backend-reference", action="store_true")
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    if int(os.environ["LOCAL_RANK"]) == 0:
        print(json.dumps({"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                          "cann": os.getenv("ASCEND_HOME_PATH"), "worker": str(Path(__file__).resolve())}), flush=True)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        run(args.backend, args.budget, args.result_dir, tokens=args.tokens,
            hidden=args.hidden, intermediate=args.intermediate, top_k=args.top_k,
            same_backend_reference=args.same_backend_reference, replica_transport=args.replica_transport,
            benchmark_iterations=args.benchmark_iterations)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


def test_native_hot_replica_npu() -> None:
    """Exercise native EP with no multicore import or payload."""
    _pytest_case("native")


def test_push_hot_replica_npu() -> None:
    """Exercise push replication and real dynamic heap growth."""
    _pytest_case("push")


def test_pull_hot_replica_npu() -> None:
    """Exercise pull replication without receive-heap growth."""
    _pytest_case("pull")


def _pytest_case(backend: str) -> None:
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        run(backend, 1, os.getenv("HP_HOT_REPLICA_RESULTS", "./logs/hot_replica"))
    finally:
        dist.destroy_process_group()
