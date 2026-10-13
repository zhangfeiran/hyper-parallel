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

from __future__ import annotations

import importlib
import json
import os
import sys
from contextlib import nullcontext
from datetime import timedelta
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import init_device_mesh
from hyper_parallel.components.modules.moe import GroupedExperts
from hyper_parallel.core.expert_parallel import ExpertParallel
from hyper_parallel.core.expert_parallel.hot_replica import (
    ExpertReplicaConfig,
)
from hyper_parallel.core.expert_parallel.hot_replica.native import (
    combine_native_replicas,
    dispatch_native_replicas,
    native_replica_experts,
)
from hyper_parallel.core.expert_parallel.hot_replica.signal_transport import (
    KERNEL_GRADIENT_TRANSPORT,
)
from tests.common.port_utils import allocate_port
from tests.torch.expert_parallel.hot_replica_checks import (
    deferred_ids,
    execution_identity,
    file_identity,
    route_evidence,
)
from tests.torch.expert_parallel.test_hot_replica import TRAINING_CASES


def _local(tensor):
    """Read the local EP shard when a parameter or gradient is a DTensor."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _expert_forward(module, executor, values, ids, probabilities, experts):
    """Use real EP hooks for candidates and FP32 expert gradients for the reference."""
    if executor not in (None, "native_fp32"):
        packed = torch.cat((_local(module.w1), _local(module.w3)), dim=1).transpose(1, 2).contiguous()
        down = _local(module.w2).transpose(1, 2).contiguous()
        return executor(values, ids, probabilities, expert_weights=(packed, down))
    order = torch.argsort(ids.flatten(), stable=True)
    expanded = values[:, None, :].expand(-1, ids.shape[1], -1).reshape(-1, values.shape[-1])
    counts = torch.bincount(ids.flatten(), minlength=experts)
    if executor is None:
        output = module(expanded[order], counts)
    else:
        config = ExpertReplicaConfig(experts, dist.get_world_size(), 0)
        inputs, state = dispatch_native_replicas((expanded[order], counts), config, dist.group.WORLD)
        packed = torch.cat((_local(module.w1), _local(module.w3)), dim=1).transpose(1, 2).contiguous()
        down = _local(module.w2).transpose(1, 2).contiguous()
        output = native_replica_experts(inputs[0], packed, down, inputs[1], state.route)
        output = combine_native_replicas(output, state)
    restored = torch.empty_like(output)
    restored[order] = output
    return (restored.reshape(values.shape[0], ids.shape[1], -1) * probabilities.unsqueeze(-1)).sum(1)


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


def _deferred_backward(base, candidate, executor, experts, rank, device, tokens, hidden, top_k, reference,
                       replica_planner, *, growth=False, cross_stream=False):
    """Keep different plans live, then run backward in reverse invocation order."""
    # Isolate saved-plan lifetime from tolerated optimizer-rounding drift.
    with torch.no_grad():
        for name in ("w1", "w2", "w3"):
            _local(getattr(candidate, name)).copy_(_local(getattr(base, name)))
    base.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    pending = []
    plans = []
    capacities = []
    streams = [torch.npu.Stream(), torch.npu.Stream()] if cross_stream else []
    main_stream = torch.npu.current_stream() if cross_stream else None
    for invocation in range(2):
        torch.manual_seed(833 + rank + invocation)
        values = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device).requires_grad_()
        other = values.detach().clone().requires_grad_()
        ids = deferred_ids(tokens, top_k, experts, invocation, device)
        if growth and invocation == 0:
            ids = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k).remainder(experts).long()
        probabilities = torch.full((tokens, top_k), 1.0 / top_k, device=device, requires_grad=True)
        other_probabilities = probabilities.detach().clone().requires_grad_()
        stream = None if not streams else streams[invocation % 2]
        if stream is not None:
            stream.wait_stream(main_stream)
        with nullcontext() if stream is None else torch.npu.stream(stream):
            expected = _expert_forward(base, reference, values, ids, probabilities, experts)
            actual = _expert_forward(candidate, executor, other, ids, other_probabilities, experts)
        plans.append(route_evidence(actual, replica_planner))
        if growth:
            resources = executor._get_execution_resources(other)
            capacities.append(resources.workspace.capacity_floor)
        pending.append((expected, actual, values, other, probabilities, other_probabilities))
    active = plans[1]["transfers"] if growth else all(plan["transfers"] for plan in plans)
    if plans[0] == plans[1] or not active:
        raise AssertionError(f"Deferred acceptance requires distinct plans and active replicas: {plans}")
    if growth and capacities[1] <= capacities[0]:
        raise AssertionError(f"Expected heap growth with an old forward alive: {capacities}")
    checks = []
    previous_backward = None
    for invocation, (expected, actual, values, other, probabilities, other_probabilities) in reversed(
            list(enumerate(pending))):
        gradient = torch.randn_like(expected)
        stream = None if not streams else streams[1 - invocation % 2]
        if stream is not None:
            stream.wait_stream(main_stream)
            stream.wait_stream(streams[invocation % 2])
            if previous_backward is not None:
                stream.wait_stream(previous_backward)
        with nullcontext() if stream is None else torch.npu.stream(stream):
            expected.backward(gradient)
            actual.backward(gradient)
            checks.append({"output": _check(actual, expected, "deferred output"),
                           "dx": _check(other.grad, values.grad, "deferred dx"),
                           "dprob": _check(other_probabilities.grad, probabilities.grad, "deferred dprob"),
                           **{name: _check(_local(getattr(candidate, name).grad), _local(getattr(base, name).grad),
                                           "deferred accumulated " + name, elementwise=False)
                              for name in ("w1", "w2", "w3")}})
        previous_backward = stream
    if previous_backward is not None:
        main_stream.wait_stream(previous_backward)
    return {"invocations": checks, "plans": plans, "capacities": capacities,
            "streams": [stream.npu_stream for stream in streams]}


def _kernel_wait_stress(base, candidate, executor, experts, rank, device, tokens, hidden, top_k,
                        reference, replica_planner):
    """Publish ready words only after submitting the fused consumer kernel.

    Fault injection delays only ready publication, retaining its original SDMA
    stream and tensor lifetimes. It never calls wait_weights before computation.
    """
    runtime = importlib.import_module("hyper_parallel.core.multicore.shmem")
    function = importlib.import_module("hyper_parallel.core.multicore.modules.mega_moe.function")
    original_put = runtime.put
    pending, publications = [], []

    def _delayed_put(destination, source, peer, **options):
        if destination.dtype == torch.int32 and destination.numel() == 1:
            pending.append((torch.npu.current_stream(), destination, source, peer, options))
        else:
            original_put(destination, source, peer, **options)

    def _after_launch(launch, direction):
        @wraps(launch)
        def _wrapped(*args, **kwargs):
            result = launch(*args, **kwargs)
            # Publication remains on the already-forked SDMA stream, never
            # behind a consumer waiting for it on the main stream.
            publications.append({"direction": direction, "ready_words": len(pending)})
            for stream, destination, source, peer, options in pending:
                with torch.npu.stream(stream):
                    original_put(destination, source, peer, **options)
                    source.record_stream(stream)
            pending.clear()
            return result
        return _wrapped

    with (patch.object(runtime, "put", _delayed_put),
          patch.object(function, "_launch_forward_kernel", _after_launch(function._launch_forward_kernel, "forward")),
          patch.object(function, "_launch_backward_kernel",
                       _after_launch(function._launch_backward_kernel, "backward"))):
        checks = _deferred_backward(base, candidate, executor, experts, rank, device,
                                    tokens, hidden, top_k, reference, replica_planner)
    totals = torch.tensor([sum(item["ready_words"] for item in publications if item["direction"] == direction)
                           for direction in ("forward", "backward")], device=device)
    dist.all_reduce(totals)
    if not bool((totals > 0).all()) or pending:
        raise AssertionError("Delayed kernel-ready publication was not exercised in both directions")
    return {"checks": checks, "publications": publications, "global_ready_words": totals.cpu().tolist()}


def _run(backend: str, top_k: int, planner: str, transport: str, result_dir: Path) -> None:
    """Compare B=1 training and saved-plan lifetimes with the native FP32 reference."""
    rank, size = dist.get_rank(), dist.get_world_size()
    tokens = hidden = intermediate = 128
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    experts = size * 6
    mesh = init_device_mesh(device_type="npu", mesh_shape=(size,), mesh_dim_names=("ep",))
    modules = []
    for budget in (0, 1):
        torch.manual_seed(371)
        module = GroupedExperts(hidden, intermediate, experts, use_grouped_mm=True).to(
            device=device, dtype=torch.bfloat16)
        ExpertParallel(replica_slots_per_rank=budget if backend == "native" else 0,
                       replica_planner=planner if budget else "cpu",
                       replica_minimum_load_reduction=0.0).apply(module, mesh)
        modules.append(module)
    base, candidate = modules[0], modules[1]
    executor = None
    if backend != "native":
        endpoint = [f"tcp://127.0.0.1:{allocate_port()}" if rank == 0 else None]
        dist.broadcast_object_list(endpoint, src=0)
        os.environ["HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT"] = endpoint[0]
        mega_moe = importlib.import_module("hyper_parallel.core.multicore").MegaMoeExperts
        executor = mega_moe(local_num_tokens=tokens, hidden_size=hidden, intermediate_size=intermediate,
                           num_experts=experts, top_k=top_k, ep_size=size, create_parameters=False,
                           ep_group=None if planner == "cpu" else mesh.get_group(), dispatch_mode=backend,
                           replica_slots_per_rank=1, replica_transport=transport, replica_planner=planner,
                           replica_minimum_load_reduction=0.0,
                           **({"initial_capacity_factor": 1.0} if backend == "push" else {}))
    optimizer_base = torch.optim.SGD(base.parameters(), lr=0.01, momentum=0.9)
    optimizer_candidate = torch.optim.SGD(candidate.parameters(), lr=0.01, momentum=0.9)
    steps = []
    try:
        arguments = (base, candidate, executor, experts, rank, device, tokens, hidden, top_k, "native_fp32", planner)
        growth = _deferred_backward(*arguments, growth=True) if backend == "push" else None
        for pattern in ("balanced", "home_hot", "one_hot_pair", "balanced"):
            torch.manual_seed(719 + rank)
            x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16).requires_grad_()
            other = x.detach().clone().requires_grad_()
            modulus = {"balanced": experts, "home_hot": max(6, top_k), "one_hot_pair": top_k}[pattern]
            ids = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k).remainder(modulus).long()
            probs = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1).requires_grad_()
            other_probs = probs.detach().clone().requires_grad_()
            base.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            expected = _expert_forward(base, "native_fp32", x, ids, probs, experts)
            actual = _expert_forward(candidate, executor, other, ids, other_probs, experts)
            route = candidate._hot_replica_dispatch.route if backend == "native" else None
            plan = route_evidence(actual, planner, home_route=route if route and not route.plan.transfers else None)
            gradient = torch.randn_like(expected)
            expected.backward(gradient)
            actual.backward(gradient)
            checks = {"pattern": pattern, "plan": plan, "output": _check(actual, expected, "output"),
                      "dx": _check(other.grad, x.grad, "dx"), "dprob": _check(other_probs.grad, probs.grad, "dprob")}
            for name in ("w1", "w2", "w3"):
                checks[name] = _check(_local(getattr(candidate, name).grad), _local(getattr(base, name).grad), name)
            optimizer_base.step()
            optimizer_candidate.step()
            for name in ("w1", "w2", "w3"):
                expected_parameter, actual_parameter = getattr(base, name), getattr(candidate, name)
                checks[name + "_updated"] = _check(_local(actual_parameter), _local(expected_parameter), name)
                checks[name + "_momentum"] = _check(
                    _local(optimizer_candidate.state[actual_parameter]["momentum_buffer"]),
                    _local(optimizer_base.state[expected_parameter]["momentum_buffer"]), name + " momentum")
            steps.append(checks)
        deferred = _deferred_backward(*arguments, cross_stream=backend != "native")
        kernel_wait = _kernel_wait_stress(*arguments) if transport == KERNEL_GRADIENT_TRANSPORT else None
        identity = execution_identity()
        identity.update(torch_npu=torch_npu.__version__, backend=backend, budget=1,
                        shape=[tokens, hidden, intermediate, top_k], requested_planner=planner,
                        actual_planner=steps[0]["plan"]["planner"], requested_transport=transport,
                        actual_transport=transport, reference="native-fp32")
        if planner == "device":
            library = importlib.import_module(
                "hyper_parallel.core.expert_parallel.hot_replica.device_kernel")._planner_library()
            identity["planner_payload"] = {**file_identity(Path(library._name)), "abi": library.planner_abi_version()}
        if executor is None:
            if any(name.startswith("hyper_parallel.core.multicore") for name in sys.modules):
                raise AssertionError("Native execution imported multicore")
        else:
            resources = executor._get_execution_resources(other)
            provider = resources.workspace.replica_provider
            if transport == KERNEL_GRADIENT_TRANSPORT:
                if provider.gradient_scratch_bytes > hidden * intermediate * 3 * 4:
                    raise AssertionError(f"Gradient scratch exceeds one FP32 expert: {provider.gradient_scratch_bytes}")
                identity["kernel_gradient_matrices"] = provider.last_kernel_gradient_matrices
                if sorted(identity["kernel_gradient_matrices"]) != [0, 1]:
                    raise AssertionError(f"Both projections must return through the kernel: {identity}")
            loader = importlib.import_module("hyper_parallel.core.multicore._loader")
            vendor, adapter = loader.get_multicore_paths()
            identity["multicore_payload"] = [file_identity(adapter),
                                             file_identity(vendor / "op_api/lib/libcust_opapi.so")]
            identity["multicore_kernels"] = [file_identity(path) for path in sorted(vendor.rglob("*.o"))]
            identity["multicore_transport_abi"] = torch.ops.hyper_parallel.mega_moe_transport_version()
            identity["shmem_binding"] = file_identity(Path(sys.modules["hyper_parallel_shmem_torch"].__file__))
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"{backend}-b1-rank{rank}.json").write_text(json.dumps(
            {"identity": identity, "steps": steps, "growth_deferred": growth,
             "deferred": deferred, "kernel_wait": kernel_wait}, indent=2) + "\n", encoding="utf-8")
    finally:
        if executor is not None:
            executor.close()


@pytest.mark.parametrize("case,backend,planner,transport", TRAINING_CASES,
                         ids=[case[0] for case in TRAINING_CASES])
def test_replica_training_npu(case: str, backend: str, planner: str, transport: str) -> None:
    """Exercise final transports, planners and invocation/stream lifetimes."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        topks = (1, 2, 3, 4, 5, 6, 8) if case == "native_cpu" else (2 if backend == "native" else 8,)
        for top_k in topks:
            directory = Path(os.getenv("HP_HOT_REPLICA_RESULTS", "./logs/hot_replica")) / case / f"k{top_k}"
            _run(backend, top_k, planner, transport, directory)
    finally:
        dist.destroy_process_group()
