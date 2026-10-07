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

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from functools import wraps
from typing import Any
from unittest.mock import patch
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
from hyper_parallel.core.expert_parallel.hot_replica import (
    ExpertReplicaConfig, ExpertReplicaCostModel, build_expert_replica_plan,
)
from hyper_parallel.core.expert_parallel.hot_replica.native import (
    dispatch_native_replicas, combine_native_replicas, native_replica_experts,
)
from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute
from hyper_parallel.core.expert_parallel.hot_replica.signal_transport import (
    SIGNAL_TRANSPORT_MODES, SignalReplicaTransport,
)
from hyper_parallel.core.expert_parallel.hot_replica.transport import prefetch_weights, return_gradients
from tests.common.port_utils import allocate_port
from tests.torch.expert_parallel.hot_replica_checks import (
    deferred_ids, execution_identity, file_identity, route_evidence,
)


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


def _native_fp32_forward(module, values, ids, probabilities, experts):
    """Use native B=0 execution with FP32 dW, without modifying production hooks."""
    order = torch.argsort(ids.flatten(), stable=True)
    expanded = values[:, None, :].expand(-1, ids.shape[1], -1).reshape(-1, values.shape[-1])
    counts = torch.bincount(ids.flatten(), minlength=experts)
    config = ExpertReplicaConfig(experts, dist.get_world_size(), 0)
    inputs, state = dispatch_native_replicas((expanded[order], counts), config, dist.group.WORLD)
    packed = torch.cat((_local(module.w1), _local(module.w3)), dim=1).transpose(1, 2).contiguous()
    down = _local(module.w2).transpose(1, 2).contiguous()
    output = native_replica_experts(inputs[0], packed, down, inputs[1], state.route)
    combined = combine_native_replicas(output, state)
    restored = torch.empty_like(combined)
    restored[order] = combined
    return (restored.reshape(values.shape[0], ids.shape[1], -1) * probabilities.unsqueeze(-1)).sum(1)


def _expert_forward(module, executor, values, ids, probabilities, experts):
    """Select the controlled reference executor without changing parameters."""
    if executor == "native_fp32":
        return _native_fp32_forward(module, values, ids, probabilities, experts)
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


def _deferred_backward(base, candidate, executor, experts, rank, device, tokens, hidden, top_k, reference,
                       replica_planner, *, growth=False, require_transfers=True, cross_stream=False):
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
        plans.append(route_evidence(actual, replica_planner) if require_transfers else None)
        if growth:
            resources = executor._get_execution_resources(other)
            capacities.append(resources.workspace.capacity_floor)
        pending.append((expected, actual, values, other, probabilities, other_probabilities))
    if require_transfers:
        active = plans[1]["transfers"] if growth else all(plan["transfers"] for plan in plans)
        if plans[0] == plans[1] or not active:
            raise AssertionError("Deferred acceptance requires different plans and active replicas")
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
        previous_backward = stream
    if previous_backward is not None:
        main_stream.wait_stream(previous_backward)
    for hook in hooks:
        hook.remove()
    # Elementwise tolerances apply to each incoming partial above. BF16 summation
    # can amplify relative error at cancellation; exact accumulation is checked
    # against the captured partials, with a separate normwise comparison here.
    accumulated_error = {name: _check(_local(getattr(candidate, name).grad), _local(getattr(base, name).grad),
                                      "accumulated " + name, elementwise=False)
                         for name in ("w1", "w2", "w3")}
    return {"invocations": checks, "accumulated": accumulated_error, "plans": plans, "capacities": capacities}


def _cross_stream_deferred(base, candidate, executor, experts, rank, device, tokens, hidden, top_k,
                           reference, replica_planner, *, runtime_version=2):
    """Check per-launch tails and image ownership on two streams with reverse backward."""
    function = importlib.import_module("hyper_parallel.core.multicore.modules.mega_moe.function")
    launches, snapshots, images = [], [], []

    original_image = function._runtime_image

    @wraps(original_image)
    def _image(*args, **kwargs):
        result = original_image(*args, **kwargs)
        images.append((result, args[1]))
        return result

    def _observe(original, direction, execution_index):
        @wraps(original)
        def _launch(*args, **kwargs):
            result = original(*args, **kwargs)
            execution = args[execution_index]
            image, expected = images[-1]
            snapshot = image[execution.profile_call.runtime_config.numel():].clone()
            snapshots.append((snapshot, expected))
            launches.append({"direction": direction, "stream": torch.npu.current_stream().npu_stream,
                             "image_pointer": image.data_ptr(), "suffix": expected.hex()})
            return result
        return _launch

    with (patch.object(function, "_runtime_image", _image),
          patch.object(function, "_launch_forward_kernel", _observe(function._launch_forward_kernel, "forward", 5)),
          patch.object(function, "_launch_backward_kernel", _observe(function._launch_backward_kernel, "backward", 3))):
        checks = _deferred_backward(base, candidate, executor, experts, rank, device, tokens, hidden, top_k,
                                    reference, replica_planner, cross_stream=True)
    for direction in ("forward", "backward"):
        records = [item for item in launches if item["direction"] == direction]
        _check_runtime_direction(records, runtime_version, direction)
    for snapshot, expected in snapshots:
        if bytes(snapshot.cpu().tolist()) != expected:
            raise AssertionError("Runtime tail was overwritten before its consumer completed")
    return {"checks": checks, "launches": launches}


def _check_runtime_direction(records, runtime_version, direction):
    """Require two actual streams and the expected image and epoch ownership."""
    if len(records) != 2 or len({item["stream"] for item in records}) != 2:
        raise AssertionError(f"Expected two actual {direction} streams: {records}")
    expected_images = 1 if runtime_version == 2 else 2
    if len({item["image_pointer"] for item in records}) != expected_images:
        raise AssertionError(f"Unexpected {direction} runtime image ownership: {records}")
    tails = [bytes.fromhex(item["suffix"]) for item in records]
    if any(int.from_bytes(tail[4:8], "little") != runtime_version for tail in tails):
        raise AssertionError(f"Unexpected runtime version: {records}")
    if runtime_version == 2 and len(set(tails)) != 1:
        raise AssertionError(f"Static suffix changed: {records}")
    if runtime_version == 4 and tails[0][-8:] == tails[1][-8:]:
        raise AssertionError(f"Projection epochs did not advance: {records}")


def _cross_layer_pool(backend, budget, mesh, executor, reference, tokens, hidden, intermediate, top_k,
                      replica_min_rows=0, replica_planner="cpu", cost_model=None):
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
                           replica_min_rows=replica_min_rows if hot else 0,
                           replica_planner=replica_planner if hot else "cpu",
                           replica_cost_model=cost_model if hot and backend == "native" else None).apply(module, mesh)
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


def _run(backend: str, budget: int, result_dir: str, *, tokens: int = 128,
        hidden: int = 128, intermediate: int = 128, top_k: int = 2,
        same_backend_reference: bool = False, replica_transport: str = "p2p",
        benchmark_iterations: int = 0, replica_min_rows: int = 0, replica_planner: str = "cpu",
        default_group: bool = False, fp32_reference: bool = False,
        cost_model: ExpertReplicaCostModel | None = None, cross_stream: bool = False) -> None:
    """Compare full forward/backward with native B=0, preserving EP ownership."""
    if cross_stream and (backend == "native" or replica_transport not in ("p2p", "shmem_signal_sdma_projection")):
        raise ValueError("Runtime cross-stream acceptance requires MegaMoe P2P or projection")
    if fp32_reference and same_backend_reference:
        raise ValueError("Select only one reference backend")
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
    if backend == "native" and replica_transport != "p2p":
        raise ValueError("Native hot replicas use HCCL P2P; one-sided transports require MegaMoe")
    if backend != "native":
        endpoint = [f"tcp://127.0.0.1:{allocate_port()}" if rank == 0 else None]
        dist.broadcast_object_list(endpoint, src=0)
        os.environ["HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT"] = endpoint[0]
    ExpertParallel().apply(base, mesh)
    ExpertParallel(replica_slots_per_rank=budget if backend == "native" else 0,
                   replica_min_rows=replica_min_rows, replica_planner=replica_planner,
                   replica_cost_model=cost_model if backend == "native" else None).apply(candidate, mesh)
    executor = None
    reference = "native_fp32" if fp32_reference else None
    if backend != "native":
        mega_moe = importlib.import_module("hyper_parallel.core.multicore").MegaMoeExperts
        options = {"initial_capacity_factor": 1.0} if backend == "push" else {}
        executor = mega_moe(local_num_tokens=tokens, hidden_size=hidden, intermediate_size=intermediate,
                                 num_experts=experts, top_k=top_k, ep_size=size,
                                 ep_group=None if default_group else mesh.get_group(),
                                 create_parameters=False, dispatch_mode=backend, replica_slots_per_rank=budget,
                                 replica_transport=replica_transport, replica_min_rows=replica_min_rows,
                                 replica_planner=replica_planner, replica_cost_model=cost_model,
                                 **options)
        if same_backend_reference:
            reference = mega_moe(local_num_tokens=tokens, hidden_size=hidden, intermediate_size=intermediate,
                                 num_experts=experts, top_k=top_k, ep_size=size,
                                 ep_group=None if default_group else mesh.get_group(),
                                 create_parameters=False, dispatch_mode=backend, replica_slots_per_rank=0,
                                 **options)
    results = []
    optimizer_base = torch.optim.SGD(base.parameters(), lr=0.01, momentum=0.9)
    optimizer_candidate = torch.optim.SGD(candidate.parameters(), lr=0.01, momentum=0.9)
    try:
        growth = None
        if backend == "push" and budget == 1 and top_k >= 5 and replica_min_rows == 0:
            growth = _deferred_backward(base, candidate, executor, experts, rank, device,
                                        tokens, hidden, top_k, reference, replica_planner, growth=True)
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
            plan_record = route_evidence(actual, replica_planner) if budget else None
            gradient = torch.randn_like(expected)
            expected.backward(gradient)
            actual.backward(gradient)
            evidence = {"pattern": pattern, "plan": plan_record, "output": _check(actual, expected, "output"),
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
                active_provider = resources.workspace.replica_provider
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
            base, candidate, executor, experts, rank, device, tokens, hidden, top_k, reference, replica_planner,
            require_transfers=budget > 0 and replica_min_rows == 0)
        if rank == 0:
            print(json.dumps({"backend": backend, "B": budget, "deferred": deferred}), flush=True)
        if backend == "native" and any(
                name.startswith("hyper_parallel.core.multicore") for name in sys.modules):
            raise AssertionError("native execution imported multicore")
        cross_stream_result = None
        if cross_stream:
            cross_stream_result = _cross_stream_deferred(base, candidate, executor, experts, rank, device,
                                                        tokens, hidden, top_k, reference, replica_planner,
                                                        runtime_version=2 if replica_transport == "p2p" else 4)
        cross_layer = []
        if hidden <= 128:
            cross_layer = _cross_layer_pool(backend, budget, mesh, executor, reference,
                                           tokens, hidden, intermediate, top_k, replica_min_rows,
                                           replica_planner, cost_model)
        signal_stress = None
        if replica_transport in SIGNAL_TRANSPORT_MODES and hidden <= 128:
            active_provider = resources.workspace.replica_provider
            signal_stress = _signal_stress(active_provider, mesh, hidden, intermediate, budget)
        kernel_wait = None
        if (budget and replica_min_rows == 0 and
                replica_transport in ("shmem_signal_sdma_projection", "shmem_signal_kernel_gradient")):
            kernel_wait = _kernel_wait_stress(base, candidate, executor, experts, rank, device,
                                              tokens, hidden, top_k, reference, replica_planner)
        timing = None if not benchmark_iterations else _benchmark(
            candidate, executor, experts, tokens, hidden, top_k, benchmark_iterations)
        identity = execution_identity()
        identity.update(torch_npu=torch_npu.__version__, ep_members=dist.get_process_group_ranks(mesh.get_group()),
                        backend=backend, budget=budget, shape=[tokens, hidden, intermediate, top_k],
                        requested_planner=replica_planner,
                        calibration=None if cost_model is None else asdict(cost_model),
                        actual_planner=results[0]["plan"]["planner"] if budget else "disabled",
                        requested_transport=replica_transport, reference=("native-fp32" if fp32_reference else
                        "same-backend" if same_backend_reference else "native"), default_group=default_group)
        if replica_planner == "device":
            library = importlib.import_module(
                "hyper_parallel.core.expert_parallel.hot_replica.device_kernel")._planner_library()
            identity["planner_payload"] = {**file_identity(Path(library._name)),
                                           "abi": library.planner_abi_version()}
        if executor is not None:
            identity["actual_transport"] = resources.spec.replica_transport
            loader = importlib.import_module("hyper_parallel.core.multicore._loader")
            vendor, adapter = loader.get_multicore_paths()
            identity["multicore_payload"] = [file_identity(adapter),
                                             file_identity(vendor / "op_api/lib/libcust_opapi.so")]
            identity["multicore_transport_abi"] = torch.ops.hyper_parallel.mega_moe_transport_version()
            identity["multicore_schemas"] = [str(torch.ops.hyper_parallel.mega_moe.default._schema),
                                             str(torch.ops.hyper_parallel.mega_moe_grad.default._schema)]
            identity["multicore_kernels"] = [file_identity(path) for path in sorted(vendor.rglob("*.o"))]
            provider = resources.workspace.replica_provider
            identity["provider"] = type(provider).__name__
            if replica_transport in SIGNAL_TRANSPORT_MODES:
                identity["provider_options"] = {"overlap_home": provider.overlap_home,
                                                "projection_ready": provider._projection_ready,
                                                "kernel_gradients": provider.kernel_gradients}
                expected_projection = replica_transport in (
                    "shmem_signal_sdma_projection", "shmem_signal_kernel_gradient")
                if (provider._projection_ready != expected_projection or
                        provider.kernel_gradients != (replica_transport == "shmem_signal_kernel_gradient")):
                    raise AssertionError("Requested signal transport did not execute")
            binding = sys.modules.get("hyper_parallel_shmem_torch")
            if binding is not None:
                identity["shmem_binding"] = file_identity(Path(binding.__file__))
        else:
            identity["actual_transport"] = "p2p"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        Path(result_dir, f"{backend}-b{budget}-rank{rank}.json").write_text(
            json.dumps({"identity": identity, "growth_deferred": growth, "steps": results,
                        "deferred": deferred, "cross_layer": cross_layer, "cross_stream": cross_stream_result,
                        "replica_transport": replica_transport, "replica_min_rows": replica_min_rows,
                        "signal_stress": signal_stress, "kernel_wait": kernel_wait, "timing": timing},
                       indent=2) + "\n", encoding="utf-8")
    finally:
        if executor is not None:
            executor.close()
        if reference is not None and not isinstance(reference, str):
            reference.close()


def run(backend: str, budget: int, result_dir: str, **options: Any) -> None:
    """Run acceptance with native provider isolation and optional delayed readiness."""
    with ExitStack() as stack:
        if backend == "native":
            stack.enter_context(patch.object(SignalReplicaTransport, "__init__", side_effect=AssertionError(
                "Native constructed a one-sided provider")))
        _run(backend, budget, result_dir, **options)


def main() -> None:
    """Initialize one worker group and run the selected executor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "push", "pull"), default="native")
    parser.add_argument("--replica-transport",
                        choices=("p2p", "shmem", *SIGNAL_TRANSPORT_MODES),
                        default="p2p")
    parser.add_argument("--replica-planner", choices=("cpu", "device"), default="cpu")
    parser.add_argument("--cross-stream", action="store_true")
    parser.add_argument("--cost-model")
    parser.add_argument("--default-group", action="store_true")
    parser.add_argument("--fp32-reference", action="store_true")
    parser.add_argument("--benchmark-iterations", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--replica-min-rows", type=int, default=0)
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
            benchmark_iterations=args.benchmark_iterations, replica_min_rows=args.replica_min_rows,
            replica_planner=args.replica_planner, default_group=args.default_group, fp32_reference=args.fp32_reference,
            cross_stream=args.cross_stream, cost_model=None if args.cost_model is None else ExpertReplicaCostModel(
                **json.loads(Path(args.cost_model).read_text(encoding="utf-8"))["model"]))
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


def _pytest_case(backend: str, **options) -> None:
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        run(backend, 1, os.getenv("HP_HOT_REPLICA_RESULTS", "./logs/hot_replica"), **options)
    finally:
        dist.destroy_process_group()


def test_native_device_hot_replica_npu() -> None:
    """Run native device planning with ordinary P2P and no multicore load."""
    _pytest_case("native", replica_planner="device", fp32_reference=True)


def test_push_default_group_npu() -> None:
    """Reach actual default-group P2P after balanced warmup."""
    _pytest_case("push", default_group=True, fp32_reference=True)


def test_pull_default_group_npu() -> None:
    """Exercise default-group P2P in the pull forward and backward."""
    _pytest_case("pull", default_group=True, fp32_reference=True)


def test_push_device_projection_npu() -> None:
    """Validate actual device planning and projection execution."""
    _pytest_case("push", replica_planner="device", replica_transport="shmem_signal_sdma_projection",
                 top_k=8, fp32_reference=True)


def test_push_device_kernel_gradient_npu() -> None:
    """Validate actual device planning and kernel_gradient execution."""
    _pytest_case("push", replica_planner="device", replica_transport="shmem_signal_kernel_gradient",
                 top_k=8, fp32_reference=True)


def test_pull_device_projection_npu() -> None:
    """Validate actual device planning and projection execution."""
    _pytest_case("pull", replica_planner="device", replica_transport="shmem_signal_sdma_projection",
                 top_k=8, fp32_reference=True)


def test_pull_device_kernel_gradient_npu() -> None:
    """Validate actual device planning and kernel_gradient execution."""
    _pytest_case("pull", replica_planner="device", replica_transport="shmem_signal_kernel_gradient",
                 top_k=8, fp32_reference=True)


def test_native_deferred_topk_npu() -> None:
    """Retain real native plans and reverse backward for every reviewed TopK."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        for top_k in (1, 2, 3, 4, 5, 6, 8):
            directory = Path(os.getenv("HP_HOT_REPLICA_RESULTS", "./logs/hot_replica")) / f"k{top_k}"
            run("native", 1, str(directory), top_k=top_k, fp32_reference=True)
    finally:
        dist.destroy_process_group()


def test_push_static_runtime_streams_npu() -> None:
    """Reuse static v2 images across streams and reversed backward in push."""
    _pytest_case("push", top_k=8, fp32_reference=True, cross_stream=True)


def test_pull_static_runtime_streams_npu() -> None:
    """Reuse static v2 images across streams and reversed backward in pull."""
    _pytest_case("pull", top_k=8, fp32_reference=True, cross_stream=True)


def test_push_projection_runtime_lifetime_npu() -> None:
    """Verify independent v4 epochs on two streams with retained push plans."""
    _pytest_case("push", top_k=8, fp32_reference=True, cross_stream=True,
                 replica_transport="shmem_signal_sdma_projection")


def test_pull_projection_runtime_lifetime_npu() -> None:
    """Verify independent v4 epochs on two streams with retained pull plans."""
    _pytest_case("pull", top_k=8, fp32_reference=True, cross_stream=True,
                 replica_transport="shmem_signal_sdma_projection")
