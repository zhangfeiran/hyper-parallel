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
"""Fused CP2 DSA and EP4 MegaMoE on one encompassing root, with delayed backward."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_root_mapping_validate import (
    _active,
)
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import (
    collective_validation_check,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import (
    assert_stable_memory,
    collect_memory,
    start_shmem_lifetime,
)


def _moe_forward(mega, common, shape, streams):
    mega.zero_grad(set_to_none=True)
    common.zero_grad(set_to_none=True)
    hidden, upstream = baseline.make_data(shape)
    ids, weights, counts = baseline.make_balanced_route(shape)
    observed_inputs = [hidden.detach().clone().requires_grad_() for _ in streams]
    expected_inputs = [hidden.detach().clone().requires_grad_() for _ in streams]
    observed_routes = [weights.detach().clone().requires_grad_() for _ in streams]
    expected_routes = [weights.detach().clone().requires_grad_() for _ in streams]
    expected = [baseline.forward_layer(common, value, ids, route, tokens_per_expert=counts)
                for value, route in zip(expected_inputs, expected_routes)]
    default = torch.npu.current_stream()
    observed = []
    for stream, value, route in zip(streams, observed_inputs, observed_routes):
        with torch.npu.stream(stream):
            stream.wait_stream(default)
            observed.append(mega(value, ids, route, tokens_per_expert=counts))
    return (observed, expected, observed_inputs, expected_inputs,
            observed_routes, expected_routes, upstream)


def _moe_backward(mega, common, pending, streams):
    observed, expected, observed_inputs, expected_inputs, observed_routes, expected_routes, upstream = pending
    for index in (1, 0):
        with torch.npu.stream(streams[index]):
            observed[index].backward(upstream)
        expected[index].backward(upstream)
    torch.npu.synchronize()

    def _compare():
        for name, actual_values, expected_values in (("output", observed, expected),
                                                    ("input", observed_inputs, expected_inputs),
                                                    ("route", observed_routes, expected_routes)):
            for index, (actual, reference) in enumerate(zip(actual_values, expected_values)):
                baseline.assert_close(f"{name}{index}", actual if name == "output" else actual.grad,
                                      reference if name == "output" else reference.grad)
        for name, actual, reference in zip(("gate_up", "down"), baseline.expert_weight_gradients(mega),
                                          baseline.expert_weight_gradients(common)):
            baseline.assert_close(name, actual, reference)
        return {"output_input_route_expert_gradients": True}
    proof = collective_validation_check(_compare)
    for layer in (mega, common):
        torch.optim.SGD(layer.parameters(), lr=1e-3).step()

    def _compare_weights():
        for name, actual, reference in zip(("gate_up", "down"), baseline.expert_weights(mega),
                                          baseline.expert_weights(common)):
            baseline.assert_close(f"updated_{name}", actual, reference)
        return {"sgd_weights": True}
    proof.update(collective_validation_check(_compare_weights))
    return proof


def _dsa_work(report, workspace, members, group, group_id, stream):
    def _work():
        if workspace.participates:
            with torch.npu.stream(stream):
                stream.wait_stream(torch.npu.default_stream())
                _active(report, workspace, members, group, group_id)
        return {"participates": workspace.participates}
    collective_validation_check(_work)


def _allocation_check(root, generation, expected):
    state = shmem.debug_state()
    if root.generation != generation or state["reference_count"] != 1 or state["allocated_count"] != expected:
        raise AssertionError(f"shared root identity/allocation count changed: {state}")
    return {"generation": generation, "allocated_count": expected}


def _scenario(report, groups, mode, partitioned, close_dsa_first):
    start_shmem_lifetime()
    root = SharedShmemRoot(baseline.DEVICE, root_group=dist.group.WORLD)
    shape = baseline.MoeShape(ep_size=4, num_experts=8)
    mega, common = baseline.new_layers(shape, dispatch_mode=mode,
                                      initial_capacity_factor=4 if mode == "push" else None)
    mega.bind_shmem_root(root, "moe", dtype=torch.bfloat16)
    group_id = int(partitioned and dist.get_rank() in groups[1][0])
    members, group = groups[group_id]
    workspace = MegaDsaWorkspace(root, "mapped-dsa", DsaWorkspaceSpec(13, 13, 13, 2), cp_group=group)
    case = {"dispatch_mode": mode, "partitioned": partitioned, "close_dsa_first": close_dsa_first,
            "participates": workspace.participates, "cases": [], "moe_steps": [], "memory": []}
    report["scenarios"].append(case)
    streams = (torch.npu.Stream(), torch.npu.Stream())
    try:
        workspace.bind()
        generation = root.generation
        for step in range(4):
            pending = _moe_forward(mega, common, shape, streams)
            _dsa_work(case, workspace, members, group, group_id, streams[1])
            case["moe_steps"].append(_moe_backward(mega, common, pending, streams))
            del pending
            collective_validation_check(lambda: _allocation_check(root, generation, 5))
            if step >= 1:
                case["memory"].append(collect_memory())
        collective_validation_check(lambda: _memory_check(case["memory"]))
        if close_dsa_first:
            workspace.close()
            collective_validation_check(lambda: _allocation_check(root, generation, 4))
            pending = _moe_forward(mega, common, shape, streams)
            case["moe_steps"].append(_moe_backward(mega, common, pending, streams))
            del pending
            mega.close()
        else:
            mega.close()
            collective_validation_check(lambda: _allocation_check(root, generation, 1))
            _dsa_work(case, workspace, members, group, group_id, streams[0])
            workspace.close()
        collective_validation_check(lambda: _allocation_check(root, generation, 0))
        root.close()
        case.update(status="passed", heap_bytes=root.heap_size_bytes, generation=generation)
        collective_validation_check(_closed_check)
    finally:
        mega.close()
        workspace.close()
        root.close()


def _memory_check(samples):
    assert_stable_memory(samples)
    return {"memory_plateau": True}


def _closed_check():
    state = shmem.debug_state()
    if state["reference_count"] != 0 or state["state"] != "Uninitialized":
        raise AssertionError(f"root final close did not shut down cleanly: {state}")
    return {"root_closed": True}


def test_mapped_cp_ep_shared_root() -> None:
    """Validate real fused DSA and MoE serial coexistence with CP2/EP4 and both close orders."""
    if dist.get_world_size() != 4:
        raise ValueError("mapped CP/EP shared-root acceptance requires four ranks")
    report = {"rank": dist.get_rank(), "status": "running", "scenarios": [],
              "cp_size": 2, "ep_size": 4, "root_size": 4, "full_model_acceptance": False,
              "ep_subgroup_mapping_acceptance": False, "performance_acceptance": False}
    try:
        groups = tuple((members, dist.new_group(list(members))) for members in ((1, 3), (0, 2)))
        for mode in ("pull", "push"):
            for partitioned in (False, True):
                for close_dsa_first in (False, True):
                    _scenario(report, groups, mode, partitioned, close_dsa_first)
        report.update(status="passed", loaded_library_sha256=_loaded_library_identity())
    except Exception as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        directory = os.environ.get("HP_MAPPED_CP_EP_REPORT_DIR")
        if directory:
            path = Path(directory) / f"rank{dist.get_rank()}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        dist.destroy_process_group()
