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
"""Real NPU MoE/DSA shared-root lifecycle and serial cross-stream acceptance."""

import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import (
    assert_stable_memory,
    collect_memory,
    environment_identity,
    start_shmem_lifetime,
)


def _prepare_invocations(workspace: MegaDsaWorkspace) -> tuple:
    workspace.bind()
    rank, size = dist.get_rank(), dist.get_world_size()
    total = 2 * size + 1
    owners = tuple(token % size for token in range(total))
    offsets = tuple(token // size for token in range(total))
    owned = tuple(token for token in range(total) if owners[token] == rank)
    meta = DsaBatchMeta((0, total), owned, tuple(reversed(owned)), owners, offsets,
                        cp_ranks=tuple(range(size)), root_pes=tuple(range(size)), cp_rank=rank,
                        heap_generation=workspace.root.generation)
    return tuple(workspace.prepare(replace(meta, invocation=step, layer=step % 2, microbatch=step % 3))
                 for step in range(8))


def _publish_and_read(workspace: MegaDsaWorkspace, invocation, direction: str) -> None:
    meta = invocation.batch_meta
    offset = meta.invocation * 512 + meta.layer * 256 + meta.microbatch * 128
    if direction == "backward":
        offset += 4096
    ids = torch.tensor(meta.kv_global_ids, dtype=torch.bfloat16, device=baseline.DEVICE) * 32 + offset
    states = tuple(ids[:, None].expand(-1, width).contiguous() for width in (512, 64, 128))
    peer = (dist.get_rank() + 1) % dist.get_world_size()
    with workspace.lease(invocation, direction=direction):
        workspace.publish_owner_states(invocation, states)
        torch.npu.synchronize()
        shmem.host_barrier()
        for name, width in (("source_compressed", 512), ("source_rope", 64), ("source_index", 128)):
            output = torch.empty(width, dtype=torch.bfloat16, device=baseline.DEVICE)
            shmem.get(output, workspace.buffers[name][0], peer)
            torch.npu.synchronize()
            torch.testing.assert_close(output, torch.full_like(output, peer * 32 + offset), rtol=0, atol=0)
        shmem.host_barrier()


def _assert_step(mega, common, shape, workspace, invocation, streams) -> None:
    mega.zero_grad(set_to_none=True)
    common.zero_grad(set_to_none=True)
    hidden, upstream = baseline.make_data(shape)
    ids, weights, counts = baseline.make_balanced_route(shape)
    actual_inputs = [hidden.detach().clone().requires_grad_() for _ in range(2)]
    expected_inputs = [hidden.detach().clone().requires_grad_() for _ in range(2)]
    actual_weights = [weights.detach().clone().requires_grad_() for _ in range(2)]
    expected_weights = [weights.detach().clone().requires_grad_() for _ in range(2)]
    reference = [baseline.forward_layer(common, value, ids, route, tokens_per_expert=counts)
                 for value, route in zip(expected_inputs, expected_weights)]
    default = torch.npu.current_stream()
    actual = []
    for index, (value, route) in enumerate(zip(actual_inputs, actual_weights)):
        with torch.npu.stream(streams[index]):
            streams[index].wait_stream(default)
            actual.append(mega(value, ids, route, tokens_per_expert=counts))
    with torch.npu.stream(streams[1]):
        _publish_and_read(workspace, invocation, "forward")
    for index in (1, 0):
        with torch.npu.stream(streams[index]):
            actual[index].backward(upstream)
        reference[index].backward(upstream)
    with torch.npu.stream(streams[0]):
        _publish_and_read(workspace, invocation, "backward")
    torch.npu.synchronize()
    for index, (observed, expected) in enumerate(zip(actual, reference)):
        baseline.assert_close(f"output{index}", observed, expected)
    for name, observed, expected in zip(("input0", "input1", "route0", "route1"),
                                       actual_inputs + actual_weights, expected_inputs + expected_weights):
        baseline.assert_close(name, observed.grad, expected.grad)
    for name, observed, expected in zip(("gate_up", "down"), baseline.expert_weight_gradients(mega),
                                       baseline.expert_weight_gradients(common)):
        baseline.assert_close(name, observed, expected)


def _coexistence(close_dsa_first: bool, dispatch_mode: str) -> dict:
    start_shmem_lifetime()
    root = SharedShmemRoot(baseline.DEVICE, root_group=dist.group.WORLD)
    size = dist.get_world_size()
    shape = baseline.MoeShape(ep_size=size, num_experts=2 * size)
    mega, common = baseline.new_layers(shape, dispatch_mode=dispatch_mode,
                                        initial_capacity_factor=size if dispatch_mode == "push" else None)
    mega.bind_shmem_root(root, "moe", dtype=torch.bfloat16)
    workspace = MegaDsaWorkspace(root, "dsa", DsaWorkspaceSpec(3, 32, 3, size))
    samples = []
    streams = (torch.npu.Stream(), torch.npu.Stream())
    try:
        invocations = _prepare_invocations(workspace)
        generation = root.generation
        for step, invocation in enumerate(invocations):
            _assert_step(mega, common, shape, workspace, invocation, streams)
            state = shmem.debug_state()
            if root.generation != generation or state["reference_count"] != 1 or state["allocated_count"] != 5:
                raise AssertionError(f"shared root identity/allocation count changed: {state}")
            if step >= 3:
                sample = collect_memory()
                sample["shmem_allocations"] = state["allocated_count"]
                samples.append(sample)
        assert_stable_memory(samples)
        if close_dsa_first:
            workspace.close()
            remaining = shmem.debug_state()["allocated_count"]
            if remaining != 4:
                raise AssertionError(f"DSA close changed MoE allocations: {remaining}")
            mega.close()
        else:
            mega.close()
            remaining = shmem.debug_state()["allocated_count"]
            if remaining != 1:
                raise AssertionError(f"MoE close changed DSA arena: {remaining}")
            workspace.close()
        if shmem.debug_state()["reference_count"] != 1 or shmem.debug_state()["allocated_count"] != 0:
            raise AssertionError("consumer close did not preserve the sole root reference")
        root.close()
        if shmem.debug_state()["reference_count"] != 0 or shmem.debug_state()["state"] != "Uninitialized":
            raise AssertionError("shared root final close did not shut down cleanly")
        return {"status": "passed", "dispatch_mode": dispatch_mode, "close_dsa_first": close_dsa_first, "steps": 8,
                "outstanding_moe_forwards": 2, "moe_vs_common": {"rtol": 0.02, "atol": 0.002},
                "dsa_remote_reads_exact": True, "generation": generation, "heap_bytes": root.heap_size_bytes,
                "stable_memory": samples, "fused_dsa_execution": False}
    finally:
        mega.close()
        workspace.close()
        root.close()


def test_shared_shmem_root_coexistence() -> None:
    """Compare MoE output/all input and weight gradients while publishing DSA on the same root."""
    cases = [_coexistence(close_dsa_first, mode) for mode in ("pull", "push") for close_dsa_first in (False, True)]
    result = {"rank": dist.get_rank(), "cases": cases, "environment": environment_identity()}
    directory = os.environ.get("HP_SHARED_SHMEM_ROOT_REPORT_DIR")
    if directory:
        path = Path(directory) / f"rank{dist.get_rank()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()
