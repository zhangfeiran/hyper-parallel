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
"""Device index-K return: exact ordered FP32 sums, sparse receipts and inbox reuse."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _local_states, _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import _loaded_library_identity
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_validate import _SCALE, _tile
from hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate import _inputs
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import collective_validation_check
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import FusedDsaCpForwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_backward import FusedDsaCpBackwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.index_return import SparseIndexGradReturnProbe
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import DsaWorkspaceSpec, MegaDsaWorkspace
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _contribution(tokens, sender, repeat):
    gradient = torch.arange(tokens*128, dtype=torch.float32).reshape(tokens, 128)/4096+.03137
    gradient *= (-1 if sender % 2 else 1)*(sender+1)
    rows = torch.arange(tokens)
    if repeat == 1:
        gradient.zero_()
        gradient[::2] = -0.
    elif repeat == 3:
        gradient.zero_()
        if sender == 0:
            gradient[0, 0] = float("nan")
            gradient[-1, 1] = float("inf")
    else:
        gradient[(rows+sender) % 3 == 0] = 0.
    return gradient


def _check(probe, result, peers, meta):
    expected = peers[0].clone()
    for contribution in peers[1:]:
        expected.add_(contribution)
    expected = expected[list(meta.kv_global_ids)]
    torch.testing.assert_close(result.owner_fp32.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    trace = result.transport_trace.cpu()
    proof = probe.validate_trace(result, trace)
    masks = tuple(value.ne(0).any(dim=1) for value in peers)
    local = masks[meta.cp_rank]
    remote = torch.tensor(meta.token_owners) != meta.cp_rank
    received = sum(int(value[list(meta.kv_global_ids)].sum()) for value in masks)
    remote_received = sum(int(value[list(meta.kv_global_ids)].sum()) for sender, value in enumerate(masks)
                          if sender != meta.cp_rank)
    expected_bytes = (int(local.sum())*512, int((local & remote).sum())*512,
                      received*512, remote_received*512)
    if tuple(trace[4:8].tolist()) != expected_bytes:
        raise RuntimeError("index-K payload receipts differ from actual input contributions")
    return {"ordered_fp32_owner_exact": True, "trace": proof}


def _native_cases(report, workspace, forward, probe, meta, lengths, pattern):
    cpu_full, _ = _inputs(32, torch.float32, lengths)
    full = tuple(value.to(forward.layout.device) for value in cpu_full)
    main_return = FusedDsaCpBackwardProbe(forward, sparse_return=True)
    size = len(meta.cp_ranks)
    for tile_name in ("owner", "interior", "first_sequence", "empty"):
        tile = _tile(meta, tile_name)
        layout = forward.prepare_local_queries(tile.q_global_ids)
        states, index = _local_states(tile, full[:4], (full[4], full[5][:, None], full[6]))
        invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch+1))
        saved = forward.forward_local(invocation, states, index, query_layout=layout, selected_kl=True)
        partials = (saved.kl_key_partials*(.3/meta.global_valid_queries)*-2.5).contiguous()
        for repeat in range(3):
            report["stage"] = {"lengths": lengths, "pattern": pattern, "repeat": repeat,
                               "tile": tile_name, "producer": "native_fused_kl", "main_index_interleaved": True}
            main_result = main_return.backward_local(saved, torch.full_like(saved.output, 0. if repeat == 1 else .25))
            contribution = torch.zeros_like(partials) if repeat == 1 else partials
            invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch+1))
            result = probe.return_owner(invocation, contribution)
            torch.npu.synchronize()
            peers = [None]*size
            dist.all_gather_object(peers, contribution.cpu())
            evidence = collective_validation_check(partial(_check, probe, result, peers, meta))
            proof = collective_validation_check(partial(main_return.validate_trace, main_result,
                                                         tuple(value.cpu() for value in main_result.phase_traces),
                                                         main_result.transport_trace.cpu()))
            report["cases"].append({**report["stage"], **evidence, "main_trace": proof})


def _fixture(report, lengths, pattern):
    rank, size = dist.get_rank(), dist.get_world_size()
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    meta = _metadata(lengths, pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "sparse-index-return", DsaWorkspaceSpec(capacity, sum(lengths), capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    try:
        forward = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=32, attention_scale=_SCALE,
                                        schedule=MixedSfaSchedule(7))
        probe = SparseIndexGradReturnProbe(forward)
        for repeat in range(4):
            report["stage"] = {"lengths": lengths, "pattern": pattern, "repeat": repeat, "producer": "controlled"}
            contribution = _contribution(sum(lengths), rank, repeat).to(device)
            invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch+1))
            result = probe.return_owner(invocation, contribution)
            torch.npu.synchronize()
            peers = [None]*size
            dist.all_gather_object(peers, contribution.cpu())
            evidence = collective_validation_check(partial(_check, probe, result, peers, meta))
            report["cases"].append({**report["stage"], **evidence})
        _native_cases(report, workspace, forward, probe, meta, lengths, pattern)
    finally:
        workspace.close()
        root.close()


def main() -> None:
    """Archive transport correctness separately from model and performance acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--long-history", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "cases": [], "full_model_acceptance": False, "performance_acceptance": False}
    try:
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("hccl", timeout=timedelta(minutes=10))
        report.update(rank=dist.get_rank(), cp_size=dist.get_world_size())
        for pattern in (("strided",) if args.smoke else ("contiguous", "strided", "zigzag", "empty")):
            _fixture(report, (3, 10), pattern)
        if args.long_history:
            _fixture(report, (3, 2113), "strided")
        report.update(status="passed", loaded_library_sha256=_loaded_library_identity())
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
