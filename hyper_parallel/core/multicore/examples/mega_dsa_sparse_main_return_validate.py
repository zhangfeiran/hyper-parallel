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
"""Sparse main-gradient return: exact ordered FP32 owner sums and zero-row inbox reuse."""

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

from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import (
    _local_states,
    _metadata,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _difference,
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_validate import (
    _SCALE,
    _tile,
)
from hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate import _inputs
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import (
    collective_validation_check,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_backward import (
    FusedDsaCpBackwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _merged(partials):
    return torch.cat((partials[:, :512]+partials[:, 576:], partials[:, 512:576]), dim=1)


def _check(sparse, dense, result, baseline, peers, meta, zero):
    contributions = tuple(_merged(value) for value in peers)
    summed = contributions[0].clone()
    for value in contributions[1:]:
        summed = summed+value
    owned = summed[list(meta.kv_global_ids)]
    actual = torch.cat(result.owner_fp32, dim=1).cpu()
    torch.testing.assert_close(actual, owned, rtol=0, atol=0)
    nonzero = tuple((value != 0).any(dim=1) for value in contributions)
    rank = meta.cp_rank
    rows = nonzero[rank]
    remote_mask = torch.tensor(meta.token_owners) != rank
    received = sum(int(value[list(meta.kv_global_ids)].sum()) for value in nonzero)
    remote_received = sum(int(value[list(meta.kv_global_ids)].sum()) for sender, value in enumerate(nonzero)
                          if sender != rank)
    trace = result.transport_trace.cpu()
    proof = sparse.validate_trace(result, tuple(value.cpu() for value in result.phase_traces), trace)
    dense_proof = dense.validate_trace(baseline, tuple(value.cpu() for value in baseline.phase_traces),
                                       baseline.transport_trace.cpu())
    expected = (int(rows.sum())*2816, int((rows & remote_mask).sum())*2816, received*2816, remote_received*2816)
    if tuple(trace[4:8].tolist()) != expected:
        raise RuntimeError("sparse payload counters differ from actual producer FP32 contributions")
    if zero and (bool(actual.count_nonzero()) or bool(trace[4:8].count_nonzero())):
        raise RuntimeError("zero-cotangent replay consumed stale owner inbox data")
    comparisons = [_difference(value, reference) for value, reference in zip(result.gradients, baseline.gradients)]
    return {"ordered_fp32_owner_exact": True, "native_dense_comparison": comparisons, "trace": proof,
            "dense_trace": dense_proof, "zero_cotangent": zero,
            "payload_bytes_saved": dense_proof["remote_sent_bytes"]-proof["remote_sent_bytes"]}


def _fixture(report, lengths, heads, pattern, smoke):
    rank, size = dist.get_rank(), dist.get_world_size()
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    meta = _metadata(lengths, pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "sparse-main-return", DsaWorkspaceSpec(capacity, sum(lengths), capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    cpu_full, _ = _inputs(heads, torch.bfloat16, lengths)
    full = tuple(value.to(device) for value in cpu_full)
    try:
        for groups in ((7,) if smoke else (1, 7, 19)):
            forward = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=heads,
                                             attention_scale=_SCALE, schedule=MixedSfaSchedule(groups))
            dense = FusedDsaCpBackwardProbe(forward)
            sparse = FusedDsaCpBackwardProbe(forward, sparse_return=True)
            for tile_name in ("owner", "interior", "first_sequence", "empty"):
                tile = _tile(meta, tile_name)
                layout = forward.prepare_local_queries(tile.q_global_ids)
                states, index = _local_states(tile, tuple(full[:4]), (full[4], full[5][:, None], full[6]))
                invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch+1))
                saved = forward.forward_local(invocation, states, index, query_layout=layout)
                cotangent = torch.full_like(saved.output, .25)
                for repeat, zero in enumerate((False, True, False)):
                    report["stage"] = {"heads": heads, "lengths": lengths, "pattern": pattern, "groups": groups,
                                       "tile": tile_name, "repeat": repeat, "zero_cotangent": zero}
                    current = torch.zeros_like(cotangent) if zero else cotangent
                    baseline = dense.backward_local(saved, current)
                    result = sparse.backward_local(saved, current)
                    torch.npu.synchronize()
                    peers = [None]*size
                    dist.all_gather_object(peers, result.partials.cpu())
                    evidence = collective_validation_check(partial(_check, sparse, dense, result, baseline,
                                                                   peers, meta, zero))
                    report["cases"].append({**report["stage"], **evidence})
    finally:
        workspace.close()
        root.close()


def main() -> None:
    """Archive exact same-producer owner checks separately from performance and model acceptance."""
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
        shapes = [(32, "strided")] if args.smoke else [
            (heads, pattern) for heads in (32, 64) for pattern in ("contiguous", "strided", "zigzag", "empty")]
        for heads, pattern in shapes:
            _fixture(report, (3, 10), heads, pattern, args.smoke)
        if args.long_history:
            _fixture(report, (3, 2113), 32, "strided", True)
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
