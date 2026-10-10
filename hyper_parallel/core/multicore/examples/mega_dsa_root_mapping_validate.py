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
"""Root-wide DSA storage with subgroup-only computation and explicit remote root PEs."""

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
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_validate import _SCALE, _oracle, _tile
from hyper_parallel.core.multicore.examples.mega_dsa_sparse_index_return_validate import _check as _check_index
from hyper_parallel.core.multicore.examples.mega_dsa_sparse_main_return_validate import _check as _check_main
from hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate import _inputs
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import collective_validation_check
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import FusedDsaCpForwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_backward import FusedDsaCpBackwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.index_return import SparseIndexGradReturnProbe
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import DsaWorkspaceSpec, MegaDsaWorkspace
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _active(report, workspace, members, group, group_id):
    cp_rank = dist.get_rank(group)
    meta = _metadata((3, 10), "strided", len(members), cp_rank)
    meta = replace(meta, cp_ranks=members, root_pes=tuple(workspace.root.members.index(rank) for rank in members),
                   heap_generation=workspace.root.generation, layer=group_id)
    backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=32, attention_scale=_SCALE,
                                     schedule=MixedSfaSchedule(7))
    dense, sparse = FusedDsaCpBackwardProbe(backend), FusedDsaCpBackwardProbe(backend, sparse_return=True)
    index_return = SparseIndexGradReturnProbe(backend)
    cpu_full, _ = _inputs(32, torch.float32, (3, 10))
    cpu_full = (cpu_full[0]*(group_id+1), *cpu_full[1:])
    full = tuple(value.to(backend.layout.device) for value in cpu_full)
    for tile_name in ("owner", "interior", "first_sequence", "empty"):
        tile = _tile(meta, tile_name)
        layout = backend.prepare_local_queries(tile.q_global_ids)
        states, index = _local_states(tile, full[:4], (full[4], full[5][:, None], full[6]))
        invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch+1))
        saved = backend.forward_local(invocation, states, index, query_layout=layout, selected_kl=True)
        oracle = collective_validation_check(partial(_oracle, saved, cpu_full[:4], tile), group=group)
        for repeat in range(3):
            stage = {"group": list(members), "root_pes": list(meta.root_pes), "cp_rank": cp_rank,
                     "tile": tile_name, "repeat": repeat}
            report["stage"] = stage
            cotangent = torch.full_like(saved.output, 0. if repeat == 1 else .25)
            baseline = dense.backward_local(saved, cotangent)
            main_result = sparse.backward_local(saved, cotangent)
            contribution = (torch.zeros_like(saved.kl_key_partials) if repeat == 1
                            else saved.kl_key_partials*(.3/13)*-2.5)
            invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch+1))
            index_result = index_return.return_owner(invocation, contribution.contiguous())
            torch.npu.synchronize()
            peers_main, peers_index = [None]*len(members), [None]*len(members)
            dist.all_gather_object(peers_main, main_result.partials.cpu(), group=group)
            dist.all_gather_object(peers_index, contribution.cpu(), group=group)
            main_proof = collective_validation_check(partial(_check_main, sparse, dense, main_result, baseline,
                                                              peers_main, meta, repeat == 1), group=group)
            index_proof = collective_validation_check(partial(_check_index, index_return, index_result,
                                                               peers_index, meta), group=group)
            report["cases"].append({**stage, "forward_oracle": oracle,
                                    "main": main_proof, "index": index_proof})


def _scenario(report, groups, partitioned):
    rank = dist.get_rank()
    root = SharedShmemRoot(torch.device("npu", int(os.environ.get("LOCAL_RANK", "0"))), root_group=dist.group.WORLD)
    if partitioned and rank in groups[1][0]:
        group_id = 1
    else:
        group_id = 0
    members, group = groups[group_id]
    workspace = MegaDsaWorkspace(root, "mapped-dsa", DsaWorkspaceSpec(13, 13, 13, 2), cp_group=group)
    try:
        workspace.bind()
        report["scenarios"].append({"partitioned": partitioned, "participates": workspace.participates,
                                    "arena_bytes": workspace.arena_bytes, "root_generation": root.generation})
        def _work():
            if workspace.participates:
                _active(report, workspace, members, group, group_id)
            return {"participates": workspace.participates}
        collective_validation_check(_work)
    finally:
        workspace.close()
        root.close()


def main() -> None:
    """Archive subgroup transport correctness without claiming complete CP/EP coexistence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "running", "cases": [], "scenarios": [], "full_model_acceptance": False,
              "cp_ep_coexistence_acceptance": False, "performance_acceptance": False}
    try:
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("hccl", timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError("root mapping validator requires four root members")
        report.update(rank=dist.get_rank(), root_size=4)
        groups = tuple((members, dist.new_group(list(members))) for members in ((1, 3), (0, 2)))
        for partitioned in (False, True):
            _scenario(report, groups, partitioned)
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
