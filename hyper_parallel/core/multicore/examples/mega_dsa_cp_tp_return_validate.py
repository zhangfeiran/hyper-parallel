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
"""Joint native CP index-K return followed by TP shared owner SUM, without early cast."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import (
    _SCALE,
    _metadata,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_sparse_index_return_validate import (
    _contribution,
)
from hyper_parallel.core.multicore.examples.mega_dsa_tp_owner_return_validate import (
    _fields,
)
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import (
    collective_validation_check,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cp_tp_return import (
    CpTpSharedGradientReturnProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.index_return import (
    SparseIndexGradReturnProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.tp_owner_return import (
    TpOwnerGradientReturnProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumProbe,
    TpTileSumSpec,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _expected_key(meta, tp_members, cp_sets, replay):
    expected = torch.zeros(len(meta.kv_global_ids), 128)
    for peer in tp_members:
        cp_sum = torch.zeros(meta.global_valid_queries, 128)
        for sender in cp_sets[peer]:
            cp_sum.add_(_contribution(meta.global_valid_queries, sender, replay))
        expected.add_(cp_sum[list(meta.kv_global_ids)])
    return expected


def _case(scenario, backend, workspace, meta, cp_sets, presence, epoch, replay):
    meta = replace(meta, invocation=epoch)
    invocation = backend.prepare(workspace.prepare(meta), epoch=epoch, presence=presence)
    root_rank = workspace.root.rank
    tp_rank = dist.get_rank(backend.tp_return.transport.group)
    device = workspace.root.device
    index = _contribution(meta.global_valid_queries, root_rank, replay).to(device) if presence[2] else None
    cpu_main = _fields(len(meta.kv_global_ids), tp_rank, (*presence[:2], False))[:2]
    main_fields = tuple(None if value is None else value.to(device) for value in cpu_main)
    result = backend.return_owner(
        index, main_fields, invocation, input_dtypes=(torch.bfloat16, torch.bfloat16, torch.float32))
    torch.npu.synchronize()
    wanted = [None if value is None else torch.zeros_like(value) for value in cpu_main]
    for peer in range(len(backend.tp_return.transport.members)):
        for total, value in zip(wanted, _fields(len(meta.kv_global_ids), peer, (*presence[:2], False))[:2]):
            if total is not None:
                total.add_(value)
    wanted.append(_expected_key(meta, backend.tp_return.transport.members, cp_sets, replay) if presence[2] else None)
    for fp32, gradient, expected, dtype in zip(result.shared_tp.owner_fp32, result.shared_tp.gradients, wanted,
                                              (torch.bfloat16, torch.bfloat16, torch.float32)):
        if expected is not None:
            torch.testing.assert_close(fp32.cpu(), expected, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(gradient.cpu(), expected.to(dtype), rtol=0, atol=0, equal_nan=True)
    cp_proof = (backend.index_return.validate_trace(result.index_cp, result.index_cp.transport_trace.cpu())
                if result.index_cp is not None else None)
    transport = backend.tp_return.transport
    offset = transport.spec.control_offset
    controls = transport.arena[offset:offset+transport.spec.slots*640].view(torch.int64).reshape(-1, 80).cpu()
    sequence = epoch*2**32+invocation.tp.transport.rounds
    for control in controls:
        if int(control[0]) != sequence or any(int(control[(2+peer)*8]) != sequence
                                             for peer in range(len(transport.members))):
            raise RuntimeError('CP/TP shared return lacks final TP publication or reader ACK')
    scenario['cases'].append({'passed': True, 'presence': presence, 'replay': replay, 'epoch': epoch,
        'owner_ids': meta.kv_global_ids, 'native_cp_index_receipt': cp_proof,
        'exact_cp_then_tp_fp32_sum': True, 'exact_final_cast': True, 'final_tp_readers_acknowledged': True,
        'tp_rounds': invocation.tp.transport.rounds})
    return {'passed': True}


def _run(report, topology, groups, cp_sets, lengths, pattern, slots, *, smoke):
    cp_members, cp_group, tp_group = groups
    root = SharedShmemRoot(torch.device('npu', int(os.environ['LOCAL_RANK'])), root_group=dist.group.WORLD)
    meta = _metadata(lengths, pattern, len(cp_members), dist.get_rank(cp_group))
    meta = replace(meta, cp_ranks=cp_members, root_pes=cp_members, kv_global_ids=tuple(reversed(meta.kv_global_ids)))
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(len(cp_members))))
    workspace = MegaDsaWorkspace(root, 'cp-index', DsaWorkspaceSpec(capacity, sum(lengths), capacity, len(cp_members)),
                                 cp_group=cp_group)
    transport = TpTileSumProbe(root, 'tp-owner', TpTileSumSpec(slots, 512), tp_group=tp_group)
    scenario = {'topology': topology, 'cp_members': cp_members, 'tp_members': transport.members,
                'lengths': lengths, 'pattern': pattern, 'slots': slots, 'cases': []}
    report['scenarios'].append(scenario)
    try:
        workspace.bind()
        transport.bind()
        meta = replace(meta, heap_generation=root.generation)
        forward = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=32,
                                         attention_scale=_SCALE, schedule=MixedSfaSchedule(7))
        backend = CpTpSharedGradientReturnProbe(SparseIndexGradReturnProbe(forward),
                                                TpOwnerGradientReturnProbe(transport))
        epoch = 0
        masks = ((False, False, True),) if smoke else ((False, False, True), (True, True, True), (True, True, False))
        for presence in masks:
            for replay in range(3):
                epoch += 1
                collective_validation_check(partial(_case, scenario, backend, workspace, meta, cp_sets,
                                                    presence, epoch, replay))
        scenario['status'] = 'passed'
    finally:
        transport.close()
        workspace.close()
        root.close()


def main() -> None:
    """Archive joint CP/TP index transport plus already CP-returned main owner fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    report = {'status': 'running', 'scenarios': [], 'scope': 'joint native CP then TP shared-gradient owner return',
              'native_main_cp_return_integration': False, 'native_kl_integration': False,
              'cp_tp_model_acceptance': False, 'performance_acceptance': False}
    try:
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError('CP/TP shared-gradient validation requires four root members')
        rank = dist.get_rank()
        single = tuple(((peer,), dist.new_group([peer])) for peer in range(4))
        cp_pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 1), (2, 3)))
        tp_pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 2), (1, 3)))
        cp_pair = next(item for item in cp_pairs if rank in item[0])
        tp_pair = next(item for item in tp_pairs if rank in item[0])
        topologies = [('cp4_tp1', (tuple(range(4)), dist.group.WORLD, single[rank][1]), (tuple(range(4)),)*4),
                      ('cp2_tp2', (*cp_pair, tp_pair[1]), ((0, 1), (0, 1), (2, 3), (2, 3))),
                      ('cp1_tp4', (*single[rank], dist.group.WORLD), tuple((peer,) for peer in range(4)))]
        lengths_list, patterns, slots_list = ((3, 10), (3, 2113)), ('strided', 'empty'), (1, 14, 38)
        if args.smoke:
            lengths_list, patterns, slots_list = ((3, 10),), ('strided',), (14,)
        for topology, groups, cp_sets in topologies:
            for lengths in lengths_list:
                for pattern in patterns:
                    for slots in slots_list:
                        _run(report, topology, groups, cp_sets, lengths, pattern, slots, smoke=args.smoke)
        report.update(status='passed', loaded_library_sha256=_loaded_library_identity())
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
