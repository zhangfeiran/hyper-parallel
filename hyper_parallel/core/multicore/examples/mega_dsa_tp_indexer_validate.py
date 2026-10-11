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
"""Native TP pre-Top-K score exchange against stock full-head LI on explicit CP query tiles."""

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

from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_indexer_validate import (
    _query_meta,
)
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_indexer_validate import (
    _fixture,
)
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import (
    collective_validation_check,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_indexer import (
    validate_fused_indexer_traces,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.tp_indexer import (
    TpLocalIndexerProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumProbe,
    TpTileSumSpec,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _states(lengths, name, device):
    base = 'random_signed_fp32_weights' if name == 'random_signed_fp32_weights' else (
        'random_signed' if name == 'random_signed' else 'increasing')
    meta, layout, cpu, tensors = _fixture(lengths, base, device)
    if name == 'balanced_heads':
        query, key, weight = (tensor.clone() for tensor in cpu)
        query.copy_(query[:, :1].expand_as(query))
        weight.fill_(1 / 64)
        cpu = query, key, weight
        tensors = tuple(tensor.to(device) for tensor in cpu)
    elif name == 'global_winner':
        query, key, weight = (torch.zeros_like(tensor) for tensor in cpu)
        query[:, 0, 0] = 1
        query[:, 32, 1] = 1
        weight[:, 0] = weight[:, 32] = 1
        key[:2048, 0, 0] = 10
        key[2048:4096, 0, 1] = 10
        key[4096, 0, :2] = 6
        # Exact binary low bits remove the unrelated stock/local tiling tie ambiguity.
        # Each shard still has 2048 candidates strictly above the eventual global winner.
        for bit in range(12):
            query[:, 0, 2 + bit] = 2**(bit - 14)
            key[:4096, 0, 2 + bit] = ((torch.arange(4096) >> bit) & 1).bfloat16()
        cpu = query, key, weight
        tensors = tuple(tensor.to(device) for tensor in cpu)
    stock = torch.ops.npu.npu_lightning_indexer(*tensors, actual_seq_lengths_query=layout.length_tensor,
        actual_seq_lengths_key=layout.length_tensor, layout_query='TND', layout_key='TND',
        sparse_count=2048, sparse_mode=3, return_value=True)
    return meta, cpu, tensors, stock


def _query_tile(lengths, pattern, cp_members, cp_rank, generation):
    meta = _query_meta(lengths, pattern, len(cp_members), cp_rank)
    wanted = set()
    for start, end in zip(meta.global_cu_seqlens, meta.global_cu_seqlens[1:]):
        length = end - start
        wanted.update(start + position for position in (0, min(1, length - 1), length // 2,
                                                         max(0, length - 2), length - 1))
    return replace(meta, q_global_ids=tuple(token for token in meta.q_global_ids if token in wanted),
                   kv_global_ids=tuple(range(meta.global_valid_queries)), cp_ranks=cp_members,
                   root_pes=cp_members, heap_generation=generation)


def _case(scenario, backend, global_meta, tensors, stock, lengths, pattern, name, cp_members, cp_rank, replay):
    meta = _query_tile(lengths, pattern, cp_members, cp_rank, backend.transport.root.generation)
    invocation = backend.prepare(meta, epoch=replay + 1)
    rank = dist.get_rank(backend.transport.group)
    heads = slice(rank * backend.local_heads, (rank + 1) * backend.local_heads)
    ids = list(meta.q_global_ids)
    inputs = tensors[0][ids, heads].contiguous(), tensors[1][:, 0].contiguous(), tensors[2][ids, heads].contiguous()
    result = backend.forward(*inputs, invocation)
    actual = result.global_indices.cpu()
    starts = torch.tensor([token - meta.sequence_position(token)[1] for token in ids], dtype=torch.int32).reshape(-1, 1)
    expected_relative = stock[0][ids, 0].cpu()
    expected = torch.where(expected_relative >= 0, expected_relative + starts, -1)
    torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0)
    torch.testing.assert_close(result.values.cpu().sort(-1).values.view(torch.int16),
                               stock[1][ids, 0].cpu().sort(-1).values.view(torch.int16), rtol=0, atol=0)
    counts = (actual >= 0).sum(-1)
    torch.testing.assert_close(counts, torch.tensor([min(2048, meta.sequence_position(token)[1] + 1)
                                                   for token in ids], dtype=torch.long))
    for row, token in enumerate(ids):
        sequence, position = meta.sequence_position(token)
        selected = actual[row][actual[row] >= 0].tolist()
        if len(selected) != len(set(selected)) or any(global_meta.sequence_position(key)[0] != sequence
                or global_meta.sequence_position(key)[1] > position for key in selected):
            raise RuntimeError('TP LI selected a duplicate, cross-sequence or future key')
        if name == 'global_winner' and token == 4096 and actual[row, 0].item() != 4096:
            raise RuntimeError('pre-Top-K head SUM failed to retain the global winner')
    long_history = any(meta.sequence_position(token)[1] + 1 > 2048 for token in ids)
    phases = validate_fused_indexer_traces(tuple(trace.cpu() for trace in result.phase_trace.unbind()),
        backend.schedule, require_ld=long_history)
    offset = backend.transport.spec.control_offset
    controls = backend.transport.arena[offset:offset + backend.transport.spec.slots * 640]
    controls = controls.view(torch.int64).reshape(-1, 80).cpu()
    active_slots = 0
    for control in controls:
        sequence = control[0].item()
        if sequence // 2**32 != replay + 1:
            continue
        active_slots += 1
        for peer in range(len(backend.transport.members)):
            if control[(2 + peer) * 8].item() != sequence:
                raise RuntimeError('TP LI source lacks final reader ACK')
    scenario['cases'].append({'passed': True, 'pattern': pattern, 'fixture': name, 'replay': replay,
        'groups': backend.schedule.compute_groups, 'local_queries': len(ids), 'global_keys': sum(lengths),
        'stock_selected_sets_exact': True, 'stock_bf16_values_exact': True, 'causal_global_ids': True,
        'active_transport_slots': active_slots, 'final_readers_acknowledged': True, 'phase_evidence': phases})
    return {'passed': True}


def _run(report, topology, groups, lengths, name, pattern, compute_groups):
    cp_members, cp_group, tp_group = groups
    root = SharedShmemRoot(torch.device('npu', int(os.environ.get('LOCAL_RANK', '0'))), root_group=dist.group.WORLD)
    transport = TpTileSumProbe(root, 'tp-li', TpTileSumSpec(2 * compute_groups, 512), tp_group=tp_group)
    backend = TpLocalIndexerProbe(transport, schedule=MixedSfaSchedule(compute_groups))
    scenario = {'topology': topology, 'cp_members': cp_members, 'tp_members': transport.members, 'cases': []}
    report['scenarios'].append(scenario)
    try:
        transport.bind()
        prepared = collective_validation_check(lambda: {'states': _states(lengths, name, root.device)})
        meta, _cpu, tensors, stock = prepared['states']
        for replay in range(2):
            collective_validation_check(partial(_case, scenario, backend, meta, tensors, stock, lengths, pattern,
                                                name, cp_members, dist.get_rank(cp_group), replay))
        scenario['status'] = 'passed'
    finally:
        transport.close()
        root.close()


def main() -> None:
    """Validate genuine CP-local query tiles and TP head shards, including empty owners."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    report = {'status': 'running', 'scenarios': [], 'scope': 'native pre-Top-K TP head SUM with replicated global K',
              'native_tp_kl_acceptance': False, 'cp_tp_model_acceptance': False, 'performance_acceptance': False}
    try:
        torch.npu.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError('native TP LI validator requires four root members')
        report['rank'] = dist.get_rank()
        single = tuple(((rank,), dist.new_group([rank])) for rank in range(4))
        cp_pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 1), (2, 3)))
        tp_pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 2), (1, 3)))
        rank = dist.get_rank()
        cp_pair = next(item for item in cp_pairs if rank in item[0])
        tp_pair = next(item for item in tp_pairs if rank in item[0])
        topologies = [('cp4_tp1', (tuple(range(4)), dist.group.WORLD, single[rank][1])),
                      ('cp2_tp2', (*cp_pair, tp_pair[1])), ('cp1_tp4', (*single[rank], dist.group.WORLD))]
        fixtures = [((3, 10), 'random_signed'), ((3, 10), 'random_signed_fp32_weights'),
                    ((3, 2113), 'balanced_heads'), ((4097,), 'global_winner')]
        patterns, schedules = ('strided', 'interior', 'empty'), (1, 7, 19)
        if args.smoke:
            fixtures, patterns, schedules = fixtures[:1], ('strided', 'empty'), (7,)
        for topology, groups in topologies:
            for lengths, name in fixtures:
                for pattern in patterns:
                    for compute_groups in schedules:
                        _run(report, topology, groups, lengths, name, pattern, compute_groups)
        report.update(status='passed', loaded_library_sha256=_loaded_library_identity())
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
