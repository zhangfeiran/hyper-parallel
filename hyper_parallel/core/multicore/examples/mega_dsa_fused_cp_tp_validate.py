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
"""Native fused CP/TP local forward and selected KL against full-head stock and FP32."""

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

from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_tp_kl_validate import (
    _oracle,
    _stock,
)
from hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate import (
    _SCALE,
    _inputs,
    _metrics,
)
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import (
    collective_validation_check,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_tp import (
    FusedDsaCpTpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.selected_requests import (
    validate_selected_training_traces,
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


def _attention_stock(full, lengths, selected, meta, heads):
    layout = CannDsaLayout(DsaBatchMeta.packed(lengths), full[0].device)
    indices = torch.ops.npu.npu_lightning_indexer(
        full[4], full[5][:, None], full[6], actual_seq_lengths_query=layout.length_tensor,
        actual_seq_lengths_key=layout.length_tensor, layout_query='TND', layout_key='TND',
        sparse_count=2048, sparse_mode=3, return_value=False)[0]
    rows = list(meta.q_global_ids)
    starts = torch.tensor([token-meta.sequence_position(token)[1] for token in rows], dtype=torch.int32,
                          device=full[0].device).reshape(-1, 1)
    indices[rows, 0] = torch.where(selected >= 0, selected-starts, -1)
    output = torch.ops.npu.npu_sparse_flash_attention(
        full[0], full[1][:, None], full[1][:, None], indices, _SCALE,
        actual_seq_lengths_query=layout.length_tensor, actual_seq_lengths_kv=layout.length_tensor,
        query_rope=full[2], key_rope=full[3][:, None], sparse_block_size=1, layout_query='TND', layout_kv='TND',
        sparse_mode=3, attention_mode=2, return_softmax_lse=True)[0]
    return output[rows, heads]


def _case(scenario, backend, meta, cpu, full, lengths, with_kl, epoch):
    invocation = backend.prepare(backend.cp.workspace.prepare(replace(meta, invocation=epoch)),
                                 epoch=epoch, query_ids=meta.q_global_ids, selected_kl=with_kl)
    rows, owners = list(meta.q_global_ids), list(meta.kv_global_ids)
    rank = dist.get_rank(backend.li.group)
    main_heads = slice(rank*backend.cp.heads, (rank+1)*backend.cp.heads)
    index_heads = slice(rank*backend.index_heads, (rank+1)*backend.index_heads)
    main_states = (full[0][rows, main_heads].contiguous(), full[1][owners].contiguous(),
            full[2][rows, main_heads].contiguous(), full[3][owners].contiguous())
    index = (full[4][rows, index_heads].contiguous(), full[5][owners].contiguous(),
             full[6][rows, index_heads].contiguous())
    result = backend.forward(main_states, index, invocation)
    torch.npu.synchronize()
    result.saved.validate_versions()
    selection = result.global_indices[:, 0]
    selected_cpu = selection.cpu()
    selections = [None]*len(backend.li.members)
    dist.all_gather_object(selections, selected_cpu, group=backend.li.group)
    if any(not torch.equal(value, selected_cpu) for value in selections):
        raise RuntimeError('fused TP peers disagree on ordered LI selected IDs')
    counts = (selected_cpu >= 0).sum(-1)
    expected = torch.tensor([min(meta.sequence_position(token)[1]+1, 2048) for token in rows], dtype=torch.int64)
    torch.testing.assert_close(counts, expected, rtol=0, atol=0)
    output = _metrics(result.output, _attention_stock(full, lengths, selection, meta, main_heads))
    evidence = validate_selected_training_traces(tuple(trace.cpu() for trace in result.phase_traces),
                                                backend.cp.schedule, require_ld=False, with_kl=with_kl)
    comparisons = []
    if with_kl and rows:
        _stats, stock = _stock(full, meta, selection, main_heads)
        oracle_meta = replace(meta, kv_global_ids=tuple(range(meta.global_valid_queries)))
        oracle, shared_oracle = _oracle(cpu, oracle_meta, selection, index_heads)
        shared = result.kl_key_partials.clone()
        dist.all_reduce(shared, group=backend.li.group)
        comparisons = [*_metrics_list((result.kl_gradients[0], result.kl_gradients[2], result.kl_loss),
                                       (oracle[0], oracle[2], oracle[3])), _metrics(shared, shared_oracle)]
        native = _metrics_list((result.kl_gradients[0], shared, result.kl_gradients[2], result.kl_loss),
                               (stock[0][:, index_heads], stock[1], stock[2][:, index_heads], stock[3]))
        if not all(item['pointwise_pass'] for item in native):
            raise RuntimeError('fused TP KL differs from full-head stock')
    passed = output['pointwise_pass'] and all(item['relative_l2'] <= .02 or item['max_abs'] <= 2e-5
                                             for item in comparisons)
    scenario['cases'].append({'passed': passed, 'queries': len(rows), 'with_kl': with_kl, 'epoch': epoch,
                              'output_stock': output, 'kl_fp32': comparisons, 'phases': evidence,
                              'tp_ordered_selection_equal': True, 'causal_cardinality_exact': True})
    if not passed:
        raise RuntimeError('fused CP TP numerical mismatch; thresholds unchanged')
    return {'passed': True}


def _metrics_list(actual, expected):
    return [_metrics(value, reference) for value, reference in zip(actual, expected)]


def _run(report, topology, groups, pattern):
    cp_members, cp_group, tp_group = groups
    lengths = (3, 10)
    size = dist.get_world_size(tp_group)
    global_heads = 64 if size == 1 else 128
    root = SharedShmemRoot(torch.device('npu', int(os.environ['LOCAL_RANK'])), root_group=dist.group.WORLD)
    meta = _metadata(lengths, pattern, len(cp_members), dist.get_rank(cp_group))
    meta = replace(meta, cp_ranks=cp_members, root_pes=cp_members)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(len(cp_members))))
    workspace = MegaDsaWorkspace(root, 'fused-cp', DsaWorkspaceSpec(capacity, sum(lengths), capacity, len(cp_members)),
                                 cp_group=cp_group)
    li = TpTileSumProbe(root, 'fused-li', TpTileSumSpec(14, 512), tp_group=tp_group)
    kl = TpTileSumProbe(root, 'fused-kl', TpTileSumSpec(14, 2048), tp_group=tp_group)
    scenario = {'topology': topology, 'pattern': pattern, 'cases': []}
    report['scenarios'].append(scenario)
    try:
        workspace.bind()
        li.bind()
        kl.bind()
        meta = replace(meta, heap_generation=root.generation)
        cp = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=global_heads//size,
                                   attention_scale=_SCALE, schedule=MixedSfaSchedule(7))
        backend = FusedDsaCpTpForwardProbe(cp, li, kl)
        cpu, _cotangent = _inputs(global_heads, torch.bfloat16, lengths)
        full = tuple(value.to(root.device) for value in cpu)
        for epoch, with_kl in enumerate((False, True), start=1):
            collective_validation_check(partial(_case, scenario, backend, meta, cpu, full, lengths, with_kl, epoch))
        scenario['status'] = 'passed'
    finally:
        kl.close()
        li.close()
        workspace.close()
        root.close()


def main() -> None:
    """Archive four-card fused local forward/KL smoke without claiming model acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'scenarios': [], 'cp_tp_model_acceptance': False,
              'performance_acceptance': False, 'scope': 'fused local CP TP forward and saved selected KL'}
    try:
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError('fused CP TP validator requires four root members')
        rank = dist.get_rank()
        single = tuple(((peer,), dist.new_group([peer])) for peer in range(4))
        cp_pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 1), (2, 3)))
        tp_pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 2), (1, 3)))
        cp_pair = next(item for item in cp_pairs if rank in item[0])
        tp_pair = next(item for item in tp_pairs if rank in item[0])
        topologies = [('cp4_tp1', (tuple(range(4)), dist.group.WORLD, single[rank][1])),
                      ('cp2_tp2', (*cp_pair, tp_pair[1])), ('cp1_tp4', (*single[rank], dist.group.WORLD))]
        for topology, groups in topologies:
            for pattern in ('strided', 'empty'):
                _run(report, topology, groups, pattern)
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
