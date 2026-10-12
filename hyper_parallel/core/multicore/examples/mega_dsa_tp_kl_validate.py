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
"""Native TP LI plus selected KL against full-head stock and independent FP32 VJP."""

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
from torch.nn import functional

from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_tp_indexer_validate import (
    _query_tile,
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
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_kl import (
    mixed_kl_native_key_cast,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule,
    validate_mixed_trace,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    selected_kl_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_indexer import (
    TpLocalIndexerProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_kl import TpLocalKlProbe
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumProbe,
    TpTileSumSpec,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _oracle(cpu, meta, selected, head_slice):
    rows = list(meta.q_global_ids)
    iq = cpu[4][rows].float().requires_grad_()
    ik = cpu[5].float().requires_grad_()
    weight = cpu[6][rows].float().requires_grad_()
    if not rows:
        empty = (iq[:, head_slice].detach(), torch.zeros_like(ik), weight[:, head_slice].detach(), torch.zeros(()))
        return empty, torch.zeros_like(ik)
    selected = selected.cpu().to(torch.int32)
    valid = selected >= 0
    offsets = selected.clamp_min(0).long()
    gathered = ik[offsets]
    raw = torch.einsum('thd,tkd->thk', iq, gathered)
    logits = (raw.relu()*weight[:, :, None]).sum(1).masked_fill(~valid, -torch.inf)
    query, key, rope, key_rope = (value.float() for value in cpu[:4])
    teacher_scores = (torch.einsum('thd,tkd->thk', query[rows], key[offsets])
                      + torch.einsum('thd,tkd->thk', rope[rows], key_rope[offsets]))*_SCALE
    probability = teacher_scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
    teacher = probability.sum(1)
    teacher = teacher / teacher.sum(-1, keepdim=True)
    log_probability = logits.log_softmax(-1).masked_fill(~valid, 0)
    loss = functional.kl_div(log_probability, teacher, reduction='sum')
    normalization = DsaLossNormalization(meta.global_valid_queries, meta.global_valid_queries)
    reference = selected_kl_reference(iq, ik, weight, query[rows], key, rope[rows], key_rope, selected, meta,
                                      attention_scale=_SCALE, normalization=normalization)
    torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-6)
    grad_q, grad_k, grad_w, grad_raw = torch.autograd.grad(loss, (iq, ik, weight, raw))
    selected_partials = torch.einsum('thk,thd->tkd', grad_raw[:, head_slice], iq[:, head_slice])
    partials = torch.zeros_like(ik)
    partials.index_add_(0, offsets[valid], selected_partials[valid])
    return (grad_q[:, head_slice], partials, grad_w[:, head_slice], loss.detach()), grad_k


def _stock(full, meta, selected, heads):
    rows = list(meta.q_global_ids)
    lengths = tuple(b-a for a, b in zip(meta.global_cu_seqlens, meta.global_cu_seqlens[1:]))
    layout = CannDsaLayout(DsaBatchMeta.packed(lengths), full[0].device)
    query, key, rope, key_rope, iq, ik, weight = (value.detach().clone() for value in full)
    for value in (query, rope, iq, weight):
        preserved = value[rows].clone()
        value.zero_()
        value[rows] = preserved
    indices = torch.ops.npu.npu_lightning_indexer(
        full[4], full[5][:, None], full[6], actual_seq_lengths_query=layout.length_tensor,
        actual_seq_lengths_key=layout.length_tensor, layout_query='TND', layout_key='TND',
        sparse_count=2048, sparse_mode=3, return_value=False)[0]
    starts = torch.tensor([token-meta.sequence_position(token)[1] for token in rows], device=query.device,
                          dtype=torch.int32).reshape(len(rows), 1)
    indices[rows, 0] = torch.where(selected >= 0, selected-starts, -1)
    _output, maximum, denominator = torch.ops.npu.npu_sparse_flash_attention(
        query, key[:, None], key[:, None], indices, _SCALE, actual_seq_lengths_query=layout.length_tensor,
        actual_seq_lengths_kv=layout.length_tensor, query_rope=rope, key_rope=key_rope[:, None],
        sparse_block_size=1, layout_query='TND', layout_kv='TND', sparse_mode=3,
        attention_mode=2, return_softmax_lse=True)
    gradient = torch.ops.npu.npu_sparse_lightning_indexer_grad_kl_loss(
        query, key[:, None], iq, ik[:, None], weight, indices, maximum, denominator, _SCALE,
        query_rope=rope, key_rope=key_rope[:, None], actual_seq_qlen=list(layout.cumulative_lengths),
        actual_seq_klen=list(layout.cumulative_lengths), layout='TND', sparse_mode=3)
    stats = (maximum[:, rows, heads].contiguous(), denominator[:, rows, heads].contiguous())
    return stats, (gradient[0][rows], gradient[1][:, 0], gradient[2][rows], gradient[3].reshape(()))


def _case(scenario, li, kl, cpu, full, lengths, pattern, cp_members, cp_group, replay, check_selection=False):
    meta = _query_tile(lengths, pattern, cp_members, dist.get_rank(cp_group), li.transport.root.generation)
    meta = replace(meta, q_global_ids=tuple(reversed(meta.q_global_ids)), invocation=replay+1)
    rows = list(meta.q_global_ids)
    rank = dist.get_rank(li.transport.group)
    index_heads = slice(rank*li.local_heads, (rank+1)*li.local_heads)
    main_heads = slice(rank*kl.local_heads, (rank+1)*kl.local_heads)
    selection = li.forward(full[4][rows, index_heads].contiguous(), full[5],
                           full[6][rows, index_heads].contiguous(), li.prepare(meta, epoch=replay+1))
    if check_selection:
        selected_cpu = selection.global_indices.cpu()
        selections = [None] * dist.get_world_size(li.transport.group)
        dist.all_gather_object(selections, selected_cpu, group=li.transport.group)
        if any(not torch.equal(value, selected_cpu) for value in selections):
            dump = Path(os.environ.get('MEGA_DSA_TP_KL_DIAGNOSTICS', '.'))
            dump.mkdir(parents=True, exist_ok=True)
            torch.save({'meta': meta, 'tp_ordered_selections': selections},
                       dump/f'selection_rank{dist.get_rank()}.pt')
            raise RuntimeError('TP LI peers produced different ordered selected IDs before KL')
    stats, stock = _stock(full, meta, selection.global_indices, main_heads)
    result = kl.gradients((full[0][rows, main_heads], full[1], full[2][rows, main_heads], full[3]),
                          (full[4][rows, index_heads], full[5], full[6][rows, index_heads]), stats,
                          kl.prepare(meta, selection.global_indices, epoch=replay+1))
    torch.npu.synchronize()
    oracle, shared_oracle = _oracle(cpu, meta, selection.global_indices, index_heads)
    actual = (*result.gradients, result.loss)
    independent = [_metrics(value, expected) for value, expected in zip(actual, oracle)]
    shared = result.key_partials.clone()
    dist.all_reduce(shared, group=kl.transport.group)
    native = [_metrics(result.gradients[0], stock[0][:, index_heads]),
              _metrics(shared, stock[1]), _metrics(result.gradients[2], stock[2][:, index_heads]),
              _metrics(result.loss, stock[3])]
    shared_metric = _metrics(shared, shared_oracle)
    torch.testing.assert_close(mixed_kl_native_key_cast(result.key_partials.cpu()), result.gradients[1].cpu(),
                               rtol=0, atol=0)
    phases = [validate_mixed_trace(trace.cpu(), kl.schedule) for trace in result.traces]
    offset = kl.transport.spec.control_offset
    controls = kl.transport.arena[offset:offset+kl.transport.spec.slots*640].view(torch.int64).reshape(-1, 80).cpu()
    active = 0
    for control in controls:
        sequence = int(control[0])
        if sequence // 2**32 == replay+1:
            active += 1
            if any(int(control[(2+peer)*8]) != sequence for peer in range(len(kl.transport.members))):
                raise RuntimeError('TP KL transport lacks final reader ACK')
    passed = (all(item['relative_l2'] <= .02 or item['max_abs'] <= 2e-5 for item in (*independent, shared_metric))
              and all(item['pointwise_pass'] for item in native))
    scenario['cases'].append({'passed': passed, 'pattern': pattern, 'replay': replay, 'queries': len(rows),
        'oracle': independent, 'stock': native, 'tp_summed_fp32_key_oracle': shared_metric,
        'key_cast_bits_exact': True, 'active_slots': active, 'final_readers_acknowledged': True,
        'phase_evidence': phases})
    if not passed:
        dump = Path(os.environ.get('MEGA_DSA_TP_KL_DIAGNOSTICS', '.'))
        dump.mkdir(parents=True, exist_ok=True)
        torch.save({'meta': meta, 'selected': selection.global_indices.cpu(),
                    'main': tuple(value.cpu() for value in (full[0][rows, main_heads], full[1],
                                                          full[2][rows, main_heads], full[3])),
                    'index': tuple(value.cpu() for value in (full[4][rows, index_heads], full[5],
                                                           full[6][rows, index_heads])),
                    'statistics': tuple(value.cpu() for value in stats),
                    'gradients': tuple(value.cpu() for value in result.gradients),
                    'loss': result.loss.cpu(), 'key_partials': result.key_partials.cpu(),
                    'oracle': oracle, 'stock': tuple(value.cpu() for value in stock)},
                   dump/f'rank{dist.get_rank()}.pt')
        raise RuntimeError('TP KL numerical mismatch; inspect archived per-field metrics')
    return {'passed': True}


def _run(report, topology, groups, lengths, pattern, compute_groups, weight_dtype, *, check_selection=False):
    cp_members, cp_group, tp_group = groups
    size = dist.get_world_size(tp_group)
    global_heads = 64 if size == 1 else 128
    root = SharedShmemRoot(torch.device('npu', int(os.environ['LOCAL_RANK'])), root_group=dist.group.WORLD)
    li_transport = TpTileSumProbe(root, 'tp-li', TpTileSumSpec(2*compute_groups, 512), tp_group=tp_group)
    kl_transport = TpTileSumProbe(root, 'tp-kl', TpTileSumSpec(2*compute_groups, 2048), tp_group=tp_group)
    schedule = MixedSfaSchedule(compute_groups)
    li = TpLocalIndexerProbe(li_transport, schedule=schedule)
    kl = TpLocalKlProbe(kl_transport, global_heads=global_heads, attention_scale=_SCALE, schedule=schedule)
    scenario = {'topology': topology, 'cp_members': cp_members, 'tp_members': kl_transport.members,
                'heads': global_heads, 'lengths': lengths, 'groups': compute_groups,
                'weight_dtype': str(weight_dtype), 'cases': []}
    report['scenarios'].append(scenario)
    try:
        li_transport.bind()
        kl_transport.bind()
        cpu, _cotangent = _inputs(global_heads, weight_dtype, lengths)
        if weight_dtype == torch.float32:
            cpu = (*cpu[:6], cpu[6]*1.003+0.000013)
        full = tuple(value.to(root.device) for value in cpu)
        for replay in range(2):
            collective_validation_check(partial(_case, scenario, li, kl, cpu, full, lengths, pattern,
                                                cp_members, cp_group, replay, check_selection=check_selection))
        scenario['status'] = 'passed'
    finally:
        kl_transport.close()
        li_transport.close()
        root.close()


def main() -> None:
    """Archive native local-head VJP, global-head objective and final protocol acknowledgments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--focus-tp4-long-g19', action='store_true')
    parser.add_argument('--check-tp-selection', action='store_true',
                        help='Add a diagnostic host synchronization and ordered TP selection check before KL')
    args = parser.parse_args()
    report = {'status': 'running', 'scenarios': [], 'pre_kl_selection_sync': args.check_tp_selection,
              'scope': 'native TP LI plus selected KL with replicated global K',
              'cp_tp_model_acceptance': False, 'performance_acceptance': False}
    try:
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError('TP KL validator requires four root members')
        rank = dist.get_rank()
        single = tuple(((peer,), dist.new_group([peer])) for peer in range(4))
        cp_pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 1), (2, 3)))
        tp_pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 2), (1, 3)))
        cp_pair = next(item for item in cp_pairs if rank in item[0])
        tp_pair = next(item for item in tp_pairs if rank in item[0])
        topologies = [('cp4_tp1', (tuple(range(4)), dist.group.WORLD, single[rank][1])),
                      ('cp2_tp2', (*cp_pair, tp_pair[1])), ('cp1_tp4', (*single[rank], dist.group.WORLD))]
        lengths_list, patterns, schedules = ((3, 10), (3, 2113)), ('strided', 'interior', 'empty'), (1, 7, 19)
        dtypes = (torch.bfloat16, torch.float32)
        if args.smoke:
            lengths_list, patterns, schedules, dtypes = ((3, 10),), ('strided', 'empty'), (7,), (torch.bfloat16,)
        if args.focus_tp4_long_g19:
            topologies = topologies[-1:]
            lengths_list, patterns, schedules, dtypes = ((3, 2113),), ('strided',), (19,), (torch.bfloat16,)
        for topology, groups in topologies:
            for lengths in lengths_list:
                for pattern in patterns:
                    for compute_groups in schedules:
                        for dtype in dtypes:
                            _run(report, topology, groups, lengths, pattern, compute_groups, dtype,
                                 check_selection=args.check_tp_selection)
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
