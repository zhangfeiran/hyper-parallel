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
"""Four-process CP2 x TP2 mathematical boundary validation, without native fusion."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.cp_reference import DsaCpLayout
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    indexer_reference,
    selected_kl_reference,
    sparse_attention_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_collectives import (
    DsaTpHeadCollectives,
)


def _selection(boundary, query, keys, weights, meta, tile_size):
    rows = len(meta.q_global_ids)
    best = query.new_full((rows, 3), -torch.inf)
    ids = torch.full((rows, 3), -1, dtype=torch.int32)
    for start in range(0, len(keys), tile_size):
        end = min(start + tile_size, len(keys))
        scores = boundary.index_scores(query.detach(), keys[start:end].detach(), weights.detach())
        valid = torch.tensor([[meta.sequence_position(token)[0] == meta.sequence_position(key)[0]
                               and meta.sequence_position(key)[1] <= meta.sequence_position(token)[1]
                               for key in range(start, end)] for token in meta.q_global_ids], dtype=torch.bool)
        valid = valid.reshape(rows, end - start)
        scores = scores.masked_fill(~valid, -torch.inf)
        candidates = torch.arange(start, end, dtype=torch.int32).expand(rows, -1).masked_fill(~valid, -1)
        best = torch.cat((best, scores), -1)
        ids = torch.cat((ids, candidates), -1)
        id_order = ids.argsort(dim=-1, stable=True)
        best, ids = best.gather(-1, id_order), ids.gather(-1, id_order)
        order = best.argsort(dim=-1, descending=True, stable=True)[:, :3]
        best, ids = best.gather(-1, order), ids.gather(-1, order)
    return ids


def _metadata(empty, members, group):
    meta = DsaBatchMeta.packed((3, 4))
    owners = tuple(0 if empty else token % 2 for token in range(7))
    offsets = tuple(sum(owner == owners[token] for owner in owners[:token]) for token in range(7))
    rank = dist.get_rank(group)
    ids = tuple(token for token in reversed(range(7)) if owners[token] == rank)
    return replace(meta, token_owners=owners, token_local_offsets=offsets, q_global_ids=ids, kv_global_ids=ids,
                   cp_ranks=members, root_pes=members, cp_rank=rank)


def _fixture(cp_members, cp_group, tp_group, empty, tile_size, coefficient, divisor):
    generator = torch.Generator().manual_seed(20261011)
    shapes = ((7, 4, 3), (7, 3), (7, 4, 2), (7, 2), (7, 4, 3), (7, 3), (7, 4))
    full = tuple(torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_() for shape in shapes)
    meta = _metadata(empty, cp_members, cp_group)
    layout = DsaCpLayout(meta, 'cpu', group=cp_group)
    tp_rank = dist.get_rank(tp_group)
    heads = slice(tp_rank * 2, tp_rank * 2 + 2)
    rows = torch.tensor(meta.q_global_ids, dtype=torch.long)
    local = tuple((tensor[rows, heads] if field in (0, 2, 4, 6) else tensor[rows]).detach().clone().requires_grad_()
                  for field, tensor in enumerate(full))
    gathered = layout.gather_fields(local, tuple(tensor.shape[1:] for tensor in local))
    query, compressed, query_rope, key_rope, index_query, index_key, weight = gathered
    local_query, local_rope = query[rows], query_rope[rows]
    local_index, local_weight = index_query[rows], weight[rows]
    global_meta = replace(meta, kv_global_ids=tuple(range(7)))
    signature = (meta.global_cu_seqlens, meta.q_global_ids, global_meta.kv_global_ids, meta.cache_identity)
    boundary = DsaTpHeadCollectives(tp_group, row_signature=signature)
    selected = _selection(boundary, local_index, index_key, local_weight, meta, tile_size)
    reference_meta = DsaBatchMeta.packed((3, 4))
    wanted_selection = indexer_reference(*full[4:], reference_meta, sparse_count=3)
    torch.testing.assert_close(selected, wanted_selection[rows])
    output, _ = sparse_attention_reference(local_query, compressed, local_rope, key_rope, selected, global_meta,
                                           attention_scale=0.37)
    valid = selected != -1
    selected_keys = selected.clamp_min(0).long()
    logits = (torch.einsum('thd,tkd->thk', local_query, compressed[selected_keys])
              + torch.einsum('thd,tkd->thk', local_rope, key_rope[selected_keys])) * 0.37
    teacher = boundary.teacher(logits, valid)
    partial = (torch.einsum('thd,tkd->thk', local_index, index_key[selected_keys]).relu()
               * local_weight[..., None]).sum(1)
    # The selected-key order differs per query, so share the same head-SUM VJP as dense tiles.
    scores = boundary.sum_selected_scores(partial)
    normalization = DsaLossNormalization(7, divisor)
    loss = boundary.selected_kl(scores, teacher, valid, normalization=normalization, loss_coeff=coefficient)
    objective = output.square().sum() / (7 * 4 * 3) + loss * -2.5
    gradients = torch.autograd.grad(objective, local)
    shared = boundary.sum_shared_gradients(tuple(gradients[field] for field in (1, 3, 5)))
    gradients = tuple(shared[(1, 3, 5).index(field)] if field in (1, 3, 5) else gradient
                      for field, gradient in enumerate(gradients))
    wanted_output, _ = sparse_attention_reference(*full[:4], wanted_selection, reference_meta, attention_scale=0.37)
    wanted_loss = selected_kl_reference(*full[4:], *full[:4], wanted_selection, reference_meta,
                                        attention_scale=0.37, normalization=normalization, loss_coeff=coefficient)
    wanted_gradients = torch.autograd.grad(wanted_output.square().sum() / (7 * 4 * 3) + wanted_loss * -2.5, full)
    torch.testing.assert_close(output, wanted_output[rows, heads], rtol=1e-10, atol=1e-12)
    for field, (actual, expected) in enumerate(zip(gradients, wanted_gradients)):
        wanted = expected[rows, heads] if field in (0, 2, 4, 6) else expected[rows]
        torch.testing.assert_close(actual, wanted, rtol=1e-10, atol=1e-12)
    total_loss = loss.detach().clone()
    dist.all_reduce(total_loss, group=cp_group)
    torch.testing.assert_close(total_loss, wanted_loss.detach(), rtol=1e-10, atol=1e-12)
    return {'empty_owner': empty, 'candidate_tile': tile_size, 'loss_coeff': coefficient,
            'reducer_divisor': divisor, 'local_queries': len(rows), 'passed': True}


def main() -> None:
    """Compare all seven input gradients with a full-head, full-token FP64 oracle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group('gloo', timeout=timedelta(minutes=3))
    report = {'status': 'running', 'rank': dist.get_rank(), 'cases': [], 'cp': 2, 'tp': 2,
              'native_fusion_acceptance': False, 'npu_acceptance': False, 'full_model_acceptance': False}
    try:
        if dist.get_world_size() != 4:
            raise ValueError('CP2 x TP2 validator requires four processes')
        cp_groups = tuple((members, dist.new_group(list(members))) for members in ((0, 2), (1, 3)))
        tp_groups = tuple((members, dist.new_group(list(members))) for members in ((0, 1), (2, 3)))
        cp_members, cp_group = next(item for item in cp_groups if dist.get_rank() in item[0])
        tp_group = next(item[1] for item in tp_groups if dist.get_rank() in item[0])
        for empty in (False, True):
            for tile_size in (1, 2, 7):
                for coefficient in (0.0, 0.37):
                    for divisor in (1, 2):
                        report['cases'].append(_fixture(cp_members, cp_group, tp_group, empty, tile_size,
                                                         coefficient, divisor))
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2) + '\n')
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
