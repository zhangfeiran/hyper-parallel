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
"""Explicit offline captures of the real CP/TP model's sparse primitive and projection boundaries."""

from __future__ import annotations

from functools import partial
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.examples import (
    mega_dsa_cp_tp_model_validate as validation,
)
from hyper_parallel.core.multicore.examples.mega_dsa_fused_model_cp_validate import (
    _IndependentStockReference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import model_inputs
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta

_FIELDS = ('query', 'compressed', 'query_rope', 'key_rope', 'index_query', 'index_key', 'merge_weight')


def _cpu(tensor):
    return tensor.detach().cpu().clone()


def _gradient(target, name, gradient):
    target[name] = _cpu(gradient)


def _inputs(target, tensors, names):
    target.setdefault('inputs', {}).update({name: _cpu(value) for name, value in zip(names, tensors)})
    gradients = target.setdefault('gradients', {})
    for name, value in zip(names, tensors):
        if value.requires_grad:
            value.register_hook(partial(_gradient, gradients, name))


def _output(target, output):
    target['sparse_output'] = _cpu(output)
    if output.requires_grad:
        output.register_hook(partial(_gradient, target, 'sparse_cotangent'))


def _case(original, scenario, template, groups, workspace, forward, owner, meta, mode, replay, output_dir):
    if mode != 'joint' or replay != 'plain':
        raise ValueError('boundary capture requires the explicit joint plain smoke matrix')
    capture = {'native': {}, 'stock': {}, 'owner_ids': meta.kv_global_ids,
               'topology': scenario['topology'], 'pattern': scenario['pattern'],
               'initial_state': {name: _cpu(value) for name, value in template.state_dict().items()},
               'global_heads': template.num_heads, 'scope': 'diagnostic CPU snapshots; extra synchronization'}
    hidden, embeddings = model_inputs(DsaBatchMeta.packed((3, 10)), dtype=torch.bfloat16)
    capture.update(hidden=hidden, embeddings=embeddings)
    original_execute = validation._execute
    original_reduce = validation._reduce_parameters
    original_indexer = _IndependentStockReference.indexer
    original_attention = _IndependentStockReference.attention
    def _indexer(backend, *index):
        _inputs(capture['stock'], index, _FIELDS[4:])
        return original_indexer(backend, *index)
    def _attention(backend, *args):
        _inputs(capture['stock'], args[:4], _FIELDS[:4])
        output, stats = original_attention(backend, *args)
        _output(capture['stock'], output)
        return output, stats
    def _reduce(model, gradients, group):
        reduced = original_reduce(model, gradients, group)
        capture['parameter_gradients'] = {name: None if value is None else _cpu(value)
                                         for (name, _parameter), value in zip(model.named_parameters(), reduced[1:])}
        layout = model.dsa_tp_projection_layout
        capture.update(tp_rank=layout.rank, tp_members=layout.members, shards=layout.shards,
                       local_heads=layout.local_heads, local_index_heads=layout.local_index_heads,
                       hidden_gradient=None if reduced[0] is None else _cpu(reduced[0]))
        return reduced
    def _execute(model, probe, *args):
        original_forward = probe.forward
        def _forward(invocation, main_inputs, index):
            _inputs(capture['native'], (*main_inputs, *index), _FIELDS)
            output, loss = original_forward(invocation, main_inputs, index)
            _output(capture['native'], output)
            capture['native']['loss'] = _cpu(loss)
            return output, loss
        with patch.object(probe, 'forward', side_effect=_forward):
            result = original_execute(model, probe, *args)
        capture['native']['global_indices'] = result[2][0][2]
        return result
    with patch.object(_IndependentStockReference, 'indexer', _indexer), \
            patch.object(_IndependentStockReference, 'attention', _attention), \
            patch.object(validation, '_execute', _execute), \
            patch.object(validation, '_reduce_parameters', _reduce):
        proof = original(scenario, template, groups, workspace, forward, owner, meta, mode, replay, output_dir)
    for target in (capture['native'], capture['stock']):
        if set(target['inputs']) != set(_FIELDS) or set(target['gradients']) != set(_FIELDS):
            raise RuntimeError('boundary hooks did not capture every active sparse input and cotangent')
        if 'sparse_cotangent' not in target:
            raise RuntimeError('boundary hooks did not capture the compressed output cotangent')
    path = output_dir/f"{scenario['topology']}_{scenario['pattern']}_boundary_rank{dist.get_rank()}.pt"
    torch.save(capture, path)
    return proof


def main() -> None:
    """Run the existing four-card smoke with explicit CPU boundary hooks and unchanged acceptance limits."""
    original = validation._case
    with patch.object(validation, '_case', side_effect=partial(_case, original)):
        validation.main()


if __name__ == '__main__':
    main()
