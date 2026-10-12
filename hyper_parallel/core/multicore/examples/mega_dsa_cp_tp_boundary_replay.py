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
"""Fixed-state CP/TP sparse operator replay and projection VJP attribution from explicit CPU captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_boundary_snapshot import (
    boundary_metrics as _metrics,
)
from hyper_parallel.core.multicore.examples.mega_dsa_boundary_snapshot import (
    merge_boundary_parameters as _merge_parameters,
)
from hyper_parallel.core.multicore.examples.mega_dsa_boundary_snapshot import (
    merge_boundary_rows as _merge_rows,
)
from hyper_parallel.core.multicore.examples.mega_dsa_layer_diagnose import (
    _operator_replay,
    _project,
    _projection_vjp,
    _relu_crossings,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _OracleDsaReference,
    _rotary_mul_oracle,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_validate import (
    _StockReference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_restore_diagnose import (
    _restoration_replay,
)
from hyper_parallel.core.multicore.examples.mega_dsa_tp_model_common import (
    tp_model_fixture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    selected_kl_reference,
)

_FIELDS = ('query', 'compressed', 'query_rope', 'key_rope', 'index_query', 'index_key', 'merge_weight')
_OPERATOR_FIELDS = ('q_nope', 'compressed_kv', 'q_rope', 'k_rope', 'index_q', 'index_k', 'merge_weight')


def _replay(native, meta, model, device):
    values = tuple(native['inputs'][name] for name in _FIELDS)
    indices = native['indices']
    backend = (_OracleDsaReference(meta, attention_scale=model.scaling, indices=indices) if device.type == 'cpu'
               else _StockReference(CannDsaLayout(meta, device), attention_scale=model.scaling, indices=indices))
    return _operator_replay(values, native['sparse_cotangent'], backend, indices, auxiliary_scale=-1.75)


class _FixedReluBackward(torch.autograd.Function):
    """Keep FP32 forward values and force only a supplied diagnostic ReLU backward branch."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """Save the fixed branch without altering the ordinary ReLU output."""
        ctx.save_for_backward(positive)
        return value.clamp_min(0)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple:
        """Use the native projected branch as an explicit counterfactual, never a production VJP."""
        return gradient*ctx.saved_tensors[0], None


def _index_branch_control(model, main_states, index, meta, native, parameters):
    indices = native['indices'].long().clamp_min(0)
    inputs = native['inputs']
    dots = torch.einsum('thd,tkd->thk', inputs['index_query'].float(), inputs['index_key'].float()[indices])
    positive = dots > 0
    targets = dict(model.named_parameters())
    names = tuple(name for name in targets if name.startswith('indexer.'))
    def _loss():
        return selected_kl_reference(*index, *main_states, native['indices'], meta, attention_scale=model.scaling,
                                     normalization=DsaLossNormalization(13), loss_coeff=.3)
    baseline_loss = _loss()
    baseline = torch.autograd.grad(baseline_loss*-1.75, tuple(targets[name] for name in names), retain_graph=True)
    def _relu(value):
        if value.shape != positive.shape:
            raise ValueError('counterfactual must affect only the selected indexer score ReLU')
        return _FixedReluBackward.apply(value, positive)
    with patch.object(torch.Tensor, 'relu', _relu):
        forced_loss = _loss()
        forced = torch.autograd.grad(forced_loss*-1.75, tuple(targets[name] for name in names))
    torch.testing.assert_close(forced_loss, baseline_loss, rtol=0, atol=0)
    actual = {name: parameters[name] for name in names}
    return {'forward_loss_unchanged': True,
            'baseline': _metrics(actual, dict(zip(names, baseline))),
            'native_projected_relu_backward_branch': _metrics(actual, dict(zip(names, forced))),
            'scope': 'FP32 projection and teacher held; only ReLU backward branch forced; diagnostic only'}


def _restoration_control(model, native, parameters, hidden):
    generator = torch.Generator().manual_seed(20261011)
    cotangent = (torch.randn(hidden.shape, generator=generator)*.01).bfloat16()
    restored = _restoration_replay(model, {'sparse_input': native['sparse_output'],
                                         'hidden_shape': list(hidden.shape), 'output_cotangent': cotangent}, 'combined')
    weight = parameters['kv_b_proj.weight'].reshape(model.num_heads, -1, 512)
    actual = {'input_cotangent': native['sparse_cotangent'], 'W_UV': weight[:, model.qk_nope_head_dim:],
              'o_proj.weight': parameters['o_proj.weight']}
    return _metrics(actual, {name: restored[name] for name in actual})


def _projection_control(model, hidden, embeddings, meta, native, expected):
    hidden = hidden.float().requires_grad_()
    with patch('hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul',
               side_effect=_rotary_mul_oracle):
        main_states, index, _weight = _project(model, hidden, tuple(value.float() for value in embeddings))
        cotangents = tuple(native['gradients'][name].float() for name in _FIELDS)
        vjp = _projection_vjp((*main_states, *index), cotangents, hidden, model)
    actual_projection = {name: value for name, value in expected.items() if name != 'o_proj.weight'}
    vjp.pop('o_proj.weight')
    for fields in (actual_projection, vjp):
        fields['kv_b_proj.weight'] = fields['kv_b_proj.weight'].reshape(model.num_heads, -1, 512)[
            :, :model.qk_nope_head_dim]
    source_states = tuple(native['inputs'][name].float() for name in _FIELDS)
    crossings = _relu_crossings(source_states, tuple(value.detach() for value in (*main_states, *index)), meta)
    return {'fixed_native_cotangent_fp32_projection_vjp': _metrics(actual_projection, vjp),
            'projected_state_errors': _metrics(dict(zip(_FIELDS, source_states)),
                                               dict(zip(_FIELDS, (*main_states, *index)))),
            'legal_relu_crossings': crossings,
            'relu_branch_counterfactual': _index_branch_control(model, main_states, index, meta, native, expected),
            'fixed_sparse_and_output_cotangent_restoration': _restoration_control(model, native, expected, hidden)}


def _case(paths, device):
    peers = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
    first = peers[0]
    declarations = {(peer['topology'], peer['pattern'], peer['global_heads']) for peer in peers}
    if len(declarations) != 1 or len(peers) != 4:
        raise ValueError('boundary replay requires one complete four-root-rank capture group')
    replicas = []
    native = {section: {name: _merge_rows(peers, section, name, replicas) for name in _FIELDS}
              for section in ('inputs', 'gradients')}
    for field in ('sparse_output', 'sparse_cotangent'):
        native[field] = _merge_rows(peers, None, field, replicas)
    native['indices'] = torch.full((13, 2048), -1, dtype=torch.int32)
    for peer in peers:
        native['indices'][list(peer['owner_ids'])] = peer['native']['global_indices']
    native['loss'] = sum(peer['native']['loss'] for peer in peers if peer['tp_rank'] == 0)
    expected_parameters = _merge_parameters(peers, replicas)
    meta = DsaBatchMeta.packed((3, 10))
    model = tp_model_fixture(heads=first['global_heads'], dtype=torch.float32)
    model.load_state_dict(first['initial_state'])
    replay = _replay(native, meta, model, device)
    observed = {'output': native['sparse_output'], 'kl_loss': native['loss']}
    observed.update({name: native['gradients'][field] for name, field in zip(_OPERATOR_FIELDS, _FIELDS)})
    result = {'topology': first['topology'], 'pattern': first['pattern'],
              'fixed_native_states_and_cotangent': _metrics(observed, replay),
              'replica_checks': replicas, 'root_cause_proven': False,
              'full_model_acceptance': False,
              'capture_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    stock = first['stock']
    result['native_stock_projection_states'] = _metrics(native['inputs'], stock['inputs'])
    result['native_stock_projected_state_exact'] = {
        name: bool(torch.equal(native['inputs'][name], stock['inputs'][name])) for name in _FIELDS}
    result['native_stock_sparse_output_and_cotangent'] = _metrics(
        {name: native[name] for name in ('sparse_output', 'sparse_cotangent')},
        {name: stock[name] for name in ('sparse_output', 'sparse_cotangent')})
    result['native_stock_activation_vjps'] = _metrics(native['gradients'], stock['gradients'])
    if device.type == 'cpu':
        result.update(_projection_control(model, first['hidden'], first['embeddings'], meta, native,
                                          expected_parameters))
        stock_native = {'inputs': stock['inputs'], 'sparse_cotangent': stock['sparse_cotangent'],
                        'indices': native['indices']}
        stock_replay = _replay(stock_native, meta, model, device)
        stock_observed = {'output': stock['sparse_output']}
        stock_observed.update({name: stock['gradients'][field] for name, field in zip(_OPERATOR_FIELDS, _FIELDS)})
        result['fixed_stock_states_and_cotangent'] = _metrics(stock_observed, stock_replay)
    return result


def main() -> None:
    """Replay all captured topologies at identical input values and fixed compressed-output cotangents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'npu'), default='cpu')
    args = parser.parse_args()
    device = torch.device('npu', 0) if args.device == 'npu' else torch.device('cpu')
    if args.device == 'npu':
        torch.npu.set_device(0)
    groups = {}
    for path in sorted(args.captures.glob('*_boundary_rank*.pt')):
        groups.setdefault(path.name.split('_boundary_rank')[0], []).append(path)
    if not groups:
        raise ValueError('boundary replay requires explicit archived tensor captures')
    report = {'status': 'running', 'cases': [], 'device': args.device,
              'scope': 'fixed-state sparse primitive and projection VJP diagnostic; extra CPU synchronization',
              'full_model_acceptance': False, 'performance_acceptance': False}
    try:
        for paths in groups.values():
            report['cases'].append(_case(paths, device))
        report['status'] = 'diagnosed'
        if args.device == 'npu':
            report['loaded_library_sha256'] = _loaded_library_identity()
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
