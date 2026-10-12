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
"""Offline attribution of archived CP/TP model failures without launching native kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.core.multicore.examples.mega_dsa_local_training_oracle import (
    projection_vjp_metrics,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _OracleDsaReference,
    _rotary_mul_oracle,
    unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_tp_model_common import (
    tp_model_fixture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.model_boundary import (
    CannDsaReferenceAttention,
)


def _metric(actual, expected):
    if actual is None or expected is None:
        return {'passed': actual is None and expected is None, 'availability_only': True}
    return projection_vjp_metrics(actual, expected)


def _formula_control(capture):
    meta = DsaBatchMeta.packed((3, 10))
    heads = capture['initial_state']['q_b_proj.weight'].shape[0]//192
    source = tp_model_fixture(heads=heads, dtype=torch.float32)
    source.load_state_dict(capture['initial_state'])
    hidden = capture['hidden'].float().requires_grad_()
    embeddings = tuple(value.float() for value in capture['embeddings'])
    selected = torch.full((13, 2048), -1, dtype=torch.int32)
    for token in range(13):
        start = meta.global_cu_seqlens[meta.sequence_position(token)[0]]
        selected[token, :token-start+1] = torch.arange(start, token+1)
    set_aux_loss_scale(torch.tensor(-1.75))
    cotangent = capture['cotangent']
    with patch('hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul',
               side_effect=_rotary_mul_oracle):
        expected, expected_loss = unabsorbed_model_reference(source, hidden, embeddings, meta, selected)
        gradients = torch.autograd.grad((expected*cotangent).sum()+expected_loss*-1.75,
                                        (hidden, *source.parameters()))
        candidate = CannDsaReferenceAttention(module=source)
        backend = _OracleDsaReference(meta, attention_scale=source.scaling, indices=selected)
        observed = candidate(hidden, position_embeddings=embeddings, dsa_reference=backend)[0]
        repeated = torch.autograd.grad((observed*cotangent).sum(), (hidden, *candidate.parameters()))
    names = ('hidden', *dict(source.named_parameters()))
    fields = {'output': _metric(observed, expected), 'loss': _metric(backend.last_loss, expected_loss)}
    fields.update({name: _metric(left, right) for name, left, right in zip(names, repeated, gradients)})
    return {'passed': all(metric['passed'] for metric in fields.values()), 'fields': fields,
            'scope': 'CPU FP32 absorbed versus unabsorbed global model at archived quantized weights and inputs'}


def _diagnose(path, *, formula_control):
    capture = torch.load(path, map_location='cpu', weights_only=False)
    fields = {}
    for name, oracle in capture['oracle'].items():
        actual, stock = capture['actual'][name], capture['stock'][name]
        fields[name] = {'native_vs_stock': _metric(actual, stock), 'native_vs_fp32': _metric(actual, oracle),
                        'stock_vs_fp32': _metric(stock, oracle)}
    inherited = [name for name, metric in fields.items()
                 if not metric['native_vs_fp32']['passed'] and not metric['stock_vs_fp32']['passed']]
    native_only = [name for name, metric in fields.items()
                   if not metric['native_vs_stock']['passed']]
    result = {'capture': path.name, 'capture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'fields': fields, 'fp32_failures_also_present_in_stock': inherited,
              'native_stock_mismatch_fields': native_only,
              'root_cause_proven': False, 'native_model_acceptance': False}
    if formula_control:
        result['fp32_formula_control'] = _formula_control(capture)
    return result


def main() -> None:
    """Compare captured native/stock/oracle tensors and independently check the FP32 formula."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--formula-control', action='store_true')
    args = parser.parse_args()
    paths = sorted(args.captures.glob('*_joint_plain_rank*.pt'))
    if not paths:
        raise ValueError('diagnosis requires archived joint plain model captures')
    cases = []
    controlled = set()
    for path in paths:
        topology = '_'.join(path.stem.split('_')[:2])
        control = args.formula_control and topology not in controlled
        cases.append(_diagnose(path, formula_control=control))
        if control:
            controlled.add(topology)
    report = {'status': 'diagnosed', 'capture_count': len(cases), 'cases': cases,
              'scope': 'offline attribution; no native invocation or performance measurement',
              'native_model_acceptance': False, 'root_cause_proven': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
