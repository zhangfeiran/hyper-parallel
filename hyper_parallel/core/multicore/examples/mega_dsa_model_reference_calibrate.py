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
"""Independent stock BF16 versus FP32 full-model reference calibration; no fused candidate execution."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch_npu

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.core.multicore.examples.mega_dsa_fused_model_cp_validate import (
    _complete_indices,
    _configure,
    _IndependentStockReference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _OracleDsaReference,
    _rotary_mul_oracle,
    model_inputs,
    unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_tp_model_common import (
    tp_model_fixture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.model_boundary import (
    CannDsaReferenceAttention,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_reference_precision import (
    fit_model_reference_profile,
    model_reference_policy,
    model_reference_shapes,
    policy_sha256,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    gradient_error_metrics,
)


def _fields(model, output, loss, gradients):
    return {'output': output, 'hidden': gradients[0], 'loss': loss,
            **{f'gradient/{name}': value for (name, _parameter), value in zip(model.named_parameters(), gradients[1:])}}


def _oracle_check(template, mode, hidden_cpu, embeddings_cpu, cotangent, meta, expected, policy):
    source = copy.deepcopy(template).float()
    _configure(source, mode)
    model = CannDsaReferenceAttention(module=source)
    hidden = hidden_cpu.float().requires_grad_()
    backend = _OracleDsaReference(meta, attention_scale=model.scaling, indices=_complete_indices(meta))
    set_aux_loss_scale(torch.tensor(policy['auxiliary']))
    with patch('hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul',
               side_effect=_rotary_mul_oracle):
        output = model(hidden, position_embeddings=tuple(value.float() for value in embeddings_cpu),
                       dsa_reference=backend)[0]
        gradients = torch.autograd.grad((output*cotangent).sum(), (hidden, *model.parameters()), allow_unused=True)
    actual = _fields(model, output, backend.last_loss, gradients)
    for name, value in actual.items():
        target = expected[name]
        if value is None or target is None:
            if value is not target:
                return False
        else:
            metric = gradient_error_metrics(value.detach(), target.detach())
            if not metric['finite'] or (metric['relative_l2'] or 0) > 1e-4 or metric['max_abs'] > 1e-6:
                return False
    return True


def _measure(heads, mode, seed, policy, device, *, sgd_geometry=False):
    meta = DsaBatchMeta.packed(tuple(policy['lengths']))
    template = tp_model_fixture(heads=heads, seed=seed)
    stock = CannDsaReferenceAttention(module=copy.deepcopy(template).to(device))
    oracle = copy.deepcopy(template).float()
    for model in (stock, oracle):
        _configure(model, mode)
    hidden_cpu, embeddings_cpu = model_inputs(meta, dtype=torch.bfloat16, seed=seed+2)
    hidden_stock = hidden_cpu.to(device).requires_grad_()
    hidden_oracle = hidden_cpu.float().requires_grad_()
    generator = torch.Generator().manual_seed(seed+3)
    cotangent = torch.randn(hidden_cpu.shape, generator=generator)*.01
    backend = _IndependentStockReference(CannDsaLayout(meta, device), attention_scale=stock.scaling,
                                         indices=_complete_indices(meta))
    set_aux_loss_scale(torch.tensor(policy['auxiliary'], device=device))
    output_stock = stock(hidden_stock, position_embeddings=tuple(value.to(device) for value in embeddings_cpu),
                         dsa_reference=backend)[0]
    gradients_stock = torch.autograd.grad((output_stock.float()*cotangent.to(device)).sum(),
                                          (hidden_stock, *stock.parameters()), allow_unused=True)
    with patch('hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul',
               side_effect=_rotary_mul_oracle):
        output_oracle, loss_oracle = unabsorbed_model_reference(
            oracle, hidden_oracle, tuple(value.float() for value in embeddings_cpu), meta, _complete_indices(meta))
        objective = (output_oracle*cotangent).sum()
        if loss_oracle is not None:
            objective = objective+loss_oracle*policy['auxiliary']
        gradients_oracle = torch.autograd.grad(objective, (hidden_oracle, *oracle.parameters()), allow_unused=True)
    actual = _fields(stock, output_stock, backend.last_loss, gradients_stock)
    expected = _fields(oracle, output_oracle, loss_oracle, gradients_oracle)
    oracle_verified = _oracle_check(template, mode, hidden_cpu, embeddings_cpu, cotangent, meta, expected, policy)
    selected = backend.last_selection.to_global_indices().cpu()
    selection_verified = torch.equal(selected.sort(-1).values, _complete_indices(meta).sort(-1).values)
    initial_stock = ({name: parameter.detach().clone() for name, parameter in stock.named_parameters()}
                     if sgd_geometry else {})
    for model, gradients in ((stock, gradients_stock), (oracle, gradients_oracle)):
        for parameter, gradient in zip(model.parameters(), gradients[1:]):
            parameter.grad = gradient
        torch.optim.SGD(model.parameters(), lr=policy['learning_rate'], foreach=False).step()
    actual.update({f'sgd/{name}': parameter for name, parameter in stock.named_parameters()})
    expected.update({f'sgd/{name}': parameter for name, parameter in oracle.named_parameters()})
    shapes = model_reference_shapes(heads)
    fields = {}
    for name, left in actual.items():
        right = expected[name]
        available = left is not None and right is not None
        if available:
            if list(left.shape) != shapes[name] or left.shape != right.shape:
                raise RuntimeError('reference model field does not have its declared complete global shape')
            metrics = gradient_error_metrics(left.detach().float().cpu(), right.detach().float().cpu())
        else:
            metrics = {}
        fields[name] = {**metrics, 'shape': shapes[name], 'availability_only': not available,
                        'availability_passed': available or left is right,
                        'oracle_verified': oracle_verified, 'selection_verified': selection_verified}
    if sgd_geometry:
        for field, (name, parameter) in enumerate(stock.named_parameters(), 1):
            control = initial_stock[name]
            actual_gradient = gradients_stock[field]
            if actual_gradient is not None:
                control.add_(actual_gradient, alpha=-policy['learning_rate'])
            reference_parameter = expected['sgd/'+name].detach().cpu().double()
            reference_gradient = gradients_oracle[field]
            active = reference_gradient is not None
            reference_gradient = None if not active else reference_gradient.detach().cpu().double()
            fields['sgd/'+name]['sgd_geometry'] = {
                'numel': reference_parameter.numel(),
                'parameter_norm': float(reference_parameter.norm()),
                'parameter_max_abs': float(reference_parameter.abs().max()),
                'gradient_active': active,
                'gradient_norm': 0. if not active else float(reference_gradient.norm()),
                'gradient_max_abs': 0. if not active else float(reference_gradient.abs().max()),
                'single_axpy_verified': bool(torch.equal(parameter, control))}
    return fields


def _environment(policy):
    return {'torch': str(torch.__version__), 'torch_npu': torch_npu.__version__,
            'cpu_reference_threads': torch.get_num_threads(),
            'soc': torch.npu.get_device_name(0),
            'cann_environment': {name: os.environ.get(name) for name in
                                 ('ASCEND_HOME_PATH','ASCEND_OPP_PATH','ASCEND_CUSTOM_OPP_PATH')},
            'loaded_library_sha256': _loaded_library_identity(),
            'policy_sha256': policy_sha256(policy)}


def main() -> None:
    """Collect disjoint reference samples, persist failures and never fit from a fused candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.policy.read_text())
    if policy != model_reference_policy():
        raise ValueError('reference worker requires the exact predeclared immutable policy')
    report = {'status': 'running', 'candidate_measurements': False, 'policy': policy,
              'policy_sha256': policy_sha256(policy), 'cases': [], 'full_model_acceptance': False}
    try:
        torch.npu.set_device(0)
        device = torch.device('npu', 0)
        for heads in policy['heads']:
            for mode in policy['modes']:
                for seed in policy['fit_seeds']+policy['held_out_seeds']:
                    samples = [_measure(heads, mode, seed, policy, device) for _ in range(policy['repeats'])]
                    report['cases'].append({'heads': heads, 'mode': mode, 'seed': seed, 'samples': samples,
                                            'partition': 'calibration' if seed in policy['fit_seeds'] else 'held_out'})
                    args.output.write_text(json.dumps(report, indent=2)+'\n')
        report['environment'] = _environment(policy)
        report = fit_model_reference_profile(policy, report['cases'], report['environment'])
        if report['status'] != 'validated':
            raise RuntimeError('independent held-out model reference exceeds the frozen profile; margins unchanged')
    except Exception as error:
        report['error'] = repr(error)
        if report['status'] == 'running':
            report['status'] = 'error'
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
