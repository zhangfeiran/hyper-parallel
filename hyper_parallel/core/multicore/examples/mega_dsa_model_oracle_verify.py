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
"""CPU formula identity and exact FP32 provenance for independent model calibration."""

from __future__ import annotations

import copy
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _OracleDsaReference,
    _rotary_mul_oracle,
    unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.model_boundary import (
    CannDsaReferenceAttention,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    gradient_error_metrics,
)


def _fields(model, output, loss, gradients):
    return {'output': output, 'hidden': gradients[0], 'loss': loss,
            **{f'gradient/{name}': value for (name, _parameter), value in zip(model.named_parameters(), gradients[1:])}}


def _evaluate(template, mode, hidden, embeddings, cotangent, meta, indices, policy, dtype, *, absorbed):
    source = copy.deepcopy(template).to(dtype=dtype)
    source.train(mode != 'eval')
    source.freeze_dsa = mode == 'freeze'
    source.dsa_loss_coeff = 0. if mode == 'zero' else .3
    model = CannDsaReferenceAttention(module=source) if absorbed else source
    inputs = hidden.to(dtype=dtype).detach().requires_grad_()
    rotary = 'hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul'
    scale = 'hyper_parallel.components.functional.aux_loss._AuxLossAutoScaler.main_loss_backward_scale'
    with patch(rotary, side_effect=_rotary_mul_oracle), patch(scale, torch.tensor(policy['auxiliary'], dtype=dtype)):
        if absorbed:
            backend = _OracleDsaReference(meta, attention_scale=model.scaling, indices=indices)
            output = model(inputs, position_embeddings=tuple(value.to(dtype=dtype) for value in embeddings),
                           dsa_reference=backend)[0]
            loss = backend.last_loss
        else:
            output, loss = unabsorbed_model_reference(model, inputs,
                tuple(value.to(dtype=dtype) for value in embeddings), meta, indices)
        objective = (output*cotangent.to(dtype=dtype)).sum()
        if not absorbed and loss is not None:
            objective = objective+loss*policy['auxiliary']
        gradients = torch.autograd.grad(objective, (inputs, *model.parameters()), allow_unused=True)
    return _fields(model, output, loss, gradients)


def verify_model_oracle(template: torch.nn.Module, mode: str, hidden: torch.Tensor,
                        embeddings: tuple[torch.Tensor, torch.Tensor], cotangent: torch.Tensor,
                        meta: DsaBatchMeta, indices: torch.Tensor, expected: dict, policy: dict) -> dict:
    """Prove both formulas in FP64 and exactly replay the FP32 fields used for calibration.

    Matrix reassociation is tested at the FP64 tolerances declared before
    device sampling. Calibration still compares BF16 with the unchanged FP32
    unabsorbed objective. The exact replay also catches changed or substituted
    expected tensors; FP64 values are never substituted into fitted samples.
    """
    if any(value.device.type != 'cpu' for value in (hidden, *embeddings, cotangent, indices)):
        raise ValueError('independent formula verification requires CPU inputs')
    if torch.get_num_threads() != policy['cpu_reference_threads']:
        raise ValueError('independent exact FP32 replay requires its declared CPU thread count')
    left = _evaluate(template, mode, hidden, embeddings, cotangent, meta, indices, policy,
                     torch.float64, absorbed=True)
    right = _evaluate(template, mode, hidden, embeddings, cotangent, meta, indices, policy,
                      torch.float64, absorbed=False)
    replay = _evaluate(template, mode, hidden, embeddings, cotangent, meta, indices, policy,
                       torch.float32, absorbed=False)
    if set(expected) != set(left):
        raise ValueError('independent formula verification requires all global FP32 reference fields')
    fields = {}
    for name, actual in left.items():
        target, original, recomputed = right[name], expected[name], replay[name]
        available = actual is not None and target is not None
        metric = gradient_error_metrics(actual, target) if available else {}
        identity = ((metric['finite'] and (metric['relative_l2'] or 0.) <= policy['oracle_relative_l2']
                     and metric['max_abs'] <= policy['oracle_max_abs']) if available else actual is target)
        exact = (original is recomputed if original is None or recomputed is None else
                 original.dtype == torch.float32 and original.device.type == 'cpu'
                 and bool(torch.equal(original.detach(), recomputed.detach())))
        fields[name] = {**metric, 'formula_identity_passed': identity, 'fp32_replay_exact': exact}
    return {'kind': policy['oracle_proof'], 'passed': all(
        field['formula_identity_passed'] and field['fp32_replay_exact'] for field in fields.values()), 'fields': fields}
