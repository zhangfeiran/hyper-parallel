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
"""Temporary RMSNorm activation/cotangent capture and identical-state VJP replay."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional

from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    gradient_error_metrics,
)


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    return {'bit_exact': bool(torch.equal(actual, expected)),
            'metrics': gradient_error_metrics(actual.detach().cpu(), expected.detach().cpu())}


class NormVjpTrace:
    """Observe norm inputs and cotangents without replacing the original forward or backward.

    This diagnostic adds copies and hooks and is unsuitable for performance
    measurements. Each observed input must feed only its declared norm branch.
    Captures stay on the original device until explicit post-backward diagnosis.
    """

    def __init__(self, modules: Mapping[str, nn.RMSNorm]) -> None:
        """Attach temporary observers to one-dimensional direct-scale nn.RMSNorm modules."""
        if not modules or any(not isinstance(module, nn.RMSNorm) or len(module.normalized_shape) != 1
                              or module.eps is None for module in modules.values()):
            raise ValueError('norm VJP trace requires one-dimensional nn.RMSNorm with explicit epsilon')
        self._phase = 'isolated'
        self._records = {name: [] for name in modules}
        self._handles = []
        for name, module in modules.items():
            self._handles.append(module.register_forward_hook(self._observer(name)))

    def set_phase(self, phase: str) -> None:
        """Label subsequent forward records as isolated or delayed."""
        if phase not in ('isolated', 'delayed'):
            raise ValueError('norm VJP trace phase must be isolated or delayed')
        self._phase = phase

    def _observer(self, name):
        def observe(module: nn.RMSNorm, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            """Capture the norm state and attach cotangent observers."""
            record = {'phase': self._phase, 'input': inputs[0].detach().clone(),
                      'weight': module.weight.detach().clone(), 'output': output.detach().clone(),
                      'shape': module.normalized_shape, 'eps': module.eps}
            self._records[name].append(record)

            def capture_input(gradient: torch.Tensor) -> torch.Tensor:
                """Record the input cotangent while preserving the original gradient."""
                record['input_vjp'] = gradient.detach().clone()
                return gradient

            def capture_output(gradient: torch.Tensor) -> torch.Tensor:
                """Record the upstream cotangent while preserving the original gradient."""
                record['upstream'] = gradient.detach().clone()
                return gradient

            if inputs[0].requires_grad:
                self._handles.append(inputs[0].register_hook(capture_input))
            if output.requires_grad:
                self._handles.append(output.register_hook(capture_output))

        return observe

    @staticmethod
    def _formula(record):
        inputs, weight, upstream = (record[name].detach().cpu().double() for name in ('input', 'weight', 'upstream'))
        inverse = torch.rsqrt(inputs.square().mean(-1, keepdim=True)+record['eps'])
        product = upstream*weight
        grad_input = inverse*product-inputs*inverse.pow(3)*(product*inputs).mean(-1, keepdim=True)
        grad_weight = (upstream*inputs*inverse).reshape(-1, weight.numel()).sum(0)
        return grad_input, grad_weight

    @classmethod
    def _replay(cls, record, repeats):
        formula_input, formula_weight = cls._formula(record)
        rows = []
        for _repeat in range(repeats):
            inputs = record['input'].detach().clone().requires_grad_()
            weight = record['weight'].detach().clone().requires_grad_()
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                output = functional.rms_norm(inputs, record['shape'], weight, record['eps'])
                grad_input, grad_weight = torch.autograd.grad(output, (inputs, weight), record['upstream'])
            rows.append({'source_output': _comparison(output, record['output']),
                         'source_input_vjp': _comparison(grad_input, record['input_vjp']),
                         'input_vjp_fp64_formula': gradient_error_metrics(grad_input.cpu(), formula_input),
                         'weight_vjp_fp64_formula': gradient_error_metrics(grad_weight.cpu(), formula_weight)})
        return rows

    def diagnose(self, *, repeats: int = 3) -> dict:
        """Replay identical state and compare isolated/delayed norm operands and input VJPs.

        Returns JSON metrics only. The FP64 formula is an independent mathematical
        derivative of the captured operands, without fitted acceptance thresholds.
        """
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise ValueError('norm VJP replay requires a positive repeat count')
        result = {'full_model_acceptance': False, 'norms': {}}
        for name, records in self._records.items():
            active = [record for record in records if 'upstream' in record and 'input_vjp' in record]
            isolated = [record for record in active if record['phase'] == 'isolated']
            delayed = [record for record in active if record['phase'] == 'delayed']
            if len(isolated) != len(delayed) or not isolated:
                raise RuntimeError('norm VJP trace requires matching complete isolated and delayed backward records')
            paired = [{field: _comparison(right[field], left[field])
                       for field in ('input', 'weight', 'output', 'upstream', 'input_vjp')}
                      for left, right in zip(isolated, delayed)]
            result['norms'][name] = {'forward_records': len(records), 'active_backward_records': len(active),
                                     'isolated_delayed': paired,
                                     'replays': [{'phase': record['phase'], 'repeats': self._replay(record, repeats)}
                                                 for record in active]}
        return result

    def close(self) -> None:
        """Remove every module and intermediate hook after the diagnostic backward completes."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
