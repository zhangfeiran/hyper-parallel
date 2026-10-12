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
"""Actual norm observation, detached-state replay and independent FP64 formulas on CPU."""

import unittest

import torch
from torch import nn

from hyper_parallel.core.multicore.examples.mega_dsa_norm_vjp_trace import (
    NormVjpTrace,
)


class TestNormVjpTrace(unittest.TestCase):
    """Norm observers preserve actual model gradients while exposing an upstream change."""

    def test_delayed_reverse_observation_and_independent_formula(self):
        """Forward-record order survives reverse backward; closed hooks do not affect later forwards."""
        norm = nn.RMSNorm(4, eps=1e-6, dtype=torch.float64)
        trace = NormVjpTrace({'norm': norm})
        inputs = [torch.arange(1., 9., dtype=torch.float64).reshape(2, 4)+offset for offset in (0., 1.)]
        isolated = []
        for value in inputs:
            local = value.clone().requires_grad_()
            output = norm(local)
            isolated.append(torch.autograd.grad(output.sum(), (local, norm.weight)))
        trace.set_phase('delayed')
        delayed_inputs = [value.clone().requires_grad_() for value in inputs]
        outputs = [norm(value) for value in delayed_inputs]
        outputs[1].sum().backward()
        outputs[0].sum().backward()
        with torch.no_grad():
            norm.weight.add_(1.)
        report = trace.diagnose(repeats=2)['norms']['norm']
        self.assertEqual(report['active_backward_records'], 4)
        for comparison in report['isolated_delayed']:
            self.assertTrue(all(field['bit_exact'] for field in comparison.values()))
        for item in report['replays']:
            for repeat in item['repeats']:
                self.assertTrue(repeat['source_output']['bit_exact'])
                self.assertTrue(repeat['source_input_vjp']['bit_exact'])
                self.assertLess(repeat['input_vjp_fp64_formula']['max_abs'], 1e-12)
                self.assertLess(repeat['weight_vjp_fp64_formula']['max_abs'], 1e-12)
        torch.testing.assert_close(norm.weight.grad, isolated[0][1]+isolated[1][1], rtol=0, atol=0)
        trace.close()
        self.assertFalse(norm._forward_hooks)
        self.assertIsNone(norm.weight._backward_hooks)
        norm(torch.ones(2, 4, dtype=torch.float64)).sum().backward()

    def test_upstream_change_is_separate_from_identical_state_replay(self):
        """A changed cotangent changes the real VJP while each same-state replay still matches."""
        norm = nn.RMSNorm(4, eps=1e-6, dtype=torch.float64)
        trace = NormVjpTrace({'norm': norm})
        for phase, scale in (('isolated', 1.), ('delayed', 2.)):
            trace.set_phase(phase)
            local = torch.arange(1., 5., dtype=torch.float64).requires_grad_()
            output = norm(local)
            torch.autograd.grad(output.sum()*scale, (local, norm.weight))
        report = trace.diagnose()['norms']['norm']
        comparison = report['isolated_delayed'][0]
        self.assertTrue(comparison['input']['bit_exact'])
        self.assertTrue(comparison['output']['bit_exact'])
        self.assertFalse(comparison['upstream']['bit_exact'])
        self.assertFalse(comparison['input_vjp']['bit_exact'])
        self.assertTrue(all(repeat['source_input_vjp']['bit_exact'] for item in report['replays']
                            for repeat in item['repeats']))
        trace.close()

    def test_incomplete_trace_and_invalid_configuration_are_rejected(self):
        """Refuse a missing delayed backward or unsupported normalization contract."""
        with self.assertRaisesRegex(ValueError, 'nn.RMSNorm'):
            NormVjpTrace({'norm': nn.LayerNorm(4)})
        norm = nn.RMSNorm(4, eps=1e-6)
        trace = NormVjpTrace({'norm': norm})
        with self.assertRaisesRegex(ValueError, 'phase'):
            trace.set_phase('other')
        with self.assertRaisesRegex(ValueError, 'repeat count'):
            trace.diagnose(repeats=0)
        with self.assertRaisesRegex(RuntimeError, 'complete isolated and delayed'):
            trace.diagnose()
        trace.close()
