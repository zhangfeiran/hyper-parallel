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
"""Source-rounded norm identity and independent full-precision derivatives."""

import unittest

import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.modules.mega_dsa.parameter_vjp import (
    ParameterVjpCapture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.source_rms_norm import (
    source_rms_norm_with_fp32_vjp,
)


class TestSourceRmsNorm(unittest.TestCase):
    """Check the source primal separately from the declared mixed-precision derivative."""

    def test_source_primal_and_independent_float_double_derivatives(self):
        """Empty/nonempty BF16 and FP64 cases retain unrounded upstream and original Parameters."""
        generator = torch.Generator().manual_seed(20261255)
        for source_dtype in (torch.bfloat16, torch.float64):
            for rows in (0, 3):
                for replay in ('plain', 'checkpoint', 'retained'):
                    with self.subTest(dtype=source_dtype, rows=rows, replay=replay):
                        dtype = torch.float64 if source_dtype == torch.float64 else torch.float32
                        norm = nn.RMSNorm(8, eps=1e-6).to(source_dtype)
                        with torch.no_grad():
                            norm.weight.copy_(torch.randn(8, generator=generator))
                        parameter = norm.weight
                        capture = ParameterVjpCapture({'weight': parameter})
                        inputs = torch.randn(rows, 8, generator=generator).to(source_dtype).to(dtype).requires_grad_()
                        upstream = torch.randn(rows, 8, generator=generator).to(dtype)

                        def _forward(value, collector=capture, module=norm, compute=dtype):
                            return source_rms_norm_with_fp32_vjp(
                                value, collector.view('weight', dtype=compute), module)

                        with torch.autocast('cpu', dtype=torch.bfloat16):
                            output = (checkpoint(_forward, inputs, use_reentrant=False)
                                      if replay == 'checkpoint' else _forward(inputs))
                        expected = norm(inputs.to(source_dtype)).to(dtype)
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        reference_inputs = inputs.detach().double().requires_grad_()
                        reference_weight = torch.Tensor.detach(parameter).double().requires_grad_()
                        reference = functional.rms_norm(reference_inputs, (8,), reference_weight, norm.eps)
                        oracle = torch.autograd.grad(reference, (reference_inputs, reference_weight), upstream.double())
                        observed = []
                        for _repeat in range(2 if replay == 'retained' else 1):
                            capture.begin_backward()
                            gradients = torch.autograd.grad(output, (inputs, parameter), upstream,
                                                            retain_graph=replay == 'retained')
                            capture.finish_backward()
                            weight_vjp = capture.take()['weight']
                            torch.testing.assert_close(gradients[0].double(), oracle[0], rtol=2e-5, atol=2e-6)
                            torch.testing.assert_close(weight_vjp.double(), oracle[1], rtol=2e-5, atol=2e-6)
                            torch.testing.assert_close(gradients[1], weight_vjp.to(source_dtype), rtol=0, atol=0)
                            observed.append((gradients[0], weight_vjp))
                        self.assertIs(norm.weight, parameter)
                        if replay == 'retained':
                            for first, second in zip(*observed):
                                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_original_parameter_version_and_rejected_norms(self):
        """Saved scale mutation fails, and unsupported norms/epsilon cannot enter the candidate."""
        norm = nn.RMSNorm(8, eps=1e-6).bfloat16()
        inputs = torch.ones(2, 8, requires_grad=True)
        output = source_rms_norm_with_fp32_vjp(inputs, norm.weight.float(), norm)
        with torch.no_grad():
            norm.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
            output.sum().backward()
        for epsilon in (None, 0., -1., float('nan'), float('inf')):
            with self.subTest(epsilon=epsilon):
                module = nn.RMSNorm(8, eps=epsilon).bfloat16()
                with self.assertRaises(ValueError):
                    source_rms_norm_with_fp32_vjp(inputs, module.weight.float(), module)
        with self.assertRaises(ValueError):
            source_rms_norm_with_fp32_vjp(inputs.bfloat16(), norm.weight.float(), norm)

    def test_second_order_rejected(self):
        """The candidate declares a first-order contract rather than silently returning higher derivatives."""
        norm = nn.RMSNorm(8, eps=1e-6).bfloat16()
        inputs = torch.ones(2, 8, requires_grad=True)
        output = source_rms_norm_with_fp32_vjp(inputs, norm.weight.float(), norm)
        with self.assertRaisesRegex(ValueError, 'first-order'):
            torch.autograd.grad(output.sum(), inputs, create_graph=True)
