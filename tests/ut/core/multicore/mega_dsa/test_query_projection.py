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
"""Source forward identity, first-order full-precision VJPs and saved-state checks on CPU."""

import unittest

import torch
from torch import nn
from torch.nn import functional

from hyper_parallel.core.multicore.modules.mega_dsa.parameter_vjp import (
    ParameterVjpCapture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.query_projection import (
    project_query_with_fp32_vjp,
)
from hyper_parallel.core.multicore.modules.mega_dsa.source_linear import (
    source_linear_with_full_precision_weight_vjp,
)


class TestQueryProjection(unittest.TestCase):
    """Compare source forward values and independent matrix VJPs without a process group."""

    def test_captured_weight_vjp_preserves_source_and_tp_input_gradient(self):
        """Weight VJP uses rounded forward inputs; TP input VJP stays unrounded in compute dtype."""
        generator = torch.Generator().manual_seed(20261251)
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            for rows in (0, 3):
                with self.subTest(dtype=dtype, rows=rows):
                    compute = torch.float64 if dtype == torch.float64 else torch.float32
                    latent = torch.randn(2, rows, 4, generator=generator).to(compute).requires_grad_()
                    weight = nn.Parameter(torch.randn(5, 4, generator=generator).to(dtype))
                    bias = nn.Parameter(torch.randn(5, generator=generator).to(dtype))
                    capture = ParameterVjpCapture({'weight': weight, 'bias': bias})
                    baseline = project_query_with_fp32_vjp(latent, weight, bias)
                    output = source_linear_with_full_precision_weight_vjp(
                        latent, capture.view('weight', dtype=compute), weight,
                        capture.view('bias', dtype=compute), bias, full_precision_inputs=True)
                    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
                    upstream = torch.randn(output.shape, generator=generator).to(dtype)
                    expected_grads = torch.autograd.grad(baseline, (latent, weight, bias), upstream)
                    capture.begin_backward()
                    grads = torch.autograd.grad(output, (latent, weight, bias), upstream)
                    capture.finish_backward()
                    observed = capture.take()
                    for observed_grad, expected_grad in zip(grads, expected_grads):
                        torch.testing.assert_close(observed_grad, expected_grad, rtol=0, atol=0)
                    expected_weight = (upstream.to(compute).reshape(-1, 5).T
                                       @latent.detach().to(dtype).to(compute).reshape(-1, 4))
                    torch.testing.assert_close(observed['weight'], expected_weight, rtol=0, atol=0)
                    torch.testing.assert_close(observed['bias'], upstream.to(compute).sum((0, 1)), rtol=0, atol=0)
                    self.assertEqual(grads[0].dtype, compute)

    def test_source_forward_and_full_precision_matrix_vjps(self):
        """VJPs use the quantized operands actually consumed by the original forward."""
        generator = torch.Generator().manual_seed(20261011)
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            for biased in (False, True):
                with self.subTest(dtype=dtype, biased=biased):
                    compute = torch.float32 if dtype == torch.bfloat16 else dtype
                    latent = torch.randn(2, 3, 4, dtype=compute, generator=generator).requires_grad_()
                    weight = torch.randn(5, 4, generator=generator).to(dtype).requires_grad_()
                    bias = torch.randn(5, generator=generator).to(dtype).requires_grad_() if biased else None
                    upstream = torch.randn(2, 3, 5, generator=generator).to(dtype)
                    with torch.autocast('cpu', enabled=False):
                        expected_output = functional.linear(  # pylint: disable=not-callable
                            latent.to(dtype), weight, bias)
                    output = project_query_with_fp32_vjp(latent, weight, bias)
                    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
                    parameters = (latent, weight) if bias is None else (latent, weight, bias)
                    gradients = torch.autograd.grad(output, parameters, grad_outputs=upstream)
                    grad_input = (upstream.to(compute).reshape(-1, 5)@weight.detach().to(compute)).reshape(latent.shape)
                    grad_weight = (upstream.to(compute).reshape(-1, 5).transpose(0, 1)
                                   @latent.detach().to(dtype).to(compute).reshape(-1, 4)).to(dtype)
                    torch.testing.assert_close(gradients[0], grad_input, rtol=0, atol=0)
                    torch.testing.assert_close(gradients[1], grad_weight, rtol=0, atol=0)
                    if bias is not None:
                        expected_bias = upstream.to(compute).sum((0, 1)).to(dtype)
                        torch.testing.assert_close(gradients[2], expected_bias, rtol=0, atol=0)

    def test_empty_rows_and_frozen_weight(self):
        """Empty owners produce zero leaf VJPs and frozen weights still propagate input VJPs."""
        for rows in (0, 2):
            with self.subTest(rows=rows):
                latent = torch.ones(1, rows, 4, requires_grad=True)
                weight = torch.ones(5, 4, dtype=torch.bfloat16)
                bias = torch.zeros(5, dtype=torch.bfloat16, requires_grad=True)
                output = project_query_with_fp32_vjp(latent, weight, bias)
                gradients = torch.autograd.grad(output.sum(), (latent, bias))
                torch.testing.assert_close(gradients[0], torch.full_like(latent, 5.), rtol=0, atol=0)
                torch.testing.assert_close(gradients[1], torch.full_like(bias, rows), rtol=0, atol=0)
                self.assertIsNone(weight.grad)

    def test_invalid_operands_and_original_weight_mutation(self):
        """Reject mismatched precision/shapes and retain autograd's original-weight version guard."""
        weight = torch.ones(5, 4, dtype=torch.bfloat16, requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'full-precision latent'):
            project_query_with_fp32_vjp(torch.ones(1, 4, dtype=torch.bfloat16), weight)
        with self.assertRaisesRegex(ValueError, 'full-precision latent'):
            project_query_with_fp32_vjp(torch.ones(1, 3), weight)
        with self.assertRaisesRegex(ValueError, 'bias must match'):
            project_query_with_fp32_vjp(torch.ones(1, 4), weight, torch.ones(5))
        latent = torch.ones(1, 4, requires_grad=True)
        output = project_query_with_fp32_vjp(latent, weight)
        with torch.no_grad():
            weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
            output.sum().backward()

    def test_second_order_backward_rejected(self):
        """Match the existing native and TP projection first-order contract."""
        latent = torch.ones(1, 4, requires_grad=True)
        weight = torch.ones(5, 4, dtype=torch.bfloat16, requires_grad=True)
        output = project_query_with_fp32_vjp(latent, weight)
        with self.assertRaisesRegex(ValueError, 'first-order backward only'):
            torch.autograd.grad(output.sum(), latent, create_graph=True)
