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
"""Source linear forward identity and pre-leaf weight VJPs without device setup."""

import unittest

import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.modules.mega_dsa.parameter_vjp import (
    ParameterVjpCapture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.source_linear import (
    source_linear_with_full_precision_weight_vjp,
)


class TestSourceLinear(unittest.TestCase):
    """Compare the captured projection with independent source forwards and full-precision matrix derivatives."""

    def test_delayed_checkpoint_accumulation_preserves_pre_leaf_residual(self):
        """Two saved forwards and both backward orders capture the FP32 sum, including checkpoint replay."""
        for replay in ('ordered', 'reverse', 'checkpoint'):
            for full_inputs in (False, True):
                with self.subTest(replay=replay, full_inputs=full_inputs):
                    weight = nn.Parameter(torch.ones(1, 1, dtype=torch.bfloat16))
                    capture = ParameterVjpCapture({'weight': weight})
                    dtype = torch.float32 if full_inputs else torch.bfloat16
                    inputs = [torch.tensor([[value]], dtype=dtype, requires_grad=True) for value in (1.5, 1.)]

                    def _project(value, collector=capture, parameter=weight, full=full_inputs):
                        return source_linear_with_full_precision_weight_vjp(
                            value, collector.view('weight', dtype=torch.float32), parameter,
                            full_precision_inputs=full)

                    outputs = [(checkpoint(_project, value, use_reentrant=False)
                                if replay == 'checkpoint' else _project(value)) for value in inputs]
                    gradients = [torch.tensor([[value]], dtype=torch.bfloat16) for value in (.66796875, -1.)]
                    order = (0, 1) if replay == 'ordered' else (1, 0)
                    capture.begin_backward()
                    for index in order:
                        outputs[index].backward(gradients[index])
                    capture.finish_backward()
                    self.assertEqual(float(capture.take()['weight']), .001953125)
                    self.assertEqual(float(weight.grad), 0.)
                    for value, gradient in zip(inputs, gradients):
                        torch.testing.assert_close(value.grad, gradient.to(dtype), rtol=0, atol=0)
                    with self.assertRaisesRegex(RuntimeError, 'not ready'):
                        capture.take()

    def test_source_forward_and_independent_weight_input_and_bias_vjps(self):
        """BF16 values stay exact while matrix derivatives reach the capture before leaf casting."""
        generator = torch.Generator().manual_seed(20261251)
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            for rows in (0, 3):
                with self.subTest(dtype=dtype, rows=rows):
                    compute = torch.float64 if dtype == torch.float64 else torch.float32
                    inputs = torch.randn(rows, 4, generator=generator).to(dtype).requires_grad_()
                    weight = nn.Parameter(torch.randn(7, 4, generator=generator).to(dtype))
                    bias = nn.Parameter(torch.randn(7, generator=generator).to(dtype))
                    capture = ParameterVjpCapture({'weight': weight, 'bias': bias})
                    output = source_linear_with_full_precision_weight_vjp(
                        inputs, capture.view('weight', dtype=compute), weight,
                        capture.view('bias', dtype=compute), bias)
                    expected = functional.linear(inputs, weight, bias)  # pylint: disable=not-callable
                    self.assertTrue(torch.equal(output, expected))
                    upstream = torch.randn(output.shape, generator=generator).to(dtype)
                    capture.begin_backward()
                    grads = torch.autograd.grad(output, (inputs, weight, bias), upstream)
                    capture.finish_backward()
                    observed = capture.take()
                    grad_weight = upstream.to(compute).T@inputs.detach().to(compute)
                    grad_bias = upstream.to(compute).sum(0)
                    grad_input = (upstream.to(compute)@weight.to(compute)).to(dtype)
                    self.assertTrue(torch.equal(observed['weight'], grad_weight))
                    self.assertTrue(torch.equal(observed['bias'], grad_bias))
                    self.assertTrue(torch.equal(grads[0], grad_input))
                    self.assertTrue(torch.equal(grads[1], grad_weight.to(dtype)))
                    self.assertTrue(torch.equal(grads[2], grad_bias.to(dtype)))

    def test_parameter_versions_invalid_views_and_second_order_are_rejected(self):
        """The source Parameter itself remains version tracked even when its compute view is copied."""
        inputs = torch.ones(2, 3, dtype=torch.bfloat16, requires_grad=True)
        weight = nn.Parameter(torch.ones(4, 3, dtype=torch.bfloat16))
        capture = ParameterVjpCapture({'weight': weight})
        view = capture.view('weight', dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, 'matching dense'):
            source_linear_with_full_precision_weight_vjp(inputs.float(), view, weight)
        output = source_linear_with_full_precision_weight_vjp(inputs, view, weight)
        with torch.no_grad():
            weight.add_(1)
        capture.begin_backward()
        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
            output.sum().backward()
        capture.discard()
        output = source_linear_with_full_precision_weight_vjp(
            inputs, capture.view('weight', dtype=torch.float32), weight)
        capture.begin_backward()
        with self.assertRaisesRegex(ValueError, 'first-order'):
            torch.autograd.grad(output.sum(), inputs, create_graph=True)
        capture.discard()


    def test_compute_output_keeps_nonrepresentable_cotangent(self):
        """Compute representation preserves the BF16 primal while avoiding a downstream cotangent cast."""
        weight = nn.Parameter(torch.tensor([[1.5, -0.75]], dtype=torch.bfloat16))
        capture = ParameterVjpCapture({'weight': weight})
        inputs = torch.tensor([[0.3, 1.2]], requires_grad=True)
        output = source_linear_with_full_precision_weight_vjp(
            inputs, capture.view('weight', dtype=torch.float32), weight,
            full_precision_inputs=True, full_precision_output=True)
        expected = functional.linear(inputs.bfloat16(), weight).float()  # pylint: disable=not-callable
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        upstream = torch.tensor([[0.1234567]])
        capture.begin_backward()
        output.backward(upstream)
        capture.finish_backward()
        torch.testing.assert_close(inputs.grad, upstream@torch.Tensor.detach(weight).float(), rtol=0, atol=0)
        torch.testing.assert_close(capture.take()['weight'], upstream.T@inputs.detach().bfloat16().float(),
                                   rtol=0, atol=0)
