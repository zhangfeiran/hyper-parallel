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
"""Shared key/value weight-view VJPs and source batched product identity on CPU."""

import unittest

import torch
from torch import nn

from hyper_parallel.core.multicore.modules.mega_dsa.parameter_vjp import (
    ParameterVjpCapture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.source_bmm import (
    source_bmm_with_full_precision_weight_vjp,
)


class TestSourceBmm(unittest.TestCase):
    """Check noncontiguous shared slices, empty owners and original weight version guards."""

    def test_shared_key_value_views_meet_before_the_original_leaf_cast(self):
        """Both MLA up-projection paths must reach one full-precision captured Parameter gradient."""
        generator = torch.Generator().manual_seed(20261251)
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            for rows in (0, 3):
                with self.subTest(dtype=dtype, rows=rows):
                    compute = torch.float64 if dtype == torch.float64 else torch.float32
                    parameter = nn.Parameter(torch.randn(2*5, 4, generator=generator).to(dtype))
                    capture = ParameterVjpCapture({'weight': parameter})
                    view = capture.view('weight', dtype=compute).view(2, 5, 4)
                    query = torch.randn(2, rows, 3, generator=generator).to(dtype).requires_grad_()
                    compressed = torch.randn(2, rows, 4, generator=generator).to(dtype).requires_grad_()
                    key = source_bmm_with_full_precision_weight_vjp(query, view[:, :3], parameter)
                    value = source_bmm_with_full_precision_weight_vjp(
                        compressed, view[:, 3:].transpose(1, 2), parameter)
                    self.assertTrue(torch.equal(key, torch.bmm(query, parameter.view(2, 5, 4)[:, :3])))
                    expected_value = torch.bmm(compressed, parameter.view(2, 5, 4)[:, 3:].transpose(1, 2))
                    self.assertTrue(torch.equal(value, expected_value))
                    gkey = torch.randn(key.shape, generator=generator).to(dtype)
                    gvalue = torch.randn(value.shape, generator=generator).to(dtype)
                    capture.begin_backward()
                    actual = torch.autograd.grad((key, value), (query, compressed, parameter), (gkey, gvalue))
                    capture.finish_backward()
                    expected = torch.zeros_like(view)
                    expected[:, :3] = torch.bmm(query.detach().to(compute).transpose(1, 2), gkey.to(compute))
                    expected[:, 3:] = torch.bmm(gvalue.to(compute).transpose(1, 2), compressed.detach().to(compute))
                    observed = capture.take()['weight']
                    self.assertTrue(torch.equal(observed, expected.reshape(parameter.shape)))
                    self.assertTrue(torch.equal(actual[2], observed.to(dtype)))
                    expected_query = torch.bmm(gkey.to(compute), view[:, :3].transpose(1, 2)).to(dtype)
                    self.assertTrue(torch.equal(actual[0], expected_query))
                    self.assertTrue(torch.equal(actual[1], torch.bmm(gvalue.to(compute), view[:, 3:]).to(dtype)))

    def test_versions_invalid_geometry_and_second_order_are_rejected(self):
        """The original 2D shared Parameter remains guarded even when BMM consumes transposed slices."""
        parameter = nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))
        inputs = torch.ones(1, 2, 2, dtype=torch.bfloat16, requires_grad=True)
        capture = ParameterVjpCapture({'weight': parameter})
        view = capture.view('weight', dtype=torch.float32)[None]
        with self.assertRaisesRegex(ValueError, 'dense matching'):
            source_bmm_with_full_precision_weight_vjp(inputs.float(), view, parameter)
        output = source_bmm_with_full_precision_weight_vjp(inputs, view, parameter)
        capture.begin_backward()
        with torch.no_grad():
            parameter.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
            output.sum().backward()
        capture.discard()
        output = source_bmm_with_full_precision_weight_vjp(
            inputs, capture.view('weight', dtype=torch.float32)[None], parameter)
        capture.begin_backward()
        with self.assertRaisesRegex(ValueError, 'first-order'):
            torch.autograd.grad(output.sum(), inputs, create_graph=True)
        capture.discard()


    def test_compute_input_retains_source_primal_and_unrounded_input_vjp(self):
        """Source absorption consumes rounded operands but passes its matrix derivative to an FP32 query."""
        weight = nn.Parameter(torch.tensor([[[1.5], [-0.75]]], dtype=torch.bfloat16))
        inputs = torch.tensor([[[0.3, 1.2]]], requires_grad=True)
        view = weight.float()
        output = source_bmm_with_full_precision_weight_vjp(inputs, view, weight, full_precision_inputs=True)
        torch.testing.assert_close(output, torch.bmm(inputs.bfloat16(), weight), rtol=0, atol=0)
        upstream = torch.tensor([[[0.123]]], dtype=torch.bfloat16)
        grad_inputs, grad_view = torch.autograd.grad(output, (inputs, view), upstream)
        torch.testing.assert_close(grad_inputs, torch.bmm(upstream.float(), view.detach().transpose(1, 2)),
                                   rtol=0, atol=0)
        torch.testing.assert_close(grad_view, torch.bmm(inputs.detach().bfloat16().float().transpose(1, 2),
                                                      upstream.float()), rtol=0, atol=0)


    def test_compute_output_preserves_source_primal_and_nonrepresentable_upstream(self):
        """An FP32 dQ reaches absorption before activation rounding and yields FP32 matrix derivatives."""
        original=nn.Parameter(torch.tensor([[[1.5],[-.75]]],dtype=torch.bfloat16))
        inputs=torch.tensor([[[.3,1.2]]],requires_grad=True)
        weight=original.float()
        output=source_bmm_with_full_precision_weight_vjp(inputs,weight,original,
                                                       full_precision_inputs=True,full_precision_output=True)
        torch.testing.assert_close(output,torch.bmm(inputs.bfloat16(),original).float(),rtol=0,atol=0)
        upstream=torch.tensor([[[.1234567]]])
        grad_inputs,grad_weight=torch.autograd.grad(output,(inputs,weight),upstream)
        torch.testing.assert_close(grad_inputs,torch.bmm(upstream,weight.transpose(1,2)),rtol=0,atol=0)
        expected=torch.bmm(inputs.detach().bfloat16().float().transpose(1,2),upstream)
        torch.testing.assert_close(grad_weight,expected,rtol=0,atol=0)
