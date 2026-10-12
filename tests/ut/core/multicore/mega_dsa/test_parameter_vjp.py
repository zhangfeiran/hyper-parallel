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
"""Full-precision parameter VJP capture before CP reduction and leaf casting."""

import unittest

import torch
from torch import nn
from torch.nn import functional

from hyper_parallel.core.multicore.modules.mega_dsa.parameter_vjp import (
    ParameterVjpCapture,
)


class TestParameterVjp(unittest.TestCase):
    """Prove lost CP residuals, delayed accumulation and safe explicit capture lifetime."""

    def test_cp_sum_retains_a_residual_lost_by_local_bf16_leaf_casts(self):
        """A two-owner BF16 GEMM counterexample must agree with the unsharded gradient."""
        parameter = nn.Parameter(torch.ones(1, 1, dtype=torch.bfloat16))
        inputs = torch.tensor([[1.5], [1.]], dtype=torch.bfloat16)
        upstream = torch.tensor([[.66796875], [-1.]], dtype=torch.bfloat16)
        captured, leaves = [], []
        for row in range(2):
            collector = ParameterVjpCapture({'weight': parameter})
            output = functional.linear(  # pylint: disable=not-callable
                inputs[row:row+1].float(), collector.view('weight', dtype=torch.float32))
            collector.begin_backward()
            leaf, = torch.autograd.grad(output, parameter, upstream[row:row+1].float())
            collector.finish_backward()
            leaves.append(leaf)
            captured.append(collector.take()['weight'])
        expected = upstream.T@inputs
        self.assertEqual(float(sum(value.float() for value in leaves)), 0.)
        late = sum(captured).bfloat16()
        self.assertTrue(torch.equal(late, expected))
        self.assertEqual(float(late), .001953125)
        self.assertIsNone(parameter._backward_hooks)

    def test_delayed_forward_accumulation_and_retained_cycles_do_not_reuse_old_gradients(self):
        """Each explicitly bracketed backward observes its own complete sum before leaf rounding."""
        parameter = nn.Parameter(torch.ones(1, 1, dtype=torch.bfloat16))
        collector = ParameterVjpCapture({'weight': parameter})
        views = [collector.view('weight', dtype=torch.float32) for _ in range(2)]
        outputs = [functional.linear(torch.tensor([[value]]), view)  # pylint: disable=not-callable
                   for value, view in zip((1.5, 1.), views)]
        objective = outputs[0].sum()*.66796875-outputs[1].sum()
        for repeat in range(2):
            with self.subTest(repeat=repeat):
                collector.begin_backward()
                objective.backward(retain_graph=repeat == 0)
                collector.finish_backward()
                self.assertEqual(float(collector.take()['weight']), .001953125)
        with self.assertRaisesRegex(RuntimeError, 'not ready'):
            collector.take()
        self.assertEqual(float(parameter.grad), 0.)

    def test_lifecycle_mutation_unused_parameters_and_compute_dtype_are_checked(self):
        """Reject stale weights, missing brackets and incompatible views without installing leaf hooks."""
        parameter = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        collector = ParameterVjpCapture({'weight': parameter})
        with self.assertRaisesRegex(ValueError, 'full-precision'):
            collector.view('weight', dtype=torch.bfloat16)
        view = collector.view('weight', dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, 'active backward'):
            view.sum().backward()
        collector.discard()
        view = collector.view('weight', dtype=torch.float32)
        collector.begin_backward()
        with torch.no_grad():
            parameter.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'changed after'):
            view.sum().backward()
        collector.discard()
        collector.begin_backward()
        with self.assertRaisesRegex(RuntimeError, 'unfinished'):
            collector.begin_backward()
        collector.finish_backward()
        self.assertIsNone(collector.take()['weight'])
        frozen = nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False)
        collector = ParameterVjpCapture({'weight': frozen})
        self.assertFalse(collector.view('weight', dtype=torch.float32).requires_grad)
        collector.begin_backward()
        collector.finish_backward()
        self.assertIsNone(collector.take()['weight'])
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                parameter = nn.Parameter(torch.ones(2, dtype=dtype))
                collector = ParameterVjpCapture({'weight': parameter})
                value = collector.view('weight', dtype=dtype)
                collector.begin_backward()
                value.sum().backward()
                collector.finish_backward()
                self.assertEqual(collector.take()['weight'].dtype, dtype)
                self.assertIsNone(parameter._backward_hooks)
