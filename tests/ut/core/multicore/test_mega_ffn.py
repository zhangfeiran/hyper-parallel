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
"""Dense FFN parameter ownership, gradients and training lifecycle on CPU."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.core.multicore.modules.mega_ffn.module import MegaFFN
from tests.common.mark_utils import arg_mark


def _reference(x, gate, up, down):
    gate_projection = x @ gate.t()
    up_projection = x @ up.t()
    activated = (functional.silu(gate_projection.float()) * up_projection.float()).to(x.dtype)
    return activated @ down.t()


class TestMegaFFN(unittest.TestCase):
    """Check the standard training surface with independent three-Linear references."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_forward_and_all_gradients_match_three_linear_reference(self):
        """Feature: Dense FFN training.
        Description: Compare batched forward and input/weight gradients to separate projections.
        Expectation: BF16 outputs and every gradient agree within the declared component tolerance.
        """
        torch.manual_seed(7)
        module = MegaFFN(8, 6)
        x = torch.randn(2, 7, 8, dtype=torch.bfloat16, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        gate = module.gate_up[:, :6].t().detach().clone().requires_grad_()
        up = module.gate_up[:, 6:].t().detach().clone().requires_grad_()
        down = module.down.t().detach().clone().requires_grad_()
        actual = module(x)
        expected = _reference(reference_x, gate, up, down)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
        cotangent = torch.randn_like(actual)
        actual.backward(cotangent)
        expected.backward(cotangent)
        for actual_grad, expected_grad in (
            (x.grad, reference_x.grad),
            (module.gate_up.grad, torch.cat((gate.grad.t(), up.grad.t()), dim=1)),
            (module.down.grad, down.grad.t()),
        ):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0.02, atol=0.002)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_borrowed_weights_remain_invocation_owned(self):
        """Feature: Caller-owned FFN parameters.
        Description: Use two distinct weight pairs before either backward.
        Expectation: No parameters are duplicated and both invocations receive their own gradients.
        """
        module = MegaFFN(8, 6, create_parameters=False)
        self.assertEqual(list(module.parameters()), [])
        x = torch.randn(7, 8, dtype=torch.bfloat16, requires_grad=True)
        pairs = [tuple(t.requires_grad_() for t in (torch.randn(8, 12, dtype=x.dtype),
                                                   torch.randn(6, 8, dtype=x.dtype))) for _ in range(2)]
        first, second = (module(x, weights=weights) for weights in pairs)
        second.float().square().mean().backward(retain_graph=True)
        self.assertIsNone(pairs[0][0].grad)
        first.float().square().mean().backward()
        for weights in pairs:
            for weight in weights:
                self.assertIsNotNone(weight.grad)
        self.assertTrue(all(not hasattr(executable, "weights") for executable in module._executables.values()))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_close_preserves_pending_backward_and_rejects_new_forward(self):
        """Feature: FFN resource lifecycle.
        Description: Close the module after forward while its graph is still live.
        Expectation: Backward completes and a later forward fails explicitly.
        """
        module = MegaFFN(8, 6)
        x = torch.randn(7, 8, dtype=torch.bfloat16, requires_grad=True)
        output = module(x)
        module.close()
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(module.gate_up.grad)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            module(x)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    @patch.object(DefaultDeviceType, "_default_device_type", "cpu")
    def test_checkpoint_recompute_and_variable_tokens(self):
        """Feature: Activation checkpoint integration.
        Description: Recompute two token shapes through standard Torch checkpoint.
        Expectation: All parameters receive finite gradients and cached plans remain shape specific.
        """
        module = MegaFFN(8, 6)
        for rows in (1, 7, 0):
            with self.subTest(rows=rows):
                module.zero_grad(set_to_none=True)
                x = torch.randn(rows, 8, dtype=torch.bfloat16, requires_grad=True)
                checkpoint(module, x, use_reentrant=False).sum().backward()
                for parameter in module.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(len(module._executables), 3)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_independent_optimizer_trajectories(self):
        """Feature: Independent dense FFN training trajectories.
        Description: Evolve packed and separate-projection parameters with independent AdamW histories.
        Expectation: Loss and parameter trajectories stay aligned without any cross-run resynchronization.
        """
        torch.manual_seed(19)
        module = MegaFFN(8, 6)
        reference = [torch.nn.Parameter(module.gate_up[:, :6].t().detach().clone()),
                     torch.nn.Parameter(module.gate_up[:, 6:].t().detach().clone()),
                     torch.nn.Parameter(module.down.t().detach().clone())]
        actual_optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, eps=1e-4, foreach=False)
        reference_optimizer = torch.optim.AdamW(reference, lr=1e-3, eps=1e-4, foreach=False)
        x = torch.randn(7, 8, dtype=torch.bfloat16)
        target = torch.randn(7, 8)
        for step in range(8):
            with self.subTest(step=step):
                actual_optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                loss = (module(x).float() - target).square().mean()
                reference_loss = (_reference(x, *reference).float() - target).square().mean()
                torch.testing.assert_close(loss, reference_loss, rtol=0.02, atol=0.002)
                loss.backward()
                reference_loss.backward()
                actual_optimizer.step()
                reference_optimizer.step()
                torch.testing.assert_close(module.gate_up, torch.cat((reference[0].t(), reference[1].t()), dim=1),
                                           rtol=0.02, atol=0.002)
                torch.testing.assert_close(module.down, reference[2].t(), rtol=0.02, atol=0.002)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_state_dict_roundtrip_and_parameter_placement(self):
        """Feature: FFN checkpoint and placement.
        Description: Reload state into a fresh module and move parameters after plan creation.
        Expectation: Outputs are restored and executable caches are invalidated by placement.
        """
        module = MegaFFN(8, 6)
        x = torch.randn(7, 8, dtype=torch.bfloat16)
        expected = module(x)
        restored = MegaFFN(8, 6)
        restored.load_state_dict(module.state_dict())
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
        self.assertTrue(restored._executables)
        restored.to("cpu")
        self.assertFalse(restored._executables)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_invalid_ownership_dtype_and_shapes_are_rejected(self):
        """Feature: Dense FFN input contract.
        Description: Pass a missing borrowed pair, an extra owned pair and invalid token types/shapes.
        Expectation: Each invalid invocation fails before primitive execution.
        """
        owned, borrowed = MegaFFN(8, 6), MegaFFN(8, 6, create_parameters=False)
        x = torch.randn(7, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "weight pair"):
            borrowed(x)
        with self.assertRaisesRegex(ValueError, "create_parameters"):
            owned(x, weights=(owned.gate_up, owned.down))
        for value in (x.float(), x[:, :7], x.t()):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaisesRegex(ValueError, "contiguous BF16"):
                    owned(value)
