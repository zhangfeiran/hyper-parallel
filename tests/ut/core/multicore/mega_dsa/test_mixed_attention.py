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
"""Saved-state lifetime and shared K/V autograd ownership over mocked native phases."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import mixed_attention
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


class TestMixedAttention(unittest.TestCase):
    """Keep hardware math gates separate from Python ownership/ordering checks."""

    def setUp(self) -> None:
        """Prepare independent CPU states and mocked native launch boundaries."""
        self.calls = []
        self.forward_buffers = []
        self.schedule = MixedSfaSchedule(7)
        self.config = self.schedule.runtime_config("cpu")
        self.indices = torch.full((2, 1, 2048), -1, dtype=torch.int32)
        self.indices[0, 0, 0] = 0
        self.indices[1, 0, :2] = torch.tensor([0, 1], dtype=torch.int32)
        self.lengths = torch.tensor([2], dtype=torch.int32)
        self.mocks = [patch.object(mixed_attention, "_ensure_native"),
                      patch.object(mixed_attention, "mixed_sfa_grad_workspace_bytes", return_value=64),
                      patch.object(torch.ops.hyper_parallel, "dsa_mixed_tile_out",
                                   side_effect=self._forward, create=True),
                      patch.object(torch.ops.hyper_parallel, "dsa_mixed_grad_out",
                                   side_effect=self._backward, create=True)]
        for mock in self.mocks:
            mock.start()
            self.addCleanup(mock.stop)

    def _forward(self, *args):
        query = args[0]
        output, maximum, denominator = args[-3], args[-2], args[-1]
        output.fill_(query[0, 0, 0].item())
        maximum.zero_()
        denominator.fill_(1)
        self.forward_buffers.append((output, maximum, denominator))

    def _backward(self, *args):
        retained, phase = args[14], args[16]
        if phase == 0:
            retained[0] = 13
        elif phase == 1:
            self.assertEqual(retained[0].item(), 13)
            retained[0] = 29
        else:
            self.assertEqual(retained[0].item(), 29)
            scale = args[0][0, 0, 0].item() * args[4].float().mean().item()
            for gradient, coefficient in zip(args[17:], (2, 3, 5, 7, 11)):
                gradient.fill_(scale * coefficient)
        self.calls.append((phase, retained, args[5]))

    def _apply(self, signature):
        shapes = ((2, 32, 512), (2, 512), (2, 32, 64), (2, 64))
        inputs = tuple(torch.full(shape, signature, dtype=torch.bfloat16, requires_grad=True) for shape in shapes)
        output = mixed_attention._MixedAttention.apply(
            *inputs, self.indices, self.lengths, self.config, 192**-0.5, self.schedule)
        return output, inputs

    def test_delayed_reverse_backward_and_unique_saved_outputs(self):
        """Two outstanding forwards retain independent stats and sum dK+dV exactly once."""
        first, first_inputs = self._apply(1)
        second, second_inputs = self._apply(2)
        second_gradients = torch.autograd.grad((second * 5).sum(), second_inputs)
        first_gradients = torch.autograd.grad((first * 3).sum(), first_inputs)
        for actual, coefficient in zip(first_gradients, (2, 8, 7, 11)):
            self.assertTrue(torch.equal(actual, torch.full_like(actual, 3 * coefficient)))
        for actual, coefficient in zip(second_gradients, (2, 8, 7, 11)):
            self.assertTrue(torch.equal(actual, torch.full_like(actual, 10 * coefficient)))
        self.assertEqual([call[0] for call in self.calls], [0, 1, 2, 0, 1, 2])
        self.assertIs(self.calls[0][1], self.calls[1][1])
        self.assertIsNot(self.calls[0][1], self.calls[3][1])
        for first_buffer, second_buffer in zip(*self.forward_buffers):
            self.assertNotEqual(first_buffer.data_ptr(), second_buffer.data_ptr())

    def test_retain_graph_acquires_fresh_backward_scratch(self):
        """A second backward starts initialization with a fresh independent accumulator."""
        output, inputs = self._apply(1)
        first = torch.autograd.grad(output.sum(), inputs, retain_graph=True)
        second = torch.autograd.grad(output.sum(), inputs)
        for actual, expected in zip(first, second):
            self.assertTrue(torch.equal(actual, expected))
        self.assertIsNot(self.calls[0][1], self.calls[3][1])

    def test_saved_input_version_and_higher_derivative_rejected(self):
        """Fail before native work when saved inputs mutate or higher derivatives are requested."""
        output, inputs = self._apply(1)
        with torch.no_grad():
            inputs[0].add_(1)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace"):
            torch.autograd.grad(output.sum(), inputs)
        self.assertFalse(self.calls)
        output, inputs = self._apply(1)
        with self.assertRaisesRegex(ValueError, "first-order"):
            torch.autograd.grad(output.sum(), inputs, create_graph=True)
        self.assertFalse(self.calls)

    def test_failed_main_never_runs_post(self):
        """Propagate phase failures instead of casting incomplete accumulators."""
        output, inputs = self._apply(1)
        with patch.object(torch.ops.hyper_parallel, "dsa_mixed_grad_out",
                          side_effect=[None, RuntimeError("main failed")]) as launch:
            with self.assertRaisesRegex(RuntimeError, "main failed"):
                torch.autograd.grad(output.sum(), inputs)
            self.assertEqual(launch.call_count, 2)


class TestMixedGradBudget(unittest.TestCase):
    """Capacity arithmetic includes all shared gradient planes and the larger scatter ring."""

    def test_independent_known_capacities_and_admission(self):
        """Manual pinned-layout capacities and noninteger geometry rejection."""
        # Restore the real calculator; these tests do not allocate the large arenas.
        self.assertEqual(mixed_attention.mixed_sfa_grad_workspace_bytes(1, 32), 642789888)
        self.assertEqual(mixed_attention.mixed_sfa_grad_workspace_bytes(20, 64), 667046912)
        for tokens, heads in ((0, 32), (True, 32), (1.0, 32), (1, 16), (1, False)):
            with self.subTest(tokens=tokens, heads=heads), self.assertRaises(ValueError):
                mixed_attention.mixed_sfa_grad_workspace_bytes(tokens, heads)
