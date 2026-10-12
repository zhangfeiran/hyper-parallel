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
"""Precision and ownership controls without hardware or process groups."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import mixed_fp32_control as control
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


class TestMixedFp32Control(unittest.TestCase):
    """Preserve fixed saved statistics and accumulate before activation conversion."""

    def setUp(self) -> None:
        """Prepare tiny reusable planes standing in for the native scratch."""
        self.key = torch.empty((2, 576), dtype=torch.float32)
        self.value = torch.empty((2, 512), dtype=torch.float32)
        self.calls = []
        self.partials = [(1.003, -1.0)]

    @staticmethod
    def _fixture(heads: int) -> tuple:
        states = tuple(torch.zeros(shape, dtype=torch.bfloat16) for shape in
                       ((2, heads, 512), (2, 512), (2, heads, 64), (2, 64)))
        output = torch.zeros_like(states[0])
        maximum = torch.arange(2 * heads, dtype=torch.float32).view(1, 2, heads)
        forward = (output, maximum, maximum + 1)
        return states, torch.ones_like(output), forward

    def _producer(self, states, _indices, _lengths, _config, cotangent, forward,
                  _scale, _schedule, retained):
        self.assertFalse(torch.is_grad_enabled())
        self.calls.append((states, cotangent, forward, retained))
        key, value = self.partials[len(self.calls) - 1]
        self.key.fill_(key)
        self.value.fill_(value)
        gradients = tuple(torch.full_like(tensor, 4) for tensor in
                          (states[0], states[1][:, None], states[1][:, None], states[2], states[3][:, None]))
        return gradients, tuple(torch.zeros((20, 64), dtype=torch.int64) for _ in range(3)), self.key

    def _run(self, heads: int) -> tuple:
        states, cotangent, forward = self._fixture(heads)
        with patch.object(control, 'mixed_sfa_backward_probe', side_effect=self._producer), \
             patch.object(control, 'mixed_sfa_gradient_partials', side_effect=lambda *_a, **_k: (self.key, self.value)):
            result = control.mixed_sfa_fp32_backward_control(
                states, torch.zeros((2, 1, 2048), dtype=torch.int32), torch.tensor([2], dtype=torch.int32),
                cotangent, forward, attention_scale=192**-0.5, schedule=MixedSfaSchedule(1))
        return result, forward

    def test_sum_before_cast_and_owned_snapshots(self) -> None:
        """Cancellation exposes premature BF16 casting; scratch overwrite cannot alter the result."""
        result, _ = self._run(32)
        expected = (torch.tensor(1.003) - 1).bfloat16()
        self.assertTrue(torch.equal(result.gradients[1], torch.full_like(result.gradients[1], expected)))
        self.assertNotEqual(expected.item(), (torch.tensor(1.003).bfloat16() - 1).item())
        self.assertEqual(result.head_ranges, ((0, 32),))
        self.key.zero_()
        self.value.zero_()
        self.assertTrue(torch.equal(result.key_partials, torch.full_like(result.key_partials, 1.003)))
        self.assertTrue(torch.equal(result.value_partials, torch.full_like(result.value_partials, -1)))

    def test_h128_slices_original_stats_and_sums_heads_in_fp32(self) -> None:
        """Two H64 invocations reuse scratch without losing the first FP32 contribution."""
        self.partials = [(1.003, 0), (-1.0, 0)]
        result, forward = self._run(128)
        self.assertEqual(result.head_ranges, ((0, 64), (64, 128)))
        self.assertEqual(len(result.traces), 6)
        self.assertIs(self.calls[1][3], self.key)
        for call, (start, end) in zip(self.calls, result.head_ranges):
            self.assertTrue(torch.equal(call[2][1], forward[1][:, :, start:end]))
            self.assertTrue(torch.equal(call[2][2], forward[2][:, :, start:end]))
        expected = (torch.tensor(1.003) - 1).bfloat16()
        for gradient in (result.gradients[1], result.gradients[3]):
            self.assertTrue(torch.equal(gradient, torch.full_like(gradient, expected)))
        self.assertEqual(result.gradients[0].shape, (2, 128, 512))
        self.assertEqual(result.gradients[2].shape, (2, 128, 64))

    def test_admission_fails_before_native_work(self) -> None:
        """Unsupported geometry, stats, dtype or repeated phases never launch native work."""
        states, cotangent, forward = self._fixture(64)
        bad_stats = (forward[0], forward[1].transpose(1, 2), forward[2])
        with patch.object(control, 'mixed_sfa_backward_probe') as launch:
            for args, schedule in (((states, cotangent, bad_stats), MixedSfaSchedule(1)),
                                   ((states, cotangent.float(), forward), MixedSfaSchedule(1)),
                                   ((states, cotangent, forward), MixedSfaSchedule(1, rounds=2))):
                with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                    control.mixed_sfa_fp32_backward_control(
                        args[0], torch.empty(0), torch.empty(0), args[1], args[2],
                        attention_scale=1, schedule=schedule)
            launch.assert_not_called()

    def test_second_chunk_failure_propagates(self) -> None:
        """A partial H128 replay never returns a seemingly complete VJP."""
        states, cotangent, forward = self._fixture(128)

        def _fail_second(*args):
            if self.calls:
                raise RuntimeError('producer failed')
            return self._producer(*args)

        def _partial_views(*_args, **_kwargs):
            return self.key, self.value
        with patch.object(control, 'mixed_sfa_backward_probe', side_effect=_fail_second), \
             patch.object(control, 'mixed_sfa_gradient_partials', side_effect=_partial_views), \
             self.assertRaisesRegex(RuntimeError, 'producer failed'):
            control.mixed_sfa_fp32_backward_control(
                states, torch.empty(0), torch.empty(0), cotangent, forward,
                attention_scale=1, schedule=MixedSfaSchedule(1))
        self.assertEqual(len(self.calls), 1)
