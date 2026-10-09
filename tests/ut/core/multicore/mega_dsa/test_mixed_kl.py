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
"""CPU ownership/lifetime checks for selected-KL retained phase submission."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import mixed_kl
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaStats
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


class TestMixedKl(unittest.TestCase):
    """Keep raw phase ownership checks separate from device numerical acceptance."""

    def setUp(self) -> None:
        """Prepare CPU buffers and mock the native submission boundary."""
        self.phases = []
        self.storage = []
        self.mocks = [patch.object(mixed_kl, "_load_native"),
                      patch.object(mixed_kl, "mixed_kl_workspace_bytes", return_value=128),
                      patch.object(torch.ops.hyper_parallel, "dsa_mixed_kl_version", return_value=1, create=True),
                      patch.object(torch.ops.hyper_parallel, "dsa_mixed_kl_out", side_effect=self._native, create=True)]
        for mock in self.mocks:
            mock.start()
            self.addCleanup(mock.stop)

    def _native(self, *args):
        retained, phase = args[14], args[16]
        trace = args[13]
        self.assertEqual(trace.storage_offset(), 0)
        self.assertEqual(trace.untyped_storage().nbytes(), trace.numel() * trace.element_size())
        for tensor in (*args[:5], args[8], args[9]):
            self.assertFalse(tensor.requires_grad)
        if phase == 0:
            retained[0] = 13
            self.storage.append(retained)
        elif phase == 1:
            self.assertEqual(retained[0].item(), 13)
            retained[0] = 29
        else:
            self.assertEqual(retained[0].item(), 29)
            for output, multiplier in zip(args[17:], (2, 3, 5, 7)):
                output.fill_(args[0][0, 0, 0].item() * multiplier)
        self.phases.append(phase)

    @staticmethod
    def _run(signature):
        shapes = ((2, 32, 512), (2, 512), (2, 32, 64), (2, 64), (2, 64, 128), (2, 128), (2, 64))
        inputs = tuple(torch.full(shape, signature, dtype=torch.bfloat16, requires_grad=True) for shape in shapes)
        indices = torch.full((2, 1, 2048), -1, dtype=torch.int32)
        lengths = (2,)
        stats = CannDsaStats(torch.zeros((1, 2, 32)), torch.ones((1, 2, 32)))
        config = MixedSfaSchedule(7).runtime_config("cpu")
        return mixed_kl._submit_mixed_kl(inputs[:4], inputs[4:], indices, lengths, stats, config, 192**-.5)

    def test_raw_derivatives_and_loss_remain_owned_across_invocations(self):
        """Later submissions cannot overwrite retained derivatives or scale the raw objective."""
        first, second = self._run(1), self._run(3)
        self.assertEqual(self.phases, [0, 1, 2, 0, 1, 2])
        self.assertIsNot(self.storage[0], self.storage[1])
        for result, signature in ((first, 1), (second, 3)):
            for gradient, multiplier in zip(result.gradients, (2, 3, 5)):
                torch.testing.assert_close(gradient, torch.full_like(gradient, signature * multiplier))
            self.assertEqual(result.loss.item(), signature * 7)
            self.assertEqual(len(result.traces), 3)
            self.assertEqual(len({trace.untyped_storage().data_ptr() for trace in result.traces}), 3)
            self.assertFalse(result.loss.requires_grad)

    def test_invalid_abi_and_deterministic_mode_reject_before_submission(self):
        """Unsupported execution modes must not enter partially initialized phases."""
        with patch.object(torch.ops.hyper_parallel, "dsa_mixed_kl_version", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "ABI mismatch"):
                self._run(1)
        with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
            with self.assertRaisesRegex(ValueError, "non-deterministic"):
                self._run(1)
        self.assertEqual(self.phases, [])
