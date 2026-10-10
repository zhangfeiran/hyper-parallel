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

import gc
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


class TestKlFp32KeyExport(unittest.TestCase):
    """Byte-plane admission and storage ownership are independent of Native numerical proof."""

    def test_fp32_plane_preserves_values_before_bf16_and_survives_wrapper_release(self):
        """FP32 partials retain sub-BF16 contributions and ordinary invocation-owned storage."""
        for heads in (32, 64):
            with self.subTest(heads=heads):
                required = mixed_kl.mixed_kl_workspace_bytes(3, heads)
                retained = torch.empty(required, dtype=torch.uint8)
                offset = required - 3 * 128 * 4
                retained[offset - 64:offset].fill_(0xA5)
                expected = torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128) / 1024 + 256
                retained[offset:].copy_(expected.contiguous().view(torch.uint8).flatten())
                derivatives = (torch.zeros(1, 64, 128), expected.bfloat16(), torch.zeros(1, 64))
                result = mixed_kl.MixedKlResult(derivatives, torch.zeros(()), (), retained, heads)
                partials = result.key_partials
                self.assertEqual(partials.dtype, torch.float32)
                self.assertEqual(partials.untyped_storage().data_ptr(), retained.untyped_storage().data_ptr())
                torch.testing.assert_close(partials, expected, rtol=0, atol=0)
                self.assertFalse(torch.equal(partials, derivatives[1].float()))
                self.assertTrue(bool((retained[offset - 64:offset] == 0xA5).all()))
                del result, retained
                gc.collect()
                torch.testing.assert_close(partials, expected, rtol=0, atol=0)

    def test_wrong_geometry_dtype_and_unaligned_storage_reject(self):
        """Wrong head counts and byte views cannot silently decode another scratch region."""
        required = mixed_kl.mixed_kl_workspace_bytes(2, 64)
        retained = torch.empty(required, dtype=torch.uint8)
        variants = ((retained, 2, 32), (retained, 1, 64), (retained[:-1], 2, 64),
                    (torch.empty(2, dtype=torch.float32), 2, 64), (retained.reshape(2, -1), 2, 64),
                    (torch.empty(required + 1, dtype=torch.uint8)[1:], 2, 64))
        for workspace, tokens, heads in variants:
            with self.subTest(tokens=tokens, heads=heads, shape=workspace.shape), self.assertRaises(ValueError):
                mixed_kl.mixed_kl_key_partials(workspace, tokens=tokens, heads=heads)
        with self.assertRaisesRegex(ValueError, "metadata"):
            _ = mixed_kl.MixedKlResult((), torch.zeros(()), (), retained).key_partials
