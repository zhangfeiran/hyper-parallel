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
"""Bounded mock collectives prove reduction order, input ownership and async waits."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.parameter_sum import (
    ordered_parameter_sum,
)


class TestParameterSum(unittest.TestCase):
    """Exercise CPU logic without a process group or hardware backend."""

    def setUp(self) -> None:
        """Mock only group metadata; each test controls communication completion."""
        self.group = object()
        prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.parameter_sum.dist'
        self.addCleanup(patch.stopall)
        self.initialized = patch(prefix+'.is_initialized', return_value=True).start()
        self.size = patch(prefix+'.get_world_size', return_value=4).start()
        self.gather = patch(prefix+'.all_gather_into_tensor').start()

    def _make_gather(self, peers):
        size = len(peers)
        state = SimpleNamespace(cursor=0, pointers=[], widths=[])

        def _gather(output, inputs, *, group, async_op):
            self.assertIs(group, self.group)
            self.assertTrue(async_op)
            count = inputs.numel()
            start = state.cursor
            self.assertLessEqual(count, 4)
            self.assertEqual(output.numel(), size*count)
            state.pointers.append(output.data_ptr())
            state.widths.append(count)
            output.fill_(float('nan'))

            def _complete():
                for rank, peer in enumerate(peers):
                    output.view(size, count)[rank].copy_(peer.reshape(-1)[start:start+count])
                state.cursor += count

            return SimpleNamespace(wait=_complete)

        return _gather, state

    def test_chunks_wait_before_reading_and_reuse_bounded_scratch(self):
        """Delayed mock completion exposes an omitted wait and verifies short-tail chunk ordering."""
        for dtype in (torch.float32, torch.float64):
            for size in (2, 4, 8):
                with self.subTest(dtype=dtype, size=size):
                    self.size.return_value = size
                    peers = [(torch.arange(15, dtype=dtype).reshape(3, 5)+rank).T for rank in range(size)]
                    original = peers[0].clone()
                    gather, state = self._make_gather(peers)
                    self.gather.side_effect = gather
                    observed = ordered_parameter_sum(peers[0], self.group, chunk_elements=4)
                    expected = peers[0].clone()
                    for peer in peers[1:]:
                        expected.add_(peer)
                    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                    torch.testing.assert_close(peers[0], original, rtol=0, atol=0)
                    self.assertTrue(observed.is_contiguous())
                    self.assertEqual(state.cursor, 15)
                    self.assertEqual(state.widths, [4, 4, 4, 3])
                    self.assertEqual(len(set(state.pointers)), 1)

    def test_rank_order_cancellation_is_a_sum_without_leaf_rounding(self):
        """Explicit FP32 order distinguishes this result from a differently associated sum or mean."""
        contributions = torch.tensor([2**24, 1., -(2**24), 1.])

        def _gather(output, _inputs, **_kwargs):
            return SimpleNamespace(wait=lambda: output.copy_(contributions))

        self.gather.side_effect = _gather
        value = ordered_parameter_sum(contributions[:1], self.group)
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(float(value), 1.)
        reversed_sum = contributions[-1].clone()
        for index in (2, 1, 0):
            reversed_sum.add_(contributions[index])
        self.assertEqual(float(reversed_sum), 2.)

    def test_single_member_and_empty_inputs_do_not_communicate_or_alias(self):
        """A single member still returns owned storage, while empty tensors preserve their shape."""
        for size, shape in ((1, (3, 2)), (1, (0, 2)), (8, (2, 0))):
            with self.subTest(size=size, shape=shape):
                self.size.return_value = size
                inputs = torch.ones(shape)
                output = ordered_parameter_sum(inputs, self.group)
                torch.testing.assert_close(output, inputs, rtol=0, atol=0)
                if inputs.numel():
                    self.assertNotEqual(output.data_ptr(), inputs.data_ptr())
                self.gather.assert_not_called()

    def test_positive_scalar_loss_preserves_identical_rank_order_across_replicas(self):
        """A real positive loss fixture differs by two ULPs under another FP32 association."""
        peers = [torch.tensor(value) for value in
                 (0.001309438026510179, 0.0006013101665303111,
                  0.0004632215714082122, 0.0008541629649698734)]
        expected = torch.tensor(0.003228132613003254)
        regrouped = (peers[0]+peers[3])+(peers[1]+peers[2])
        self.assertNotEqual(float(regrouped), float(expected))
        for _replica in range(2):
            gather, state = self._make_gather(peers)
            self.gather.side_effect = gather
            original = peers[0].clone()
            observed = ordered_parameter_sum(peers[0], self.group)
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            torch.testing.assert_close(peers[0], original, rtol=0, atol=0)
            self.assertEqual(observed.shape, torch.Size([]))
            self.assertEqual(state.widths, [1])
            self.assertNotEqual(observed.data_ptr(), peers[0].data_ptr())

    def test_invalid_dtype_autograd_group_and_chunk_contracts(self):
        """Reject local misuse before a collective is launched."""
        for value in (torch.ones(2, dtype=torch.bfloat16), torch.ones(2, requires_grad=True)):
            with self.assertRaisesRegex(ValueError, 'detached dense'):
                ordered_parameter_sum(value, self.group)
        for chunk in (True, 0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                ordered_parameter_sum(torch.ones(2), self.group, chunk_elements=chunk)
        with self.assertRaisesRegex(ValueError, 'explicit initialized'):
            ordered_parameter_sum(torch.ones(2), None)
        self.initialized.return_value = False
        with self.assertRaisesRegex(ValueError, 'explicit initialized'):
            ordered_parameter_sum(torch.ones(2), self.group)
        self.initialized.return_value = True
        for size in (-1, 3):
            self.size.return_value = size
            with self.assertRaisesRegex(ValueError, 'participating group'):
                ordered_parameter_sum(torch.ones(2), self.group)
        self.gather.assert_not_called()
