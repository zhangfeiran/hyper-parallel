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
"""TP head reductions against an independent unsharded KL and gradient oracle."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.tp_collectives import (
    DsaTpHeadCollectives,
)


class TestTpCollectives(unittest.TestCase):
    """Mock transport while preserving real score/teacher computation and autograd."""

    def setUp(self) -> None:
        """Prepare two ordered TP members with a synchronous mock transport."""
        self.transport = patch("hyper_parallel.core.multicore.modules.mega_dsa.tp_collectives.dist")
        self.dist = self.transport.start()
        self.addCleanup(self.transport.stop)
        self.dist.is_initialized.return_value = True
        self.dist.get_process_group_ranks.return_value = (1, 3)
        self.dist.get_rank.return_value = 0
        self.dist.all_gather_object.side_effect = lambda outputs, value, **_kwargs: outputs.__setitem__(
            slice(None), [value, value])
        self.group = object()
        self.boundary = DsaTpHeadCollectives(self.group, row_signature=((0, 2), (0, 1, 2), 7))

    def test_global_winner_absent_from_local_topk_union(self):
        """A candidate second on both head shards wins only after score SUM."""
        query = torch.ones((1, 1, 1), requires_grad=True)
        keys = torch.tensor([[10.0], [0.5], [6.0]], requires_grad=True)
        weights = torch.ones((1, 1), requires_grad=True)
        remote = torch.tensor([[0.5, 10.0, 6.0]])
        self.dist.all_reduce.side_effect = lambda tensor, **_kwargs: tensor.add_(remote)
        scores = self.boundary.index_scores(query, keys, weights)
        self.assertEqual(scores.argmax(-1).item(), 2)
        self.assertNotIn(2, (keys.argmax().item(), remote.argmax().item()))
        scores.sum().backward()
        torch.testing.assert_close(keys.grad, torch.ones_like(keys))
        torch.testing.assert_close(weights.grad, torch.tensor([[16.5]]))
        self.assertEqual(self.dist.all_reduce.call_count, 1)
        self.assertIs(self.dist.all_reduce.call_args.kwargs["group"], self.group)

    def test_teacher_softmax_precedes_head_sum_and_is_detached(self):
        """Unequal head counts and empty/masked rows agree with concatenated heads."""
        local = torch.tensor([[[8.0, 1.0, -2.0]], [[1.0, 2.0, 3.0]]], requires_grad=True)
        remote = torch.tensor([[[0.0, 4.0, 7.0], [2.0, 3.0, 0.0]], [[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]])
        valid = torch.tensor([[True, True, False], [False, False, False]])
        remote_probability = remote[0, :, :2].softmax(-1).sum(0)
        remote_sum = torch.tensor([[remote_probability[0], remote_probability[1], 0.0], [0.0, 0.0, 0.0]])
        self.dist.all_reduce.side_effect = lambda tensor, **_kwargs: tensor.add_(remote_sum)
        teacher = self.boundary.teacher(local, valid)
        expected = torch.cat((local.detach(), remote), 1)[0, :, :2].softmax(-1).mean(0)
        torch.testing.assert_close(teacher[0, :2], expected)
        self.assertFalse(teacher.requires_grad)
        torch.testing.assert_close(teacher[1], torch.zeros(3))
        wrong = torch.cat((local.detach(), remote), 1)[0, :, :2].sum(0).softmax(-1)
        self.assertFalse(torch.allclose(expected, wrong))

    def test_kl_gradients_match_unsharded_objective_without_tp_division(self):
        """Signed merge weights, CP divisor and upstream scale reach every local head once."""
        generator = torch.Generator().manual_seed(108)
        full = tuple(torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_()
                     for shape in ((2, 4, 3), (3, 3), (2, 4)))
        query, keys, weights = full
        teacher = torch.tensor([[0.2, 0.8, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
        valid = torch.tensor([[True, True, False], [False, False, False]])
        normalization = DsaLossNormalization(7, 2)
        all_scores = (torch.einsum("thd,kd->thk", query, keys).relu() * weights[..., None]).sum(1)
        safe = all_scores[0, :2].log_softmax(-1)
        expected = (teacher[0, :2] * (teacher[0, :2].log() - safe)).sum() * (0.37 * 2 / 7 * -2.5)
        wanted = torch.autograd.grad(expected, full)
        shared_partials = []
        for head_slice in (slice(0, 2), slice(2, 4)):
            with self.subTest(head_slice=head_slice):
                local = tuple(tensor.detach().clone().requires_grad_() for tensor in
                              (query[:, head_slice], keys, weights[:, head_slice]))
                remote = all_scores.detach() - (
                    torch.einsum("thd,kd->thk", local[0].detach(), local[1].detach()).relu()
                    * local[2].detach()[..., None]).sum(1)
                self.dist.all_reduce.side_effect = lambda tensor, peer=remote, **_kwargs: tensor.add_(peer)
                scores = self.boundary.index_scores(*local)
                loss = self.boundary.selected_kl(scores, teacher, valid, normalization=normalization, loss_coeff=0.37)
                gradients = torch.autograd.grad(loss * -2.5, local)
                torch.testing.assert_close(gradients[0], wanted[0][:, head_slice])
                torch.testing.assert_close(gradients[2], wanted[2][:, head_slice])
                shared_partials.append(gradients[1])
        self.dist.all_reduce.side_effect = lambda tensor, **_kwargs: tensor.add_(shared_partials[1].flatten())
        returned = self.boundary.sum_shared_gradients((None, shared_partials[0]))
        self.assertIsNone(returned[0])
        torch.testing.assert_close(returned[1], wanted[1])

    def test_shared_gradients_sum_before_cast_and_preserve_inputs(self):
        """FP32 contributions that vanish in BF16 survive the collective SUM."""
        partial = torch.tensor([1.0, 0.002], dtype=torch.float32)
        peer = torch.tensor([0.002, 1.0], dtype=torch.float32)
        saved = partial.clone()
        self.dist.all_reduce.side_effect = lambda tensor, **_kwargs: tensor.add_(peer)
        actual = self.boundary.sum_shared_gradients((partial, None, torch.empty(0)))
        torch.testing.assert_close(actual[0], saved + peer)
        torch.testing.assert_close(partial, saved)
        self.assertIsNone(actual[1])
        self.assertEqual(actual[2].numel(), 0)
        self.assertEqual(self.dist.all_reduce.call_count, 1)

    def test_reject_implicit_group_stale_rows_and_active_detached_sum(self):
        """Admission fails before mixing unrelated candidate rows or discarding gradients."""
        with self.assertRaisesRegex(ValueError, "explicit"):
            DsaTpHeadCollectives(None, row_signature=())
        self.dist.all_gather_object.side_effect = lambda outputs, value, **_kwargs: outputs.__setitem__(
            slice(None), [value, ("other invocation",)])
        with self.assertRaisesRegex(ValueError, "disagree"):
            DsaTpHeadCollectives(self.group, row_signature=(7,))
        with self.assertRaisesRegex(ValueError, "autograd"):
            self.boundary.sum_detached(torch.ones(3, requires_grad=True))
        with self.assertRaisesRegex(ValueError, "FP32/FP64"):
            self.boundary.sum_shared_gradients((torch.ones(3, dtype=torch.bfloat16),))
