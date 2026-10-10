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
"""FP64 local-query LM+KL objective contracts, independent of native CP kernels."""

import unittest
from dataclasses import replace

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    selected_kl_reference,
    sparse_attention_reference,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20261011)
    shapes = ((7, 2, 3), (7, 3), (7, 2, 2), (7, 2), (7, 4), (4, 6), (4, 3), (4, 2), (6, 4))
    return tuple((torch.randn(shape, generator=generator, dtype=torch.float64) * 0.2).requires_grad_()
                 for shape in shapes)


def _loss(inputs, meta, indices, *, divisor=1, coefficient=0.3, lm=True, freeze_main=False):
    query, compressed, query_rope, key_rope, hidden, query_weight, key_weight, merge_weight, output_weight = inputs
    rows, keys = list(meta.q_global_ids), list(meta.kv_global_ids)
    main = (query[rows], compressed[keys], query_rope[rows], key_rope[keys])
    if freeze_main:
        main = tuple(tensor.detach() for tensor in main)
    # Detach before projection preserves index parameter derivatives and isolates the trunk.
    index = ((hidden.detach() @ query_weight).reshape(7, 2, 3)[rows],
             (hidden.detach() @ key_weight)[keys], (hidden.detach() @ merge_weight)[rows])
    selected = indices[rows]
    normalization = DsaLossNormalization(7, divisor)
    kl = selected_kl_reference(*index, *main, selected, meta, attention_scale=0.37,
                               normalization=normalization, loss_coeff=coefficient)
    loss = kl * -2.5
    if lm:
        output, _ = sparse_attention_reference(*main, selected, meta, attention_scale=0.37)
        loss = loss + (output.flatten(1) @ output_weight).square().sum() * (1.7 * normalization.local_sum_scale)
    return loss


def _partition_meta(partition, rank):
    owners = [0] * 7
    offsets = [0] * 7
    for owner, ids in enumerate(partition):
        for offset, token in enumerate(ids):
            owners[token], offsets[token] = owner, offset
    return DsaBatchMeta((0, 3, 7), partition[rank], (5, 0, 6, 2, 4, 1, 3), tuple(owners), tuple(offsets),
                        cp_ranks=tuple(range(len(partition))), root_pes=tuple(range(len(partition))), cp_rank=rank)


class TestLocalTrainingContract(unittest.TestCase):
    """Uneven/empty CP query sums reproduce one global LM+KL objective and projection gradients."""

    def setUp(self) -> None:
        """Prepare fixed selections with padding, packed boundaries and reordered slots."""
        self.inputs = _inputs()
        self.indices = torch.tensor([
            [0, -1, 1, 3], [1, 0, -1, 3], [2, -1, 0, 1],
            [-1, 3, 2, 4], [4, 3, -1, 0], [3, 5, 4, -1], [6, 3, -1, 5],
        ], dtype=torch.int32)
        self.meta = DsaBatchMeta.packed((3, 4))

    def test_local_lm_kl_matches_global_loss_and_every_projection_gradient(self):
        """Reducer compensation occurs once across CP1/2/4, including an empty rank."""
        partitions = (((6, 4, 0, 2, 1, 5, 3),), ((4, 0), (6, 1, 5, 3, 2)),
                      ((4, 0), (6,), (), (2, 5, 1, 3)))
        modes = ((True, False, 0.3), (True, False, 0.0), (False, False, 0.3), (True, True, 0.3))
        for partition in partitions:
            for divisor in sorted({1, len(partition)}):
                for lm, freeze_main, coefficient in modes:
                    with self.subTest(cp=len(partition), divisor=divisor, mode=(lm, freeze_main, coefficient)):
                        expected = _loss(self.inputs, self.meta, self.indices, lm=lm,
                                         freeze_main=freeze_main, coefficient=coefficient)
                        expected_grad = torch.autograd.grad(expected, self.inputs, allow_unused=True)
                        local = [_loss(self.inputs, _partition_meta(partition, rank), self.indices,
                                       divisor=divisor, lm=lm, freeze_main=freeze_main, coefficient=coefficient)
                                 for rank in range(len(partition))]
                        reduced = sum(local) / divisor
                        torch.testing.assert_close(reduced, expected, rtol=1e-11, atol=1e-12)
                        actual_grad = torch.autograd.grad(reduced, self.inputs, allow_unused=True)
                        for actual, wanted in zip(actual_grad, expected_grad):
                            if wanted is None:
                                self.assertIsNone(actual)
                            else:
                                torch.testing.assert_close(actual, wanted, rtol=1e-10, atol=1e-12)
                        self.assertIsNone(actual_grad[4])
                        if coefficient != 0:
                            for gradient in actual_grad[5:8]:
                                self.assertGreater(gradient.norm(), 0)

    def test_query_tiles_accumulate_without_a_second_cp_division(self):
        """Multiple caller-ordered tiles use the same global denominator and aux scale."""
        partition = ((4, 0), (6, 1, 5, 3, 2))
        expected = _loss(self.inputs, self.meta, self.indices)
        expected_grad = torch.autograd.grad(expected, self.inputs, allow_unused=True)
        contributions = []
        for rank, rows in enumerate(partition):
            meta = _partition_meta(partition, rank)
            for tile in (rows[::2], (), rows[1::2]):
                contributions.append(_loss(self.inputs, replace(meta, q_global_ids=tile), self.indices, divisor=2))
        actual = sum(contributions) / 2
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        gradients = torch.autograd.grad(actual, self.inputs, allow_unused=True)
        for gradient, wanted in zip(gradients, expected_grad):
            if wanted is None:
                self.assertIsNone(gradient)
            else:
                torch.testing.assert_close(gradient, wanted, rtol=1e-10, atol=1e-12)

    def test_averaging_local_means_is_a_different_objective(self):
        """The fixture detects per-rank mean weighting and redundant CP division."""
        partition = ((4, 0), (6, 1, 5, 3, 2))
        expected = _loss(self.inputs, self.meta, self.indices)
        parts = [_loss(self.inputs, _partition_meta(partition, rank), self.indices)
                 for rank in range(2)]
        wrong_local_means = sum(loss * 7 / len(rows) for loss, rows in zip(parts, partition)) / 2
        self.assertFalse(torch.isclose(wrong_local_means, expected, rtol=1e-5, atol=1e-8))
        self.assertFalse(torch.isclose(sum(parts) / 2, expected, rtol=1e-5, atol=1e-8))
