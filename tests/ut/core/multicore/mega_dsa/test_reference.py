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
"""Independent numerical and gradient acceptance for the small DSA oracle."""

import unittest
from dataclasses import replace

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    indexer_reference,
    selected_kl_reference,
    sparse_attention_reference,
)


def _leaves(shapes: tuple, seed: int = 23) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_() for shape in shapes)


class TestSparseAttentionReference(unittest.TestCase):
    """Compare to dense attention, separate K/V gradients and gradcheck."""

    def test_dense_limit_and_all_input_gradients(self):
        """Covering all legal history matches independent dense packed attention."""
        meta = DsaBatchMeta.packed((3, 2))
        query, compressed, query_rope, key_rope = _leaves(((5, 2, 3), (5, 3), (5, 2, 2), (5, 2)))
        indices = torch.arange(5, dtype=torch.int32).expand(5, -1)
        result, stats = sparse_attention_reference(query, compressed, query_rope, key_rope, indices, meta,
                                                 attention_scale=0.37)
        dense_rows = []
        for token in range(5):
            sequence, position = meta.sequence_position(token)
            start = meta.global_cu_seqlens[sequence]
            keys = slice(start, start + position + 1)
            scores = (query[token] @ compressed[keys].T + query_rope[token] @ key_rope[keys].T) * 0.37
            dense_rows.append(scores.softmax(-1) @ compressed[keys])
        expected = torch.stack(dense_rows)
        torch.testing.assert_close(result, expected)
        self.assertTrue(torch.isfinite(stats.lse).all())
        actual_grad = torch.autograd.grad(result.square().sum(), (query, compressed, query_rope, key_rope))
        expected_grad = torch.autograd.grad(expected.square().sum(), (query, compressed, query_rope, key_rope))
        for actual, wanted in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, wanted)

    def test_compressed_gradient_contains_key_and_value_contributions(self):
        """The merged c gradient equals independently computed dK + dV."""
        meta = DsaBatchMeta.packed((3,))
        query, compressed, query_rope, key_rope = _leaves(((3, 2, 3), (3, 3), (3, 2, 2), (3, 2)))
        indices = torch.tensor([[0, -1], [0, 1], [0, 2]], dtype=torch.int32)
        result, _ = sparse_attention_reference(query, compressed, query_rope, key_rope, indices, meta,
                                              attention_scale=0.5)
        merged_grad = torch.autograd.grad(result.square().sum(), compressed)[0]
        separate_key = compressed.detach().clone().requires_grad_()
        separate_value = compressed.detach().clone().requires_grad_()
        outputs = []
        for token, row in enumerate(indices.tolist()):
            selected = [key for key in row if key >= 0]
            score = (query[token] @ separate_key[selected].T + query_rope[token] @ key_rope[selected].T) * 0.5
            outputs.append(score.softmax(-1) @ separate_value[selected])
        key_grad, value_grad = torch.autograd.grad(torch.stack(outputs).square().sum(),
                                                 (separate_key, separate_value))
        self.assertGreater(key_grad.norm(), 0)
        self.assertGreater(value_grad.norm(), 0)
        torch.testing.assert_close(merged_grad, key_grad + value_grad)

    def test_fixed_selection_gradcheck(self):
        """Gradcheck validates all continuous inputs at fixed Top-K."""
        meta = DsaBatchMeta.packed((2,))
        inputs = _leaves(((2, 1, 2), (2, 2), (2, 1, 1), (2, 1)))
        indices = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)

        def _evaluate(*values: torch.Tensor) -> torch.Tensor:
            return sparse_attention_reference(*values, indices, meta, attention_scale=0.3)[0]

        self.assertTrue(torch.autograd.gradcheck(_evaluate, inputs))

    def test_reordered_storage_interior_and_zigzag_queries(self):
        """Global causal positions survive noncontiguous query/key storage."""
        base = DsaBatchMeta.packed((3, 2))
        inputs = _leaves(((5, 2, 3), (5, 3), (5, 2, 2), (5, 2)))
        indices = torch.arange(5, dtype=torch.int32).expand(5, -1)
        full, _ = sparse_attention_reference(*inputs, indices, base, attention_scale=0.4)
        q_ids, k_ids = (4, 1), (3, 1, 4, 0, 2)
        shard = replace(base, q_global_ids=q_ids, kv_global_ids=k_ids,
                        cp_ranks=(7, 2), root_pes=(2, 0), token_owners=(1, 0, 1, 1, 0),
                        token_local_offsets=(0, 1, 1, 2, 0), layout_id="zigzag")
        local, _ = sparse_attention_reference(inputs[0][list(q_ids)], inputs[1][list(k_ids)],
                                             inputs[2][list(q_ids)], inputs[3][list(k_ids)],
                                             indices[list(q_ids)], shard, attention_scale=0.4)
        torch.testing.assert_close(local, full[list(q_ids)])

    def test_padding_and_empty_candidates_have_zero_gradient(self):
        """-1, future tokens and other sequences never contribute KV gradients."""
        meta = replace(DsaBatchMeta.packed((2, 1)), q_global_ids=(0,))
        inputs = _leaves(((1, 1, 2), (3, 2), (1, 1, 1), (3, 1)))
        indices = torch.tensor([[-1, 1, 2]], dtype=torch.int32)
        result, stats = sparse_attention_reference(*inputs, indices, meta, attention_scale=0.5)
        self.assertEqual(result.count_nonzero(), 0)
        self.assertTrue(torch.isneginf(stats.maximum).all())
        self.assertTrue(torch.isneginf(stats.lse).all())
        self.assertEqual(stats.denominator.count_nonzero(), 0)
        for gradient in torch.autograd.grad(result.sum(), inputs):
            self.assertEqual(gradient.count_nonzero(), 0)

    def test_cp_query_contributions_sum_to_unsharded_kv_gradients(self):
        """Unequal zigzag Q shards give one summed dK+dV contribution per owner token."""
        meta = DsaBatchMeta.packed((3, 2))
        inputs = _leaves(((5, 2, 3), (5, 3), (5, 2, 2), (5, 2)))
        indices = torch.arange(5, dtype=torch.int32).expand(5, -1)
        full, _ = sparse_attention_reference(*inputs, indices, meta, attention_scale=0.4)
        full_grad = torch.autograd.grad(full.square().sum(), inputs)
        losses = []
        key_order = [3, 1, 4, 0, 2]
        for rank, rows in enumerate(((4, 1), (0, 2, 3))):
            shard = replace(meta, q_global_ids=rows, kv_global_ids=tuple(key_order),
                            cp_ranks=(7, 2), root_pes=(2, 0), cp_rank=rank,
                            token_owners=(1, 0, 1, 1, 0), token_local_offsets=(0, 1, 1, 2, 0))
            output, _ = sparse_attention_reference(inputs[0][list(rows)], inputs[1][key_order],
                                                  inputs[2][list(rows)], inputs[3][key_order],
                                                  indices[list(rows)], shard, attention_scale=0.4)
            losses.append(output.square().sum())
        for actual, expected in zip(torch.autograd.grad(sum(losses), inputs), full_grad):
            torch.testing.assert_close(actual, expected)

    def test_invalid_selection_and_missing_kv_fail(self):
        """Reject unknown IDs, duplicates and missing remote causal keys."""
        meta = DsaBatchMeta.packed((2,))
        inputs = _leaves(((2, 1, 2), (2, 2), (2, 1, 1), (2, 1)))
        for indices in (torch.tensor([[0], [-2]], dtype=torch.int32),
                        torch.tensor([[0], [2]], dtype=torch.int32),
                        torch.tensor([[0, 0], [1, -1]], dtype=torch.int32),
                        torch.tensor([[0], [1]], dtype=torch.int64)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                sparse_attention_reference(*inputs, indices, meta, attention_scale=0.5)
        shard = replace(meta, kv_global_ids=(1,))
        with self.assertRaisesRegex(ValueError, "missing"):
            sparse_attention_reference(inputs[0], inputs[1][1:], inputs[2], inputs[3][1:],
                                       torch.tensor([[0], [1]], dtype=torch.int32), shard, attention_scale=0.5)

    def test_bfloat16_promotes_to_float32(self):
        """The oracle has an explicit FP32 working path for BF16."""
        inputs = tuple(tensor.detach().bfloat16() for tensor in _leaves(((1, 1, 2), (1, 2), (1, 1, 1), (1, 1))))
        result, _ = sparse_attention_reference(*inputs, torch.tensor([[0]], dtype=torch.int32),
                                              DsaBatchMeta.packed((1,)), attention_scale=0.5)
        self.assertEqual(result.dtype, torch.float32)

    def test_absorption_restoration_preserves_projection_gradients(self):
        """Absorbed MLA agrees with unabsorbed attention including W_UK/W_UV/o_proj."""
        meta = DsaBatchMeta.packed((3,))
        query, compressed, query_rope, key_rope, key_up, value_up, output_weight = _leaves(
            ((3, 2, 4), (3, 3), (3, 2, 2), (3, 2), (2, 4, 3), (2, 3, 2), (4, 5)))
        indices = torch.tensor([[0, -1], [1, 0], [0, 2]], dtype=torch.int32)
        absorbed = torch.einsum("thd,hdc->thc", query, key_up)
        output, _ = sparse_attention_reference(absorbed, compressed, query_rope, key_rope, indices, meta,
                                              attention_scale=6**-0.5)
        restored = torch.einsum("thc,hcv->thv", output, value_up).flatten(1) @ output_weight
        full_key = torch.einsum("sc,hdc->shd", compressed, key_up)
        full_value = torch.einsum("sc,hcv->shv", compressed, value_up)
        dense_rows = []
        for token, row in enumerate(indices.tolist()):
            selected = [key for key in row if key >= 0]
            scores = (torch.einsum("hd,shd->hs", query[token], full_key[selected])
                      + query_rope[token] @ key_rope[selected].T) * 6**-0.5
            dense_rows.append(torch.einsum("hs,shv->hv", scores.softmax(-1), full_value[selected]))
        expected = torch.stack(dense_rows).flatten(1) @ output_weight
        torch.testing.assert_close(restored, expected)
        leaves = (query, compressed, query_rope, key_rope, key_up, value_up, output_weight)
        actual_grad = torch.autograd.grad(restored.square().sum(), leaves)
        expected_grad = torch.autograd.grad(expected.square().sum(), leaves)
        for actual, wanted in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, wanted)


class TestIndexerAndKlReference(unittest.TestCase):
    """Validate signed weights, exact candidates, detached teacher and scaling."""

    def setUp(self) -> None:
        """Create one complete deterministic FP64 batch."""
        self.meta = DsaBatchMeta.packed((5,))
        self.main = _leaves(((5, 2, 3), (5, 3), (5, 2, 2), (5, 2)))
        self.index = _leaves(((5, 2, 3), (5, 3), (5, 2)), seed=17)
        self.indices = torch.arange(5, dtype=torch.int32).expand(5, -1)

    def _loss(self, index=None, main=None, meta=None, indices=None, divisor=1, coefficient=1.0):
        return selected_kl_reference(*(self.index if index is None else index),
                                     *(self.main if main is None else main),
                                     self.indices if indices is None else indices,
                                     self.meta if meta is None else meta, attention_scale=0.37,
                                     normalization=DsaLossNormalization(5, divisor), loss_coeff=coefficient)

    def test_negative_weights_ties_and_padding(self):
        """Negative merge weights stay signed; ties use ascending global IDs."""
        meta = replace(DsaBatchMeta.packed((3,)), kv_global_ids=(2, 0, 1))
        query = torch.ones(3, 1, 1)
        key = torch.tensor([[3.0], [1.0], [2.0]])
        negative = indexer_reference(query, key, -torch.ones(3, 1), meta, sparse_count=4)
        self.assertEqual(negative.tolist(), [[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1]])
        ties = indexer_reference(query, key, torch.zeros(3, 1), meta, sparse_count=2)
        self.assertEqual(ties[-1].tolist(), [0, 1])

    def test_global_best_candidates_can_all_belong_to_one_owner(self):
        """Global K cannot be replaced by K/CP per owner."""
        meta = DsaBatchMeta((0, 4), (3,), (0, 1, 2, 3), (0, 0, 1, 1), (0, 1, 0, 1),
                            cp_ranks=(0, 1), root_pes=(0, 1), cp_rank=1)
        indices = indexer_reference(torch.ones(1, 1, 1), torch.tensor([[9.0], [8.0], [1.0], [2.0]]),
                                    torch.ones(1, 1), meta, sparse_count=2)
        self.assertEqual(indices.tolist(), [[0, 1]])
        with self.assertRaisesRegex(ValueError, "full global"):
            indexer_reference(torch.ones(1, 1, 1), torch.ones(2, 1), torch.ones(1, 1),
                              replace(meta, kv_global_ids=(2, 3)), sparse_count=2)

    def test_kl_teacher_is_independent_per_head_and_detached(self):
        """An independent per-head calculation matches KL and its three gradients."""
        loss = self._loss()
        terms = []
        for token in range(5):
            selected = slice(0, token + 1)
            scores = (self.main[0][token] @ self.main[1][selected].T
                      + self.main[2][token] @ self.main[3][selected].T) * 0.37
            teacher = scores.softmax(-1).mean(0).detach()
            scores_index = ((self.index[0][token] @ self.index[1][selected].T).relu()
                            * self.index[2][token, :, None]).sum(0)
            terms.append((teacher * (teacher.log() - scores_index.log_softmax(0))).sum())
        expected = torch.stack(terms).mean()
        torch.testing.assert_close(loss, expected)
        gradients = torch.autograd.grad(loss, self.index + self.main, allow_unused=True, retain_graph=True)
        expected_grad = torch.autograd.grad(expected, self.index)
        for actual, wanted in zip(gradients[:3], expected_grad):
            torch.testing.assert_close(actual, wanted)
        self.assertEqual(gradients[3:], (None, None, None, None))

    def test_kl_gradcheck(self):
        """Indexer derivatives pass gradcheck away from ReLU/Top-K boundaries."""
        self.assertTrue(torch.autograd.gradcheck(lambda *index: self._loss(index=index), self.index))

    def test_loss_coefficient_and_upstream_scale_applied_once(self):
        """Nonunit aux scale multiplies the indexer gradients exactly once."""
        gradients = torch.autograd.grad(self._loss(), self.index)
        scaled = torch.autograd.grad(self._loss(coefficient=0.3) * 7, self.index)
        for actual, expected in zip(scaled, gradients):
            torch.testing.assert_close(actual, expected * 2.1)
        for actual in torch.autograd.grad(self._loss(coefficient=0), self.index):
            self.assertEqual(actual.count_nonzero(), 0)

    def test_unequal_cp_shards_recover_global_objective_and_gradients(self):
        """Two- and three-query sums reproduce global gradients for sum/average reducers."""
        full_loss = self._loss()
        full_grad = torch.autograd.grad(full_loss, self.index)
        for divisor in (1, 2):
            parts = []
            for rank, rows in enumerate(((0, 1), (2, 3, 4))):
                meta = replace(self.meta, q_global_ids=rows, cp_ranks=(7, 2), root_pes=(2, 0), cp_rank=rank,
                               token_owners=(0, 0, 1, 1, 1), token_local_offsets=(0, 1, 0, 1, 2))
                index = (self.index[0][list(rows)], self.index[1], self.index[2][list(rows)])
                main = (self.main[0][list(rows)], self.main[1], self.main[2][list(rows)], self.main[3])
                parts.append(self._loss(index=index, main=main, meta=meta,
                                        indices=self.indices[list(rows)], divisor=divisor))
            reduced = sum(parts) / divisor
            torch.testing.assert_close(reduced, full_loss)
            for actual, expected in zip(torch.autograd.grad(reduced, self.index), full_grad):
                torch.testing.assert_close(actual, expected)

    def test_empty_selected_set_has_finite_zero_loss_and_gradients(self):
        """No softmax NaNs or last-token gradient can arise from -1 padding."""
        loss = self._loss(indices=torch.full((5, 2), -1, dtype=torch.int32))
        self.assertEqual(loss, 0)
        for gradient in torch.autograd.grad(loss, self.index):
            self.assertEqual(gradient.count_nonzero(), 0)

    def test_projection_detach_preserves_index_parameter_gradients(self):
        """Detach before projection isolates the trunk while training the indexer."""
        hidden, weight = _leaves(((5, 3), (3, 6)), seed=11)
        index_query = (hidden.detach() @ weight).reshape(5, 2, 3)
        loss = self._loss(index=(index_query, self.index[1], self.index[2]))
        hidden_grad, weight_grad = torch.autograd.grad(loss, (hidden, weight), allow_unused=True)
        self.assertIsNone(hidden_grad)
        self.assertGreater(weight_grad.norm(), 0)
