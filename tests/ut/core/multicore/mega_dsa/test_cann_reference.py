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
"""CPU contract tests for the explicitly mocked CANN reference boundary."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
    CannDsaStats,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)

_MODULE = "hyper_parallel.core.multicore.modules.mega_dsa.cann_reference"


class TestCannDsaLayout(unittest.TestCase):
    """Verify packed namespace conversion independently of any NPU operator."""

    def test_sequence_namespace_roundtrip_preserves_padding(self):
        """The second packed sequence uses a local KV index base of zero."""
        layout = CannDsaLayout(DsaBatchMeta.packed((2, 3)), "cpu")
        global_ids = torch.tensor([[0, -1], [1, 0], [2, -1], [3, 2], [4, 3]], dtype=torch.int32)
        native = layout.global_to_sequence_indices(global_ids)
        self.assertEqual(native[:, 0].tolist(), [[0, -1], [1, 0], [0, -1], [1, 0], [2, 1]])
        torch.testing.assert_close(layout.sequence_to_global_indices(native), global_ids)
        self.assertEqual(layout.cumulative_lengths, (2, 5))
        self.assertEqual(layout.length_tensor.tolist(), [2, 5])

    def test_causal_filter_compacts_holes_in_original_slot_order(self):
        """CANN's first -1 terminator follows all surviving causal selections."""
        layout = CannDsaLayout(DsaBatchMeta.packed((2, 2)), "cpu")
        indices = torch.tensor([[-1, 0, 1, -2], [1, -1, 0, 99],
                                [0, -1, 2, 3], [3, 0, -1, 2]], dtype=torch.int32)
        native = layout.global_to_sequence_indices(indices)
        self.assertEqual(native[:, 0].tolist(), [[0, -1, -1, -1], [1, 0, -1, -1],
                                                [0, -1, -1, -1], [1, 0, -1, -1]])

    def test_cp_partial_and_reordered_storage_are_rejected(self):
        """Stock right-down masking must never guess an interior shard's offset."""
        base = DsaBatchMeta.packed((3,))
        cases = (replace(base, q_global_ids=(1, 2)), replace(base, kv_global_ids=(2, 0, 1)),
                 replace(base, cp_ranks=(0, 1), root_pes=(0, 1)))
        for meta in cases:
            with self.subTest(meta=meta), self.assertRaisesRegex(ValueError, "CP=1"):
                CannDsaLayout(meta, "cpu")

    def test_bad_index_rank_dtype_and_nkv_fail(self):
        """Only declared canonical/native index layouts are accepted."""
        layout = CannDsaLayout(DsaBatchMeta.packed((2,)), "cpu")
        for indices in (torch.ones(2, 2, dtype=torch.int64), torch.ones(1, 2, dtype=torch.int32),
                        torch.ones(2, 1, 2, dtype=torch.int32), torch.ones(2, 0, dtype=torch.int32)):
            with self.subTest(shape=indices.shape), self.assertRaises(ValueError):
                layout.global_to_sequence_indices(indices)
        with self.assertRaisesRegex(ValueError, "Nkv=1"):
            layout.sequence_to_global_indices(torch.ones(2, 2, 4, dtype=torch.int32))

    def test_cpu_layout_cannot_launch_backend_calls(self):
        """Preparing CPU metadata never enables a production CPU fallback."""
        backend = CannDsaReference(CannDsaLayout(DsaBatchMeta.packed((2,)), "cpu"), attention_scale=0.1)
        with self.assertRaisesRegex(ValueError, "NPU layout"):
            backend.indexer(torch.zeros(2, 8, 128), torch.zeros(2, 128), torch.zeros(2, 8))


class TestCannDsaReference(unittest.TestCase):
    """Mock only device execution to test ABI wiring and the precomputed-grad bridge."""

    def setUp(self) -> None:
        """Create canonical BF16 tensors and explicit device/operator mocks."""
        self.layout = CannDsaLayout(DsaBatchMeta.packed((2, 1)), "cpu")
        self.backend = CannDsaReference(self.layout, attention_scale=192**-0.5)
        self.main = tuple(torch.ones(shape, dtype=torch.bfloat16).requires_grad_()
                          for shape in ((3, 32, 512), (3, 512), (3, 32, 64), (3, 64)))
        self.index = tuple(torch.ones(shape, dtype=torch.bfloat16).requires_grad_()
                           for shape in ((3, 8, 128), (3, 128), (3, 8)))
        self.indices = torch.full((3, 2048), -1, dtype=torch.int32)
        self.indices[:, 0] = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.stats = CannDsaStats(torch.zeros(1, 3, 32), torch.ones(1, 3, 32))
        self.ops = SimpleNamespace(
            npu_lightning_indexer_enhance=Mock(), npu_sparse_flash_attention_enhance=Mock(),
            npu_sparse_lightning_indexer_grad_kl_loss_enhance=Mock(),
        )
        device_patch = patch.object(CannDsaReference, "_require_npu")
        ops_patch = patch(f"{_MODULE}._load_custom_ops", return_value=self.ops)
        device_patch.start()
        ops_patch.start()
        self.addCleanup(device_patch.stop)
        self.addCleanup(ops_patch.stop)

    def test_indexer_uses_tensor_lengths_and_returns_global_namespace(self):
        """Native sequence-local IDs become canonical IDs without another score scale."""
        native = torch.full((3, 1, 2048), -1, dtype=torch.int32)
        native[:, 0, 0] = torch.tensor([0, 1, 0], dtype=torch.int32)
        self.ops.npu_lightning_indexer_enhance.return_value = (native, torch.empty(0))
        selected = self.backend.indexer(*self.index)
        self.assertEqual(selected[:, 0].tolist(), [0, 1, 2])
        args, kwargs = self.ops.npu_lightning_indexer_enhance.call_args
        self.assertIs(args[2], self.index[2])
        self.assertEqual(args[1].shape, (3, 1, 128))
        self.assertIs(kwargs["actual_seq_lengths_query"], self.layout.length_tensor)
        self.assertEqual(kwargs["sparse_count"], 2048)

    def test_sparse_attention_reuses_one_key_value_and_one_backward(self):
        """Torch adds the separate native K/V contributions through the shared input."""
        def _native_attention(query, key, value, *_args, **_kwargs):
            return query + key[:, 0, None, :] + value[:, 0, None, :], self.stats.maximum, self.stats.denominator

        self.ops.npu_sparse_flash_attention_enhance.side_effect = _native_attention
        output, stats = self.backend.attention(*self.main, self.indices)
        output.sum().backward()
        torch.testing.assert_close(self.main[1].grad, torch.full_like(self.main[1], 64))
        args, kwargs = self.ops.npu_sparse_flash_attention_enhance.call_args
        self.assertIs(args[1], args[2])
        self.assertEqual(args[3][:, 0, 0].tolist(), [0, 1, 0])
        self.assertEqual(args[4], 192**-0.5)
        self.assertEqual(kwargs["attention_mode"], 2)
        self.assertEqual(stats.maximum.shape, (1, 3, 32))
        self.ops.npu_sparse_flash_attention_enhance.assert_called_once()

    def test_bsnd_boundary_restores_padding_and_projection_gradients(self):
        """Reshaping and zero RoPE padding retain the original model autograd edge."""
        self.ops.npu_sparse_flash_attention_enhance.return_value = (
            self.main[0] * 2, self.stats.maximum, self.stats.denominator)
        query, compressed, qr, kr = self.main
        output, _ = self.backend.attention_bsnd(query[None], compressed[None, :, None], qr[None],
                                              kr[None, :, None], self.indices)
        self.assertEqual(output.shape, (1, 3, 32, 576))
        self.assertEqual(output[..., 512:].count_nonzero(), 0)
        output[..., :512].sum().backward()
        torch.testing.assert_close(query.grad, torch.full_like(query, 2))

    def _kl_outputs(self, factor=1):
        return (torch.full_like(self.index[0], factor), torch.full_like(self.index[1][:, None], factor * 2),
                torch.full_like(self.index[2], factor * 3), torch.tensor([15.0]))

    def _loss(self, coefficient=0.3):
        return self.backend.kl_loss(*self.index, self.main, self.indices, self.stats,
                                    normalization=DsaLossNormalization(3), loss_coeff=coefficient)

    def test_kl_list_lengths_detached_teacher_and_scale_once(self):
        """The ABI receives Python lengths and backward only scales saved index gradients."""
        self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.return_value = self._kl_outputs()
        loss = self._loss()
        (loss * 7).backward()
        self.assertEqual(loss, 1.5)
        for tensor, multiplier in zip(self.index, (1, 2, 3)):
            torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 0.7 * multiplier))
        self.assertTrue(all(tensor.grad is None for tensor in self.main))
        _, kwargs = self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.call_args
        self.assertEqual(kwargs["actual_seq_qlen"], [2, 3])
        self.assertEqual(kwargs["actual_seq_klen"], [2, 3])
        self.assertFalse(kwargs["query"].requires_grad)
        self.assertFalse(kwargs["key"].requires_grad)
        self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.assert_called_once()

    def test_multiple_forwards_delayed_backward_and_retain_graph(self):
        """Each invocation keeps its own precomputed gradients without another KL launch."""
        self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.side_effect = [
            self._kl_outputs(1), self._kl_outputs(4)]
        first, second = self._loss(coefficient=1), self._loss(coefficient=1)
        second_grad = torch.autograd.grad(second, self.index, retain_graph=True)
        first_grad = torch.autograd.grad(first, self.index)
        repeated = torch.autograd.grad(second, self.index)
        for actual, expected, again in zip(second_grad, first_grad, repeated):
            torch.testing.assert_close(actual, expected * 4)
            torch.testing.assert_close(again, actual)
        self.assertEqual(self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.call_count, 2)

    def test_bad_dimensions_stats_and_normalization_fail_before_launch(self):
        """Unsupported support-matrix entries cannot reach the native operator."""
        with self.assertRaisesRegex(ValueError, "C=512"):
            self.backend.attention(self.main[0][..., :256], *self.main[1:], self.indices)
        with self.assertRaisesRegex(ValueError, "K=2048"):
            self.backend.attention(*self.main, self.indices[:, :1024])
        with self.assertRaisesRegex(ValueError, "complete packed"):
            self.backend.kl_loss(*self.index, self.main, self.indices, self.stats,
                                 normalization=DsaLossNormalization(5))
        with self.assertRaisesRegex(ValueError, "native.*statistics"):
            self.backend.kl_loss(*self.index, self.main, self.indices,
                                 CannDsaStats(torch.zeros(3, 32), torch.ones(3, 32)),
                                 normalization=DsaLossNormalization(3))
        self.ops.npu_sparse_flash_attention_enhance.assert_not_called()
        self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.assert_not_called()
