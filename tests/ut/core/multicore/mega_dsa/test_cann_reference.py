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

from hyper_parallel.components.functional.aux_loss import (
    aux_loss_auto_scale,
    set_aux_loss_scale,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
    CannDsaSelection,
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
        self.indices[1, 1] = 0
        self.selection = self.backend.prepare_selection(self.indices)
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
        native[1, 0, 1] = 0
        self.ops.npu_lightning_indexer_enhance.return_value = (native, torch.empty(0))
        selected = self.backend.indexer(*self.index)
        self.assertIsInstance(selected, CannDsaSelection)
        self.assertEqual(selected.to_global_indices()[:, :2].tolist(), [[0, -1], [1, 0], [2, -1]])
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
        output, stats = self.backend.attention(*self.main, self.selection)
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
                                              kr[None, :, None], self.selection)
        self.assertEqual(output.shape, (1, 3, 32, 576))
        self.assertEqual(output[..., 512:].count_nonzero(), 0)
        output[..., :512].sum().backward()
        torch.testing.assert_close(query.grad, torch.full_like(query, 2))

    def _kl_outputs(self, factor=1):
        return (torch.full_like(self.index[0], factor), torch.full_like(self.index[1][:, None], factor * 2),
                torch.full_like(self.index[2], factor * 3), torch.tensor([15.0]))

    def _loss(self, coefficient=0.3):
        return self.backend.kl_loss(*self.index, self.main, self.selection, self.stats,
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

    def test_existing_auxiliary_scaler_controls_injected_index_gradients(self):
        """Trainer grad_aux is independent of the main objective's scalar multiplier."""
        self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.return_value = self._kl_outputs()
        scale_attribute = "hyper_parallel.components.functional.aux_loss._AuxLossAutoScaler.main_loss_backward_scale"
        for auxiliary_scale, coefficient in ((1, 0.3), (7, 0.3), (7, 0)):
            with self.subTest(scale=auxiliary_scale, coefficient=coefficient), patch(scale_attribute):
                set_aux_loss_scale(torch.tensor(float(auxiliary_scale)))
                output = torch.ones(3, 2, requires_grad=True)
                loss = self._loss(coefficient=coefficient)
                attached = aux_loss_auto_scale(output, loss)
                gradients = torch.autograd.grad(attached.square().mean() * 13, (output, *self.index))
                torch.testing.assert_close(attached, output)
                torch.testing.assert_close(gradients[0], torch.full_like(output, 26 / output.numel()))
                for actual, multiplier in zip(gradients[1:], (1, 2, 3)):
                    expected = multiplier * coefficient / 3 * auxiliary_scale
                    torch.testing.assert_close(actual, torch.full_like(actual, expected))
                self.assertTrue(all(tensor.grad is None for tensor in self.main))

    def test_bad_dimensions_stats_and_normalization_fail_before_launch(self):
        """Unsupported support-matrix entries cannot reach the native operator."""
        with self.assertRaisesRegex(ValueError, "C=512"):
            self.backend.attention(self.main[0][..., :256], *self.main[1:], self.selection)
        with self.assertRaisesRegex(ValueError, "K=2048"):
            self.backend.prepare_selection(self.indices[:, :1024])
        with self.assertRaisesRegex(ValueError, "complete packed"):
            self.backend.kl_loss(*self.index, self.main, self.selection, self.stats,
                                 normalization=DsaLossNormalization(5))
        with self.assertRaisesRegex(ValueError, "native.*statistics"):
            self.backend.kl_loss(*self.index, self.main, self.selection,
                                 CannDsaStats(torch.zeros(3, 32), torch.ones(3, 32)),
                                 normalization=DsaLossNormalization(3))
        self.ops.npu_sparse_flash_attention_enhance.assert_not_called()
        self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.assert_not_called()

    def test_prepare_rejects_underfilled_empty_unknown_and_duplicate_selections(self):
        """Invalid effective counts/IDs cannot reach native attention or KL."""
        cases = []
        partial = self.indices.clone()
        partial[1, 1] = -1
        cases.append((partial, "requires 2 legal keys; got 1"))
        cases.append((torch.full_like(self.indices, -1), "requires 1 legal keys; got 0"))
        for invalid, message in ((-2, "unknown"), (3, "unknown"), (1, "duplicate")):
            bad = self.indices.clone()
            bad[1, 1] = invalid
            cases.append((bad, message))
        for indices, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.backend.prepare_selection(indices)
        self.ops.npu_sparse_flash_attention_enhance.assert_not_called()
        self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.assert_not_called()

    def test_prepare_filters_holes_future_and_cross_sequence_before_count_check(self):
        """Full legal histories survive stable filtering without admitting an underfilled row."""
        indices = torch.full_like(self.indices, -1)
        indices[0, :3] = torch.tensor([-1, 1, 0], dtype=torch.int32)
        indices[1, :4] = torch.tensor([2, 1, -1, 0], dtype=torch.int32)
        indices[2, :3] = torch.tensor([0, -1, 2], dtype=torch.int32)
        selected = self.backend.prepare_selection(indices)
        torch.testing.assert_close(selected.to_global_indices(), self.indices)
        indices[1, 3] = -1
        with self.assertRaisesRegex(ValueError, "requires 2 legal keys; got 1"):
            self.backend.prepare_selection(indices)

    def test_prepare_requires_an_explicit_host_snapshot(self):
        """Preparation cannot silently synchronize or read device selections."""
        for indices in (self.indices.float(), self.indices[:, :1024], self.indices[None],
                        torch.empty(3, 2048, dtype=torch.int32, device="meta"), None):
            with self.subTest(indices=type(indices)), self.assertRaises(ValueError):
                self.backend.prepare_selection(indices)

    def test_raw_tensors_are_rejected_by_all_execution_boundaries(self):
        """Even a full-history tensor needs preparation before attention/BSND/KL."""
        calls = (
            lambda: self.backend.attention(*self.main, self.indices),
            lambda: self.backend.kl_loss(*self.index, self.main, self.indices, self.stats,
                                         normalization=DsaLossNormalization(3)),
            lambda: self.backend.attention_bsnd(self.main[0][None], self.main[1][None, :, None],
                                                self.main[2][None], self.main[3][None, :, None], self.indices),
        )
        for call in calls:
            with self.assertRaisesRegex(TypeError, "raw indices"):
                call()
        self.ops.npu_sparse_flash_attention_enhance.assert_not_called()
        self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.assert_not_called()

    def test_selection_is_owned_and_exports_do_not_alias(self):
        """Changing the host source or an exported tensor cannot invalidate admitted storage."""
        expected = self.indices.clone()
        self.indices.fill_(-1)
        exported = self.selection.to_global_indices()
        exported.fill_(-1)
        torch.testing.assert_close(self.selection.to_global_indices(), expected)

    def test_foreign_layout_mutated_storage_and_direct_construction_fail(self):
        """A proof cannot be reused for another layout or after internal storage mutation."""
        other = CannDsaReference(CannDsaLayout(self.layout.batch_meta, "cpu"), attention_scale=192**-0.5)
        with self.assertRaisesRegex(ValueError, "different prepared layout"):
            other.attention(*self.main, self.selection)
        with self.assertRaisesRegex(ValueError, "prepare_selection or indexer"):
            CannDsaSelection(self.layout, self.indices[:, None])
        self.selection._native_indices[0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "storage was modified"):
            self.backend.attention(*self.main, self.selection)
        self.ops.npu_sparse_flash_attention_enhance.assert_not_called()

    def test_native_indexer_and_execution_do_not_read_host_tensor_values(self):
        """Native-produced selections need no cpu/item/tolist calls or repeated compaction."""
        native = self.selection._native_indices.clone()
        self.ops.npu_lightning_indexer_enhance.return_value = (native, torch.empty(0))
        self.ops.npu_sparse_flash_attention_enhance.return_value = (
            self.main[0] * 2, self.stats.maximum, self.stats.denominator)
        self.ops.npu_sparse_lightning_indexer_grad_kl_loss_enhance.return_value = self._kl_outputs()
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")), \
                patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar read")), \
                patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host values")), \
                patch.object(self.layout, "global_to_sequence_indices", side_effect=AssertionError("compaction")):
            selected = self.backend.indexer(*self.index)
            self.backend.attention(*self.main, selected)
            self.backend.kl_loss(*self.index, self.main, selected, self.stats, normalization=DsaLossNormalization(3))

    def test_selection_preparation_in_inference_mode_retains_version_checks(self):
        """Inference-mode callers still receive owned, versioned index storage."""
        with torch.inference_mode():
            selection = self.backend.prepare_selection(self.indices)
            exported = selection.to_global_indices()
        torch.testing.assert_close(exported, self.indices)

    def test_long_prefix_accepts_exact_k_not_the_full_prefix(self):
        """For causal histories beyond K, preparation admits K unique legal keys."""
        total = 2049
        backend = CannDsaReference(CannDsaLayout(DsaBatchMeta.packed((total,)), "cpu"), attention_scale=0.1)
        ids = torch.arange(2048, dtype=torch.int32).expand(total, -1).clone()
        ids.masked_fill_(ids > torch.arange(total, dtype=torch.int32)[:, None], -1)
        ids[-1, -1] = 2048
        selection = backend.prepare_selection(ids)
        torch.testing.assert_close(selection.to_global_indices(), ids)
        ids[-1, -1] = -1
        with self.assertRaisesRegex(ValueError, "requires 2048 legal keys; got 2047"):
            backend.prepare_selection(ids)
