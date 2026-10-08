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
"""Mocked CP layout/communication tests; no distributed setup or hardware."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.cp_reference import (
    CannDsaCpReference,
    DsaCpLayout,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)

_MODULE = "hyper_parallel.core.multicore.modules.mega_dsa.cp_reference"


def _meta(rank=0):
    ids = ((2, 0), (4, 1, 3))[rank]
    return DsaBatchMeta((0, 2, 5), ids, tuple(reversed(ids)), (0, 1, 0, 1, 1), (1, 2, 0, 1, 0),
                        cp_ranks=(3, 7), root_pes=(1, 4), cp_rank=rank)


class TestDsaCpLayout(unittest.TestCase):
    """Explicit namespace, owner accumulation and construction-time disagreement."""

    def _distributed(self, rank=0):
        mocked = patch(f"{_MODULE}.dist").start()
        self.addCleanup(patch.stopall)
        mocked.is_initialized.return_value = True
        mocked.get_process_group_ranks.return_value = [3, 7]
        mocked.get_rank.return_value = rank
        mocked.all_gather_object.side_effect = lambda output, value, **_: output.__setitem__(slice(None), [value] * 2)
        return mocked

    def test_uneven_reordered_owner_permutations(self):
        """Storage offsets and global causality remain independent."""
        self._distributed()
        layout = DsaCpLayout(_meta(), "cpu")
        self.assertEqual(layout.counts, (2, 3))
        self.assertEqual(layout.global_order.tolist(), [1, 5, 0, 4, 3])
        self.assertEqual(layout.query_order.tolist(), [0, 1])
        self.assertEqual(layout.key_order.tolist(), [1, 0])
        self.assertEqual(layout.local_query_ids.tolist(), [2, 0])

    def test_gather_restores_global_fields_and_one_backward(self):
        """One packed gather carries independent Q and KV orderings."""
        mocked = self._distributed()
        layout = DsaCpLayout(_meta(), "cpu")
        tensors = tuple(torch.tensor([[2.], [0.]], requires_grad=True) for _ in range(4))
        tensors = (tensors[0], tensors[1].flip(0), tensors[2], tensors[3].flip(0))

        def _gather(output, local, **_):
            torch.testing.assert_close(local, torch.tensor([[2.] * 4, [0.] * 4, [0.] * 4]))
            output.copy_(torch.tensor([[2.] * 4, [0.] * 4, [0.] * 4, [4.] * 4, [3.] * 4, [1.] * 4]))

        def _scatter(output, full, **_):
            self.assertEqual(full.dtype, torch.float32)
            output.copy_(full[:3])

        mocked.all_gather_into_tensor.side_effect = _gather
        mocked.reduce_scatter_tensor.side_effect = _scatter
        fields = layout.gather_fields(tensors, ((1,),) * 4)
        for field in fields:
            torch.testing.assert_close(field[:, 0], torch.arange(5, dtype=torch.float32))
        sum(field.square().sum() for field in fields).backward()
        self.assertEqual(mocked.all_gather_into_tensor.call_count, 1)
        self.assertEqual(mocked.reduce_scatter_tensor.call_count, 1)

    def test_owner_sum_is_fp32_then_cast_once(self):
        """BF16 owner return must retain a small contribution between cancelling peers."""
        mocked = self._distributed()
        layout = DsaCpLayout(_meta(), "cpu")
        local = torch.ones((2, 1), dtype=torch.bfloat16, requires_grad=True)
        mocked.all_gather_into_tensor.side_effect = lambda output, padded, **_: output.copy_(padded.repeat(2, 1))

        def _scatter(output, gradient, **_):
            self.assertEqual(gradient.dtype, torch.float32)
            peer = gradient.new_full(gradient.shape, 256)
            other_peer = gradient.new_full(gradient.shape, -256)
            output.copy_((peer + gradient + other_peer)[:3])

        mocked.reduce_scatter_tensor.side_effect = _scatter
        sum(field.sum() for field in layout.gather_fields((local,) * 4, ((1,),) * 4)).backward()
        torch.testing.assert_close(local.grad, torch.full_like(local, 4))
        self.assertEqual(local.grad.dtype, torch.bfloat16)

    def test_fp64_oracle_accumulation_is_preserved(self):
        """CPU gradcheck precision must not be lowered by the exchange."""
        mocked = self._distributed()
        layout = DsaCpLayout(_meta(), "cpu")
        local = torch.ones((2, 1), dtype=torch.float64, requires_grad=True)
        mocked.all_gather_into_tensor.side_effect = lambda output, padded, **_: output.copy_(padded.repeat(2, 1))

        def _scatter(output, gradient, **_):
            self.assertEqual(gradient.dtype, torch.float64)
            output.copy_(gradient[:3])

        mocked.reduce_scatter_tensor.side_effect = _scatter
        sum(field.sum() for field in layout.gather_fields((local,) * 4, ((1,),) * 4)).backward()
        torch.testing.assert_close(local.grad, torch.full_like(local, 4))

    def test_collective_metadata_disagreement(self):
        """Different invocation IDs are rejected by every participant during preparation."""
        mocked = self._distributed()

        def _declarations(output, value, **_):
            if value is None:
                output[:] = [None, None]
            else:
                different = (*value[0][:-3], value[0][-3] + 1, *value[0][-2:])
                output[:] = [value, (different, None)]

        mocked.all_gather_object.side_effect = _declarations
        with self.assertRaisesRegex(ValueError, "disagree on global metadata"):
            DsaCpLayout(_meta(), "cpu")

    def test_partial_shard_collectively_rejected(self):
        """CP baseline cannot silently drop owner queries or keys."""
        self._distributed()
        with self.assertRaisesRegex(ValueError, "complete owner-local"):
            DsaCpLayout(replace(_meta(), kv_global_ids=(0,)), "cpu")

    def test_group_order_and_membership_rejected(self):
        """A same-sized unrelated group is not the metadata CP group."""
        mocked = self._distributed()
        mocked.get_process_group_ranks.return_value = [7, 3]
        with self.assertRaisesRegex(ValueError, "membership/order/rank"):
            DsaCpLayout(_meta(), "cpu")

    def test_cp_requires_initialized_group(self):
        """A multi-owner layout cannot default to a single-process calculation."""
        with patch(f"{_MODULE}.dist.is_initialized", return_value=False), \
             self.assertRaisesRegex(ValueError, "initialized process group"):
            DsaCpLayout(_meta(), "cpu")

    def test_empty_owner_supported(self):
        """Padded communication can represent a rank with no valid query or KV rows."""
        self._distributed(rank=1)
        meta = DsaBatchMeta((0, 2), (), (), (0, 0), (0, 1), cp_ranks=(3, 7), root_pes=(1, 4), cp_rank=1)
        layout = DsaCpLayout(meta, "cpu")
        self.assertEqual(layout.local_tokens, 0)
        self.assertEqual(layout.padded_tokens, 2)
        self.assertEqual(layout.local_query_ids.numel(), 0)

    def test_cp1_reordered_fields_and_gradcheck(self):
        """The same logical owner mapping works without distributed initialization."""
        with patch(f"{_MODULE}.dist.is_initialized", return_value=False):
            meta = replace(DsaBatchMeta.packed((2, 3)), q_global_ids=(3, 0, 1, 4, 2),
                           kv_global_ids=(2, 4, 0, 3, 1))
            layout = DsaCpLayout(meta, "cpu")
            tensors = tuple(torch.randn(5, 1, dtype=torch.float64, requires_grad=True) for _ in range(4))
            self.assertTrue(torch.autograd.gradcheck(lambda *args: layout.gather_fields(args, ((1,),) * 4), tensors))

    def test_field_shape_and_dtype_rejected_before_communication(self):
        """Static descriptor mismatch cannot enter the device collective."""
        mocked = self._distributed()
        layout = DsaCpLayout(_meta(), "cpu")
        fields = (torch.zeros(2, 1),) * 4
        for invalid in ((torch.zeros(1, 1), *fields[1:]), (fields[0].double(), *fields[1:])):
            with self.subTest(shapes=[item.shape for item in invalid]), self.assertRaises(ValueError):
                layout.gather_fields(invalid, ((1,),) * 4)
        mocked.all_gather_into_tensor.assert_not_called()


class TestCannDsaCpReference(unittest.TestCase):
    """Native API routing and KL replica weighting tested using mocked calls."""

    def setUp(self) -> None:
        """Build a CP=1 CPU layout; no native call is allowed in these tests."""
        with patch(f"{_MODULE}.dist.is_initialized", return_value=False):
            self.layout = DsaCpLayout(DsaBatchMeta.packed((2, 3)), "cpu")
        self.backend = CannDsaCpReference(self.layout, attention_scale=0.1)
        self.inputs = tuple(torch.randn(5, *shape, dtype=torch.bfloat16, requires_grad=True)
                            for shape in self.backend.shapes)

    def test_global_normalization_and_kl_replica_division(self):
        """Each replica contributes global KL / CP, with the original coefficient once."""
        self.layout.cp_size = 4
        self.layout.gather_fields = lambda inputs, _: inputs
        with patch.object(self.backend.backend, "indexer") as indexer, \
             patch.object(self.backend.backend, "attention") as attention, \
             patch.object(self.backend.backend, "kl_loss") as kl:
            indexer.return_value.to_global_indices.return_value = torch.zeros(5, 2048, dtype=torch.int32)
            stats = unittest.mock.Mock(maximum=torch.ones(1, 5, 32), denominator=torch.ones(1, 5, 32))
            attention.return_value = (self.inputs[0], stats)
            kl.return_value = torch.tensor(8., requires_grad=True)
            norm = DsaLossNormalization(5, reducer_divisor=4)
            result = self.backend.forward(self.inputs[:4], index_inputs=self.inputs[4:],
                                          normalization=norm, loss_coeff=0.3)
            self.assertEqual(result.kl_loss, 2)
            self.assertIs(kl.call_args.kwargs["normalization"], norm)
            self.assertEqual(kl.call_args.kwargs["loss_coeff"], 0.3)
            result.kl_loss.backward()
            self.assertEqual(kl.return_value.grad, 0.25)

    def test_attention_only_requires_selection(self):
        """Disabling the indexer must still supply an admitted external selection."""
        backend = CannDsaCpReference(self.layout, attention_scale=0.1, with_indexer=False)
        with self.assertRaisesRegex(ValueError, "external selection"):
            backend.forward(self.inputs[:4])

    def test_packed_exchange_preserves_kl_teacher_gradient_absence(self):
        """KL-only backward must leave all main teacher gradients None."""
        gathered = self.layout.gather_fields(self.inputs, self.backend.shapes)
        sum(field.float().sum() for field in gathered[4:]).backward()
        for teacher in self.inputs[:4]:
            self.assertIsNone(teacher.grad)
        for index in self.inputs[4:]:
            torch.testing.assert_close(index.grad, torch.ones_like(index))

    def test_schema_and_normalization_errors(self):
        """Rank schema and global count are explicit rather than inferred from local lengths."""
        with self.assertRaisesRegex(ValueError, "field"):
            self.backend.forward(self.inputs[:4])
        with self.assertRaisesRegex(ValueError, "explicit global"):
            self.backend.forward(self.inputs[:4], index_inputs=self.inputs[4:])
        with self.assertRaisesRegex(ValueError, "complete global"):
            self.backend.forward(self.inputs[:4], index_inputs=self.inputs[4:], normalization=DsaLossNormalization(3))

    def test_invalid_schema_switch_and_coefficient(self):
        """Only a declared boolean schema and finite nonnegative KL coefficient are admitted."""
        with self.assertRaisesRegex(TypeError, "boolean"):
            CannDsaCpReference(self.layout, attention_scale=0.1, with_indexer="yes")
        for coefficient in (-1, float("nan"), float("inf")):
            with self.subTest(coefficient=coefficient), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                self.backend.forward(self.inputs[:4], index_inputs=self.inputs[4:],
                                     normalization=DsaLossNormalization(5), loss_coeff=coefficient)

    def test_native_cardinality_contract_preserved(self):
        """CP setup cannot use underfilled candidates forbidden by native backward/KL."""
        with self.assertRaisesRegex(ValueError, "requires 1 legal keys; got 0"):
            self.backend.prepare_selection(torch.full((5, 2048), -1, dtype=torch.int32))
