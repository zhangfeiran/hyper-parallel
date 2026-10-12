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
"""FP32 KL precision, TP head partition and immutable saved-state contracts."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import fp32_index_kl
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta


class TestFp32IndexKl(unittest.TestCase):
    """Mock submissions while independently checking the selected KL derivative formula."""

    def setUp(self) -> None:
        """Prepare stable positive dot products and a detached non-unit teacher mass."""
        self.meta = DsaBatchMeta.packed((3,))
        self.indices = torch.full((3, 2048), -1, dtype=torch.int32)
        for row in range(3):
            self.indices[row, :row + 1] = torch.arange(row + 1)
        generator = torch.Generator().manual_seed(541)
        self.query = torch.rand((3, 4, 128), generator=generator) * .1 + .01
        self.key = torch.rand((3, 128), generator=generator) * .1 + .01
        self.weight = torch.randn((3, 4), generator=generator) * .1
        self.teacher = torch.zeros((3, 2048), dtype=torch.float32)
        for row in range(3):
            self.teacher[row, :row + 1] = .75 / (row + 1)
        for mock in (patch.object(fp32_index_kl, '_load_native'),
                     patch.object(torch.ops.hyper_parallel, 'dsa_fp32_index_kl_version', return_value=1, create=True),
                     patch.object(torch.ops.hyper_parallel, 'dsa_fp32_index_kl_out',
                                  side_effect=self._submit, create=True)):
            mock.start()
            self.addCleanup(mock.stop)

    @staticmethod
    def _logits(query, key, weight, indices):
        gathered = key[indices.clamp_min(0).long()]
        dots = torch.einsum('thd,tkd->thk', query, gathered).relu()
        return (dots * weight[..., None]).sum(dim=1).masked_fill(indices < 0, 0)

    def _submit(self, query, key, weight, teacher, global_logits, indices, phase,
                local_logits, grad_query, grad_key, grad_weight, loss, trace):
        inputs = tuple(value.detach().requires_grad_() for value in (query, key, weight))
        local = self._logits(*inputs, indices)
        if phase == 0:
            local_logits.copy_(local.detach())
        else:
            combined = local + (global_logits - local).detach()
            log_probability = combined.masked_fill(indices < 0, -torch.inf).log_softmax(dim=-1)
            safe_log = torch.where(teacher > 0, teacher.log() - log_probability, torch.zeros_like(teacher))
            terms = torch.where(teacher > 0, teacher * safe_log, torch.zeros_like(teacher))
            value = terms.sum(dim=-1)
            gradients = torch.autograd.grad(value.sum(), inputs)
            for dst, src in zip((grad_query, grad_key, grad_weight), gradients):
                dst.copy_(src)
            loss.copy_(value.detach())
        for block in range(40):
            rows = tuple(range(block, query.shape[0], 40))
            trace[block] = torch.tensor([phase + 1, len(rows), sum(row + 1 for row in rows),
                                         rows[-1] if rows else -1])

    def test_tp_partition_preserves_global_logits_and_local_vjps(self) -> None:
        """Two local-head derivatives sum to the complete key derivative with no extra TP divisor."""
        probe = fp32_index_kl.Fp32IndexKlProbe(self.meta, self.indices, 'cpu')
        full = probe.scores(self.query, self.key, self.weight)
        left = probe.scores(self.query[:, :2], self.key, self.weight[:, :2])
        right = probe.scores(self.query[:, 2:], self.key, self.weight[:, 2:])
        summed = left.logits + right.logits
        torch.testing.assert_close(summed, full.logits, rtol=1e-6, atol=1e-7)
        expected = probe.vjp(full, self.teacher, summed)
        first = probe.vjp(left, self.teacher, summed)
        second = probe.vjp(right, self.teacher, summed)
        torch.testing.assert_close(first.gradients[1] + second.gradients[1], expected.gradients[1])
        torch.testing.assert_close(torch.cat((first.gradients[0], second.gradients[0]), dim=1), expected.gradients[0])
        torch.testing.assert_close(torch.cat((first.gradients[2], second.gradients[2]), dim=1), expected.gradients[2])
        self.assertTrue(torch.equal(first.loss, second.loss))
        receipt = fp32_index_kl.validate_fp32_index_kl_trace(first.trace, query_tokens=3, phase=1)
        self.assertEqual(receipt, {'phase': 1, 'queries': 3, 'native_launch': True})

    def test_fp32_values_and_saved_input_mutation(self) -> None:
        """A representable FP32 perturbation survives; modified forward inputs fail before backward."""
        probe = fp32_index_kl.Fp32IndexKlProbe(self.meta, self.indices, 'cpu')
        self.query[0, 0, 0] = 1.003
        full = probe.scores(self.query, self.key, self.weight)
        rounded = self._logits(self.query.bfloat16().float(), self.key, self.weight, self.indices)
        self.assertFalse(torch.equal(full.logits, rounded))
        self.query.add_(.01)
        with self.assertRaisesRegex(ValueError, 'saved inputs'):
            probe.vjp(full, self.teacher, full.logits)

    def test_selection_and_dtype_admission(self) -> None:
        """Reject cross-sequence, future, duplicate and non-tail padding IDs, plus lossy dtype conversion."""
        for row in ([1, -1], [0, 0], [-1, 0]):
            indices = self.indices.clone()
            indices[0, :2] = torch.tensor(row)
            with self.subTest(row=row), self.assertRaises(ValueError):
                fp32_index_kl.Fp32IndexKlProbe(self.meta, indices, 'cpu')
        probe = fp32_index_kl.Fp32IndexKlProbe(self.meta, self.indices, 'cpu')
        with self.assertRaisesRegex(ValueError, 'preserves FP32'):
            probe.scores(self.query.bfloat16(), self.key, self.weight)
        saved = probe.scores(self.query, self.key, self.weight)
        probe.indices[0, 0] = -1
        with self.assertRaises(ValueError):
            probe.vjp(saved, self.teacher, saved.logits)

    def test_phase_trace_rejects_missing_rows(self) -> None:
        """Receipt checks independently reject missing query work rather than trusting a launch return."""
        probe = fp32_index_kl.Fp32IndexKlProbe(self.meta, self.indices, 'cpu')
        result = probe.scores(self.query, self.key, self.weight)
        result.trace[1, 1] = 0
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            fp32_index_kl.validate_fp32_index_kl_trace(result.trace, query_tokens=3, phase=0)

    def test_empty_query_keeps_complete_zero_key_return_without_native_launch(self) -> None:
        """An empty CP query shard still supplies a complete FP32 zero key contribution."""
        meta = replace(self.meta, q_global_ids=())
        probe = fp32_index_kl.Fp32IndexKlProbe(meta, self.indices[:0], 'cpu')
        with patch.object(torch.ops.hyper_parallel, 'dsa_fp32_index_kl_out') as launch:
            scores = probe.scores(self.query[:0], self.key, self.weight[:0])
            result = probe.vjp(scores, self.teacher[:0], scores.logits)
            launch.assert_not_called()
        self.assertTrue(torch.equal(result.gradients[1], torch.zeros_like(self.key)))
        self.assertEqual(result.loss.shape, (0,))
        self.assertEqual(fp32_index_kl.validate_fp32_index_kl_trace(result.trace, query_tokens=0, phase=1),
                         {'phase': 1, 'queries': 0, 'native_launch': False})
