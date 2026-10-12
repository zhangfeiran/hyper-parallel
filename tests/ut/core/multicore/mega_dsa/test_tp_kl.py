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
"""CPU admission, packing and raw-VJP contracts for native TP selected KL."""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.tp_kl import TpLocalKlProbe
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumInvocation,
    TpTileSumProbe,
    TpTileSumSpec,
)


class TestTpKl(unittest.TestCase):
    """Mock native math/communication while exercising real query and head layouts."""

    def setUp(self) -> None:
        """Prepare mocked TP transport and native execution for reordered local rows."""
        self.prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.tp_kl'
        self.transport = Mock(spec=TpTileSumProbe)
        self.transport.spec = TpTileSumSpec(14, 2048)
        self.transport.members = (0, 2)
        self.transport.participates = True
        self.transport.group = object()
        self.transport.arena = torch.zeros(self.transport.spec.arena_bytes, dtype=torch.uint8)
        self.transport.lease.side_effect = lambda *_args: nullcontext()
        self.transport.prepare.side_effect = self._prepare
        self.backend = TpLocalKlProbe(self.transport, global_heads=128, attention_scale=0.125,
                                      schedule=MixedSfaSchedule(7))
        self.meta = replace(DsaBatchMeta.packed((3, 3)), q_global_ids=(5, 1, 3), heap_generation=1)
        self.selection = torch.full((3, 2048), -1, dtype=torch.int32)
        self.selection[0, :3] = torch.tensor([3, 4, 5])
        self.selection[1, :2] = torch.tensor([0, 1])
        self.selection[2, :1] = 3
        for name in ('dist.all_gather_object', '_load_native', 'torch.ops.hyper_parallel'):
            patcher = patch(self.prefix + '.' + name, Mock())
            value = patcher.start()
            self.addCleanup(patcher.stop)
            if name == 'dist.all_gather_object':
                value.side_effect = lambda outputs, item, **_kwargs: outputs.__setitem__(slice(None), [item, item])
            elif name == 'torch.ops.hyper_parallel':
                self.ops = value
        self.ops.dsa_tp_local_mixed_kl_version.return_value = 1
        self.ops.dsa_tp_local_mixed_kl_out.side_effect = self._native
        self.ops.dsa_local_mixed_kl_out.side_effect = self._native
        partials = patch(self.prefix + '.mixed_kl_key_partials', return_value=torch.full((6, 128), 0.75))
        partials.start()
        self.addCleanup(partials.stop)

    def _prepare(self, *, rounds, epoch, identity):
        words = self.transport.spec.metadata(rounds=rounds, epoch=epoch, rank=0, root_pes=(0, 2), root_size=4,
                                             identity=identity)
        meta = torch.tensor(words, dtype=torch.int64)
        return TpTileSumInvocation(self.transport, rounds, epoch, 1, meta, meta._version)

    @staticmethod
    def _native(*args):
        scale, phase = args[-6], args[-5]
        grad_query, grad_key = args[-4], args[-3]
        grad_weight, loss = args[-2], args[-1]
        if phase == 2:
            grad_query.fill_(2.5)
            grad_key.fill_(3.5)
            grad_weight.fill_(4.5)
            loss.fill_(5.5)
        if scale != 0.125:
            raise ValueError('wrong caller scale')

    @staticmethod
    def _inputs(rows=3):
        main = (torch.randn(rows, 64, 512, dtype=torch.bfloat16), torch.randn(6, 512, dtype=torch.bfloat16),
                torch.randn(rows, 64, 64, dtype=torch.bfloat16), torch.randn(6, 64, dtype=torch.bfloat16))
        index = (torch.arange(rows * 32 * 128).reshape(rows, 32, 128).bfloat16(),
                 torch.randn(6, 128, dtype=torch.bfloat16), torch.full((rows, 32), -2.5))
        stats = (torch.randn(1, rows, 64), torch.ones(1, rows, 64))
        return main, index, stats

    def test_tp_head_padding_and_phase_specific_exchange_preserve_raw_vjp(self):
        """Exchange runs only in phase one; no extra TP factor changes any local derivative."""
        invocation = self.backend.prepare(self.meta, self.selection, epoch=1)
        main, index, stats = self._inputs()
        result = self.backend.gradients(main, index, stats, invocation)
        self.ops.dsa_tp_local_mixed_kl_out.assert_called_once()
        self.assertEqual(self.ops.dsa_local_mixed_kl_out.call_count, 2)
        self.transport.lease.assert_called_once_with(invocation.transport)
        args = self.ops.dsa_tp_local_mixed_kl_out.call_args.args
        torch.testing.assert_close(args[0], main[0][[1, 2, 0]])
        torch.testing.assert_close(args[2][:, :32], index[0][[1, 2, 0]])
        torch.testing.assert_close(args[4][:, :32], index[2][[1, 2, 0]])
        self.assertEqual(args[2][:, 32:].count_nonzero(), 0)
        self.assertEqual(args[4][:, 32:].count_nonzero(), 0)
        self.assertEqual(args[5][:, 0, :3].tolist(), [[0, 1, -1], [0, -1, -1], [0, 1, 2]])
        self.assertEqual(args[10], (1, 3))
        self.assertEqual(args[11], (3, 6))
        self.assertEqual(invocation.transport.rounds, 3)
        for tensor, value in zip(result.gradients, (2.5, 3.5, 4.5)):
            torch.testing.assert_close(tensor, torch.full_like(tensor, value))
        self.assertEqual(result.gradients[0].shape, (3, 32, 128))
        self.assertEqual(result.key_partials.dtype, torch.float32)
        self.assertEqual(result.loss.item(), 5.5)

    def test_empty_queries_keep_true_zero_rows_and_zero_shared_key_partials(self):
        """An empty owner claims its epoch without fabricating any native query task."""
        invocation = self.backend.prepare(replace(self.meta, q_global_ids=()), self.selection[:0], epoch=1)
        result = self.backend.gradients(*self._inputs(0), invocation)
        self.ops.dsa_tp_local_mixed_kl_out.assert_not_called()
        self.ops.dsa_local_mixed_kl_out.assert_not_called()
        self.assertEqual(result.gradients[0].shape, (0, 32, 128))
        self.assertEqual(result.gradients[1].count_nonzero(), 0)
        self.assertEqual(result.key_partials.count_nonzero(), 0)
        self.assertEqual(result.loss.item(), 0)
        self.assertEqual(result.traces, ())

    def test_mutated_selection_and_missing_capability_fail_before_claim(self):
        """Mutable saved selection and incompatible payload cannot enter communication."""
        invocation = self.backend.prepare(self.meta, self.selection, epoch=1)
        self.selection[0, 0] = 4
        with self.assertRaisesRegex(ValueError, 'modified'):
            self.backend.gradients(*self._inputs(), invocation)
        self.transport.lease.assert_not_called()
        invocation = self.backend.prepare(self.meta, self.selection, epoch=2)
        self.ops.dsa_tp_local_mixed_kl_version.return_value = 0
        with self.assertRaisesRegex(RuntimeError, 'version 1'):
            self.backend.gradients(*self._inputs(), invocation)
        self.transport.lease.assert_not_called()

    def test_reject_nondivisible_heads_incomplete_keys_and_peer_disagreement(self):
        """Invalid global geometry or different TP row declarations cannot publish data."""
        with self.assertRaisesRegex(ValueError, 'H32/H64'):
            TpLocalKlProbe(self.transport, global_heads=129, attention_scale=0.125, schedule=MixedSfaSchedule(7))
        with self.assertRaisesRegex(ValueError, 'globally ordered'):
            self.backend.prepare(replace(self.meta, kv_global_ids=(0, 1)), self.selection, epoch=1)
        with patch(self.prefix + '.dist.all_gather_object') as gather:
            gather.side_effect = lambda outputs, item, **_kwargs: outputs.__setitem__(slice(None), [item, ()])
            with self.assertRaisesRegex(ValueError, 'disagree'):
                self.backend.prepare(self.meta, self.selection, epoch=1)
        self.transport.prepare.assert_not_called()
