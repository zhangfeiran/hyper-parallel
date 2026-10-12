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
"""CPU contracts for native TP owner reduction and the final projection-boundary cast."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.tp_owner_return import (
    TpOwnerGradientReturnProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumInvocation,
    TpTileSumProbe,
    TpTileSumSpec,
)


class TestTpOwnerReturn(unittest.TestCase):
    """Mock transport while retaining real owner ordering, padding and dtype conversion."""

    def setUp(self) -> None:
        """Prepare reordered owner-local rows and a single bounded transport slot."""
        self.prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.tp_owner_return'
        self.transport = Mock(spec=TpTileSumProbe)
        self.transport.spec = TpTileSumSpec(1, 512)
        self.transport.participates = True
        self.transport.members = (0, 2)
        self.transport.group = object()
        self.transport.arena = torch.zeros(self.transport.spec.arena_bytes, dtype=torch.uint8)
        self.transport.prepare.side_effect = self._prepare
        self.backend = TpOwnerGradientReturnProbe(self.transport)
        self.meta = replace(DsaBatchMeta.packed((3,)), kv_global_ids=(2, 0, 1), heap_generation=1)
        self.words = 3*(512+64+128)
        self.transport.run.side_effect = self._run
        for name in ('dist.all_gather_object', '_load_native', 'torch.ops.hyper_parallel'):
            patcher = patch(self.prefix + '.' + name, Mock())
            value = patcher.start()
            self.addCleanup(patcher.stop)
            if name == 'dist.all_gather_object':
                value.side_effect = lambda output, signature, **_kwargs: output.__setitem__(
                    slice(None), [signature, signature])
            elif name == 'torch.ops.hyper_parallel':
                value.dsa_tp_tile_sum_version.return_value = 1
                self.ops = value

    def _prepare(self, *, rounds, epoch, identity):
        words = self.transport.spec.metadata(rounds=rounds, epoch=epoch, rank=0, root_pes=(0, 2), root_size=4,
                                             identity=identity)
        metadata = torch.tensor(words, dtype=torch.int64)
        return TpTileSumInvocation(self.transport, rounds, epoch, 1, metadata, metadata._version)

    def _run(self, partial, invocation):
        self.assertEqual(partial.shape, (invocation.rounds, 1, 512))
        self.assertEqual(partial.reshape(-1)[self.words:].count_nonzero(), 0)
        result = partial.clone()
        result.reshape(-1)[:self.words] -= 1
        return result

    @staticmethod
    def _inputs():
        return tuple(torch.full((3, width), value) for width, value in zip((512, 64, 128), (1.003, 2.007, 3.011)))

    def test_sum_fp32_before_single_cast_preserves_cancellation_and_owner_order(self):
        """A small residual lost by premature BF16 cast survives the native owner SUM."""
        invocation = self.backend.prepare(self.meta, epoch=1)
        partials = self._inputs()
        originals = tuple(value.clone() for value in partials)
        result = self.backend.return_owner(partials, invocation, input_dtypes=(torch.bfloat16,)*3)
        self.assertEqual(invocation.owner_ids, (2, 0, 1))
        self.assertEqual(invocation.transport.rounds, 5)
        for source, original, summed, final in zip(partials, originals, result.owner_fp32, result.gradients):
            torch.testing.assert_close(source, original)
            torch.testing.assert_close(summed, source-1, rtol=0, atol=0)
            torch.testing.assert_close(final, (source-1).bfloat16(), rtol=0, atol=0)
            self.assertNotEqual(source.untyped_storage().data_ptr(), summed.untyped_storage().data_ptr())
        self.assertGreater(result.gradients[0].abs().min().item(), 0)
        self.assertEqual((partials[0].bfloat16()-1).count_nonzero(), 0)

    def test_absent_fields_and_empty_owner_keep_declared_shapes(self):
        """Branch absence and genuine empty owner rows retain matching collective geometry."""
        invocation = self.backend.prepare(self.meta, epoch=1, presence=(False, False, True))
        self.words = 3*128
        result = self.backend.return_owner((None, None, self._inputs()[2]), invocation,
                                          input_dtypes=(torch.bfloat16, torch.bfloat16, torch.float32))
        self.assertEqual(result.owner_fp32[:2], (None, None))
        self.assertEqual(invocation.transport.rounds, 1)
        self.assertEqual(result.gradients[2].shape, (3, 128))
        empty = replace(self.meta, kv_global_ids=())
        invocation = self.backend.prepare(empty, epoch=2)
        self.words = 0
        result = self.backend.return_owner(tuple(torch.empty(0, width) for width in (512, 64, 128)), invocation,
                                          input_dtypes=(torch.bfloat16,)*3)
        self.assertEqual([tensor.shape for tensor in result.gradients], [(0, 512), (0, 64), (0, 128)])
        self.assertEqual(invocation.transport.rounds, 1)

    def test_reject_precast_gradients_changed_presence_and_missing_capability(self):
        """Invalid dtype, branch or ABI fails before native publication."""
        invocation = self.backend.prepare(self.meta, epoch=1)
        partials = self._inputs()
        with self.assertRaisesRegex(ValueError, 'detached FP32'):
            self.backend.return_owner((partials[0].bfloat16(), *partials[1:]), invocation,
                                      input_dtypes=(torch.bfloat16,)*3)
        with self.assertRaisesRegex(ValueError, 'presence'):
            self.backend.return_owner((None, *partials[1:]), invocation, input_dtypes=(torch.bfloat16,)*3)
        self.ops.dsa_tp_tile_sum_version.return_value = 0
        with self.assertRaisesRegex(RuntimeError, 'version 1'):
            self.backend.return_owner(partials, invocation, input_dtypes=(torch.bfloat16,)*3)
        self.transport.run.assert_not_called()

    def test_peers_must_agree_on_owner_rows_and_branches(self):
        """Different TP owner storage order or branch mask cannot enter the transport protocol."""
        with patch(self.prefix + '.dist.all_gather_object') as gather:
            gather.side_effect = lambda outputs, signature, **_kwargs: outputs.__setitem__(
                slice(None), [signature, ('different owner order',)])
            with self.assertRaisesRegex(ValueError, 'disagree'):
                self.backend.prepare(self.meta, epoch=1)
        self.transport.prepare.assert_not_called()
