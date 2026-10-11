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
"""CPU contracts for native TP LI packing, global IDs and immutable prepared metadata."""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.tp_indexer import (
    TpLocalIndexerProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumInvocation,
    TpTileSumProbe,
    TpTileSumSpec,
)


class TestTpIndexer(unittest.TestCase):
    """Mock only transport/native execution; exercise real row/head preparation."""

    def setUp(self) -> None:
        """Prepare two TP members and one CP owner with reordered packed query rows."""
        self.prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.tp_indexer'
        self.transport = Mock(spec=TpTileSumProbe)
        self.transport.spec = TpTileSumSpec(14, 512)
        self.transport.members = (0, 1)
        self.transport.participates = True
        self.transport.group = object()
        self.transport.arena = torch.zeros(self.transport.spec.arena_bytes, dtype=torch.uint8)
        self.transport.root = Mock(generation=1)
        self.transport.lease.return_value = nullcontext()
        self.transport.prepare.side_effect = self._transport_prepare
        self.backend = TpLocalIndexerProbe(self.transport, schedule=MixedSfaSchedule(7))
        self.meta = replace(DsaBatchMeta.packed((3, 3)), q_global_ids=(5, 1, 3), heap_generation=1)
        gather = patch(self.prefix + '.dist.all_gather_object')
        self.gather = gather.start()
        self.addCleanup(gather.stop)
        self.gather.side_effect = lambda outputs, value, **_kwargs: outputs.__setitem__(slice(None), [value, value])
        loader = patch(self.prefix + '._load_native')
        loader.start()
        self.addCleanup(loader.stop)
        ops = patch(self.prefix + '.torch.ops.hyper_parallel', Mock())
        self.ops = ops.start()
        self.addCleanup(ops.stop)
        self.ops.dsa_tp_local_indexer_version.return_value = 1
        self.ops.dsa_tp_local_indexer_out.side_effect = self._native

    def _transport_prepare(self, *, rounds, epoch, identity):
        words = self.transport.spec.metadata(rounds=rounds, epoch=epoch, rank=0, root_pes=(0, 1), root_size=2,
                                             identity=identity)
        metadata = torch.tensor(words, dtype=torch.int64)
        return TpTileSumInvocation(self.transport, rounds, epoch, 1, metadata, metadata._version)

    def _native(self, query, key, weights, query_lengths, key_lengths, config, trace, retained, positions,
                arena, metadata, indices, values):
        self.assertEqual(query.shape[1:], (64, 128))
        self.assertEqual(key.shape, (6, 1, 128))
        self.assertEqual(config.tolist(), [0x48504453414D4958, 1, 7, 1])
        self.assertIs(arena, self.transport.arena)
        self.assertEqual(metadata.numel(), 24)
        self.assertEqual(trace.shape, (2, 20, 64))
        self.assertEqual(query[:, 32:].count_nonzero(), 0)
        self.assertEqual(weights[:, 32:].count_nonzero(), 0)
        self.assertEqual(retained.numel(), 47226880 if len(query) else 0)
        self.assertEqual(key_lengths.tolist(), [3, 6])
        self.assertEqual(query_lengths.tolist(), [1, 3] if len(query) else [0, 0])
        for row, position in enumerate(positions.tolist()):
            indices[row, 0, :position + 1] = torch.arange(position + 1)
            values[row, 0, :position + 1] = 1
        return indices, values, trace, retained

    def _inputs(self, rows=3):
        return (torch.arange(rows * 32 * 128).reshape(rows, 32, 128).bfloat16(),
                torch.randn(6, 128, dtype=torch.bfloat16), torch.full((rows, 32), -2.5, dtype=torch.float32))

    def test_pack_reordered_queries_and_restore_global_packed_ids(self):
        """Sequence-relative native IDs become global IDs before restoring caller order."""
        invocation = self.backend.prepare(self.meta, epoch=1)
        result = self.backend.forward(*self._inputs(), invocation)
        self.assertEqual(invocation.query_lengths.tolist(), [1, 3])
        self.assertEqual(invocation.positions.tolist(), [1, 0, 2])
        self.assertEqual(invocation.sequence_starts.flatten().tolist(), [0, 3, 3])
        self.assertEqual(result.global_indices[0, :4].tolist(), [3, 4, 5, -1])
        self.assertEqual(result.global_indices[1, :3].tolist(), [0, 1, -1])
        self.assertEqual(result.global_indices[2, :2].tolist(), [3, -1])
        query, _key, weights = self.ops.dsa_tp_local_indexer_out.call_args.args[:3]
        torch.testing.assert_close(query[:, :32], self._inputs()[0][[1, 2, 0]])
        torch.testing.assert_close(weights[:, :32], torch.full((3, 32), -2.5))
        self.assertIs(result._runtime_buffers[-1], self.backend.retained)
        self.transport.lease.assert_called_once_with(invocation.transport)

    def test_empty_owner_still_submits_complete_phase_protocol(self):
        """Empty queries retain genuine zero cumulative lengths and no fabricated query row."""
        invocation = self.backend.prepare(replace(self.meta, q_global_ids=()), epoch=1)
        result = self.backend.forward(*self._inputs(rows=0), invocation)
        self.assertEqual(invocation.query_lengths.tolist(), [0, 0])
        self.assertEqual(result.global_indices.shape, (0, 2048))
        self.ops.dsa_tp_local_indexer_out.assert_called_once()
        self.assertEqual(invocation.transport.rounds, 1)

    def test_reject_incomplete_kv_metadata_mutation_and_missing_capability(self):
        """Invalid setup and unsupported payloads fail before entering native protocol."""
        with self.assertRaisesRegex(ValueError, 'globally ordered'):
            self.backend.prepare(replace(self.meta, kv_global_ids=(0, 1)), epoch=1)
        invocation = self.backend.prepare(self.meta, epoch=1)
        invocation.positions[0] = 0
        with self.assertRaisesRegex(ValueError, 'modified'):
            self.backend.forward(*self._inputs(), invocation)
        self.transport.lease.assert_not_called()
        invocation = self.backend.prepare(self.meta, epoch=2)
        self.ops.dsa_tp_local_indexer_version.return_value = 0
        with self.assertRaisesRegex(RuntimeError, 'version 1'):
            self.backend.forward(*self._inputs(), invocation)
        self.transport.lease.assert_not_called()

    def test_tp_peers_must_agree_on_ordered_rows(self):
        """Different CP group ranks are allowed, but differing logical query rows are rejected."""
        self.gather.side_effect = lambda outputs, value, **_kwargs: outputs.__setitem__(
            slice(None), [value, ('different query order',)])
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.backend.prepare(self.meta, epoch=1)
        self.transport.prepare.assert_not_called()
