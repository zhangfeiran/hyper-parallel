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
"""CPU checks for owner-ID reconstruction and duplicate TP/CP diagnostic replicas."""

import copy
import unittest

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_boundary_snapshot import (
    merge_boundary_parameters,
    merge_boundary_rows,
)


class TestBoundarySnapshot(unittest.TestCase):
    """Diagnostic reconstruction must not average or multiply replicated rows/weights."""

    @staticmethod
    def _fixture():
        query = torch.arange(13*4*3, dtype=torch.float32).reshape(13, 4, 3)
        compressed = torch.arange(13*3, dtype=torch.float32).reshape(13, 3)
        hidden = torch.arange(26, dtype=torch.float32).reshape(1, 13, 2)
        weights = {'row.weight': torch.arange(12, dtype=torch.float32).reshape(4, 3),
                   'column.weight': torch.arange(8, dtype=torch.float32).reshape(2, 4),
                   'shared.weight': torch.arange(6, dtype=torch.float32).reshape(2, 3)}
        peers = []
        for tp_rank in (0, 1):
            for cp_rank in (0, 1):
                ids = list(reversed(range(cp_rank, 13, 2)))
                inputs = {'query': query[ids, tp_rank*2:(tp_rank+1)*2], 'compressed': compressed[ids]}
                peers.append({'owner_ids': ids, 'tp_rank': tp_rank, 'tp_members': (0, 2),
                              'global_heads': 4, 'native': {'inputs': inputs},
                              'initial_state': weights, 'hidden': torch.zeros_like(hidden),
                              'hidden_gradient': hidden[:, ids],
                              'shards': (('row.weight', 0, (4, 3)), ('column.weight', 1, (2, 4))),
                              'parameter_gradients': {'row.weight': weights['row.weight'].chunk(2, 0)[tp_rank],
                                                      'column.weight': weights['column.weight'].chunk(2, 1)[tp_rank],
                                                      'shared.weight': weights['shared.weight']}})
        return peers, query, compressed, hidden, weights

    def test_reordered_owner_rows_and_head_shards_restore_global_order(self):
        """Interleaved reversed owners and row-parallel heads reconstruct unique global coordinates."""
        peers, query, compressed, _hidden, _weights = self._fixture()
        replicas = []
        actual = merge_boundary_rows(list(reversed(peers)), 'inputs', 'query', replicas)
        torch.testing.assert_close(actual, query, rtol=0, atol=0)
        self.assertEqual(replicas, [])
        actual = merge_boundary_rows(peers, 'inputs', 'compressed', replicas)
        torch.testing.assert_close(actual, compressed, rtol=0, atol=0)
        self.assertEqual(len(replicas), 2)
        self.assertTrue(all(item['exact'] for item in replicas))

    def test_empty_owner_is_valid_and_uncovered_owner_is_rejected(self):
        """A zero-row peer adds no data; missing global tokens must fail instead of silently staying zero."""
        peers, query, _compressed, _hidden, _weights = self._fixture()
        for peer in peers:
            ids = [] if peer['owner_ids'][0] % 2 else list(reversed(range(13)))
            peer['owner_ids'] = ids
            rank = peer['tp_rank']
            peer['native']['inputs']['query'] = query[ids, rank*2:(rank+1)*2]
        actual = merge_boundary_rows(peers, 'inputs', 'query', [])
        torch.testing.assert_close(actual, query, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'do not cover'):
            merge_boundary_rows(peers[1:], 'inputs', 'query', [])

    def test_duplicate_replica_difference_is_recorded_without_double_counting(self):
        """Keep the first deterministic representative and make divergent replicated values visible."""
        peers, _query, compressed, _hidden, _weights = self._fixture()
        peers = copy.deepcopy(peers)
        peers[2]['native']['inputs']['compressed'] += 1
        replicas = []
        actual = merge_boundary_rows(peers, 'inputs', 'compressed', replicas)
        torch.testing.assert_close(actual, compressed, rtol=0, atol=0)
        self.assertFalse(replicas[0]['exact'])
        self.assertFalse(replicas[0]['metric']['passed'])

    def test_parameter_shards_and_hidden_rows_are_not_summed_again(self):
        """CP gradients are already reduced; reconstruct row/column shards and shared weights exactly once."""
        peers, _query, _compressed, hidden, weights = self._fixture()
        replicas = []
        actual = merge_boundary_parameters(peers, replicas)
        for name, value in {'hidden': hidden, **weights}.items():
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        self.assertEqual(len(replicas), 7)
        self.assertTrue(all(item['exact'] for item in replicas))
        with self.assertRaisesRegex(ValueError, 'every shard'):
            merge_boundary_parameters(peers[:2], [])
