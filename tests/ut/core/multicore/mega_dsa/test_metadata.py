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
"""Logical metadata contract tests, without real process groups."""

import unittest
from dataclasses import replace

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)


class TestDsaBatchMeta(unittest.TestCase):
    """Validate packed addresses and explicit CP/root mappings."""

    def test_packed_positions_do_not_require_tile_alignment(self):
        """Three and two valid tokens are accepted without MoE alignment."""
        meta = DsaBatchMeta.packed((3, 2))
        self.assertEqual(meta.global_cu_seqlens, (0, 3, 5))
        self.assertEqual([meta.sequence_position(token) for token in meta.q_global_ids],
                         [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])

    def test_noncontiguous_group_and_zigzag_storage(self):
        """Storage, owner IDs, WORLD ranks and root PEs remain distinct."""
        meta = DsaBatchMeta((0, 6), (5, 0), (4, 1, 5, 0, 3, 2),
                            (0, 1, 1, 1, 1, 0), (1, 1, 3, 2, 0, 0),
                            cp_ranks=(7, 2), root_pes=(3, 0), cp_rank=0, layout_id="zigzag")
        self.assertEqual(meta.sequence_position(meta.q_global_ids[0]), (0, 5))
        self.assertEqual(meta.token_local_offsets[5], 0)
        self.assertEqual(meta.cp_ranks[meta.token_owners[5]], 7)
        self.assertEqual(meta.root_pes[meta.token_owners[5]], 3)

    def test_empty_local_shard_is_valid(self):
        """A CP rank may own no valid queries or keys."""
        meta = DsaBatchMeta((0, 2), (), (), (1, 1), (0, 1), cp_ranks=(0, 1), root_pes=(0, 1))
        self.assertEqual(meta.global_valid_queries, 2)

    def test_invalid_metadata_rejected(self):
        """Reject malformed namespace, topology and owner addresses early."""
        meta = DsaBatchMeta.packed((3,))
        cases = [{"global_cu_seqlens": (1, 3)}, {"global_cu_seqlens": (0, 0, 3)},
                 {"q_global_ids": (0, 0)}, {"kv_global_ids": (-1,)}, {"cp_ranks": (0, 0)},
                 {"root_pes": (0, 1)}, {"cp_rank": 1}, {"token_owners": (0, 0, 1)},
                 {"token_local_offsets": (0, 0, 2)}, {"token_local_offsets": (0, 1, 3)},
                 {"invocation": -1}, {"layout_id": ""}, {"index_namespace": "buffer_offset"},
                 {"q_global_ids": [0, 1, 2]}]
        for update in cases:
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(meta, **update)

    def test_remote_queries_rejected(self):
        """A local query must belong to the declared CP member."""
        with self.assertRaisesRegex(ValueError, "local queries"):
            DsaBatchMeta((0, 2), (1,), (0, 1), (0, 1), (0, 0), cp_ranks=(0, 1), root_pes=(0, 1))

    def test_cache_identity_changes_with_invocation_or_generation(self):
        """Identical token IDs do not allow reuse across layers or invocations."""
        meta = DsaBatchMeta.packed((1,))
        for field in ("layer", "microbatch", "invocation", "heap_generation"):
            self.assertNotEqual(meta.cache_identity, replace(meta, **{field: 1}).cache_identity)
        self.assertNotEqual(meta.cache_identity, replace(meta, layout_id="reordered").cache_identity)

    def test_loss_normalization_rejects_invalid_counts(self):
        """Global objective counts and reducer divisor must be positive integers."""
        self.assertEqual(DsaLossNormalization(5, 2).local_sum_scale, 0.4)
        for count, divisor in ((0, 1), (5, 0), (-1, 2), (3.5, 1), (True, 1)):
            with self.subTest(count=count, divisor=divisor), self.assertRaises(ValueError):
                DsaLossNormalization(count, divisor)
