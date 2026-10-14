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
"""Conservation, address coverage and dedup round trips for the A3 MoonEP port."""

import unittest

import numpy as np

from hyper_parallel.core.expert_parallel.moonep_planning import (
    deduplicate_moonep, encode_moonep_routes, plan_moonep,
)


class TestMoonEPPlanning(unittest.TestCase):
    """Check routing semantics independently of NPU communication kernels."""

    def test_random_allocations_cover_exactly_every_destination_row(self):
        """Every routed token has one destination, including highly skewed inputs."""
        generator = np.random.default_rng(2026)
        for ranks in (2, 4, 16):
            experts, tokens, top_k = ranks * 6, 17, 8
            capacity = tokens * top_k
            for concentration in (0.005, 0.1, 10000):
                with self.subTest(ranks=ranks, concentration=concentration):
                    ids = np.stack([generator.choice(experts, (tokens, top_k),
                                                     p=generator.dirichlet(np.full(experts, concentration)))
                                    for _ in range(ranks)])
                    counts = np.stack([np.bincount(row.reshape(-1), minlength=experts) for row in ids])
                    placement = plan_moonep(counts, capacity)
                    packed = np.stack([encode_moonep_routes(ids[rank], counts, placement, rank, capacity)
                                       for rank in range(ranks)])
                    np.testing.assert_array_equal(np.sort(packed.reshape(-1)), np.arange(ranks * capacity))
                    np.testing.assert_array_equal(placement.compact_counts.sum(1), np.full(ranks, capacity))
                    np.testing.assert_array_equal(placement.allocation.sum(1), counts.sum(0))

    def test_dedup_scatter_and_expand_preserve_all_payloads(self):
        """Sending one hidden vector per destination reproduces dense dispatch."""
        ranks, tokens, top_k, experts = 4, 8, 4, 8
        capacity = tokens * top_k
        ids = np.tile(np.array([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=np.int32), (ranks, tokens // 2, 1))
        counts = np.stack([np.bincount(row.reshape(-1), minlength=experts) for row in ids])
        placement = plan_moonep(counts, capacity)
        raw = np.stack([encode_moonep_routes(ids[rank], counts, placement, rank, capacity) for rank in range(ranks)])
        dense = np.empty(ranks * capacity, dtype=np.int32)
        sparse = np.full_like(dense, -1)
        plans = [deduplicate_moonep(raw, rank, top_k, capacity) for rank in range(ranks)]
        for source, (encoded, _, _, _) in enumerate(plans):
            payload = np.repeat(np.arange(tokens) + source * tokens, top_k)
            dense[raw[source]] = payload
            sparse[encoded[encoded >= 0]] = payload[encoded >= 0]
            decoded = np.where(encoded < 0, -encoded - 1, encoded)
            np.testing.assert_array_equal(decoded, raw[source])
        for destination, (_, groups, loffs, lengths) in enumerate(plans):
            for primary, start, length in groups[:lengths[0]]:
                target = destination * capacity + loffs[start:start + length]
                sparse[target] = sparse[destination * capacity + primary]
        np.testing.assert_array_equal(sparse, dense)

    def test_exactly_balanced_routes_need_no_remote_weights(self):
        """An exact home balance preserves all local expert rows."""
        counts = np.full((4, 8), 2, dtype=np.int64)
        placement = plan_moonep(counts, 16)
        np.testing.assert_array_equal(placement.experts_to_copy, np.full((4, 2), -1))
        np.testing.assert_array_equal(placement.compact_counts[:, 2:], np.zeros((4, 2)))

    def test_invalid_counts_and_routes_fail_before_communication(self):
        """Invalid source sizes and histogram mismatches are rejected."""
        with self.assertRaisesRegex(ValueError, "S\\*K"):
            plan_moonep(np.array([[1, 1], [0, 1]]), 2)
        counts = np.array([[2, 0], [2, 0]])
        placement = plan_moonep(counts, 2)
        with self.assertRaisesRegex(ValueError, "histogram"):
            encode_moonep_routes(np.array([0, 1]), counts, placement, 0, 2)


if __name__ == "__main__":
    unittest.main()
