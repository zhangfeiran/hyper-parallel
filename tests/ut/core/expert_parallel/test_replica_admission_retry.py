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
"""Retry otherwise rejected preference targets without changing accepted placements."""

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from hyper_parallel.core.expert_parallel.hot_replica import planner
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel


class TestReplicaAdmissionRetry(unittest.TestCase):
    """Keep allocation bounds independent of the optional load-reduction target."""

    @staticmethod
    def diffuse_counts() -> list[list[int]]:
        """Represent 4096 distinct TopK6 routes per source with a diffuse hot owner."""
        owner = 5529
        remaining = 4096 * 6 - owner
        totals = [owner] + [remaining // 7 + int(rank < remaining % 7) for rank in range(7)]
        counts = [rows // 6 + int(expert < rows % 6) for rows in totals for expert in range(6)]
        return [list(counts) for _ in range(8)]

    def test_uncalibrated_preference_keeps_owner_without_retry(self) -> None:
        """A smaller receive peak alone does not justify the optional retry."""
        counts = self.diffuse_counts()
        for budget in (2, 6):
            with self.subTest(budget=budget):
                options = dict(adaptive=True, minimum_replica_rows=1024, minimum_load_reduction=.4,
                               capacity_limit=196608, receive_capacity=196608)
                expected = planner._build_expert_replica_plan(counts, budget, target_load=24576, **options)
                with patch.object(planner, '_uncalibrated_copies', wraps=planner._uncalibrated_copies) as select:
                    actual = planner._build_expert_replica_plan(counts, budget, target_load=30720, **options)
                actual.validate()
                self.assertTrue(expected.transfers)
                self.assertFalse(actual.transfers)
                self.assertEqual([call.args[5] for call in select.call_args_list], [30720])

    def test_calibrated_retry_must_cover_search_and_uncertainty(self) -> None:
        """A paired group estimate must cover both fixed replica and extra search costs."""
        counts = self.diffuse_counts()
        model = ExpertReplicaCostModel(
            calibration_version=2, backend='push', hidden_size=128, intermediate_size=256,
            ep_size=8, transport='p2p', forward_ms=((0, 0.), (50000, .05)),
            backward_ms=((0, 0.), (50000, .05)), weight_ms=0., gradient_ms=0.,
            effective_group_ms=(0., .001, 0., 0., 0., 16.), _rank_nodes=(0, 0, 0, 0, 1, 1, 1, 1),
            minimum_gain_ms=1., quota_search_budget_ms=2.,
        )
        options = dict(adaptive=True, target_load=30720, minimum_replica_rows=1024, minimum_load_reduction=.4,
                       capacity_limit=196608, receive_capacity=196608)

        admitted = planner._build_expert_replica_plan(counts, 2, cost_model=model, **options)
        admitted.validate()
        self.assertTrue(admitted.transfers)
        self.assertEqual(max(admitted.destination_loads), 24576)
        for budget in (0., 3.):
            with self.subTest(search_budget_ms=budget):
                rejected = planner._build_expert_replica_plan(
                    counts, 2, cost_model=replace(model, quota_search_budget_ms=budget), **options)
                rejected.validate()
                self.assertFalse(rejected.transfers)

    def test_profitable_preference_keeps_first_candidate(self) -> None:
        """Existing concentrated-hot choices must not pay for a second search."""
        counts = [[4096] * 6 + [0] * 42 for _ in range(8)]
        for budget in (2, 6):
            with self.subTest(budget=budget):
                with patch.object(planner, '_uncalibrated_copies', wraps=planner._uncalibrated_copies) as select:
                    actual = planner._build_expert_replica_plan(
                        counts, budget, adaptive=True, target_load=30720, minimum_replica_rows=1024,
                        minimum_load_reduction=.4, capacity_limit=196608, receive_capacity=196608)
                actual.validate()
                self.assertTrue(actual.transfers)
                self.assertEqual(select.call_count, 1)

    def test_impossible_fraction_keeps_owner_without_retry(self) -> None:
        """An impossible admission bound must remain a cheap owner-only decision."""
        with patch.object(planner, '_uncalibrated_copies', wraps=planner._uncalibrated_copies) as select:
            actual = planner._build_expert_replica_plan(
                self.diffuse_counts(), 6, adaptive=True, target_load=30720,
                minimum_replica_rows=1024, minimum_load_reduction=.8,
                capacity_limit=196608, receive_capacity=196608)
        self.assertFalse(actual.transfers)
        self.assertEqual(select.call_count, 0)

    def test_required_capacity_does_not_use_optional_retry(self) -> None:
        """Mandatory replicas remain valid even if optional gain is impossible."""
        counts = [[4096] * 6 + [0] * 42 for _ in range(8)]
        with patch.object(planner, '_uncalibrated_copies', wraps=planner._uncalibrated_copies) as select:
            actual = planner._build_expert_replica_plan(
                counts, 2, adaptive=True, target_load=30720, minimum_replica_rows=1000000,
                minimum_load_reduction=.99, capacity_limit=49152, receive_capacity=49152)
        actual.validate()
        self.assertTrue(actual.transfers)
        self.assertLessEqual(max(actual.destination_loads), 49152)
        self.assertEqual(select.call_count, 1)

    def test_complete_candidate_fits_two_physical_guests_and_keeps_all_routes(self) -> None:
        """A full redistribution that fits every receiver should not be quota-thinned."""
        source = [4096, 0, 9, 97, 0, 0, 4096, 0, 4096, 4096, 0, 0,
                  0, 0, 1200, 0, 0, 0, 0, 3354, 0, 11, 0, 82,
                  0, 0, 1, 0, 0, 0, 224, 0, 7, 2993, 0, 0,
                  0, 0, 0, 0, 0, 208, 0, 0, 1, 5, 0, 0]
        counts = [source] * 8
        options = dict(adaptive=True, target_load=30720, minimum_replica_rows=1024,
                       minimum_load_reduction=.4, capacity_limit=196608, receive_capacity=196608)
        with patch.dict(os.environ, {"HYPER_PARALLEL_REPLICA_COMPLETE_GUEST_SEARCH": "0"}):
            baseline = planner._build_expert_replica_plan(counts, 2, **options)
        with patch.dict(os.environ, {"HYPER_PARALLEL_REPLICA_COMPLETE_GUEST_SEARCH": "1"}):
            candidate = planner._build_expert_replica_plan(counts, 2, **options)
        candidate.validate()
        self.assertEqual(candidate.config.replica_slots_per_rank, 2)
        self.assertLessEqual(max(candidate.replica_slots_used), 2)
        self.assertEqual(max(candidate.destination_loads), 24576)
        self.assertGreater(max(baseline.destination_loads), 30720)
