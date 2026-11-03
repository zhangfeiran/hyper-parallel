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

import unittest
from unittest.mock import patch

from hyper_parallel.core.expert_parallel.hot_replica import planner


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

    def test_rejected_preference_retries_average_with_same_capacity(self) -> None:
        """A reservation target must not hide a feasible admitted quota candidate."""
        counts = self.diffuse_counts()
        for budget in (2, 6):
            with self.subTest(budget=budget):
                options = dict(adaptive=True, minimum_replica_rows=1024, minimum_load_reduction=.4,
                               capacity_limit=196608, receive_capacity=196608)
                expected = planner._build_expert_replica_plan(counts, budget, target_load=24576, **options)
                with patch.object(planner, '_uncalibrated_copies', wraps=planner._uncalibrated_copies) as select:
                    actual = planner._build_expert_replica_plan(counts, budget, target_load=30720, **options)
                actual.validate()
                self.assertTrue(actual.transfers)
                self.assertEqual(actual, expected)
                self.assertEqual([call.args[5] for call in select.call_args_list], [30720, 24576])

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
