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
"""Preserve useful completed budgets without accepting additional token-traffic pressure."""

"""Validate bounded balancing of already finalized guest matrices."""

from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaConfig, host, planner


class TestReplicaRetainedBalance(unittest.TestCase):
    """Keep matrix traffic fixed while admitting only nonworsening compute and row traffic."""

    @staticmethod
    def plan(budget: int, **options: object) -> object:
        """Construct a legal distinct-TopK histogram with unused retained-guest headroom."""
        source = options.get('source', (4096, 4096, 0, 4096, 2048, 0, 0, 0, 256, 512, 512, 256, 0, 256, 0, 0))
        return planner._build_expert_replica_plan(
            (source,) * 4, budget, adaptive=bool(budget), target_load=5 * sum(source) // 4,
            capacity_limit=4 * sum(source), receive_capacity=4 * sum(source),
            minimum_replica_rows=options.get('minimum_replica_rows', 1024),
            minimum_load_reduction=options.get('minimum_load_reduction', .4),
            _rank_nodes=(0, 0, 1, 1), _replica_transfer_row_credit=1152)

    def test_fill_final_guests_without_adding_matrix_edges(self) -> None:
        """Description: Refill the guests retained after topology placement and consolidation.

        Expectation: The expert peak falls within the old receive bound and the same three guest matrices.
        Feature: Bounded final-layout balancing.
        """
        source = (512, 0, 4096, 0, 0, 0, 0, 0, 512, 0, 1024, 0, 2048, 1024, 1024, 4096)
        actual = self.plan(1, source=source)
        actual.validate()
        self.assertLessEqual(max(actual.destination_loads), 17920)
        self.assertEqual(len(actual.transfers), 3)
        self.assertLess(max(map(max, actual.destination_counts)), 14336)
        self.assertEqual(max(actual.replica_slots_used), 1)

    def test_rank_only_improvement_keeps_the_admitted_layout(self) -> None:
        """Description: Try to reduce receive load while the largest expert fragment stays unchanged.

        Expectation: Preserve the admitted layout when retained guests cannot reduce the expert peak.
        Feature: Critical expert admission.
        """
        actual = self.plan(1)
        actual.validate()
        self.assertEqual(max(actual.destination_loads), 20160)
        self.assertEqual(max(map(max, actual.destination_counts)), 16384)

    def test_source_traffic_cannot_pay_for_a_lower_compute_peak(self) -> None:
        """Description: Try exporting more work from a source that owns every routed row.

        Expectation: The lower compute peak is rejected because the worst source sends more rows.
        Feature: Independent traffic objectives.
        """
        copies = [{0: 10}, {1: 0, 0: 1}, {2: 0, 0: 1}]
        before = deepcopy(copies)
        actual = planner._rebalance_retained_copies(copies, ((12, 0, 0), (0, 0, 0), (0, 0, 0)), 1, (0, 1, 0))
        self.assertIs(actual, copies)
        self.assertEqual(copies, before)

    def test_fill_cannot_borrow_another_home_experts_rows(self) -> None:
        """Description: Exhaust one home expert while the owner also holds unrelated work.

        Expectation: Only the matching logical quota moves, and existing guest keys remain positive.
        Feature: Logical expert conservation.
        """
        copies = [{0: 8, 1: 7}, {2: 0, 3: 0, 0: 1}, {4: 0, 5: 0, 0: 1}]
        planner._fill_critical_guest_rows(copies, 2)
        self.assertEqual(copies[0], {0: 0, 1: 7})
        self.assertEqual([row.get(0, 0) for row in copies], [0, 5, 5])

    def test_improved_rank_intersections_can_reduce_node_traffic(self) -> None:
        """Description: Give the same hot expert equal demand on three source ranks.

        Expectation: Three balanced local intersections eliminate all remote rows without new copies.
        Feature: Exact source and node intersections.
        """
        copies = [{0: 10}, {1: 0, 0: 1}, {2: 0, 0: 1}]
        counts, nodes = ((4, 0, 0),) * 3, (0, 1, 0)
        actual = planner._rebalance_retained_copies(copies, counts, 1, nodes)
        self.assertEqual([sum(row.values()) for row in actual], [4, 4, 4])
        plan = planner._materialize(counts, actual, ExpertReplicaConfig(3, 3, 1), rank_nodes=nodes)
        plan.validate()
        self.assertTrue(all(count == 0 or physical // 2 == source
                            for source, row in enumerate(plan.dispatch_counts) for physical, count in enumerate(row)))

    def test_node_ingress_increase_rejects_rank_safe_refill(self) -> None:
        """Description: Lower the compute peak without increasing any rank traffic bound.

        Expectation: Topology-aware admission rejects a rise in peak node ingress from seven to eight rows.
        Feature: Retained replica traffic bounds.
        """
        copies = [{0: 8}, {1: 2, 0: 14}, {2: 5, 0: 8}, {3: 5, 0: 7}]
        proposal = [{0: 9}, {1: 2, 0: 10}, {2: 5, 0: 8}, {3: 5, 0: 10}]
        counts = ((8, 0, 1, 1), (13, 0, 1, 3), (4, 1, 1, 1), (12, 1, 2, 0))

        rank_only = planner._accept_retained_proposal(copies, proposal, counts, ())
        topology_aware = planner._accept_retained_proposal(copies, proposal, counts, (0, 1, 0, 2))

        self.assertIs(rank_only, proposal)
        self.assertIs(topology_aware, copies)

    def test_compatibility_thresholds_skip_final_balancing(self) -> None:
        """Description: Use zero-fragment and zero-admission compatibility policies.

        Expectation: The new final-layout search is not entered for either policy.
        Feature: Compatibility admission.
        """
        with patch.object(planner, '_rebalance_retained_copies', side_effect=AssertionError('extra refinement')):
            for options in ({'minimum_replica_rows': 0}, {'minimum_load_reduction': 0.0}):
                with self.subTest(options=options):
                    self.plan(6, **options).validate()

    def test_critical_guest_can_return_rows_to_existing_fragments(self) -> None:
        """Description: Return hot guest rows to the owner and another existing guest.

        Expectation: The hot fragment shortens without replacing its matrix or losing local rows.
        Feature: Fixed-matrix critical expert refill.
        """
        copies = [{0: 1}, {1: 0, 0: 11}, {2: 0, 0: 2}]
        counts = ((4, 0, 0), (4, 0, 0), (6, 0, 0))
        actual = planner._rebalance_retained_copies(copies, counts, 1, (0, 0, 1))
        self.assertEqual([row[0] for row in actual], [5, 5, 4])
        self.assertEqual([row[0] for row in copies], [1, 11, 2])

    def test_critical_guest_keeps_locally_sourced_rows(self) -> None:
        """Description: Offer to move a guest's locally sourced rows to existing peers.

        Expectation: Keep the current layout when no hot rows can move without increasing source traffic.
        Feature: Source-local refill constraints.
        """
        copies = [{0: 1}, {1: 0, 0: 11}, {2: 0, 0: 2}]
        counts = ((1, 0, 0), (11, 0, 0), (2, 0, 0))
        actual = planner._rebalance_retained_copies(copies, counts, 1, (0, 0, 1))
        self.assertIs(actual, copies)

    def test_critical_guest_does_not_shift_the_remote_ingress_peak(self) -> None:
        """Description: Consider a shorter guest with increased remote ingress at its owner.

        Expectation: Reject the proposal if the owner would exceed the original peak remote ingress.
        Feature: Rank ingress limit in a two-node topology.
        """
        copies = [{0: 2, 1: 5}, {2: 0, 3: 0, 0: 9}, {4: 7, 5: 0, 0: 1}]
        counts = ((0, 0, 0, 0, 0, 0), (3, 5, 0, 0, 0, 0), (9, 0, 0, 0, 7, 0))
        actual = planner._rebalance_retained_copies(copies, counts, 2, (0, 1, 0))
        self.assertIs(actual, copies)

    def test_owner_only_skips_all_quota_completion(self) -> None:
        """Description: Run the same histogram with a zero slot cap.

        Expectation: The original owner-only path retains no transfers and never rebalances guests.
        Feature: Owner-only execution.
        """
        with patch.object(planner, '_rebalance_retained_copies', side_effect=AssertionError('guest refinement')):
            actual = self.plan(0)
        actual.validate()
        self.assertFalse(actual.transfers)
        self.assertEqual(max(actual.replica_slots_used), 0)

    def test_native_locality_preserves_python_decisions_and_dictionary_order(self) -> None:
        """Description: Compare native integer moves with the Python placement reference.

        Expectation: Every guest move, swap and insertion order matches on independent histograms.
        Feature: Optional native locality primitive.
        """
        native = host._native_backend()
        if native is None or not hasattr(native, 'locality_moves'):
            self.skipTest('Optional locality primitive is not built')
        generator = random.Random(202611051)
        for budget in (1, 2, 6):
            config = ExpertReplicaConfig(24, 4, budget)
            for _ in range(12):
                hot = generator.randrange(24)
                counts = tuple(tuple(generator.randrange(25) + (400 if expert == hot else 0)
                                     for expert in range(24)) for _rank in range(4))
                totals, loads, _home = planner._home_layout(counts, config)
                copies = planner._bounded_copies(loads, totals, config)
                before = deepcopy(copies)
                actual = planner._locality_copies(copies, config, (0, 1, 0, 1))
                with patch.object(planner, '_locality_moves', return_value=None):
                    expected = planner._locality_copies(copies, config, (0, 1, 0, 1))
                self.assertEqual([list(row.items()) for row in actual], [list(row.items()) for row in expected])
                self.assertEqual(copies, before)

    def test_native_locality_retains_arbitrary_precision_fallback(self) -> None:
        """Description: Supply a logical row count outside the native signed range.

        Expectation: The optional primitive declines without changing the Python input or error state.
        Feature: Exact integer fallback.
        """
        native = host._native_backend()
        if native is None or not hasattr(native, 'locality_moves'):
            self.skipTest('Optional locality primitive is not built')
        copies = [{0: 2**70}, {1: 0, 0: 1}]
        self.assertIsNone(native.locality_moves(copies, 1, 1, (0, 1)))
        actual = planner._locality_copies(copies, ExpertReplicaConfig(2, 2, 1), (0, 1))
        self.assertEqual(actual, copies)
