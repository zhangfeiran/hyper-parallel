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
"""Unit tests for bounded hot-expert placement."""

import unittest

from hyper_parallel.core.multicore.modules.mega_moe.placement import (
    ExpertPlacementPlan,
    build_constrained_expert_placement,
    remap_logical_routes,
)


class TestExpertPlacementPlan(unittest.TestCase):
    """Validate the framework-independent MoonEP-style control plane."""

    def test_zero_replica_slots_preserve_logical_expert_ids(self):
        """
        Feature: zero-sized replica pool
        Description: plan a route with no physical replica slots
        Expectation: preserve logical IDs, counts and destination loads exactly
        """
        counts = ((2, 1, 0, 1), (0, 1, 2, 1))

        plan = build_constrained_expert_placement(counts, 0)

        self.assertEqual(plan.slot_to_logical_expert, ((0, 1), (2, 3)))
        self.assertEqual(plan.dispatch_counts, counts)
        self.assertEqual(plan.physical_num_experts, 4)
        self.assertEqual(plan.destination_loads, plan.baseline_destination_loads)
        self.assertEqual(
            remap_logical_routes((0, 0, 1, 3), 0, plan),
            (0, 0, 1, 3),
        )

    def test_one_replica_turns_hot_route_into_one_wave(self):
        """
        Feature: capacity-aware hot-expert replication
        Description: split one hot expert into an otherwise cold rank
        Expectation: reduce a two-wave home route to one exact wave
        """
        counts = ((4, 0, 0, 0), (4, 0, 0, 0))

        plan = build_constrained_expert_placement(
            counts,
            1,
            receive_capacity=4,
        )

        self.assertEqual(plan.replica_expert_ids, ((-1,), (0,)))
        self.assertEqual(plan.baseline_destination_loads, (8, 0))
        self.assertEqual(plan.destination_loads, (4, 4))
        self.assertEqual(plan.physical_expert_loads, (4, 0, 0, 0, 0, 4))
        self.assertEqual(plan.group_lists_by_rank, ((4, 4, 4), (0, 0, 4)))
        self.assertEqual(plan.replica_gradient_pairs, ((5, 0),))
        self.assertEqual(
            tuple(
                (
                    transfer.logical_expert,
                    transfer.owner_rank,
                    transfer.owner_slot,
                    transfer.target_rank,
                    transfer.target_slot,
                )
                for transfer in plan.replica_transfers
            ),
            ((0, 0, 0, 1, 2),),
        )
        self.assertEqual(plan.baseline_wave_count(4), 2)
        self.assertEqual(plan.wave_count(4), 1)
        physical_routes = (
            remap_logical_routes((0, 0, 0, 0), 0, plan)
            + remap_logical_routes((0, 0, 0, 0), 1, plan)
        )
        self.assertEqual(physical_routes.count(0), 4)
        self.assertEqual(physical_routes.count(5), 4)

    def test_residual_skew_is_represented_as_multiple_waves(self):
        """
        Feature: bounded replica pool with lossless fallback
        Description: plan an extreme hot expert with too few physical copies
        Expectation: reduce skew while retaining an exact residual wave count
        """
        counts = tuple((20, 0, 0, 0, 0, 0, 0, 0) for _ in range(4))

        plan = build_constrained_expert_placement(
            counts,
            1,
            receive_capacity=10,
        )

        self.assertEqual(plan.baseline_destination_loads, (80, 0, 0, 0))
        self.assertEqual(plan.destination_loads, (20, 20, 20, 20))
        self.assertEqual(plan.baseline_wave_count(10), 8)
        self.assertEqual(plan.wave_count(10), 2)
        self.assertEqual(sum(plan.destination_loads), 80)

    def test_replica_split_targets_global_capacity_not_pairwise_midpoint(self):
        """
        Feature: exact multi-rank balancing target
        Description: split one hot expert across three ranks with one slot each
        Expectation: fill both cold ranks to the one-wave capacity exactly
        """
        counts = ((0, 0, 6), (0, 0, 0), (0, 0, 0))

        plan = build_constrained_expert_placement(
            counts,
            1,
            receive_capacity=2,
        )

        self.assertEqual(plan.destination_loads, (2, 2, 2))
        self.assertEqual(plan.wave_count(2), 1)
        self.assertEqual(plan.replica_expert_ids, ((2,), (2,), (-1,)))

    def test_chained_group_transfer_preserves_one_remote_group_bound(self):
        """
        Feature: MoonEP-style chained balancing
        Description: balance two warm home groups into one initially empty rank
        Expectation: reach equal load with one remote expert slot per rank
        """
        plan = build_constrained_expert_placement(
            ((0, 3, 3), (0, 0, 0), (0, 0, 0)),
            1,
            receive_capacity=2,
        )

        self.assertEqual(plan.destination_loads, (2, 2, 2))
        self.assertEqual(plan.wave_count(2), 1)
        self.assertTrue(
            all(len(experts) == 1 for experts in plan.replica_expert_ids)
        )

    def test_capacity_headroom_moves_only_overflow_rows(self):
        """
        Feature: capacity-aware movement volume
        Description: balance a route whose capacity exceeds its average load
        Expectation: migrate only rows above capacity instead of filling a cold rank
        """
        plan = build_constrained_expert_placement(
            ((7, 0), (0, 1)),
            1,
            receive_capacity=6,
        )

        self.assertEqual(plan.baseline_destination_loads, (7, 1))
        self.assertEqual(plan.destination_loads, (6, 2))
        self.assertEqual(plan.replica_expert_ids, ((-1,), (0,)))

    def test_capacity_stop_leaves_uniform_slots_empty(self):
        """
        Feature: capacity-aware early stop
        Description: provide more replica slots than one-wave balance requires
        Expectation: retain a uniform shape and mark unused suffix slots empty
        """
        plan = build_constrained_expert_placement(
            ((4, 0, 0, 0), (4, 0, 0, 0)),
            3,
            receive_capacity=4,
        )

        self.assertEqual(plan.physical_experts_per_rank, 5)
        self.assertEqual(plan.replica_expert_ids[0], (-1, -1, -1))
        self.assertEqual(plan.replica_expert_ids[1], (0, -1, -1))
        unused_physical_ids = (2, 3, 4, 8, 9)
        for source_counts in plan.dispatch_counts:
            self.assertTrue(
                all(source_counts[index] == 0 for index in unused_physical_ids)
            )

    def test_remap_rejects_a_route_histogram_mismatch(self):
        """
        Feature: logical-route remapping validation
        Description: apply a route sequence that disagrees with planned counts
        Expectation: reject it before a physical dispatch can be constructed
        """
        plan = build_constrained_expert_placement(
            ((2, 1, 0, 1), (0, 1, 2, 1)),
            1,
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            remap_logical_routes((0, 1, 1, 3), 0, plan)

    def test_plan_validation_rejects_routes_to_an_empty_slot(self):
        """
        Feature: physical placement validation
        Description: construct planner output that routes rows to an empty slot
        Expectation: reject the malformed source-to-physical count matrix
        """
        with self.assertRaisesRegex(ValueError, "unused physical expert"):
            ExpertPlacementPlan(
                logical_counts_by_source=((1, 0),),
                replica_slots_per_rank=1,
                slot_to_logical_expert=((0, 1, -1),),
                dispatch_counts=((0, 0, 1),),
            )

    def test_invalid_topology_is_rejected(self):
        """
        Feature: planner topology validation
        Description: supply ragged, negative and indivisible configurations
        Expectation: reject every malformed topology with a useful error
        """
        invalid_cases = (
            (((1, 2), (3,)), 0, "rectangular"),
            (((1, 2),), -1, "replica_slots_per_rank"),
            (((1, 2, 3), (0, 0, 0)), 0, "divisible"),
        )
        for counts, replicas, message in invalid_cases:
            with self.subTest(counts=counts, replicas=replicas):
                with self.assertRaisesRegex(ValueError, message):
                    build_constrained_expert_placement(counts, replicas)


if __name__ == "__main__":
    unittest.main()
