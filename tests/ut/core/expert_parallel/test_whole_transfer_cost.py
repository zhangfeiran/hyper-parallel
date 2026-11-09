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
"""Bound whole-expert transfer costs without weakening capacity or optional admission."""

import unittest

from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.plan import ExpertExecutionPlan
from hyper_parallel.core.expert_parallel.hot_replica.whole import build_whole_expert_plan, place_whole_experts
from tests.common.mark_utils import arg_mark


MODERATE_COUNTS = (
    190, 14, 532, 25, 140, 481, 300, 50, 1663, 119, 251, 1601, 92, 634, 85, 92,
    30, 86, 516, 151, 20, 77, 35, 434, 890, 73, 374, 57, 80, 304, 344, 213,
    28, 407, 691, 130, 364, 406, 405, 346, 66, 130, 397, 590, 94, 1, 623, 123,
    348, 245, 543, 145, 73, 36, 832, 197, 41, 574, 232, 340, 65, 5, 449, 427,
    981, 491, 19, 659, 2294, 32, 418, 716, 312, 100, 34, 764, 62, 400, 164, 337,
    1563, 1, 143, 100, 238, 1282, 384, 521, 114, 81, 111, 237, 469, 35, 325, 75,
)


class TestWholeTransferPruning(unittest.TestCase):
    """Keep complete source order while returning surplus guest experts home."""

    def tearDown(self) -> None:
        """Avoid retaining immutable test plans in the production bounded cache."""
        place_whole_experts.cache_clear()

    def assert_complete(self, plan: ExpertExecutionPlan) -> None:
        """Independently require one execution destination for each nonempty logical expert."""
        plan.validate()
        for expert in range(plan.config.num_experts):
            logical = tuple(row[expert] for row in plan.logical_counts)
            if not any(logical):
                continue
            slots = [slot for slot, identity in enumerate(plan.physical_to_logical)
                     if identity == expert and any(row[slot] for row in plan.dispatch_counts)]
            self.assertEqual(len(slots), 1, f"expert={expert}, expected_destinations=1, got={slots}")
            actual = tuple(row[slots[0]] for row in plan.dispatch_counts)
            self.assertEqual(actual, logical, f"expert={expert}, expected={logical}, got={actual}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_prune_surplus_transfers_with_original_optional_threshold(self) -> None:
        """Feature: Bounded complete-expert transfer costs.
        Description: A feasible optional ceiling permits fewer transfers than the unconstrained minimax optimum.
        Expectation: At most nine transfers retain the original forty-percent reduction and complete source quotas.
        """
        config = ExpertReplicaConfig(96, 16, 2)
        plan = build_whole_expert_plan(
            (MODERATE_COUNTS,) * 16, config, capacity_limit=393216,
            minimum_load_reduction=.4, minimum_replica_rows=1024, rank_nodes=(0,) * 8 + (1,) * 8,
        )
        self.assertLessEqual(len(plan.transfers), 9,
                             f"expected_transfers_at_most=9, got={len(plan.transfers)}")
        peak = max(plan.destination_loads)
        self.assertLessEqual(peak, 39724, f"expected_peak_at_most=39724, got={peak}")
        self.assert_complete(plan)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_optional_saved_rows_remains_an_independent_bound(self) -> None:
        """Feature: Bounded complete-expert transfer costs.
        Description: A stricter absolute threshold must still constrain the pruned physical loads.
        Expectation: The final peak saves at least the requested absolute rows without splitting an expert.
        """
        config = ExpertReplicaConfig(96, 16, 2)
        baseline = max(sum(MODERATE_COUNTS[rank * 6:(rank + 1) * 6]) * 16 for rank in range(16))
        plan = build_whole_expert_plan(
            (MODERATE_COUNTS,) * 16, config, capacity_limit=393216,
            minimum_load_reduction=.1, minimum_saved_rows=28000, minimum_replica_rows=1024,
        )
        peak = max(plan.destination_loads)
        self.assertGreaterEqual(baseline - peak, 28000,
                                f"expected_saved_rows_at_least=28000, got={baseline - peak}")
        self.assert_complete(plan)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_capacity_required_moves_ignore_optional_benefit_only(self) -> None:
        """Feature: Bounded complete-expert transfer costs.
        Description: Necessary capacity moves retain the hard receive bound even with impossible optional thresholds.
        Expectation: Necessary moves satisfy capacity without requiring optional benefit or optional expert-size thresholds.
        """
        plan = build_whole_expert_plan(
            (MODERATE_COUNTS,) * 16, ExpertReplicaConfig(96, 16, 2), capacity_limit=40000,
            minimum_load_reduction=1., minimum_saved_rows=1000000, minimum_replica_rows=1000000,
        )
        peak = max(plan.destination_loads)
        self.assertLessEqual(peak, 40000, f"expected_peak_at_most=40000, got={peak}")
        self.assert_complete(plan)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_balanced_and_zero_routes_remain_home_only(self) -> None:
        """Feature: Bounded complete-expert transfer costs.
        Description: Pruning cannot create optional migration without a strictly positive saving.
        Expectation: Both load patterns retain zero transfers and exact complete-expert source quotas.
        """
        for count in (0, 32):
            with self.subTest(count=count):
                plan = build_whole_expert_plan(
                    ((count,) * 24,) * 4, ExpertReplicaConfig(24, 4, 2),
                    capacity_limit=4096, minimum_load_reduction=0.,
                )
                self.assertEqual(len(plan.transfers), 0,
                                 f"expected_home_only_transfers=0, got={len(plan.transfers)}")
                self.assert_complete(plan)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_no_guest_budget_keeps_home_identity(self) -> None:
        """Feature: Bounded complete-expert transfer costs.
        Description: B0 ownership and quotas remain unchanged.
        Expectation: The original home dispatch rows and zero transfer count are preserved.
        """
        counts = ((8, 0, 4, 0),) * 4
        plan = build_whole_expert_plan(counts, ExpertReplicaConfig(4, 4, 0),
                                       capacity_limit=32, minimum_load_reduction=.4)
        self.assertEqual(plan.dispatch_counts, counts, f"expected={counts}, got={plan.dispatch_counts}")
        self.assertEqual(len(plan.transfers), 0, f"expected_transfers=0, got={len(plan.transfers)}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_impossible_capacity_is_rejected_without_splitting_K(self) -> None:
        """Feature: Bounded complete-expert transfer costs.
        Description: One indivisible expert cannot be admitted by dividing its source rows.
        Expectation: Planning rejects the indivisible expert before physical dispatch.
        """
        with self.assertRaisesRegex(ValueError, 'without splitting K'):
            build_whole_expert_plan(((100, 0, 0, 0),) * 4, ExpertReplicaConfig(4, 4, 2),
                                    capacity_limit=200, minimum_load_reduction=.4)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_topology_scale_and_slot_bounds(self) -> None:
        """Feature: Bounded complete-expert transfer costs.
        Description: Sparse EP64 plans exercise bounded greedy pruning without exponential subset search.
        Expectation: Every topology retains complete reductions, its original reduction threshold and at most two guest slots.
        """
        for ep in (2, 4, 8, 16, 64):
            with self.subTest(ep=ep):
                row = (2048, 1536, 1024) + (0,) * (6 * ep - 3)
                plan = build_whole_expert_plan(
                    (row,) * ep, ExpertReplicaConfig(6 * ep, ep, 2),
                    capacity_limit=ep * sum(row), minimum_load_reduction=.4,
                    minimum_replica_rows=1024, rank_nodes=tuple(rank // 8 for rank in range(ep)),
                )
                peak = max(plan.destination_loads)
                self.assertLessEqual(peak, int(ep * sum(row) * .6),
                                     f"expected_peak_at_most={int(ep * sum(row) * .6)}, got={peak}")
                used = max(plan.replica_slots_used)
                self.assertLessEqual(used, 2, f"expected_slots_at_most=2, got={used}")
                self.assert_complete(plan)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_failed_optional_gate_does_not_relax(self) -> None:
        """Feature: Bounded complete-expert transfer costs.
        Description: An impossible optional saving must fall back home with unchanged quotas.
        Expectation: An impossible optional reduction retains the original home-only dispatch.
        """
        counts = ((100, 0, 0, 0),) * 4
        plan = build_whole_expert_plan(counts, ExpertReplicaConfig(4, 4, 2),
                                       capacity_limit=400, minimum_load_reduction=.4)
        self.assertEqual(len(plan.transfers), 0, f"expected_transfers=0, got={len(plan.transfers)}")
        self.assert_complete(plan)
