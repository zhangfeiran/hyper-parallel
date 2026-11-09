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
"""Keep each MXFP8 expert's complete reduction on one physical destination."""

import random
import unittest
from unittest.mock import patch

from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.core.expert_parallel.hot_replica import routing, whole
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.plan import ExpertExecutionPlan
from hyper_parallel.core.expert_parallel.hot_replica.planner import _build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.whole import place_whole_experts
from tests.common.mark_utils import arg_mark


class TestWholeExpertPlacement(unittest.TestCase):
    """Validate complete-K placement against logical counts and independent home bounds."""

    def setUp(self) -> None:
        """Keep cached metadata independent of other tests."""
        place_whole_experts.cache_clear()
        self.addCleanup(place_whole_experts.cache_clear)

    def assert_complete(self, plan: ExpertExecutionPlan) -> None:
        """Verify exactly one complete physical destination for every nonempty expert."""
        plan.validate()
        counts = tuple(count for row in plan.destination_counts for count in row)
        for expert in range(plan.config.num_experts):
            with self.subTest(expert=expert):
                total = sum(row[expert] for row in plan.logical_counts)
                physical = [count for logical, count in zip(plan.physical_to_logical, counts)
                            if logical == expert and count]
                expected = [total] if total else []
                self.assertEqual(physical, expected, f"expert={expert}, actual={physical}, expected={expected}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_hot_experts_move_without_retaining_partial_home_work(self) -> None:
        """Feature: Complete expert migration.
        Description: Place two hot experts from the same owner using a bounded guest budget.
        Expectation: One expert moves completely, home K becomes zero, and all source quotas are conserved.
        """
        counts = ((100, 100, 0, 0, 0, 0, 0, 0),) * 4
        before = _build_expert_replica_plan(counts, 2, capacity_limit=1000)
        after = place_whole_experts(before, capacity_limit=1000, minimum_load_reduction=.4,
                                    rank_nodes=(0, 0, 1, 1))
        self.assert_complete(after)
        self.assertEqual(after.logical_counts, counts)
        self.assertEqual(max(after.destination_loads), 400)
        self.assertEqual(len(after.transfers), 1)
        for transfer in after.transfers:
            self.assertEqual(after.destination_counts[transfer.owner_rank][transfer.owner_slot], 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_optional_rejection_materializes_home_from_a_partial_plan(self) -> None:
        """Feature: Optional migration rejection.
        Description: Supply partial-K quotas with an unattainable optional load-reduction threshold.
        Expectation: Every expert returns home without retaining any partial-K guest.
        """
        counts = ((100, 100, 0, 0, 0, 0, 0, 0),) * 4
        before = _build_expert_replica_plan(counts, 2, capacity_limit=1000)
        self.assertTrue(before.transfers)
        after = place_whole_experts(before, capacity_limit=1000, minimum_load_reduction=.9)
        self.assert_complete(after)
        self.assertFalse(after.transfers)
        self.assertEqual(after.destination_loads, (800, 0, 0, 0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_capacity_required_moves_bypass_optional_thresholds(self) -> None:
        """Feature: Necessary capacity migration.
        Description: Require a bounded receive despite restrictive optional benefit and size thresholds.
        Expectation: Complete experts migrate and satisfy the common receive bound without dropping rows.
        """
        counts = ((45, 25, 15, 10, 5, 0),) * 3
        before = _build_expert_replica_plan(counts, 2, capacity_limit=150)
        after = place_whole_experts(before, capacity_limit=150, minimum_load_reduction=.4,
                                    minimum_saved_rows=1000, minimum_replica_rows=1000)
        self.assert_complete(after)
        self.assertLessEqual(max(after.destination_loads), 150)
        self.assertTrue(after.transfers)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_unsplittable_expert_fails_before_dispatch(self) -> None:
        """Feature: Complete reduction capacity validation.
        Description: Request a receive capacity smaller than one logical expert's complete K.
        Expectation: Planning fails explicitly rather than silently using partial-K gradients.
        """
        counts = ((100, 0, 0, 0),) * 4
        before = _build_expert_replica_plan(counts, 1, capacity_limit=100)
        with self.assertRaisesRegex(ValueError, 'complete expert'):
            place_whole_experts(before, capacity_limit=100, minimum_load_reduction=.4)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_home_budget_balanced_and_empty_routes_remain_home(self) -> None:
        """Feature: Home-only placement preservation.
        Description: Plan balanced counts, empty counts and a zero guest budget.
        Expectation: No guest is created and the zero-budget plan retains its identity.
        """
        for counts, budget in ((((10, 10, 10, 10),) * 4, 2), (((0, 0, 0, 0),) * 4, 2),
                               (((100, 0, 0, 0),) * 4, 0)):
            with self.subTest(counts=counts, budget=budget):
                before = _build_expert_replica_plan(counts, budget, capacity_limit=1000)
                after = place_whole_experts(before, capacity_limit=1000, minimum_load_reduction=.4)
                self.assert_complete(after)
                self.assertFalse(after.transfers)
                if not budget:
                    self.assertIs(after, before)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_random_counts_conserve_each_source_without_increasing_home_peak(self) -> None:
        """Feature: Placement conservation across topologies.
        Description: Generate deterministic count tables for three EP sizes and three fixed budgets.
        Expectation: Each logical expert has one complete destination without exceeding home load or slot limits.
        """
        generator = random.Random(17)
        for ep in (1, 2, 4):
            for budget in (0, 1, 2):
                for _ in range(3):
                    counts = tuple(tuple(generator.randrange(50) for _ in range(ep * 2)) for _ in range(ep))
                    before = _build_expert_replica_plan(counts, budget, capacity_limit=10000)
                    after = place_whole_experts(before, capacity_limit=10000, minimum_load_reduction=.1)
                    self.assert_complete(after)
                    home_peak = max(sum(sum(row[rank * 2:(rank + 1) * 2]) for row in counts) for rank in range(ep))
                    self.assertLessEqual(max(after.destination_loads), home_peak)
                    self.assertLessEqual(max(after.replica_slots_used), budget)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_cache_keys_include_admission_capacity_and_topology(self) -> None:
        """Feature: Bounded exact metadata caching.
        Description: Repeat a plan, then change capacity, admission, minimum sizes and node topology.
        Expectation: Only the identical invocation hits the bounded cache.
        """
        counts = ((100, 100, 0, 0, 0, 0, 0, 0),) * 4
        before = _build_expert_replica_plan(counts, 2, capacity_limit=1000)
        options = {'capacity_limit': 1000, 'minimum_load_reduction': .4}
        with patch.object(whole, '_materialize', wraps=whole._materialize) as materialize:
            first = place_whole_experts(before, capacity_limit=1000, minimum_load_reduction=.4)
            self.assertIs(place_whole_experts(before, capacity_limit=1000, minimum_load_reduction=.4), first)
            changes = ({'capacity_limit': 800}, {'minimum_load_reduction': .9}, {'minimum_saved_rows': 1000},
                       {'minimum_replica_rows': 1000}, {'rank_nodes': (0, 0, 1, 1)})
            for change in changes:
                current = {**options, **change}
                place_whole_experts(before, capacity_limit=current['capacity_limit'],
                                    minimum_load_reduction=current['minimum_load_reduction'],
                                    minimum_saved_rows=current.get('minimum_saved_rows', 0),
                                    minimum_replica_rows=current.get('minimum_replica_rows', 0),
                                    rank_nodes=current.get('rank_nodes'))
            self.assertEqual(materialize.call_count, 6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_cache_evicts_old_metadata_instead_of_retaining_unbounded_plans(self) -> None:
        """Feature: Bounded placement-cache lifetime.
        Description: Create more than sixty-four capacity signatures and request the first again.
        Expectation: The oldest metadata is recomputed with an identical physical result.
        """
        before = _build_expert_replica_plan(((100, 100, 0, 0),) * 4, 1, capacity_limit=1000)
        with patch.object(whole, '_materialize', wraps=whole._materialize) as materialize:
            first = place_whole_experts(before, capacity_limit=1000, minimum_load_reduction=.4)
            for capacity in range(1001, 1065):
                place_whole_experts(before, capacity_limit=capacity, minimum_load_reduction=.4)
            after = place_whole_experts(before, capacity_limit=1000, minimum_load_reduction=.4)
            self.assertEqual(materialize.call_count, 66)
            self.assertEqual(after, first)
            self.assertIsNot(after, first)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_invalid_bounds_and_topology_are_rejected(self) -> None:
        """Feature: Placement input validation.
        Description: Supply negative bounds, invalid fractional thresholds and an incomplete topology.
        Expectation: Every malformed placement fails before physical execution.
        """
        before = _build_expert_replica_plan(((100, 100, 0, 0),) * 4, 1, capacity_limit=1000)
        options = {'capacity_limit': 1000, 'minimum_load_reduction': .4}
        for change in ({'capacity_limit': -1}, {'minimum_load_reduction': float('nan')},
                       {'minimum_load_reduction': -1.0}, {'minimum_load_reduction': 1.1},
                       {'minimum_saved_rows': -1}, {'minimum_replica_rows': -1}, {'rank_nodes': (0, 1)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                place_whole_experts(before, **{**options, **change})


class TestWholeExpertRoute(unittest.TestCase):
    """Select complete reductions before the existing checkpoint decision is saved."""

    def setUp(self) -> None:
        """Bind one real hot logical count table and a mock process-group identity."""
        self.counts = ((100, 100, 0, 0, 0, 0, 0, 0),) * 4
        self.config = ExpertReplicaConfig(8, 4, 2)
        self.group = object()
        self.options = {'capacity_limit': 1000, 'minimum_load_reduction': .4}
        place_whole_experts.cache_clear()
        self.addCleanup(place_whole_experts.cache_clear)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_default_policy_remains_the_original_partial_plan(self) -> None:
        """Feature: Shared planner behavior preservation.
        Description: Invoke the default route without the internal MX complete-K selection.
        Expectation: Its full immutable plan equals the unchanged original partial-K planner result.
        """
        expected = _build_expert_replica_plan(self.counts, 2, **self.options)
        actual, reused = routing._plan_replica_route(self.counts, self.config, self.group, 0, **self.options)
        self.assertFalse(reused)
        self.assertEqual(actual, expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_checkpoint_replays_the_complete_plan_without_rebuilding_partial_quotas(self) -> None:
        """Feature: Complete checkpoint plan replay.
        Description: Prepare one complete expert graph in the original checkpoint and recompute it.
        Expectation: Replay returns the original complete plan identity without rerunning partial placement.
        """
        forward, replay = create_recompute_contexts()
        with patch.object(routing, 'build_expert_replica_plan', wraps=_build_expert_replica_plan) as build:
            with forward:
                first, reused = routing._plan_replica_route(
                    self.counts, self.config, self.group, 0, _complete_experts=True, **self.options)
                self.assertFalse(reused)
            with replay:
                second, reused = routing._plan_replica_route(
                    self.counts, self.config, self.group, 0, _complete_experts=True, **self.options)
                self.assertTrue(reused)
                self.assertIs(second, first)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(len(first.transfers), 1)
        self.assertEqual(max(first.destination_loads), 400)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_changed_complete_policy_misses_a_cached_partial_graph(self) -> None:
        """Feature: Checkpoint policy identity.
        Description: Change the internal complete-K selection while retaining the same count table.
        Expectation: The old partial graph is rejected and a complete physical plan is built.
        """
        forward, replay = create_recompute_contexts()
        with forward:
            partial, _ = routing._plan_replica_route(self.counts, self.config, self.group, 0, **self.options)
        with replay:
            complete, reused = routing._plan_replica_route(
                self.counts, self.config, self.group, 0, _complete_experts=True, **self.options)
            self.assertFalse(reused)
            self.assertNotEqual(complete.dispatch_counts, partial.dispatch_counts)
            self.assertEqual(len(complete.transfers), 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_partial_cost_calibration_cannot_silently_price_complete_reductions(self) -> None:
        """Feature: Calibration execution identity.
        Description: Supply a partial-K cost model to the internal complete-K route.
        Expectation: Reject the mismatched calibration before planning or device execution.
        """
        with self.assertRaisesRegex(ValueError, 'costs calibrated'), patch.object(
                routing, 'build_expert_replica_plan') as build:
            routing._plan_replica_route(self.counts, self.config, self.group, 0,
                                        _complete_experts=True, **self.options, cost_model=object())
        build.assert_not_called()
