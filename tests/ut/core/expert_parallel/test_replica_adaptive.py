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
"""Bounded adaptive placement and fixed physical topology contracts."""

from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.core.expert_parallel import ExpertParallel
from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica import native, planner
from hyper_parallel.core.expert_parallel.hot_replica.planner import _build_expert_replica_plan as build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig


class TestAdaptiveReplicas(unittest.TestCase):
    """Choose actual copies without changing resident slot addresses or losing rows."""

    @staticmethod
    def model(**options: object) -> ExpertReplicaCostModel:
        """Use an explicit linear calibration with controllable fixed transfer cost."""
        values = dict(calibration_version=2, backend='push', hidden_size=16, intermediate_size=8,
                      ep_size=4, transport='p2p', forward_ms=((0, 0), (4096, 4096)),
                      backward_ms=((0, 0), (4096, 4096)), weight_ms=0, gradient_ms=0)
        return ExpertReplicaCostModel(**(values | options))

    def test_zero_one_and_multiple_guests_share_one_physical_shape(self) -> None:
        """Balanced, single-expert and multi-expert hotspots choose different actual widths."""
        workloads = ([12] * 24, [300] + [0] * 23, [50] * 6 + [0] * 18)
        widths = []
        for row in workloads:
            counts = [row] + [[0] * 24] * 3
            plan = build_expert_replica_plan(counts, 6, adaptive=True, cost_model=self.model(),
                                             capacity_limit=300)
            plan.validate()
            self.assertEqual(plan.config.replica_slots_per_rank, 6)
            self.assertEqual(len(plan.dispatch_counts[0]), 48)
            widths.append(max(plan.replica_slots_used))
        self.assertEqual(widths, [0, 1, 2])

    def test_unprofitable_work_skips_balanced_construction(self) -> None:
        """An optimistic bound rejects expensive transfers before building candidate placements."""
        counts = [[100] * 3 + [0] * 9] + [[0] * 12] * 3
        model = self.model(replica_overhead_ms=1000)
        with patch.object(planner, '_balanced_edges', side_effect=AssertionError('Unexpected placement search')):
            plan = build_expert_replica_plan(counts, 3, adaptive=True, cost_model=model, capacity_limit=300)
        self.assertFalse(plan.transfers)

    def test_equal_rank_loads_keep_exact_owner_quotas_with_costs(self) -> None:
        """Unequal expert loads need no copies when every owner already has equal work."""
        counts = [[20, 0, 10, 10, 5, 15, 0, 20]] * 4
        for model in (None, self.model(weight_ms=100, minimum_replica_gain_ms=20)):
            with self.subTest(calibrated=model is not None):
                plan = build_expert_replica_plan(counts, 1, adaptive=True,
                                                 cost_model=model, capacity_limit=80)
                plan.validate()
                self.assertFalse(plan.transfers)
                self.assertEqual(plan.destination_loads, (80, 80, 80, 80))
                self.assertEqual(plan.logical_counts, tuple(tuple(row) for row in counts))
                for source, row in enumerate(counts):
                    expected = tuple(value for start in range(0, 8, 2) for value in (*row[start:start+2], 0))
                    self.assertEqual(plan.dispatch_counts[source], expected)
                with self.assertRaises(ValueError):
                    build_expert_replica_plan(counts, 1, adaptive=True, cost_model=model, capacity_limit=79)

    def test_cost_admission_compares_completed_load_candidate(self) -> None:
        """A thin seed can lose to owner-only even when its completed layout is profitable."""
        counts = [[100] * 6 + [0] * 18] + [[0] * 24] * 3
        model = self.model(weight_ms=100, backward_weight_ms=0, minimum_gain_ms=20,
                           minimum_replica_gain_ms=20)
        owner = build_expert_replica_plan(counts, 0, capacity_limit=600)
        with patch.object(planner, '_balanced_edges', wraps=planner._balanced_edges) as edges:
            plan = build_expert_replica_plan(counts, 1, adaptive=True, cost_model=model,
                                             capacity_limit=600, minimum_replica_rows=50)
        plan.validate()
        self.assertTrue(plan.transfers)
        self.assertLess(model.estimate_ms(plan) + model.minimum_replica_gain_ms, model.estimate_ms(owner))
        self.assertLessEqual(max(plan.replica_slots_used), 1)
        self.assertLessEqual(max(plan.destination_loads), 600)
        self.assertEqual(edges.call_count, 1)

    def test_capacity_required_copies_survive_unprofitable_costs(self) -> None:
        """Owner-only is rejected when it cannot fit the agreed receive reservation."""
        counts = [[100] * 3 + [0] * 9] + [[0] * 12] * 3
        plan = build_expert_replica_plan(counts, 3, adaptive=True,
            cost_model=self.model(weight_ms=1000, minimum_replica_gain_ms=1000),
            capacity_limit=75, receive_capacity=75, minimum_replica_rows=1000)
        plan.validate()
        self.assertTrue(plan.transfers)
        self.assertEqual(max(plan.destination_loads), 75)

    def test_over_capacity_seed_can_complete_within_the_hard_and_receive_limits(self) -> None:
        """A thin seed is not a capacity proof when its existing guests can take more rows."""
        counts = (tuple([100] * 6 + [0] * 18),) + ((0,) * 24,) * 3
        model = self.model(weight_ms=100, backward_weight_ms=0)
        for hard_limit in (300, 400):
            with self.subTest(hard_limit=hard_limit):
                plan = build_expert_replica_plan(counts, 1, adaptive=True, cost_model=model,
                                                target_load=150, capacity_limit=hard_limit, receive_capacity=300)
                plan.validate()
                self.assertEqual(plan.logical_counts, counts)
                self.assertEqual(plan.destination_loads, (300, 100, 100, 100))
                self.assertEqual(plan.replica_slots_used, (0, 1, 1, 1))
                self.assertEqual(model.estimate_ms(plan), 900.)

    def test_unmeasured_completed_seed_is_a_legal_capacity_fallback(self) -> None:
        """Lack of measured coverage cannot hide a legal mandatory completion."""
        counts = (tuple([100] * 6 + [0] * 18),) + ((0,) * 24,) * 3
        model = self.model(forward_ms=((0, 0), (75, 75)), backward_ms=((0, 0), (75, 75)))
        plan = build_expert_replica_plan(counts, 1, adaptive=True, cost_model=model,
                                        target_load=150, capacity_limit=300, receive_capacity=200)
        plan.validate()
        self.assertEqual(plan.logical_counts, counts)
        self.assertEqual(plan.destination_loads, (300, 100, 100, 100))
        self.assertEqual(plan.replica_slots_used, (0, 1, 1, 1))

    def test_soft_capacity_does_not_force_optional_growth(self) -> None:
        """A cheaper owner exceeding the resident reservation cannot replace a fitting replica plan."""
        counts = [[100] * 3 + [0] * 9] + [[0] * 12] * 3
        plan = build_expert_replica_plan(counts, 3, adaptive=True,
            cost_model=self.model(weight_ms=1000), capacity_limit=300, receive_capacity=75)
        self.assertEqual(max(plan.destination_loads), 75)

    def test_each_histogram_builds_balancing_edges_and_dispatch_once(self) -> None:
        """Reuse the cheap constructive skeleton and materialize only the winning layout."""
        counts = [[100] * 3 + [0] * 9] + [[0] * 12] * 3
        with (patch.object(planner, '_balanced_edges', wraps=planner._balanced_edges) as edges,
              patch.object(planner, '_materialize', wraps=planner._materialize) as materialize):
            plan = build_expert_replica_plan(counts, 3, adaptive=True, cost_model=self.model(), capacity_limit=300)
        plan.validate()
        self.assertEqual(edges.call_count, 1)
        self.assertEqual(materialize.call_count, 1)

    def test_identical_completed_candidates_reuse_their_score(self) -> None:
        """Increasing the slot cap must not repeatedly price an unchanged completion."""
        counts = (tuple([100] * 6 + [0] * 18),) + ((0,) * 24,) * 3
        config = ExpertReplicaConfig(24, 4, 6)
        model = self.model(weight_ms=1000)
        with patch.object(planner, '_QuotaCostState', wraps=planner._QuotaCostState) as states:
            copies, bounds = planner._adaptive_copies(counts, config, 600, None, model, 150)
        self.assertEqual([sum(row.values()) for row in copies], [600, 0, 0, 0])
        self.assertEqual(bounds, (1200.0, 300.0))
        self.assertLess(states.call_count, 1 + 2 * config.replica_slots_per_rank)

    def test_completed_score_reuse_is_isolated_between_invocations(self) -> None:
        """A new calibration must change admission for the same routing histogram."""
        counts = (tuple([100] * 6 + [0] * 18),) + ((0,) * 24,) * 3
        config = ExpertReplicaConfig(24, 4, 6)
        peaks = []
        scores = []
        for weight_ms in (1000, 0, 1000):
            copies, bounds = planner._adaptive_copies(counts, config, 600, None,
                                                      self.model(weight_ms=weight_ms), 150)
            peaks.append(max(sum(row.values()) for row in copies))
            scores.append(bounds[0])
        self.assertEqual(peaks, [600, 150, 600])
        self.assertEqual(scores, [1200.0, 300.0, 1200.0])

    def test_equal_quotas_with_changed_insertion_order_are_scored_again(self) -> None:
        """Dictionary equality alone cannot certify an identical floating-point sum."""
        counts = (tuple([100] * 6 + [0] * 18),) + ((0,) * 24,) * 3
        config = ExpertReplicaConfig(24, 4, 6)
        original_improve = planner._improve_copies
        original_state = planner._QuotaCostState
        completed = []
        scored = []

        def reordered(copies: list[dict[int, int]], policy: ExpertReplicaConfig, target: int) -> None:
            """Rotate insertion order without changing the completed quotas."""
            original_improve(copies, policy, target)
            for row in copies:
                expert = next(iter(row))
                row[expert] = row.pop(expert)
            completed.append(tuple(tuple(row.items()) for row in copies))

        def record(*args: object, **kwargs: object) -> object:
            """Record each exact ordered layout before invoking the real scorer."""
            scored.append(tuple(tuple(row.items()) for row in args[1]))
            return original_state(*args, **kwargs)

        with (patch.object(planner, '_improve_copies', side_effect=reordered),
              patch.object(planner, '_QuotaCostState', side_effect=record)):
            copies, bounds = planner._adaptive_copies(counts, config, 600, None,
                                                      self.model(weight_ms=1000), 150)
        self.assertEqual(bounds, (1200.0, 300.0))
        self.assertEqual([sum(row.values()) for row in copies], [600, 0, 0, 0])
        self.assertEqual(len(completed), config.replica_slots_per_rank)
        for signature in completed:
            self.assertIn(signature, scored)

    def test_unmeasured_candidates_never_extrapolate(self) -> None:
        """Covered mandatory splits remain usable when the original owner is outside calibration."""
        counts = [[100] * 3 + [0] * 9] + [[0] * 12] * 3
        model = self.model(forward_ms=((0, 0), (75, 75)), backward_ms=((0, 0), (75, 75)))
        plan = build_expert_replica_plan(counts, 3, adaptive=True, cost_model=model, capacity_limit=75)
        plan.validate()
        self.assertEqual(max(plan.destination_loads), 75)

    def test_random_unequal_and_empty_sources_preserve_conservation(self) -> None:
        """Every route remains lossless, deterministic and bounded by the reserved physical slots."""
        generator = random.Random(7321)
        for budget in (1, 2, 3):
            for iteration in range(12):
                counts = [[generator.randrange(40) if source else 0 for _ in range(12)] for source in range(4)]
                for cost in (0, 1000):
                    with self.subTest(budget=budget, iteration=iteration, cost=cost):
                        options = dict(adaptive=True, cost_model=self.model(weight_ms=cost), capacity_limit=2000)
                        first = build_expert_replica_plan(counts, budget, **options)
                        first.validate()
                        self.assertEqual(first, build_expert_replica_plan(counts, budget, **options))
                        self.assertTrue(all(width <= budget for width in first.replica_slots_used))

    def test_invalid_opt_in_fails_at_the_planner_boundary(self) -> None:
        """An adaptive plan needs a positive physical cap and a matching cost calibration."""
        counts = [[1] * 12] * 4
        for options in ({'adaptive': 1},
                        {'adaptive': True, 'cost_model': self.model(ep_size=2)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_expert_replica_plan(counts, 3, **options)
        with self.assertRaises(ValueError):
            build_expert_replica_plan(counts, 0, adaptive=True, cost_model=self.model())

    def test_legacy_default_and_explicit_false_are_identical(self) -> None:
        """The adaptive policy is opt-in for existing callers and calibrations."""
        counts = [[100] * 3 + [0] * 9] + [[0] * 12] * 3
        original = build_expert_replica_plan(counts, 3, cost_model=self.model())
        self.assertEqual(original, build_expert_replica_plan(counts, 3, cost_model=self.model(), adaptive=False))

    def test_native_policy_adapts_without_cost_arguments(self) -> None:
        """Native business configuration needs only its resident guest cap."""
        policy = ExpertParallel(replica_slots_per_rank=3)
        self.assertTrue(policy._replicas_enabled)
        self.assertFalse(hasattr(policy, 'replica_cost_model'))
        self.assertFalse(hasattr(policy, 'replica_adaptive'))

    def test_native_dispatch_preserves_physical_width_and_checkpoint_plan(self) -> None:
        """Native adaptive dispatch uses the shared ordered plan cache without changing row order."""
        config = ExpertReplicaConfig(6, 1, 3)
        model = self.model(backend='native', ep_size=1)
        values = torch.randn(12, 16, requires_grad=True)
        counts = torch.full((6,), 2)
        group = object()
        forward, replay = create_recompute_contexts()
        with (patch.object(native.dist, 'get_rank', return_value=0),
              patch.object(native, '_gather_replica_counts', return_value=counts.reshape(1, 6)),
              patch.object(native, 'differentiable_all_to_all_single', side_effect=lambda value, *_args: value)):
            with forward:
                first, original = native.dispatch_native_replicas((values, counts), config, group,
                                                                  _cost_model=model)
            with replay:
                second, repeated = native.dispatch_native_replicas((values, counts), config, group,
                                                                   _cost_model=model)
        self.assertIs(original.route.plan, repeated.route.plan)
        self.assertEqual(tuple(first[1].shape), (9,))
        torch.testing.assert_close(first[0], values)
        torch.testing.assert_close(second[0], values)
        second[0].sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))

    def test_infeasible_capacity_fails_without_exceeding_the_slot_cap(self) -> None:
        """A hard capacity failure must not silently add slots or return an oversized dispatch."""
        counts = [[100] * 3 + [0] * 9] + [[0] * 12] * 3
        with self.assertRaisesRegex(ValueError, 'cannot satisfy capacity_limit'):
            build_expert_replica_plan(counts, 1, adaptive=True, cost_model=self.model(), capacity_limit=74)

    def test_replay_model_and_receive_limits_are_independent(self) -> None:
        """Repeated forward costs can alter admission while physical capacity remains fixed."""
        counts = [[100] * 3 + [0] * 9] + [[0] * 12] * 3
        model = self.model(weight_ms=20, gradient_ms=10, forward_repeats=2, backward_weight_ms=0)
        plan = build_expert_replica_plan(counts, 3, adaptive=True, cost_model=model, capacity_limit=300)
        plan.validate()
        self.assertLess(model.estimate_ms(plan), model.estimate_ms(
            build_expert_replica_plan(counts, 0, capacity_limit=300)))
        self.assertEqual(replace(plan.config, replica_slots_per_rank=3), plan.config)
