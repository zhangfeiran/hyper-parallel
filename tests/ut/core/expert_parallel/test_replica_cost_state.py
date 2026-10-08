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
"""Exact full-scan policy oracles for invocation-local quota scoring."""

from dataclasses import replace
import math
import random
import unittest
from unittest.mock import patch

from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaConfig, ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica import build_expert_replica_plan, cost, planner
from tests.common.mark_utils import arg_mark
from tests.ut.core.expert_parallel import replica_cost_oracle as oracle


class TestQuotaCostState(unittest.TestCase):
    """Compare every candidate score and final quota with the frozen full scan."""

    @staticmethod
    def _model(rng: random.Random, ep_size: int) -> ExpertReplicaCostModel:
        scale = rng.choice((2**-30, 1.0, 2**30))
        tables = []
        for _ in range(2):
            duration = 0.0
            table = [(0, 0)]
            for rows in (1, 3, 8, 32, 64, 256, 1024):
                duration += rng.choice((0.0, rng.random())) * scale
                table.append((rows, duration))
            tables.append(tuple(table))
        return ExpertReplicaCostModel(
            calibration_version=2, backend="native", hidden_size=8, intermediate_size=4,
            ep_size=ep_size, transport="p2p", forward_ms=tables[0], backward_ms=tables[1],
            weight_ms=rng.random() * scale, gradient_ms=rng.random() * scale,
            token_ms_per_row=rng.random() * scale / 10,
            minimum_gain_ms=rng.choice((0.0, 0.1 * scale, 1000 * scale)))

    def _check_plan(self, counts, budget, model, limit, minimum=0) -> int:
        options = {"capacity_limit": limit, "cost_model": model, "minimum_replica_rows": minimum}
        with patch.object(planner, "refine_replica_quotas", oracle._refine_replica_quotas):
            expected = build_expert_replica_plan(counts, budget, **options)
        trial_count = 0
        trial_score = cost._QuotaCostState.trial_score

        def _checked_trial(state, owner, target, expert):
            nonlocal trial_count
            trial_count += 1
            actual = trial_score(state, owner, target, expert)
            full = oracle._rank_costs(state.model, state.copies, state.counts, expected.config)
            self.assertEqual(state.costs, full)
            self.assertEqual(actual, oracle._phase_score(full))
            return actual

        with patch.object(cost._QuotaCostState, "trial_score", _checked_trial):
            actual = build_expert_replica_plan(counts, budget, **options)
        self.assertEqual(actual, expected)
        actual.validate()
        self.assertLessEqual(max(actual.destination_loads), limit)
        copies = [{expert: rows for expert, rows in zip(slots, row) if expert >= 0}
                  for slots, row in zip(actual.slot_to_logical, actual.destination_counts)]
        full = oracle._rank_costs(model, copies, actual.logical_counts, actual.config)
        self.assertEqual(model.estimate_ms(actual), oracle._phase_score(full))
        return trial_count

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_trial_scores_and_plans_match_full_scan(self) -> None:
        """
        Feature: Calibrated quota scoring
        Description: Compare random topology, budget, capacity and asymmetric cost-table candidates.
        Expectation: Every rank score and final plan exactly match the independent full-scan policy.
        """
        rng = random.Random(51008)
        trials = 0
        for ep_size in (2, 3, 4, 8):
            home = 3
            for budget in (0, 1, 2, home, home + 2):
                for index in range(12):
                    counts = [[rng.randrange(64 if expert < home else 8)
                               for expert in range(ep_size * home)] for _ in range(ep_size)]
                    model = self._model(rng, ep_size)
                    baseline = build_expert_replica_plan(counts, budget)
                    owner_loads = [sum(sum(row[e] for row in counts)
                                      for e in range(rank * home, (rank + 1) * home)) for rank in range(ep_size)]
                    limit = rng.choice((max(baseline.destination_loads), max(owner_loads), sum(owner_loads)))
                    with self.subTest(ep_size=ep_size, budget=budget, index=index, limit=limit):
                        trials += self._check_plan(counts, budget, model, limit, minimum=index % 3)
        self.assertGreater(trials, 1000)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_removing_edges_updates_both_endpoints_and_restores_trial_state(self) -> None:
        """
        Feature: Quota score communication accounting
        Description: Remove and retain edges with simultaneous owner fan-out and guest fan-in.
        Expectation: Later trials use accepted transfer counts and preserve exact full-rank scores.
        """
        config = ExpertReplicaConfig(8, 4, 4)
        counts = tuple((10,) * 8 for _ in range(4))
        copies = [{e: 40 for e in range(rank * 2, (rank + 1) * 2)} for rank in range(4)]
        copies[0][0] = 11
        copies[2][0], copies[3][0] = 13, 16
        copies[2][4], copies[1][4] = 33, 7
        model = self._model(random.Random(229), 4)
        state = cost._QuotaCostState(model, copies, counts, config)
        self.assertEqual(state.transfers, [2, 1, 2, 1])
        for owner, target, expert, selected in ((0, 2, 0, 0), (0, 3, 0, 40), (2, 1, 4, 0)):
            total = copies[owner][expert] + copies[target][expert]
            for quota in (0, total, 1, total // 2):
                copies[target][expert], copies[owner][expert] = quota, total - quota
                score = state.trial_score(owner, target, expert)
                full = oracle._rank_costs(model, copies, counts, config)
                self.assertEqual(state.costs, full)
                self.assertEqual(score, oracle._phase_score(full))
            copies[target][expert], copies[owner][expert] = selected, total - selected
            state.accept(owner, target, expert)
            if not selected:
                del copies[target][expert]
            self.assertEqual(state.costs, oracle._rank_costs(model, copies, counts, config))
        self.assertEqual(state.transfers, [1, 0, 0, 1])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_minimum_gain_boundaries_and_uncalibrated_fallback_match(self) -> None:
        """
        Feature: Deterministic strict-gain acceptance
        Description: Probe gain thresholds and unsupported rows with tight and loose capacity.
        Expectation: Ties, zero transfers and fallback retain the original refinement decisions.
        """
        model = replace(self._model(random.Random(775), 2),
                        forward_ms=((0, 0), (1024, 0)), backward_ms=((0, 0), (1024, 0)),
                        weight_ms=1.0, gradient_ms=1.0, token_ms_per_row=0.0)
        counts = [[10, 4, 0, 0]] * 2
        for gain in (0.0, math.nextafter(3.0, 0.0), 3.0, math.nextafter(3.0, math.inf), 1000.0):
            for limit in (14, 28):
                with self.subTest(gain=gain, limit=limit):
                    calibrated = replace(model, minimum_gain_ms=gain)
                    self._check_plan(counts, 1, calibrated, limit)
                    plan = build_expert_replica_plan(counts, 1, cost_model=calibrated, capacity_limit=limit)
                    self.assertEqual(bool(plan.transfers), limit == 14 or gain >= 3.0)
        fallback = replace(model, forward_ms=((0, 0), (8, 1)), backward_ms=((0, 0), (8, 1)))
        expected = build_expert_replica_plan(counts, 1, capacity_limit=28)
        with patch.object(cost._QuotaCostState, "__init__", side_effect=AssertionError("No supported scoring")):
            actual = build_expert_replica_plan(counts, 1, capacity_limit=28, cost_model=fallback)
        self.assertEqual(actual, expected)
