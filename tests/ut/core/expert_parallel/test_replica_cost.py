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
"""Deterministic cost-guided quota selection with independent capacity checks."""

from dataclasses import replace
import random
import unittest

from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaCostModel, build_expert_replica_plan


class TestReplicaCost(unittest.TestCase):
    """Keep measured costs advisory and preserve every routing invariant."""

    @staticmethod
    def _model(**options):
        values = {"backend": "native", "hidden_size": 8, "intermediate_size": 4, "ep_size": 2, "transport": "p2p",
                  "forward_ms": ((0, 0), (8, 1), (16, 1.1), (128, 3)),
                  "backward_ms": ((0, 0), (8, 1), (16, 1.1), (128, 3)), "weight_ms": 0.0, "gradient_ms": 0.0}
        return ExpertReplicaCostModel(**(values | options))

    def test_compute_cost_can_prefer_more_rows_on_one_rank(self):
        """Fewer expert fragments may finish sooner despite a larger receive peak."""
        counts = [[10, 4, 0, 0]] * 2
        model = self._model()
        baseline = build_expert_replica_plan(counts, 1, capacity_limit=28)
        candidate = build_expert_replica_plan(counts, 1, capacity_limit=28, cost_model=model)
        self.assertLess(model.estimate_ms(candidate), model.estimate_ms(baseline))
        self.assertGreater(max(candidate.destination_loads), max(baseline.destination_loads))
        self.assertLessEqual(max(candidate.destination_loads), 28)
        self.assertEqual(candidate.transfers, baseline.transfers)
        candidate.validate()

    def test_expensive_copy_is_removed_only_when_capacity_allows(self):
        """Count full-copy costs separately from changing the quota of a retained copy."""
        counts = [[10, 4, 0, 0]] * 2
        model = self._model(weight_ms=100, gradient_ms=100)
        loose = build_expert_replica_plan(counts, 1, capacity_limit=28, cost_model=model)
        self.assertFalse(loose.transfers)
        tight = build_expert_replica_plan(counts, 1, capacity_limit=14, cost_model=model)
        self.assertTrue(tight.transfers)
        self.assertEqual(max(tight.destination_loads), 14)

    def test_uncertainty_and_unmeasured_ranges_keep_original_plan(self):
        """Do not spend routing changes on tiny predictions or unsupported row counts."""
        counts = [[10, 4, 0, 0]] * 2
        baseline = build_expert_replica_plan(counts, 1, capacity_limit=28)
        for model in (self._model(minimum_gain_ms=1000),
                      self._model(forward_ms=((0, 0), (8, 1)), backward_ms=((0, 0), (8, 1)))):
            self.assertEqual(build_expert_replica_plan(counts, 1, capacity_limit=28, cost_model=model), baseline)

    def test_cost_refinement_preserves_source_conservation_and_budget(self):
        """Random legal plans may only improve prediction within the same hard bound."""
        rng = random.Random(914)
        model = self._model(weight_ms=0.1, gradient_ms=0.2, token_ms_per_row=0.01)
        for _ in range(40):
            counts = [[rng.randrange(10) for _ in range(4)] for _ in range(2)]
            baseline = build_expert_replica_plan(counts, 1)
            limit = max(baseline.destination_loads)
            candidate = build_expert_replica_plan(counts, 1, capacity_limit=limit, cost_model=model)
            candidate.validate()
            self.assertLessEqual(max(candidate.destination_loads), limit)
            self.assertLessEqual(model.estimate_ms(candidate), model.estimate_ms(baseline))
            self.assertTrue(set(candidate.transfers).issubset(set(baseline.transfers)))
            self.assertEqual(candidate, build_expert_replica_plan(counts, 1, capacity_limit=limit, cost_model=model))

    def test_cost_table_validation_and_execution_identity(self):
        """Reject malformed times, unsupported native transports and stale shape calibration."""
        for options in ({"weight_ms": float("nan")}, {"gradient_ms": -1}, {"minimum_gain_ms": True},
                        {"forward_ms": ((0, 0), (8, 2), (16, 1))}, {"backward_ms": ((1, 0), (8, 1))},
                        {"forward_ms": ((0, 0), (8, 1), (8, 2))}, {"transport": "shmem_signal"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._model(**options)
        model = self._model()
        self.assertAlmostEqual(model.compute_ms(12), 1.05)
        with self.assertRaisesRegex(ValueError, "calibrated"):
            model.compute_ms(129)
        with self.assertRaisesRegex(ValueError, "does not match"):
            model.validate_execution("native", 16, 4, 2, "p2p")
        with self.assertRaisesRegex(ValueError, "EP size"):
            build_expert_replica_plan([[1, 1]], 1, cost_model=model)
        with self.assertRaises(ValueError):
            build_expert_replica_plan([[1, 1]], 1, cost_model=object())
        overlapping = replace(model, backend="push", transport="shmem_signal_sdma_projection", weight_ms=10)
        serial = replace(overlapping, backend="native", transport="p2p")
        plan = build_expert_replica_plan([[10, 4, 0, 0]] * 2, 1)
        self.assertLess(overlapping.estimate_ms(plan), serial.estimate_ms(plan))
