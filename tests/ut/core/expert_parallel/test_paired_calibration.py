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
"""CPU checks for B0-anchored fitting and independent prediction/admission uncertainty."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import unittest

import numpy as np

from hyper_parallel.core.expert_parallel.hot_replica import autocalibration
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica.planner import _build_expert_replica_plan


class TestPairedCalibration(unittest.TestCase):
    """Recover known effective coefficients while keeping paired observations in one fold."""

    @staticmethod
    def observations() -> tuple[list[dict], np.ndarray]:
        """Create independent numerical service fixtures after plan-feature extraction."""
        rng = np.random.default_rng(71)
        beta = np.array([3., .0006, .7, 1.8, .0001, 2.])
        rows = []
        for frame in range(36):
            owner = np.array([6., rng.uniform(2000, 18000), 0., 0., rng.uniform(2000, 10000), 0.])
            guest = owner.copy()
            if frame % 4:
                guest[[0, 1, 4]] *= (7 / 6, rng.uniform(.3, .7), rng.uniform(.4, .9))
                guest[[2, 3, 5]] = rng.integers(0, 7), rng.integers(1, 9), 1
            counts = tuple((frame + 1,) + (0,) * 7 for _ in range(4))
            for budget, values in enumerate((owner, guest)):
                features = np.zeros((2, 4, 10))
                features[0][:, [0, 1, 4, 5, 8, 9]] = values
                for sample in range(3):
                    rows.append(dict(group_leader=0, frame=frame * 3, sample=sample,
                        cv_group=f'family:{frame % 6}', features=features.copy(),
                        plan=SimpleNamespace(config=ExpertReplicaConfig(8, 4, budget), logical_counts=counts),
                        rank_nodes=(0, 0, 1, 1), observed_ms=float(values @ beta),
                        component_rank_max_ms=float(values @ beta + .5), planner_cpu_ms=float(budget)))
        return rows, beta

    def test_paired_fit_recovers_fixed_floor_and_signed_replica_deltas(self) -> None:
        """A positive fixed cost survives ragged loads and both positive and negative measured gains."""
        rows, expected = self.observations()
        actual, report = autocalibration._fit_services(rows)
        np.testing.assert_allclose(actual, expected, rtol=.001, atol=.001)
        self.assertEqual(report['parameterization'], 'b0_anchored_paired_group_nnls_v2')
        self.assertEqual(report['paired_train_samples'], 108)
        self.assertLess(report['cross_validation']['relative_error_p95'], .001)
        self.assertLess(report['cross_validation']['gain_absolute_error_p95_ms'], .001)

    def test_replica_noise_cannot_change_b0_coefficients(self) -> None:
        """Replica transfer noise cannot be traded against the owner's fixed or per-row service."""
        rows, _ = self.observations()
        before, _ = autocalibration._fit_services(rows)
        changed = deepcopy(rows)
        for row in changed:
            if row['plan'].config.replica_slots_per_rank:
                row['observed_ms'] += 20.
                row['component_rank_max_ms'] += 20.
        after, _ = autocalibration._fit_services(changed)
        np.testing.assert_array_equal(after[[0, 1, 4]], before[[0, 1, 4]])

    def test_source_floor_is_selected_on_owner_training_only(self) -> None:
        """A receive/source bottleneck is recovered without a manually chosen runtime coefficient."""
        rows, beta = self.observations()
        for row in rows:
            row['plan'].logical_counts = ((10000, 0, 0, 0, 0, 0, 0, 0),) * 4
            values = row['features'][0][:, [0, 1, 4, 5, 8, 9]].max(axis=0)
            values[1] = max(values[1], 10000)
            row['observed_ms'] = row['component_rank_max_ms'] = float(values @ beta)
        actual, fit = autocalibration._fit_services(rows)
        self.assertEqual(fit['source_row_floor'], 1.)
        np.testing.assert_allclose(actual, beta, rtol=.001, atol=.001)

    def test_runtime_source_floor_preserves_sender_work_with_empty_sources(self) -> None:
        """Balanced receive rows do not erase a nonempty source rank's measured service floor."""
        counts = ((50, 0, 50, 0), (0, 0, 0, 0))
        plan = _build_expert_replica_plan(counts, 0, capacity_limit=100)
        model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
            intermediate_size=8, ep_size=2, transport='shmem_signal_kernel_gradient_bands',
            forward_ms=((0, 0.), (100, 100.)), backward_ms=((0, 0.), (100, 100.)),
            weight_ms=0., gradient_ms=0., effective_group_ms=(0., 1., 0., 0., 0., 0.),
            effective_source_row_floor=1., _rank_nodes=(0, 1))
        self.assertEqual(model.estimate_ms(plan), 100.)
        payload = asdict(model)
        payload['effective_source_row_floor'] = 0.
        self.assertEqual(ExpertReplicaCostModel(**payload).estimate_ms(plan), 50.)
        for floor in (-1., 1.1, float('nan')):
            with self.subTest(floor=floor):
                with self.assertRaises(ValueError):
                    ExpertReplicaCostModel(**(payload | {'effective_source_row_floor': floor}))

    def test_unmatched_or_leaked_pairs_fail_before_fitting(self) -> None:
        """Pair keys, source histograms, TRAIN frames and the fitting EP group are strict boundaries."""
        rows, _ = self.observations()
        invalid = []
        invalid.append((rows[1:], 'matched B0'))
        invalid.append((rows + [rows[0]], 'Duplicate'))
        invalid.append((rows + [rows[0] | {'frame': 2}], 'TRAIN frames only'))
        invalid.append((rows + [rows[0] | {'group_leader': 4}], 'one TRAIN EP group'))
        changed = deepcopy(rows)
        changed[3]['plan'].logical_counts = ((0,) * 8,) * 4
        invalid.append((changed, 'source histogram'))
        for records, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    autocalibration._fit_services(records)

    def test_slow_component_sample_is_kept_in_gain_uncertainty(self) -> None:
        """One slow replica sample increases admission uncertainty without refitting device phases."""
        rows, _ = self.observations()
        before, initial = autocalibration._fit_services(rows)
        rows[9]['component_rank_max_ms'] += 80.
        after, changed = autocalibration._fit_services(rows)
        np.testing.assert_array_equal(after, before)
        self.assertLess(initial['decision_margin_ms'], .1)
        self.assertGreater(changed['decision_margin_ms'], 79.)
        self.assertEqual(changed['paired_train_samples'], initial['paired_train_samples'])

    def test_total_prediction_error_does_not_become_a_gain_requirement(self) -> None:
        """A common total-time offset is separately gated while an exact paired gain stays admissible."""
        counts = ((100, 0, 0, 0), (0, 0, 0, 0))
        owner = _build_expert_replica_plan(counts, 0, capacity_limit=100)
        guest = _build_expert_replica_plan(counts, 1, target_load=50, capacity_limit=100)
        model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
            intermediate_size=8, ep_size=2, transport='shmem_signal_kernel_gradient_bands',
            forward_ms=((0, 0.), (100, 100.)), backward_ms=((0, 0.), (100, 100.)),
            weight_ms=0., gradient_ms=0., effective_group_ms=(0., 3., 0., 0., 0., 0.),
            _rank_nodes=(0, 1), minimum_gain_ms=.05)
        payload = dict(model=asdict(model), fit={'absolute_error_bound_ms': 15.})
        selected, baseline = [], []
        for sample in range(4):
            common = dict(group_leader=0, frame=2, sample=sample, rank_nodes=(0, 1))
            for target, plan in ((selected, guest), (baseline, owner)):
                observed = model.estimate_ms(plan) + 10.
                target.append(common | dict(plan=plan, observed_ms=observed, component_rank_max_ms=observed))
        result = autocalibration.validate_profile(payload, selected, baseline)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['absolute_error_gate_ms'], 15.)
        self.assertEqual(result['admission_error_gate_ms'], .05)
        self.assertLess(result['worst_gain_overprediction_ms'], .001)


if __name__ == '__main__':
    unittest.main()
