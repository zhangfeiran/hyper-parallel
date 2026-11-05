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
from dataclasses import asdict, replace
from types import SimpleNamespace
import random
import unittest

import numpy as np

from hyper_parallel.core.expert_parallel.hot_replica import autocalibration
from hyper_parallel.core.expert_parallel.hot_replica import routing
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel, _QuotaCostState
from hyper_parallel.core.expert_parallel.hot_replica.planner import _build_expert_replica_plan, _materialize


class TestPairedCalibration(unittest.TestCase):
    """Recover known effective coefficients while keeping paired observations in one fold."""

    def test_costed_candidate_keeps_a_profitable_complete_layout(self) -> None:
        """A low-cap incumbent cannot hide a better eligible load-driven layout."""
        counts = ((4096,) * 6 + (0,) * 42,) * 8
        rate = (.00115 + .0004 / 6) / 3
        curve = ((0, 0.), (32768, 32768 * rate))
        model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=5120,
            intermediate_size=2304, ep_size=8, transport='shmem_signal_kernel_gradient_bands',
            forward_ms=curve, backward_ms=curve, weight_ms=0., gradient_ms=0., forward_repeats=2,
            replica_overhead_ms=3.25, effective_group_ms=(2.75, .00115, 1.06, 0., 0., 6.5, .0004),
            minimum_gain_ms=17., minimum_replica_gain_ms=17., _rank_nodes=(0,) * 8)
        options = dict(target_load=30720, capacity_limit=196608, receive_capacity=196608,
                       minimum_replica_rows=1024, minimum_load_reduction=.4, adaptive=True, _rank_nodes=(0,) * 8)
        heuristic = _build_expert_replica_plan(counts, 6, **options)
        actual = _build_expert_replica_plan(counts, 6, cost_model=model, **options)
        actual.validate()
        self.assertEqual(actual, heuristic)
        conservative = replace(model, minimum_gain_ms=200., minimum_replica_gain_ms=200.)
        home = _build_expert_replica_plan(counts, 6, cost_model=conservative, **options)
        self.assertFalse(home.transfers)

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

    def test_expert_cell_fit_separates_group_intercept_from_task_counts(self) -> None:
        """A B0-only intercept must not become a per-task charge in replica candidates."""
        rows, _ = self.observations()
        beta = np.array([20., .0005, .7, 1.8, .0001, 6., .0012])
        rng = np.random.default_rng(81)
        peaks = {}
        for row in rows:
            values = row['features'][0][:, [0, 1, 4, 5, 8, 9]].max(axis=0)
            key = row['frame'], row['plan'].config.replica_slots_per_rank
            if key not in peaks:
                peaks[key] = int(rng.integers(32, max(33, int(values[1]))))
            row['expert_peak_rows'] = peaks[key]
            values[0] = 1.
            row['observed_ms'] = row['component_rank_max_ms'] = float(np.append(values, peaks[key]) @ beta)
        actual, fit = autocalibration._fit_services(rows)
        np.testing.assert_allclose(actual, beta, rtol=.001, atol=.001)
        self.assertEqual(fit['parameterization'], 'b0_anchored_paired_expert_cell_nnls_v3')
        self.assertLess(fit['cross_validation']['relative_error_p95'], .001)

    def test_expert_peak_changes_runtime_cost_at_the_same_received_load(self) -> None:
        """Equal rank totals retain the measured distinction between spread and concentrated experts."""
        model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
            intermediate_size=8, ep_size=2, transport='shmem_signal_kernel_gradient_bands',
            forward_ms=((0, 0.), (100, 0.)), backward_ms=((0, 0.), (100, 0.)),
            weight_ms=0., gradient_ms=0., effective_group_ms=(10., .25, 2., 4., 0., 6., 1.5),
            _rank_nodes=(0, 1))
        spread = _build_expert_replica_plan(((25, 25, 25, 25), (0, 0, 0, 0)), 0, capacity_limit=100)
        concentrated = _build_expert_replica_plan(((50, 0, 50, 0), (0, 0, 0, 0)), 0, capacity_limit=100)
        self.assertEqual(spread.destination_loads, concentrated.destination_loads)
        self.assertEqual(model.estimate_ms(spread), 60.)
        self.assertEqual(model.estimate_ms(concentrated), 97.5)
        empty = _build_expert_replica_plan(((0, 0, 0, 0),) * 2, 0, capacity_limit=100)
        self.assertEqual(model.estimate_ms(empty), 0.)

    def test_expert_cell_coefficient_reaches_real_replica_admission(self) -> None:
        """The new feature changes actual host plans rather than an offline-only surrogate."""
        counts = ((100, 0, 0, 0), (0, 0, 0, 0))
        results = []
        for copy_cost in (3., 100.):
            model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
                intermediate_size=8, ep_size=2, transport='shmem_signal_kernel_gradient_bands',
                forward_ms=((0, 0.), (100, 0.)), backward_ms=((0, 0.), (100, 0.)),
                weight_ms=0., gradient_ms=0., effective_group_ms=(10., 0., copy_cost, copy_cost, 0., 6., 1.),
                _rank_nodes=(0, 1), minimum_gain_ms=2., minimum_replica_gain_ms=2.)
            plan = _build_expert_replica_plan(counts, 1, cost_model=model, adaptive=True, target_load=50,
                receive_capacity=100, capacity_limit=100, _rank_nodes=(0, 1))
            plan.validate()
            results.append(len(plan.transfers))
        self.assertEqual(results, [1, 0])

    def test_two_rank_trials_match_full_expert_cell_scores(self) -> None:
        """Moving or deleting a local or remote quota preserves the exact scorer."""
        config = ExpertReplicaConfig(8, 4, 2)
        copies = [{0: 12, 1: 5, 4: 7}, {2: 10, 3: 15, 0: 8}, {4: 10, 5: 3, 1: 9}, {6: 9, 7: 14, 2: 5}]
        totals = tuple(sum(row.get(expert, 0) for row in copies) for expert in range(8))
        counts = (totals,) + ((0,) * 8,) * 3
        for floor in (0., .6):
            model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
                intermediate_size=8, ep_size=4, transport='shmem_signal_kernel_gradient_bands',
                forward_ms=((0, 0.), (100, 0.)), backward_ms=((0, 0.), (100, 0.)), weight_ms=0., gradient_ms=0.,
                effective_group_ms=(2., .37, 1.3, 3.7, 0., 2.7, .83), effective_source_row_floor=floor,
                _rank_nodes=(0, 0, 1, 1))
            state = _QuotaCostState(model, copies, counts, config)
            for target, expert in ((0, 4), (1, 0), (2, 1), (3, 2)):
                owner = expert // 2
                previous = copies[owner][expert], copies[target][expert]
                total = sum(previous)
                trial = state.prepare_group_trial(owner, target)
                self.assertIsNotNone(trial)
                for quota in (0, 1, total // 2, total):
                    with self.subTest(floor=floor, owner=owner, target=target, quota=quota):
                        copies[owner][expert], copies[target][expert] = total - quota, quota
                        self.assertEqual(state.trial_score(owner, target, expert, (0., 0.), trial), state.score())
                copies[owner][expert], copies[target][expert] = previous

    def test_extra_guest_slots_cannot_make_received_compute_free(self) -> None:
        """A packed rank retains at least the workload per calibrated owner expert width."""
        counts = ((20, 20, 20, 20), (0, 0, 0, 0))
        plan = _materialize(counts, [{0: 10, 1: 20}, {2: 20, 3: 20, 0: 10}],
                            ExpertReplicaConfig(4, 2, 1), 50)
        plan.validate()
        model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
            intermediate_size=8, ep_size=2, transport='shmem_signal_kernel_gradient_bands',
            forward_ms=((0, 0.), (100, 0.)), backward_ms=((0, 0.), (100, 0.)),
            weight_ms=0., gradient_ms=0., effective_group_ms=(0., 0., 0., 0., 0., 0., 1.), _rank_nodes=(0, 1))
        self.assertEqual(model.estimate_ms(plan), 25.)

    def test_expert_cell_factory_keeps_search_curves_as_safe_lower_bounds(self) -> None:
        """A group intercept cannot be multiplied by every expert during quota pruning."""
        manifest = dict(hidden_size=16, intermediate_size=8, num_experts=4, top_k=1, ep_size=2)
        report = dict(backend='push', rank_nodes=(0, 1), soc_name='cpu-test', execution_signature='a' * 64,
                      schedule='fixed_queue_gradient_bands_v1')
        payload = autocalibration._candidate(manifest, report, 1,
            np.array([20., 0., 0., 0., 0., 0., .1]), 1., 100)
        model = ExpertReplicaCostModel(**payload['model'])
        self.assertAlmostEqual(model.compute_ms(1), .1 / 6)

    def test_compact_expert_cell_bound_cannot_hide_a_profitable_candidate(self) -> None:
        """A compact rejection remains consistent with the full planner across sparse quota shapes."""
        rng = random.Random(83)
        accepted = 0
        config = ExpertReplicaConfig(8, 4, 1)
        for case in range(120):
            counts = tuple(tuple(rng.randrange(20) for _ in range(8)) for _ in range(4))
            totals = list(map(sum, zip(*counts)))
            loads = [sum(totals[start:start + 2]) for start in range(0, 8, 2)]
            rows = [[0, 0, sum(source), 31, sum(source), min(source), max(source), loads[rank],
                     sum(source[rank * 2:(rank + 1) * 2])] for rank, source in enumerate(counts)]
            margin = rng.random() * 2
            model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
                intermediate_size=8, ep_size=4, transport='shmem_signal_kernel_gradient_bands',
                forward_ms=((0, 0.), (1000, 0.)), backward_ms=((0, 0.), (1000, 0.)),
                weight_ms=0., gradient_ms=0.,
                effective_group_ms=(rng.random() * 20, rng.random() * .01, rng.random() * 4,
                                    rng.random() * 8, rng.random() * .01, rng.random() * 20, rng.random() * .2),
                effective_source_row_floor=case % 3 / 2, _rank_nodes=(0, 0, 1, 1),
                minimum_gain_ms=margin, minimum_replica_gain_ms=margin)
            if routing._compact_home_cost_dominates(totals, rows, config, model):
                plan = _build_expert_replica_plan(counts, 1, cost_model=model, adaptive=True,
                    target_load=sum(loads) // 4, receive_capacity=max(loads), capacity_limit=max(loads),
                    _rank_nodes=(0, 0, 1, 1))
                self.assertFalse(plan.transfers, f'case={case}, transfers={plan.transfers}')
                accepted += 1
        self.assertGreater(accepted, 0)

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
