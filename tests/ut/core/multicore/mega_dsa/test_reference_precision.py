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
"""Independent reference budgets reject zero-division artifacts, nonfinite values and unseen regimes."""

import copy
import hashlib
import json
import unittest
from dataclasses import asdict

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    ReferenceErrorBudget, fit_reference_budget, gradient_error_metrics, reference_budget_passes,
    ordered_fp32_owner_bound, reference_absolute_limit, validate_reference_profile, validate_reference_seed_sets,
)


class TestReferencePrecision(unittest.TestCase):
    """Exercise numerical acceptance boundaries with independent hand-computed perturbations."""

    def test_zero_oracle_uses_absolute_error_and_has_no_normalized_metrics(self):
        """A tiny rounding residual is distinguishable from a real zero-gradient regression."""
        expected = torch.zeros(8, dtype=torch.float64)
        small = gradient_error_metrics(torch.full_like(expected, 1e-9), expected)
        large = gradient_error_metrics(torch.full_like(expected, 1e-3), expected)
        budget = ReferenceErrorBudget(None, None, None, 2e-5)
        self.assertIsNone(small['relative_l2'])
        self.assertIsNone(small['cosine_error'])
        self.assertTrue(reference_budget_passes(small, budget))
        self.assertFalse(reference_budget_passes(large, budget))
        empty = gradient_error_metrics(torch.empty(0), torch.empty(0))
        self.assertTrue(reference_budget_passes(empty, budget))

    def test_reference_scale_and_cancellation_do_not_use_pointwise_relative_error(self):
        """The budget compares norm and RMS-scaled maximum even when individual oracle entries vanish."""
        expected = torch.tensor([0., 1., -1., 0.], dtype=torch.float64)
        reference = expected + torch.tensor([.001, 0., 0., 0.], dtype=torch.float64)
        metric = gradient_error_metrics(reference, expected)
        budget = fit_reference_budget([metric], margin=2., zero_atol=2e-5)
        self.assertAlmostEqual(metric['relative_l2'], .001 / (2**.5))
        self.assertAlmostEqual(metric['max_abs_over_rms'], .001 / (.5**.5))
        candidate = expected + torch.tensor([.0015, 0., 0., 0.], dtype=torch.float64)
        self.assertTrue(reference_budget_passes(gradient_error_metrics(candidate, expected), budget))
        self.assertFalse(reference_budget_passes(gradient_error_metrics(expected + .1, expected), budget))
        scaled = gradient_error_metrics(reference * 7, expected * 7)
        self.assertAlmostEqual(scaled['relative_l2'], metric['relative_l2'])
        self.assertAlmostEqual(scaled['max_abs_over_rms'], metric['max_abs_over_rms'])

    def test_held_out_failure_does_not_enlarge_frozen_budget(self):
        """An independent bad sample fails against the fitted immutable limits."""
        expected = torch.tensor([1., 2.], dtype=torch.float64)
        budget = fit_reference_budget([gradient_error_metrics(expected * 1.001, expected)],
                                      margin=2., zero_atol=2e-5)
        before = budget.relative_l2
        self.assertFalse(reference_budget_passes(gradient_error_metrics(expected * 1.1, expected), budget))
        self.assertEqual(budget.relative_l2, before)
        zero_only = fit_reference_budget([gradient_error_metrics(torch.zeros(2), torch.zeros(2))],
                                         margin=2., zero_atol=2e-5)
        self.assertFalse(reference_budget_passes(gradient_error_metrics(expected, expected), zero_only))

    def test_nonfinite_values_and_invalid_contracts_reject(self):
        """NaN, infinity and malformed limits cannot produce a valid calibration or acceptance result."""
        for value in (float('nan'), float('inf')):
            metrics = gradient_error_metrics(torch.tensor([value]), torch.zeros(1))
            self.assertFalse(reference_budget_passes(metrics, ReferenceErrorBudget(None, None, None, 2e-5)))
            with self.assertRaises(ValueError):
                fit_reference_budget([metrics], margin=2., zero_atol=2e-5)
        for margin in (True, .5, float('nan')):
            with self.assertRaises(ValueError):
                fit_reference_budget([], margin=margin, zero_atol=2e-5)
        for limits in ((0., None, 0., 1e-5), (0., 0., 0., None), (float('inf'), 0., 0., 1e-5)):
            with self.assertRaises(ValueError):
                ReferenceErrorBudget(*limits)
        with self.assertRaises(ValueError):
            gradient_error_metrics(torch.zeros(2), torch.zeros(3))


class TestReferenceProfileEvidence(unittest.TestCase):
    """Consumer validation recomputes evidence instead of trusting a producer's acceptance flags."""

    def setUp(self) -> None:
        """Prepare a complete small profile with independent repeated fitting and held-out records."""
        policy = {'fit_seeds': [1], 'held_out_seeds': [2], 'heads': [32], 'cp_sizes': [1],
                  'patterns': ['contiguous'], 'tiles': ['owner'], 'margin': 2., 'zero_atol': 2e-5,
                  'candidate_measurements': False}
        expected = torch.tensor([1., 2.], dtype=torch.float64)
        metric = gradient_error_metrics(expected * 1.001, expected)
        budget = fit_reference_budget([metric], margin=2., zero_atol=2e-5)
        names = ('query', 'compressed_kv', 'query_rope', 'key_rope')
        bucket = 'H32/CP1/contiguous/rank0/owner'
        sample = {'profile_metrics': [metric] * 4, 'raw_merged_vs_autograd': [{'passed': True}] * 4}
        cases = [{'bucket': bucket, 'seed': seed, 'partition': partition,
                  'separate_and_shared_fp64_oracles_agree': True,
                  'samples': [copy.deepcopy(sample), copy.deepcopy(sample)]}
                 for seed, partition in ((1, 'calibration'), (2, 'held_out'))]
        self.profile = {'status': 'validated', 'accepted_tolerance_profile': True, 'candidate_measurements': False,
                        'policy': policy, 'policy_sha256': hashlib.sha256(
                            (json.dumps(policy, sort_keys=True, indent=2) + '\n').encode()).hexdigest(),
                        'cases': cases, 'case_count': 2, 'torch_version': 'test',
                        'loaded_library_sha256': {'primary': 'a' * 64},
                        'budgets': {bucket: {name: {'limits': asdict(budget), 'held_out_samples': 2,
                                                  'held_out_passed': True} for name in names}}}

    def _validate(self, profile):
        return validate_reference_profile(profile, expected_versions={'torch_version': 'test'},
                                          expected_library_hashes={'primary': 'a' * 64},
                                          expected_policy_sha256=self.profile['policy_sha256'])

    def test_complete_reference_evidence_returns_recomputed_limits(self):
        budgets = self._validate(self.profile)
        self.assertEqual(set(budgets), {'H32/CP1/contiguous/rank0/owner'})
        self.assertIsInstance(next(iter(budgets.values()))['query'], ReferenceErrorBudget)

    def test_expanded_sampling_keeps_held_out_separate_and_requires_all_records(self):
        """Sample counts derive from frozen seed sets, and additional validation cannot fit its own budget."""
        policy = self.profile['policy']
        policy.update(fit_seeds=[11, 12, 13], held_out_seeds=[21, 22])
        template = self.profile['cases'][0]
        self.profile['cases'] = []
        for seed in policy['fit_seeds'] + policy['held_out_seeds']:
            case = copy.deepcopy(template)
            case.update(seed=seed, partition='calibration' if seed in policy['fit_seeds'] else 'held_out')
            self.profile['cases'].append(case)
        self.profile['case_count'] = 5
        self.profile['policy_sha256'] = hashlib.sha256(
            (json.dumps(policy, sort_keys=True, indent=2) + '\n').encode()).hexdigest()
        for field in self.profile['budgets']['H32/CP1/contiguous/rank0/owner'].values():
            field['held_out_samples'] = 4
        self._validate(self.profile)
        bad = copy.deepcopy(self.profile)
        bad['cases'][-1]['samples'][0]['profile_metrics'][0]['relative_l2'] = 1.
        with self.assertRaisesRegex(ValueError, 'held-out'):
            self._validate(bad)
        incomplete = copy.deepcopy(self.profile)
        incomplete['cases'].pop()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self._validate(incomplete)

    def test_forged_flags_modified_limits_and_missing_samples_reject(self):
        mutations = ('candidate', 'policy', 'coverage', 'duplicate', 'limit', 'held_out', 'library', 'version')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                profile = copy.deepcopy(self.profile)
                field = profile['budgets']['H32/CP1/contiguous/rank0/owner']['query']
                if mutation == 'candidate':
                    profile['candidate_measurements'] = True
                elif mutation == 'policy':
                    profile['policy']['margin'] = 100.
                elif mutation == 'coverage':
                    profile['cases'].pop()
                elif mutation == 'duplicate':
                    profile['cases'][1]['seed'] = 1
                elif mutation == 'limit':
                    field['limits']['relative_l2'] = 100.
                elif mutation == 'held_out':
                    profile['cases'][1]['samples'][0]['profile_metrics'][0]['relative_l2'] = 1.
                elif mutation == 'library':
                    profile['loaded_library_sha256']['primary'] = 'b' * 64
                else:
                    profile['torch_version'] = 'another'
                with self.assertRaises(ValueError):
                    self._validate(profile)


class TestOwnerPrecisionBounds(unittest.TestCase):
    """Independent owner sums include both measured partial errors and sequential FP32 rounding."""

    def test_fp32_cancellation_and_partial_errors_fit_the_composed_bound(self):
        parts = (torch.tensor([1e8, 1.]), torch.tensor([1., 2.]), torch.tensor([-1e8, 3.]))
        ordered = parts[0] + parts[1] + parts[2]
        exact = sum(part.double() for part in parts)
        bound = ordered_fp32_owner_bound(parts, (0., 0., 0.))
        self.assertTrue(bool(((ordered.double() - exact).abs() <= bound).all()))
        self.assertEqual(float(exact[0]), 1.)
        self.assertEqual(float(ordered[0]), 0.)
        with_partial_error = ordered_fp32_owner_bound(parts, (.01, .02, .03))
        torch.testing.assert_close(with_partial_error - bound, torch.full_like(bound, .06), rtol=1e-12, atol=1e-12)
        budget = ReferenceErrorBudget(.01, .01, .02, 2e-5)
        self.assertAlmostEqual(reference_absolute_limit(torch.tensor([1., -1.]), budget), .02)
        self.assertEqual(reference_absolute_limit(torch.zeros(2), budget), 2e-5)
        with self.assertRaises(ValueError):
            ordered_fp32_owner_bound(parts, (0.,))
        with self.assertRaises(ValueError):
            ordered_fp32_owner_bound((torch.tensor([float('nan')]),), (0.,))


class TestReferenceSamplingSeeds(unittest.TestCase):
    """Sampling provenance is fixed before any reference measurement or device work."""

    def test_seed_iterators_are_materialized_into_independent_tuples(self):
        """Cold preparation captures the whole sampling plan before a generator can change."""
        fitting, held_out = validate_reference_seed_sets(iter((10, 11, 12)), iter((20, 21)))
        self.assertEqual(fitting, (10, 11, 12))
        self.assertEqual(held_out, (20, 21))

    def test_invalid_or_overlapping_sets_reject(self):
        """Empty, duplicate, overlapping and non-RNG inputs fail before fitting."""
        invalid = (((), (1,)), ((1,), ()), ((1, 1), (2,)), ((1,), (2, 2)), ((1, 2), (2, 3)),
                   ((True,), (2,)), ((-1,), (2,)), ((2**63,), (2,)), ((1.5,), (2,)), (([1],), (2,)))
        for fitting, held_out in invalid:
            with self.subTest(fitting=fitting, held_out=held_out), self.assertRaises(ValueError):
                validate_reference_seed_sets(fitting, held_out)
