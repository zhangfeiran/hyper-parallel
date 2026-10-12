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
"""Reference profile integrity, independent seed coverage and fail-closed budget checks."""

import copy
import unittest

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.model_reference_precision import (
    evaluate_global_model_reference,
    fit_model_reference_profile,
    model_reference_policy,
    model_reference_shapes,
    policy_sha256,
    validate_model_reference_profile,
)


class TestModelReferencePrecision(unittest.TestCase):
    """Independent reference fitting must not consume candidate data or silently widen frozen budgets."""

    @staticmethod
    def _samples():
        policy = model_reference_policy()
        cases = []
        for heads in policy['heads']:
            for mode in policy['modes']:
                fields = {}
                for name, shape in model_reference_shapes(heads).items():
                    inactive = mode != 'joint' and (name == 'loss' or name.startswith('gradient/indexer.'))
                    fields[name] = {'shape': shape, 'availability_only': inactive, 'availability_passed': True,
                                    'oracle_verified': True, 'selection_verified': True}
                    if not inactive:
                        fields[name].update(finite=True, expected_norm=1., relative_l2=.01, cosine_error=.00005,
                                            max_abs_over_rms=.02, max_abs=.01)
                for seed in policy['fit_seeds']+policy['held_out_seeds']:
                    cases.append({'heads': heads, 'mode': mode, 'seed': seed,
                                  'partition': 'calibration' if seed in policy['fit_seeds'] else 'held_out',
                                  'samples': [copy.deepcopy(fields) for _ in range(policy['repeats'])]})
        return policy, cases, {'reference_libraries': {'stock.so': 'frozen-sha'}}

    def test_profile_recomputes_all_model_gradient_and_sgd_fields(self):
        """Independent repeated samples certify complete global fields and preserve inactive gradients."""
        policy, cases, environment = self._samples()
        profile = fit_model_reference_profile(policy, cases, environment)
        budgets = validate_model_reference_profile(profile, expected_policy_sha256=policy_sha256(policy),
                                                   expected_environment=environment)
        self.assertEqual(profile['case_count'], 192)
        self.assertEqual(len(budgets), 8)
        self.assertEqual(len(budgets['H128/joint']), 25)
        self.assertTrue(budgets['H64/freeze']['gradient/indexer.wk.weight']['availability_only'])
        self.assertAlmostEqual(budgets['H64/joint']['hidden']['limits']['relative_l2'], .0125)
        self.assertFalse(profile['full_model_acceptance'])

    def test_failed_held_out_never_refits_limits(self):
        """An independent failure leaves fitting-only limits intact and prevents profile consumption."""
        policy, cases, environment = self._samples()
        held_out = next(case for case in cases if case['partition'] == 'held_out')
        held_out['samples'][0]['hidden']['relative_l2'] = .2
        profile = fit_model_reference_profile(policy, cases, environment)
        self.assertEqual(profile['status'], 'held_out_failed')
        self.assertAlmostEqual(profile['budgets']['H64/joint']['hidden']['limits']['relative_l2'], .0125)
        with self.assertRaisesRegex(ValueError, 'unvalidated'):
            validate_model_reference_profile(profile, expected_policy_sha256=policy_sha256(policy),
                                             expected_environment=environment)

    def test_candidate_seed_coverage_and_availability_are_rejected(self):
        """Reject candidate seeds, duplicated/missing rows, repeated samples, invalid oracle and shape scope."""
        policy, cases, environment = self._samples()
        variants = []
        variant = copy.deepcopy(cases)
        variant[0]['seed'] = policy['reserved_candidate_seeds'][0]
        variants.append(variant)
        variants.append(cases[:-1])
        variants.append(cases+[cases[0]])
        for field, value in (('shape', [13]), ('availability_passed', False), ('oracle_verified', False),
                             ('selection_verified', False)):
            variant = copy.deepcopy(cases)
            variant[0]['samples'][0]['hidden'][field] = value
            variants.append(variant)
        variant = copy.deepcopy(cases)
        variant[0]['samples'].pop()
        variants.append(variant)
        for index, variant in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                fit_model_reference_profile(policy, variant, environment)

    def test_budget_environment_policy_and_candidate_flags_cannot_be_forged(self):
        """Recompute limits and coverage rather than trusting serialized accepted flags."""
        policy, cases, environment = self._samples()
        profile = fit_model_reference_profile(policy, cases, environment)
        variants = []
        for name, value in (('candidate_measurements', True), ('case_count', 191), ('policy_sha256', 'edited')):
            changed = copy.deepcopy(profile)
            changed[name] = value
            variants.append(changed)
        changed = copy.deepcopy(profile)
        changed['budgets']['H128/joint']['output']['limits']['relative_l2'] = 100.
        variants.append(changed)
        changed = copy.deepcopy(profile)
        changed['policy']['margin'] = 100.
        variants.append(changed)
        for index, changed in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_model_reference_profile(changed, expected_policy_sha256=policy_sha256(policy),
                                                 expected_environment=environment)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            validate_model_reference_profile(profile, expected_policy_sha256=policy_sha256(policy),
                                             expected_environment={'reference_libraries': {'stock.so': 'new-sha'}})


    def test_candidate_evaluation_requires_reserved_seed_and_complete_global_tensors(self):
        """A validated global profile cannot certify old candidates, owner subsets or integer gradients."""
        policy, cases, environment = self._samples()
        profile = fit_model_reference_profile(policy, cases, environment)
        actual = {name: torch.ones(shape) for name, shape in model_reference_shapes(64).items()}
        expected = {name: value.clone() for name, value in actual.items()}
        options = {'heads': 64, 'mode': 'joint', 'candidate_seed': policy['reserved_candidate_seeds'][0],
                   'expected_policy_sha256': policy_sha256(policy), 'expected_environment': environment}
        self.assertTrue(evaluate_global_model_reference(profile, actual, expected, **options)['passed'])
        for seed in (733, policy['fit_seeds'][0], policy['held_out_seeds'][0], float(options['candidate_seed'])):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'candidate seed'):
                evaluate_global_model_reference(profile, actual, expected, **{**options, 'candidate_seed': seed})
        wrong = dict(actual)
        wrong['hidden'] = actual['hidden'][:, :7]
        with self.assertRaisesRegex(ValueError, 'complete global tensors'):
            evaluate_global_model_reference(profile, wrong, expected, **options)
        wrong['hidden'] = actual['hidden'].long()
        with self.assertRaisesRegex(ValueError, 'floating-point'):
            evaluate_global_model_reference(profile, wrong, expected, **options)
