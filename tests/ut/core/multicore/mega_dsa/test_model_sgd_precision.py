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
"""Gradient-coupled BF16 SGD arithmetic bounds and independent profile provenance on CPU."""

import copy
import math
import unittest

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.model_reference_precision import (
    model_reference_policy,
    model_reference_shapes,
    policy_sha256,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_sgd_evaluate import (
    collect_global_model_diagnostics,
    evaluate_coupled_global_model_reference,
    single_sgd_control,
    verify_candidate_reference_environment,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_sgd_precision import (
    coupled_model_reference_policy,
    coupled_sgd_budget,
    fit_coupled_model_reference_profile,
    validate_coupled_model_reference_profile,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    ReferenceErrorBudget,
    gradient_error_metrics,
    reference_budget_passes,
)


class TestModelSgdPrecision(unittest.TestCase):
    """Keep gradient accuracy, mechanical SGD and representation error as separate obligations."""

    def test_diagnostics_never_accept_even_when_every_metric_and_update_is_exact(self):
        """A diagnostic run cannot turn an unvalidated reference into a passing candidate."""
        shapes = model_reference_shapes(64)
        expected = {name: torch.tensor(0. if name.startswith('gradient/') else 1.).expand(shape)
                    for name, shape in shapes.items()}
        actual = {name: value.bfloat16() for name, value in expected.items()}
        initial = {name[4:]: value.clone() for name, value in actual.items() if name.startswith('sgd/')}
        result = collect_global_model_diagnostics(actual, expected, initial, heads=64, learning_rate=.01)
        self.assertFalse(result['passed'])
        self.assertFalse(result['profile_consumed_for_acceptance'])
        self.assertTrue(all(value['max_abs'] == 0. for value in result['fields'].values()))
        self.assertTrue(all(value['single_axpy_verified'] for name, value in result['fields'].items()
                            if name.startswith('sgd/')))
        actual['hidden'] = actual['hidden'][:, :1]
        with self.assertRaisesRegex(ValueError, 'owner-local'):
            collect_global_model_diagnostics(actual, expected, initial, heads=64, learning_rate=.01)

    def test_candidate_additional_libraries_require_frozen_identity(self):
        """Loading native transport may add explicitly frozen libraries without replacing reference libraries."""
        reference = {'torch': 'fixed', 'cpu_reference_threads': 1,
                     'loaded_library_sha256': {'/reference/stock.so': 'stock'}}
        actual = copy.deepcopy(reference)
        actual['loaded_library_sha256']['/candidate/native.so'] = 'native'
        verify_candidate_reference_environment(reference, actual, candidate_libraries={'native.so': 'native'})
        with self.assertRaisesRegex(ValueError, 'outside the frozen payload'):
            verify_candidate_reference_environment(reference, actual, candidate_libraries={})
        actual['loaded_library_sha256']['/reference/stock.so'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'libraries differ'):
            verify_candidate_reference_environment(reference, actual, candidate_libraries={'native.so': 'native'})
        actual['torch'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'environment identity differs'):
            verify_candidate_reference_environment(reference, actual, candidate_libraries={'native.so': 'native'})

    @staticmethod
    def _samples():
        policy = coupled_model_reference_policy()
        cases = []
        for heads in policy['heads']:
            for mode in policy['modes']:
                fields = {}
                for name, shape in model_reference_shapes(heads).items():
                    inactive = mode != 'joint' and (name == 'loss' or name.startswith('gradient/indexer.'))
                    fields[name] = {'shape': shape, 'availability_only': inactive, 'availability_passed': True,
                                    'oracle_verified': True, 'selection_verified': True}
                    if not inactive:
                        fields[name].update(finite=True, expected_norm=1., relative_l2=.001,
                                            cosine_error=.0000005, max_abs_over_rms=.002, max_abs=.001)
                    if name.startswith('sgd/'):
                        active = not (mode != 'joint' and name.startswith('sgd/indexer.'))
                        if not active:
                            fields[name].update(relative_l2=0., cosine_error=0., max_abs_over_rms=0., max_abs=0.)
                        fields[name]['sgd_geometry'] = {'numel': math.prod(shape), 'parameter_norm': 1.,
                                                       'parameter_max_abs': 1., 'gradient_active': active,
                                                       'gradient_norm': 1. if active else 0.,
                                                       'gradient_max_abs': 1. if active else 0.,
                                                       'single_axpy_verified': True}
                fields['output']['oracle_proof'] = {
                    'kind': policy['oracle_proof'], 'passed': True,
                    'fields': {name: {'formula_identity_passed': True, 'fp32_replay_exact': True,
                                      'finite': True, 'relative_l2': 0., 'max_abs': 0.}
                               for name in fields if not name.startswith('sgd/')}}
                for seed in policy['fit_seeds']+policy['held_out_seeds']:
                    cases.append({'heads': heads, 'mode': mode, 'seed': seed,
                                  'partition': 'calibration' if seed in policy['fit_seeds'] else 'held_out',
                                  'samples': [copy.deepcopy(fields) for _ in range(policy['repeats'])]})
        return policy, cases, {'reference_libraries': {'stock.so': 'frozen-sha'}}

    def test_arithmetic_envelope_covers_rounding_and_gradient_perturbations(self):
        """Bound an independent BF16 update at midpoint weights and a zero updated oracle."""
        generator = torch.Generator().manual_seed(20261011)
        for zero_target in (False, True):
            with self.subTest(zero_target=zero_target):
                gradient = torch.randn(257, generator=generator)*.01
                initial = (.01*gradient if zero_target else torch.randn(257, generator=generator))
                expected = initial-.01*gradient
                actual_gradient = (gradient+torch.randn(257, generator=generator)*1e-6).bfloat16().float()
                error = gradient_error_metrics(actual_gradient, gradient)
                allowance = ReferenceErrorBudget(error['relative_l2']*1.25, 1., error['max_abs_over_rms']*1.25, 2e-5)
                geometry = {'numel': 257, 'parameter_norm': float(expected.double().norm()),
                            'parameter_max_abs': float(expected.abs().max()), 'gradient_active': True,
                            'gradient_norm': float(gradient.double().norm()),
                            'gradient_max_abs': float(gradient.abs().max())}
                budget = coupled_sgd_budget(geometry, allowance, learning_rate=.01)
                updated = (initial-.01*actual_gradient).bfloat16().float()
                self.assertTrue(reference_budget_passes(gradient_error_metrics(updated, expected), budget))
                rounded_lr = torch.tensor(.01).bfloat16().float()
                coarse = (initial-(rounded_lr*actual_gradient).bfloat16().float()).bfloat16().float()
                self.assertTrue(reference_budget_passes(gradient_error_metrics(coarse, expected), budget))

    def test_profile_propagates_gradients_without_fitting_sgd_errors(self):
        """SGD error observations cannot expand a physically derived limit."""
        policy, cases, environment = self._samples()
        profile = fit_coupled_model_reference_profile(policy, cases, environment)
        self.assertEqual(profile['status'], 'validated')
        self.assertFalse(profile['sgd_error_fitting'])
        validate_coupled_model_reference_profile(profile, expected_policy_sha256=policy_sha256(policy),
                                                 expected_environment=environment)
        changed = copy.deepcopy(cases)
        changed[0]['samples'][0]['sgd/linear_qkv.weight']['relative_l2'] = 1.
        failed = fit_coupled_model_reference_profile(policy, changed, environment)
        self.assertEqual(failed['status'], 'held_out_failed')
        self.assertEqual(failed['budgets']['H64/joint']['gradient/linear_qkv.weight'],
                         profile['budgets']['H64/joint']['gradient/linear_qkv.weight'])

    def test_proofs_inactive_updates_and_profile_edits_are_rejected(self):
        """Reject missing update proofs, inactive changes and edited geometry after fitting."""
        policy, cases, environment = self._samples()
        changed = copy.deepcopy(cases)
        changed[0]['samples'][0]['output']['oracle_proof']['fields']['hidden']['fp32_replay_exact'] = False
        with self.assertRaisesRegex(ValueError, 'FP32 replay failed'):
            fit_coupled_model_reference_profile(policy, changed, environment)
        changed = copy.deepcopy(cases)
        changed[0]['samples'][0]['sgd/linear_qkv.weight']['sgd_geometry']['single_axpy_verified'] = False
        with self.assertRaisesRegex(ValueError, 'one verified update'):
            fit_coupled_model_reference_profile(policy, changed, environment)
        changed = copy.deepcopy(cases)
        inactive = next(c for c in changed if c['mode'] == 'eval')
        inactive['samples'][0]['sgd/indexer.wk.weight']['max_abs'] = 1e-8
        inactive['samples'][0]['sgd/indexer.wk.weight']['relative_l2'] = 1e-8
        failed = fit_coupled_model_reference_profile(policy, changed, environment)
        self.assertEqual(failed['status'], 'held_out_failed')
        changed = copy.deepcopy(cases)
        inactive = next(c for c in changed if c['mode'] == 'eval')
        inactive['samples'][0]['sgd/indexer.k_norm.weight']['cosine_error'] = 2.220446049250313e-16
        exact = fit_coupled_model_reference_profile(policy, changed, environment)
        self.assertEqual(exact['status'], 'validated')
        inactive['samples'][0]['sgd/indexer.k_norm.weight']['sgd_geometry']['gradient_norm'] = 1.
        with self.assertRaisesRegex(ValueError, 'inactive SGD'):
            fit_coupled_model_reference_profile(policy, changed, environment)
        profile = fit_coupled_model_reference_profile(policy, cases, environment)
        edited = copy.deepcopy(profile)
        edited['cases'][0]['samples'][0]['sgd/linear_qkv.weight']['sgd_geometry']['parameter_max_abs'] = 2.
        with self.assertRaisesRegex(ValueError, 'modified'):
            validate_coupled_model_reference_profile(edited, expected_policy_sha256=policy_sha256(policy),
                                                     expected_environment=environment)
        old = model_reference_policy()
        new = policy['fit_seeds']+policy['held_out_seeds']+policy['reserved_candidate_seeds']
        previous = old['fit_seeds']+old['held_out_seeds']+old['reserved_candidate_seeds']
        self.assertFalse(set(new)&set(previous))

    def test_zero_gradient_regime_and_invalid_geometry(self):
        """Do not extrapolate unseen nonzero gradients or silently accept inactive gradient norms."""
        geometry = {'numel': 8, 'parameter_norm': 1., 'parameter_max_abs': 1., 'gradient_active': True,
                    'gradient_norm': 1., 'gradient_max_abs': 1.}
        with self.assertRaisesRegex(ValueError, 'unseen'):
            coupled_sgd_budget(geometry, ReferenceErrorBudget(None, None, None, 2e-5), learning_rate=.01)
        geometry['gradient_active'] = False
        with self.assertRaisesRegex(ValueError, 'inactive SGD'):
            coupled_sgd_budget(geometry, None, learning_rate=.01)

    def test_global_candidate_requires_exact_step_and_fresh_complete_data(self):
        """A one-ULP wrong update fails its control even inside the physical accuracy envelope."""
        policy, cases, environment = self._samples()
        profile = fit_coupled_model_reference_profile(policy, cases, environment)
        shapes = model_reference_shapes(64)
        expected = {name: torch.ones(shape) for name, shape in shapes.items()}
        actual = {name: value.clone() for name, value in expected.items()}
        initial = {}
        for name, shape in shapes.items():
            if name.startswith('gradient/'):
                actual[name] = torch.ones(shape, dtype=torch.bfloat16)
            if name.startswith('sgd/'):
                initial[name[4:]] = torch.ones(shape, dtype=torch.bfloat16)
                expected[name] = expected[name]-.01
                actual[name] = single_sgd_control(initial[name[4:]], actual['gradient/'+name[4:]], learning_rate=.01)
        options = {'heads': 64, 'mode': 'joint', 'candidate_seed': policy['reserved_candidate_seeds'][0],
                   'expected_policy_sha256': policy_sha256(policy), 'expected_environment': environment}
        result = evaluate_coupled_global_model_reference(profile, actual, expected, initial, **options)
        self.assertTrue(result['passed'])
        wrong = dict(actual)
        field = 'sgd/q_a_layernorm.weight'
        wrong[field] = torch.nextafter(actual[field], torch.full_like(actual[field], float('inf')))
        result = evaluate_coupled_global_model_reference(profile, wrong, expected, initial, **options)
        self.assertFalse(result['fields'][field]['single_axpy_verified'])
        self.assertFalse(result['passed'])
        with self.assertRaisesRegex(ValueError, 'fresh reserved'):
            evaluate_coupled_global_model_reference(profile, actual, expected, initial,
                                                    **{**options, 'candidate_seed': policy['fit_seeds'][0]})
        wrong['hidden'] = actual['hidden'][:, :1]
        with self.assertRaisesRegex(ValueError, 'complete global'):
            evaluate_coupled_global_model_reference(profile, wrong, expected, initial, **options)
        actual = {name: value.clone() for name, value in actual.items()}
        expected = {name: value.clone() for name, value in expected.items()}
        actual['loss'] = expected['loss'] = None
        for name in actual:
            if name.startswith('gradient/indexer.'):
                actual[name] = expected[name] = None
            if name.startswith('sgd/indexer.'):
                actual[name] = initial[name[4:]].clone()
                expected[name] = initial[name[4:]].float()
        result = evaluate_coupled_global_model_reference(profile, actual, expected, initial,
                                                        **{**options, 'mode': 'freeze'})
        self.assertTrue(result['passed'])
        actual['sgd/indexer.k_norm.weight'] = torch.nextafter(actual['sgd/indexer.k_norm.weight'],
            torch.full_like(actual['sgd/indexer.k_norm.weight'], float('inf')))
        result = evaluate_coupled_global_model_reference(profile, actual, expected, initial,
                                                        **{**options, 'mode': 'freeze'})
        self.assertFalse(result['passed'])


    def test_explicit_geometry_diagnostics_keep_single_axpy_and_never_accept(self):
        """A different declared width cannot consume the small-model budget or bypass SGD checks."""
        shapes = model_reference_shapes(64)
        shapes['output'] = shapes['hidden'] = [1,5,7]
        expected = {name:torch.tensor(0. if name.startswith('gradient/') else 1.).expand(shape)
                    for name,shape in shapes.items()}
        actual = {name:value.bfloat16() for name,value in expected.items()}
        initial = {name[4:]:value.clone() for name,value in actual.items() if name.startswith('sgd/')}
        result = collect_global_model_diagnostics(actual, expected, initial, heads=64, learning_rate=.01,
                                                 declared_shapes=shapes)
        self.assertFalse(result['passed'])
        self.assertFalse(result['profile_consumed_for_acceptance'])
        self.assertTrue(all(metric.get('single_axpy_verified',True) for metric in result['fields'].values()))
        actual['sgd/q_a_layernorm.weight'] = actual['sgd/q_a_layernorm.weight']+1
        result = collect_global_model_diagnostics(actual, expected, initial, heads=64, learning_rate=.01,
                                                 declared_shapes=shapes)
        self.assertFalse(result['fields']['sgd/q_a_layernorm.weight']['single_axpy_verified'])
