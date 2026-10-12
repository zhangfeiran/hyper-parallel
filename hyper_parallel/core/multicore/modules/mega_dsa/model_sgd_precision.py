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
"""Reference-only SGD budgets derived from gradient bounds and BF16 arithmetic rounding."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict

from hyper_parallel.core.multicore.modules.mega_dsa.model_reference_precision import (
    model_reference_policy,
    model_reference_shapes,
    policy_sha256,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    ReferenceErrorBudget,
    fit_reference_budget,
    reference_budget_passes,
    validate_reference_seed_sets,
)


def coupled_model_reference_policy() -> dict:
    """Declare disjoint new references and an SGD rule before observing new errors."""
    policy = model_reference_policy()
    policy.update(schema=2, fit_seeds=list(range(20261201, 20261217)),
                  held_out_seeds=list(range(20261231, 20261239)),
                  reserved_candidate_seeds=list(range(20261251, 20261255)),
                  sgd_rule='gradient-bound propagation plus BF16 scalar/product/add rounding; independent single AXPY',
                  bf16_unit_roundoff=2**-8, bf16_half_subnormal=2**-134,
                  fp32_unit_roundoff=2**-24, sgd_momentum=0., sgd_weight_decay=0.,
                  oracle_proof='FP64 absorbed/unabsorbed identity plus exact FP32 unabsorbed replay',
                  oracle_relative_l2=1e-9, oracle_max_abs=1e-11, cpu_reference_threads=1)
    return policy


def coupled_sgd_budget(geometry: dict, gradient_budget: ReferenceErrorBudget | None,
                       *, learning_rate: float) -> ReferenceErrorBudget:
    """Propagate a validated gradient envelope through one low-precision SGD step.

    The bound permits rounding of scalar alpha, the update product and final
    addition. It is evaluated at independent oracle norms, never candidate
    error observations. Float32 reference rounding is included. An inactive
    gradient requires exact unchanged parameters. This is an arithmetic
    envelope, paired with both gradient acceptance and a single-AXPY check.
    """
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('coupled SGD requires a positive finite learning rate')
    count = geometry['numel']
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('coupled SGD requires a complete nonempty parameter tensor')
    if not isinstance(geometry['gradient_active'], bool):
        raise TypeError('coupled SGD gradient availability must be a boolean')
    for name in ('parameter_norm', 'parameter_max_abs', 'gradient_norm', 'gradient_max_abs'):
        value = geometry[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('coupled SGD oracle geometry must be finite and nonnegative')
    if not geometry['gradient_active']:
        if geometry['gradient_norm'] or geometry['gradient_max_abs']:
            raise ValueError('inactive SGD cannot declare a nonzero gradient')
        return ReferenceErrorBudget(0., 0., 0., 0.)
    if gradient_budget is None:
        raise ValueError('active SGD requires an independently fitted gradient budget')
    norm, maximum = geometry['parameter_norm'], geometry['parameter_max_abs']
    grad_norm, grad_maximum = geometry['gradient_norm'], geometry['gradient_max_abs']
    root_count = math.sqrt(count)
    if grad_norm:
        if gradient_budget.relative_l2 is None:
            raise ValueError('SGD cannot extrapolate an unseen nonzero gradient regime')
        error_norm = grad_norm*gradient_budget.relative_l2
        error_maximum = grad_norm/root_count*gradient_budget.max_abs_over_rms
    else:
        error_norm = gradient_budget.zero_atol*root_count
        error_maximum = gradient_budget.zero_atol
    unit, tiny, fp32 = 2**-8, 2**-134, 2**-24
    product = (1+unit)**2

    def _bound(reference, gradient, error, underflow):
        pre = learning_rate*(product*error+(product-1)*gradient)
        pre += 2*fp32*reference+4*fp32*learning_rate*(gradient+error)+underflow*(1+unit)
        return (1+unit)*pre+unit*reference+underflow

    norm_bound = _bound(norm, grad_norm, error_norm, tiny*root_count)
    absolute_bound = _bound(maximum, grad_maximum, error_maximum, tiny)
    if norm == 0:
        return ReferenceErrorBudget(None, None, None, absolute_bound)
    relative = norm_bound/norm
    cosine = 2. if relative >= 1 else relative**2/(1+math.sqrt(1-relative**2))
    return ReferenceErrorBudget(relative, cosine, absolute_bound/(norm/root_count), absolute_bound)


def _coupled_sgd_passes(metric, gradient_budget, learning_rate):
    bound = coupled_sgd_budget(metric['sgd_geometry'], gradient_budget, learning_rate=learning_rate)
    if not metric['sgd_geometry']['gradient_active']:
        # Zero absolute difference proves equality; cosine division can round below one for identical tensors.
        return metric['finite'] and metric['max_abs'] == 0.
    return reference_budget_passes(metric, bound)


def _groups(policy, cases):
    if policy != coupled_model_reference_policy():
        raise ValueError('coupled reference policy differs from its frozen declaration')
    fit, held = validate_reference_seed_sets(policy['fit_seeds'], policy['held_out_seeds'])
    validate_reference_seed_sets(fit+held, policy['reserved_candidate_seeds'])
    previous = model_reference_policy()
    prior_seeds = tuple(previous['fit_seeds']+previous['held_out_seeds']+previous['reserved_candidate_seeds'])
    validate_reference_seed_sets(fit+held+tuple(policy['reserved_candidate_seeds']), prior_seeds)
    expected = {(heads, mode, seed) for heads in policy['heads'] for mode in policy['modes'] for seed in fit+held}
    seen, groups = set(), {}
    for case in cases:
        identity = (case['heads'], case['mode'], case['seed'])
        if identity not in expected or identity in seen:
            raise ValueError('coupled reference coverage is duplicated or outside the declared scope')
        seen.add(identity)
        if case['partition'] != ('calibration' if case['seed'] in fit else 'held_out'):
            raise ValueError('coupled reference partition differs from its declared seeds')
        if len(case['samples']) != policy['repeats']:
            raise ValueError('coupled reference repeat coverage is incomplete')
        shapes = model_reference_shapes(case['heads'])
        for sample in case['samples']:
            if set(sample) != set(shapes):
                raise ValueError('coupled reference requires every complete global model field')
            proof = sample['output'].get('oracle_proof', {})
            proof_fields = proof.get('fields', {})
            if (proof.get('kind') != policy['oracle_proof'] or proof.get('passed') is not True
                    or set(proof_fields) != {name for name in shapes if not name.startswith('sgd/')}):
                raise ValueError('coupled reference requires a complete FP64 formula and FP32 replay proof: '
                                 f'{identity}')
            for name, field in proof_fields.items():
                if (field.get('formula_identity_passed') is not True or field.get('fp32_replay_exact') is not True
                        or (not sample[name]['availability_only'] and
                            (field.get('finite') is not True or field['max_abs'] > policy['oracle_max_abs']
                             or (field['relative_l2'] or 0.) > policy['oracle_relative_l2']))):
                    raise ValueError(f'coupled reference formula identity or FP32 replay failed: {identity}, {name}')
            for name, metric in sample.items():
                if (metric['shape'] != shapes[name] or not metric['availability_passed']
                        or not metric['oracle_verified'] or not metric['selection_verified']
                        or (not metric['availability_only'] and not metric['finite'])):
                    raise ValueError(f'coupled reference shape, finite values or independent proofs failed: '
                                     f'{identity}, {name}')
                if name.startswith('sgd/'):
                    geometry = metric['sgd_geometry']
                    if (geometry['numel'] != math.prod(shapes[name]) or geometry['single_axpy_verified'] is not True
                            or geometry['parameter_norm'] != metric['expected_norm']):
                        raise ValueError('coupled SGD requires independent oracle geometry and one verified update')
                    gradient = sample['gradient/'+name[4:]]
                    if geometry['gradient_active'] == gradient['availability_only']:
                        raise ValueError('coupled SGD gradient availability differs')
                    if geometry['gradient_active'] and geometry['gradient_norm'] != gradient['expected_norm']:
                        raise ValueError('coupled SGD gradient norm differs from its oracle field')
        groups.setdefault(f"H{case['heads']}/{case['mode']}", []).append(case)
    if seen != expected:
        raise ValueError('coupled reference seed/shape/mode coverage is incomplete')
    return groups


def fit_coupled_model_reference_profile(policy: dict, cases: list[dict], environment: dict) -> dict:
    """Fit non-SGD references and verify coupled SGD without fitting its errors."""
    if not isinstance(environment, dict) or not environment:
        raise ValueError('coupled reference requires a declared nonempty environment')
    groups = _groups(policy, cases)
    budgets, accepted = {}, True
    for bucket, values in groups.items():
        fields = {}
        names = values[0]['samples'][0]
        for name in sorted(names, key=lambda field: field.startswith('sgd/')):
            fitting = [sample[name] for case in values if case['partition'] == 'calibration'
                       for sample in case['samples']]
            checking = [sample[name] for case in values if case['partition'] == 'held_out'
                        for sample in case['samples']]
            availability = {metric['availability_only'] for metric in fitting+checking}
            if len(availability) != 1:
                raise ValueError('coupled reference active regimes changed across independent samples')
            if True in availability:
                entry = {'availability_only': True, 'held_out_passed': True}
            elif not name.startswith('sgd/'):
                budget = fit_reference_budget(fitting, margin=policy['margin'], zero_atol=policy['zero_atol'])
                passed = all(reference_budget_passes(metric, budget) for metric in checking)
                entry = {'availability_only': False, 'limits': asdict(budget), 'held_out_passed': passed}
                accepted = accepted and passed
            else:
                gradient = fields['gradient/'+name[4:]]
                budget = None if gradient['availability_only'] else ReferenceErrorBudget(**gradient['limits'])
                passed = all(_coupled_sgd_passes(metric, budget, policy['learning_rate'])
                             for metric in fitting+checking)
                entry = {'availability_only': False, 'rule': policy['sgd_rule'],
                         'gradient_field': 'gradient/'+name[4:], 'held_out_passed': passed,
                         'calibration_and_held_out_sgd_passed': passed}
                accepted = accepted and passed
            fields[name] = {**entry, 'shape': fitting[0]['shape'], 'held_out_samples': len(checking)}
        budgets[bucket] = fields
    return {'schema': 2, 'status': 'validated' if accepted else 'held_out_failed',
            'accepted_tolerance_profile': accepted, 'candidate_measurements': False,
            'policy': policy, 'policy_sha256': policy_sha256(policy), 'environment': environment,
            'cases': cases, 'budgets': budgets, 'case_count': len(cases), 'full_model_acceptance': False,
            'sgd_error_fitting': False,
            'reference_cases_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}


def validate_coupled_model_reference_profile(profile: dict, *, expected_policy_sha256: str,
                                             expected_environment: dict) -> dict:
    """Recompute provenance, non-SGD budgets and every coupled SGD check before consumption."""
    try:
        if (profile['status'] != 'validated' or profile['accepted_tolerance_profile'] is not True
                or profile['candidate_measurements'] is not False or profile['sgd_error_fitting'] is not False
                or profile['environment'] != expected_environment or not expected_environment
                or profile['policy_sha256'] != expected_policy_sha256
                or policy_sha256(profile['policy']) != expected_policy_sha256):
            raise ValueError('coupled model reference profile is unvalidated or provenance changed')
        recomputed = fit_coupled_model_reference_profile(profile['policy'], profile['cases'], expected_environment)
        if recomputed != profile:
            raise ValueError('coupled model reference profile was modified after independent fitting')
        return recomputed['budgets']
    except (KeyError, TypeError) as error:
        raise ValueError('coupled model reference profile schema is incomplete') from error
