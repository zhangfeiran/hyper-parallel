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
"""Independently fitted error profiles for complete global BF16 model tensors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    ReferenceErrorBudget,
    fit_reference_budget,
    gradient_error_metrics,
    reference_budget_passes,
    validate_reference_seed_sets,
)


def model_reference_policy() -> dict:
    """Freeze a reference-only packed13 H64/H128 model policy before collecting new samples."""
    return {'schema': 1, 'scope': 'complete global model tensors; no local-owner or partial-gradient budgets',
            'candidate_measurements': False, 'heads': [64, 128], 'modes': ['joint', 'freeze', 'zero', 'eval'],
            'fit_seeds': list(range(20261101, 20261117)), 'held_out_seeds': list(range(20261131, 20261139)),
            'reserved_candidate_seeds': list(range(20261151, 20261155)),
            'repeats': 2, 'margin': 1.25, 'zero_atol': 2e-5, 'lengths': [3, 10],
            'hidden': 32, 'q_lora': 64, 'compressed': 512, 'rope': 64, 'nope': 128, 'value': 16,
            'index_heads': 64, 'index_dim': 128, 'topk': 2048, 'coefficient': .3,
            'auxiliary': -1.75, 'learning_rate': .01,
            'fixture_rng': 'model seed; head projections seed+1; hidden seed+2; cotangent seed+3',
            'metrics': ['relative_l2', 'cosine_error', 'max_abs_over_rms'],
            'oracle': 'independent FP32 unabsorbed model at quantized BF16 parameters and inputs'}


def policy_sha256(policy: dict) -> str:
    """Hash the canonical immutable sampling and metric declaration."""
    return hashlib.sha256((json.dumps(policy, sort_keys=True, indent=2)+'\n').encode()).hexdigest()


def model_reference_shapes(heads: int) -> dict[str, list[int]]:
    """Describe every complete global model field admitted by this bounded profile."""
    if heads not in (64, 128):
        raise ValueError('model reference shapes support H64/H128 only')
    parameters = {'linear_qkv.weight': [640, 32], 'q_a_layernorm.weight': [64],
                  'kv_a_layernorm.weight': [512], 'q_b_proj.weight': [heads*192, 64],
                  'kv_b_proj.weight': [heads*144, 512], 'o_proj.weight': [32, heads*16],
                  'indexer.wq_b.weight': [64*128, 64], 'indexer.wk.weight': [128, 32],
                  'indexer.k_norm.weight': [128], 'indexer.k_norm.bias': [128],
                  'indexer.weights_proj.weight': [64, 32]}
    return {'output': [1, 13, 32], 'hidden': [1, 13, 32], 'loss': [],
            **{f'gradient/{name}': shape for name, shape in parameters.items()},
            **{f'sgd/{name}': shape for name, shape in parameters.items()}}


def _groups(policy, cases):
    if policy_sha256(policy) != policy_sha256(model_reference_policy()):
        raise ValueError('model reference policy differs from the frozen supported declaration')
    fit, held_out = validate_reference_seed_sets(policy['fit_seeds'], policy['held_out_seeds'])
    validate_reference_seed_sets(fit+held_out, policy['reserved_candidate_seeds'])
    expected = {(heads, mode, seed) for heads in policy['heads'] for mode in policy['modes'] for seed in fit+held_out}
    seen, groups = set(), {}
    for case in cases:
        identity = (case['heads'], case['mode'], case['seed'])
        if identity not in expected or identity in seen:
            raise ValueError('model reference samples are duplicated or outside the declared seed/shape scope')
        seen.add(identity)
        if case['partition'] != ('calibration' if case['seed'] in fit else 'held_out'):
            raise ValueError('model reference partition does not match the frozen seed policy')
        if len(case['samples']) != policy['repeats']:
            raise ValueError('model reference repeated sample coverage is incomplete')
        shapes = model_reference_shapes(case['heads'])
        for sample in case['samples']:
            if set(sample) != set(shapes):
                raise ValueError('model reference sample is missing an output, gradient or SGD field')
            for name, metric in sample.items():
                if (metric['shape'] != shapes[name] or not metric['availability_passed']
                        or not metric['oracle_verified'] or not metric['selection_verified']):
                    raise ValueError('model reference field shape or gradient isolation differs')
                if not metric['availability_only'] and not metric['finite']:
                    raise ValueError('model reference contains nonfinite measurements')
        groups.setdefault(f"H{case['heads']}/{case['mode']}", []).append(case)
    if seen != expected:
        raise ValueError('model reference seed/shape/mode coverage is incomplete')
    return groups


def fit_model_reference_profile(policy: dict, cases: list[dict], environment: dict) -> dict:
    """Fit reference-only envelopes and reject held-out failures without expanding the frozen margins."""
    if not isinstance(environment, dict) or not environment:
        raise ValueError('model reference fitting requires a declared nonempty environment identity')
    groups = _groups(policy, cases)
    budgets, passed = {}, True
    for bucket, values in groups.items():
        fields = {}
        for name in values[0]['samples'][0]:
            fitting = [sample[name] for case in values if case['partition'] == 'calibration'
                       for sample in case['samples']]
            checking = [sample[name] for case in values if case['partition'] == 'held_out'
                        for sample in case['samples']]
            availability = {metric['availability_only'] for metric in fitting+checking}
            if len(availability) != 1:
                raise ValueError('model reference active field regime changed between independent samples')
            if True in availability:
                entry = {'availability_only': True, 'held_out_passed': True}
            else:
                budget = fit_reference_budget(fitting, margin=policy['margin'], zero_atol=policy['zero_atol'])
                accepted = all(reference_budget_passes(metric, budget) for metric in checking)
                entry = {'availability_only': False, 'limits': asdict(budget), 'held_out_passed': accepted}
                passed = passed and accepted
            fields[name] = {**entry, 'shape': fitting[0]['shape'], 'held_out_samples': len(checking)}
        budgets[bucket] = fields
    return {'schema': 1, 'status': 'validated' if passed else 'held_out_failed',
            'accepted_tolerance_profile': passed, 'candidate_measurements': False,
            'policy': policy, 'policy_sha256': policy_sha256(policy), 'environment': environment,
            'cases': cases, 'budgets': budgets, 'case_count': len(cases), 'full_model_acceptance': False}


def validate_model_reference_profile(profile: dict, *, expected_policy_sha256: str,
                                     expected_environment: dict) -> dict:
    """Recompute the exact profile and reject edited budgets, candidate fitting or environment drift."""
    try:
        if (profile['status'] != 'validated' or profile['accepted_tolerance_profile'] is not True
                or profile['candidate_measurements'] is not False or not expected_environment
                or profile['environment'] != expected_environment
                or policy_sha256(profile['policy']) != expected_policy_sha256
                or profile['policy_sha256'] != expected_policy_sha256):
            raise ValueError('model reference profile is unvalidated or its provenance changed')
        recomputed = fit_model_reference_profile(profile['policy'], profile['cases'], expected_environment)
        if profile != recomputed:
            raise ValueError('model reference profile was modified after independent fitting')
        return recomputed['budgets']
    except (KeyError, TypeError) as error:
        raise ValueError('model reference profile schema is incomplete') from error


def evaluate_global_model_reference(profile: dict, actual: dict, expected: dict, *, heads: int, mode: str,
                                    candidate_seed: int, expected_policy_sha256: str,
                                    expected_environment: dict) -> dict:
    """Check newly reserved candidate samples using immutable complete-global-field budgets.

    Owner-local tensors, partial head shards and previously measured candidate
    seeds are rejected. This function does not assemble distributed tensors,
    select a backend, alter model parameters or change prior acceptance records.
    """
    budgets = validate_model_reference_profile(profile, expected_policy_sha256=expected_policy_sha256,
                                                expected_environment=expected_environment)
    policy = profile['policy']
    if (not isinstance(candidate_seed, int) or isinstance(candidate_seed, bool)
            or candidate_seed not in policy['reserved_candidate_seeds']):
        raise ValueError('model reference evaluation requires a reserved independent candidate seed')
    bucket = f'H{heads}/{mode}'
    if bucket not in budgets or set(actual) != set(budgets[bucket]) or set(expected) != set(actual):
        raise ValueError('model reference candidate fields do not match a complete supported global bucket')
    fields = {}
    for name, specification in budgets[bucket].items():
        left, right = actual[name], expected[name]
        if specification['availability_only']:
            fields[name] = {'passed': left is None and right is None, 'availability_only': True}
            continue
        if left is None or right is None:
            fields[name] = {'passed': False, 'availability_only': True}
            continue
        if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
                or not left.is_floating_point() or not right.is_floating_point()):
            raise ValueError('model reference evaluation requires floating-point tensors')
        if list(left.shape) != specification['shape'] or left.shape != right.shape:
            raise ValueError('model reference candidate must provide complete global tensors, not local shards')
        metric = gradient_error_metrics(left.detach().float().cpu(), right.detach().float().cpu())
        budget = ReferenceErrorBudget(**specification['limits'])
        fields[name] = {**metric, 'passed': reference_budget_passes(metric, budget)}
    return {'passed': all(field['passed'] for field in fields.values()), 'fields': fields,
            'bucket': bucket, 'candidate_seed': candidate_seed, 'policy_sha256': expected_policy_sha256}
