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
"""Consume a validated coupled model profile only for fresh complete-global tensors."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.model_global_fields import (
    _declared_shapes,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_sgd_precision import (
    coupled_sgd_budget,
    validate_coupled_model_reference_profile,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    ReferenceErrorBudget,
    gradient_error_metrics,
    reference_budget_passes,
)


def verify_candidate_reference_environment(reference: dict, actual: dict, *, candidate_libraries: dict) -> None:
    """Require the reference environment unchanged and every additional native library explicitly frozen.

    A stock calibration does not necessarily load the candidate's native
    transport libraries. Extra mapped libraries are admitted only by the
    caller's frozen filename/hash declaration and retained in the run report.
    Reference paths and hashes must all still be present exactly.
    """
    library_key = 'loaded_library_sha256'
    if (set(reference) != set(actual) or library_key not in reference
            or any(actual[name] != value for name, value in reference.items() if name != library_key)):
        raise ValueError('candidate reference environment identity differs from calibration')
    expected_libraries, observed = reference[library_key], actual[library_key]
    if not expected_libraries or any(observed.get(name) != digest for name, digest in expected_libraries.items()):
        raise ValueError('candidate reference libraries differ from calibration')
    for name, digest in observed.items():
        if name not in expected_libraries and candidate_libraries.get(Path(name).name) != digest:
            raise ValueError(f'candidate additional library is outside the frozen payload: {name}')


def single_sgd_control(initial: torch.Tensor, gradient: torch.Tensor | None, *, learning_rate: float) -> torch.Tensor:
    """Produce an independent single-AXPY control on the same device before archiving candidate data.

    The caller supplies original complete parameters and actual complete
    gradients. This controls update count and scaling independently from the
    normalized accuracy comparison with the FP32 global objective.
    """
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('single SGD control requires a positive finite learning rate')
    if initial.dtype != torch.bfloat16 or (gradient is not None and
            (gradient.shape != initial.shape or gradient.dtype != initial.dtype or gradient.device != initial.device)):
        raise ValueError('single SGD control requires matching complete BF16 parameter and gradient tensors')
    result = initial.detach().clone()
    if gradient is not None:
        result.add_(gradient.detach(), alpha=-learning_rate)
    return result


def collect_global_model_diagnostics(actual: dict, expected: dict, initial_parameters: dict, *,
                                     heads: int, learning_rate: float,
                                     declared_shapes: dict[str, list[int]] | None = None) -> dict:
    """Record complete global metrics and mechanical update checks without accepting any profile.

    This entry never consumes budgets or reports numerical acceptance. It can
    diagnose a candidate while reference calibration remains unvalidated.
    Complete original parameters are still required for same-device SGD controls.
    """
    shapes = _declared_shapes(heads, declared_shapes)
    if (set(actual) != set(shapes) or set(expected) != set(shapes)
            or set(initial_parameters) != {name[4:] for name in shapes if name.startswith('sgd/')}):
        raise ValueError('global diagnostics require all complete model fields and original parameters')
    metrics = {}
    for name, value in actual.items():
        target = expected[name]
        if value is None or target is None:
            if name.startswith('sgd/'):
                raise ValueError('global diagnostics require every updated parameter')
            metrics[name] = {'availability_passed': value is target, 'availability_only': True}
        else:
            if (not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor)
                    or list(value.shape) != shapes[name] or value.shape != target.shape):
                raise ValueError('global diagnostics reject owner-local tensors and partial head shards')
            metrics[name] = gradient_error_metrics(value.detach().float().cpu(), target.detach().float().cpu())
        if name.startswith('sgd/'):
            gradient = actual['gradient/'+name[4:]]
            control = single_sgd_control(initial_parameters[name[4:]].to(value.device),
                None if gradient is None else gradient.to(value.device), learning_rate=learning_rate)
            metrics[name]['single_axpy_verified'] = bool(torch.equal(value, control))
    return {'passed': False, 'metrics_only': True, 'profile_consumed_for_acceptance': False,
            'fields': metrics, 'full_model_acceptance': False}


def evaluate_coupled_global_model_reference(profile: dict, actual: dict, expected: dict, initial_parameters: dict, *,
                                            heads: int, mode: str, candidate_seed: int,
                                            expected_policy_sha256: str, expected_environment: dict) -> dict:
    """Require gradient accuracy, physical SGD accuracy and an exact independent update control.

    This evaluator neither assembles owner/head shards nor fits candidate
    measurements. SGD controls are computed from original parameters and actual
    gradients on the updated parameter's device with the declared learning rate.
    Local shards, old seeds, unvalidated profiles and missing controls are rejected.
    """
    budgets = validate_coupled_model_reference_profile(profile,
        expected_policy_sha256=expected_policy_sha256, expected_environment=expected_environment)
    if (isinstance(candidate_seed, bool) or not isinstance(candidate_seed, int)
            or candidate_seed not in profile['policy']['reserved_candidate_seeds']):
        raise ValueError('coupled model evaluation requires a fresh reserved candidate seed')
    bucket = f'H{heads}/{mode}'
    if bucket not in budgets or set(actual) != set(expected) or set(actual) != set(budgets[bucket]):
        raise ValueError('coupled model evaluation requires every complete global model field')
    required_controls = {name[4:] for name in actual if name.startswith('sgd/')}
    if set(initial_parameters) != required_controls:
        raise ValueError('coupled model evaluation requires every original complete parameter')
    fields = {}
    for name, entry in budgets[bucket].items():
        left, right = actual[name], expected[name]
        if entry['availability_only']:
            fields[name] = {'passed': left is None and right is None, 'availability_only': True}
            continue
        if left is None or right is None:
            fields[name] = {'passed': False, 'availability_only': True}
            continue
        if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
                or not left.is_floating_point() or not right.is_floating_point()
                or list(left.shape) != entry['shape'] or left.shape != right.shape):
            raise ValueError('coupled model evaluation requires complete global floating tensors, not local shards')
        updated = left
        left, right = left.detach().cpu(), right.detach().float().cpu()
        metric = gradient_error_metrics(left.float(), right)
        if not name.startswith('sgd/'):
            budget = ReferenceErrorBudget(**entry['limits'])
            fields[name] = {**metric, 'passed': reference_budget_passes(metric, budget)}
            continue
        initial = initial_parameters[name[4:]]
        if (not isinstance(initial, torch.Tensor) or initial.shape != left.shape
                or initial.dtype != torch.bfloat16 or left.dtype != torch.bfloat16):
            raise ValueError('coupled SGD initial and updated parameters must be complete BF16 tensors')
        if expected_environment.get('soc') and updated.device.type != 'npu':
            raise ValueError('NPU-calibrated SGD must be checked on its original NPU backend')
        actual_gradient = actual['gradient/'+name[4:]]
        control = single_sgd_control(initial.to(updated.device),
            None if actual_gradient is None else actual_gradient.to(updated.device),
            learning_rate=profile['policy']['learning_rate'])
        gradient = expected['gradient/'+name[4:]]
        gradient_entry = budgets[bucket]['gradient/'+name[4:]]
        active = gradient is not None
        geometry = {'numel': right.numel(), 'parameter_norm': float(right.double().norm()),
                    'parameter_max_abs': float(right.double().abs().max()), 'gradient_active': active,
                    'gradient_norm': 0. if not active else float(gradient.detach().cpu().double().norm()),
                    'gradient_max_abs': 0. if not active else float(gradient.detach().cpu().double().abs().max())}
        gradient_budget = (None if gradient_entry['availability_only']
                           else ReferenceErrorBudget(**gradient_entry['limits']))
        budget = coupled_sgd_budget(geometry, gradient_budget, learning_rate=profile['policy']['learning_rate'])
        exact = bool(torch.equal(left, control.detach().cpu()))
        # An exactly unchanged parameter has zero error even if cosine division rounds below one.
        accuracy = (reference_budget_passes(metric, budget) if active else
                    metric['finite'] and metric['max_abs'] == 0.)
        fields[name] = {**metric, 'single_axpy_verified': exact,
                        'passed': exact and accuracy}
    return {'passed': all(field['passed'] for field in fields.values()), 'fields': fields,
            'bucket': bucket, 'candidate_seed': candidate_seed, 'policy_sha256': expected_policy_sha256,
            'full_model_acceptance': False}
