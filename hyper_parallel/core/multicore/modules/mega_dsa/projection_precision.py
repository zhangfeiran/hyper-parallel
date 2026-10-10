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
"""CPU-only integrity checks for independently frozen short projection reference profiles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from itertools import product

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    ReferenceErrorBudget, fit_reference_budget, reference_absolute_limit,
    reference_budget_passes, validate_reference_seed_sets,
)

_FIELDS = {"lm_only": ("objective", "auxiliary_loss", "hidden", "query_projection", "compressed_projection",
                      "query_rope_projection", "key_rope_projection", "output_projection"),
           "kl_only": ("objective", "auxiliary_loss", "index_query_projection", "index_key_projection",
                       "merge_projection")}
_FIELDS["lm_plus_kl"] = tuple(dict.fromkeys((*_FIELDS["lm_only"], *_FIELDS["kl_only"])))
_POLICY = {"schema": 1, "scope": "complete-query fixed-Native-value projection VJP reference",
           "candidate_measurements": False, "margin": 2., "cosine_margin": 4., "zero_atol": 2e-5,
           "lengths": [3, 10], "heads": [32, 64], "weight_dtypes": ["torch.bfloat16", "torch.float32"],
           "query_scope": "complete global queries", "modes": ["lm_only", "kl_only", "lm_plus_kl"],
           "coefficient": .3, "auxiliary": -2.5, "repeats": 2, "attention_scale": 192**-.5,
           "main_owner_contract": "FP32 dK+dV then one BF16 cast",
           "index_owner_contract": "FP32 K scale then upstream then one BF16 cast",
           "raw_stock_consistency": {"non_deterministic": True, "rtol": .02, "atol": 2e-5, "relative_l2": .02}}


def _samples(profile, fitting, held_out):
    expected = set(product((32, 64), _POLICY["weight_dtypes"], _POLICY["modes"], fitting+held_out, (0, 1)))
    seen, grouped = set(), {}
    for sample in profile["samples"]:
        identity = tuple(sample[key] for key in ("heads", "weight_dtype", "mode", "seed", "repeat"))
        if identity not in expected or identity in seen:
            raise ValueError("projection reference has duplicate or unsupported samples")
        seen.add(identity)
        if (set(sample["metrics"]) != set(_FIELDS[sample["mode"]])
                or set(sample["original_pointwise"]) != set(sample["metrics"])
                or sample["reference_native_post_bits_exact"] is not True
                or len(sample["raw_gradients_stock_consistency"]) != 8
                or any(item["passed"] is not True for item in sample["raw_gradients_stock_consistency"])):
            raise ValueError("projection reference lacks complete gradient, conversion or stock consistency evidence")
        bucket = f"H{sample['heads']}/{sample['weight_dtype']}/{sample['mode']}"
        grouped.setdefault(bucket, []).append(sample)
    if seen != expected or set(profile["budgets"]) != set(grouped):
        raise ValueError("projection reference coverage is incomplete")
    return grouped


def validate_projection_reference_profile(profile: dict, *, expected_versions: dict,
                                          expected_library_hashes: dict, expected_policy_sha256: str) -> dict:
    """Recompute every independent budget and held-out verdict before candidate validation.

    Only the fixed full-query (3,10), default projection geometry and declared
    coefficient/upstream scope is represented. Runtime validators must reject
    subsets, different geometry or objectives before using these budgets.
    """
    try:
        policy = profile["policy"]
        if (profile["status"] != "validated" or profile["accepted_tolerance_profile"] is not True
                or profile["candidate_measurements"] is not False
                or any(policy.get(key) != value for key, value in _POLICY.items())):
            raise ValueError("projection reference requires validated independent frozen policy evidence")
        policy_text = json.dumps(policy, sort_keys=True, indent=2)+"\n"
        digest = hashlib.sha256(policy_text.encode()).hexdigest()
        if digest != profile["policy_sha256"] or digest != expected_policy_sha256:
            raise ValueError("projection reference policy identity mismatch")
        if (not expected_library_hashes or profile["loaded_library_sha256"] != expected_library_hashes
                or any(profile.get(key) != value for key, value in expected_versions.items())):
            raise ValueError("projection reference environment identity mismatch")
        fitting, held_out = validate_reference_seed_sets(policy["fit_seeds"], policy["held_out_seeds"])
        if len(fitting) != 16 or len(held_out) != 8:
            raise ValueError("projection reference requires all 16 fitting and 8 held-out seeds")
        grouped = _samples(profile, fitting, held_out)
        result = {}
        for bucket, members in grouped.items():
            if set(profile["budgets"][bucket]) != set(members[0]["metrics"]):
                raise ValueError("projection reference budget fields are incomplete")
            fields = {}
            for name in members[0]["metrics"]:
                fit_metrics = [sample["metrics"][name] for sample in members if sample["seed"] in fitting]
                held_metrics = [sample["metrics"][name] for sample in members if sample["seed"] in held_out]
                budget = fit_reference_budget(fit_metrics, margin=2., zero_atol=2e-5)
                recorded = profile["budgets"][bucket][name]
                if (recorded["limits"] != asdict(budget) or recorded["held_out_samples"] != len(held_metrics)
                        or recorded["held_out_passed"] is not True
                        or not all(reference_budget_passes(metric, budget) for metric in held_metrics)):
                    raise ValueError("projection reference budget or held-out evidence was modified")
                fields[name] = budget
            result[bucket] = fields
        return result
    except (KeyError, TypeError) as error:
        raise ValueError("projection reference schema is incomplete") from error


def validate_loaded_projection_reference(profile: dict, loaded_library_hashes: dict) -> None:
    """Require every calibrated primitive library while allowing separately audited CP transport libraries."""
    expected = profile.get("loaded_library_sha256")
    if not expected or any(loaded_library_hashes.get(name) != digest for name, digest in expected.items()):
        raise ValueError("loaded projection reference libraries differ from the frozen profile")


def coupled_projection_objective_check(
    objectives: tuple[torch.Tensor, ...], auxiliary_losses: tuple[torch.Tensor, ...],
    expected_objective: torch.Tensor, expected_auxiliary: torch.Tensor, *, upstream: float,
    divisor: int, main_budget: ReferenceErrorBudget, auxiliary_budget: ReferenceErrorBudget,
) -> dict:
    """Bound a coupled scalar loss from independent component budgets and FP32 rounding.

    A nearly cancelled LM + upstream*KL does not acquire an arbitrary relative
    tolerance. Triangle inequality propagates the separately fitted component
    errors. Two FP32 multiply/add rounds and ordered peer reductions contribute
    analytical gamma bounds. The inferred main term and auxiliary term must
    each satisfy their own bounds; projection gradient budgets remain unchanged.
    """
    tensors = (*objectives, *auxiliary_losses, expected_objective, expected_auxiliary)
    if (not objectives or len(objectives) != len(auxiliary_losses)
            or any(value.device.type != "cpu" or value.numel() != 1 or not value.is_floating_point()
                   for value in tensors) or not all(bool(torch.isfinite(value).all()) for value in tensors)
            or not isinstance(divisor, int) or isinstance(divisor, bool) or divisor <= 0
            or not math.isfinite(upstream)):
        raise ValueError("coupled objective bounds require finite CPU scalars and a positive reducer divisor")
    left = sum(value.double() for value in objectives)/divisor
    auxiliary = sum(value.double() for value in auxiliary_losses)/divisor
    main_expected = expected_objective.double()-upstream*expected_auxiliary.double()
    main_limit = reference_absolute_limit(main_expected, main_budget)
    auxiliary_limit = reference_absolute_limit(expected_auxiliary.double(), auxiliary_budget)
    epsilon = torch.finfo(torch.float32).eps
    peer_steps = len(objectives)-1
    if peer_steps*epsilon >= 1:
        raise ValueError("coupled objective peer count exceeds the FP32 gamma-bound regime")
    local_magnitude = sum(abs(float(value.double()-upstream*loss.double()))+abs(upstream*float(loss))
                          for value, loss in zip(objectives, auxiliary_losses))/divisor
    multiply_add_bound = (2*epsilon/(1-2*epsilon))*local_magnitude
    peer_magnitude = sum(abs(float(value))+abs(upstream*float(loss))
                         for value, loss in zip(objectives, auxiliary_losses))/divisor
    peer_bound = (peer_steps*epsilon/(1-peer_steps*epsilon))*peer_magnitude
    # Bounds are relative to observed magnitudes, so reverse the gamma inequality.
    rounding_bound = (multiply_add_bound+peer_bound)/(1-2*epsilon)
    error = abs(float(left-expected_objective.double()))
    main_error = abs(float(left-upstream*auxiliary-main_expected))
    auxiliary_error = abs(float(auxiliary-expected_auxiliary.double()))
    limit = main_limit+abs(upstream)*auxiliary_limit+rounding_bound
    passed = (error <= limit and main_error <= main_limit+rounding_bound
              and auxiliary_error <= auxiliary_limit+peer_bound)
    return {"passed": passed, "absolute_error": error, "absolute_limit": limit,
            "inferred_main_error": main_error, "main_reference_limit": main_limit,
            "auxiliary_error": auxiliary_error, "auxiliary_reference_limit": auxiliary_limit,
            "analytical_fp32_rounding_bound": rounding_bound}
