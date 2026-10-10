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
"""CPU-only reference error contracts; never select a training backend or clear model gates."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ReferenceErrorBudget:
    """Independent reference envelope with a separate absolute rule for a zero oracle."""

    relative_l2: float | None
    cosine_error: float | None
    max_abs_over_rms: float | None
    zero_atol: float

    def __post_init__(self) -> None:
        for value in (self.relative_l2, self.cosine_error, self.max_abs_over_rms, self.zero_atol):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError("reference error budgets must be finite nonnegative numbers")
        limits = (self.relative_l2, self.cosine_error, self.max_abs_over_rms)
        if any(value is None for value in limits) and not all(value is None for value in limits):
            raise ValueError("normalized reference limits must be present together")
        if self.zero_atol is None:
            raise ValueError("a reference budget requires an explicit zero-oracle absolute limit")


def gradient_error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Measure CPU gradients in FP64 without inventing a denominator for a zero oracle."""
    if (actual.device.type != "cpu" or expected.device.type != "cpu" or actual.shape != expected.shape
            or not actual.is_floating_point() or not expected.is_floating_point()):
        raise ValueError("reference metrics require matching floating-point CPU tensors")
    left, right = actual.detach().double(), expected.detach().double()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return {"finite": False, "expected_norm": None, "relative_l2": None,
                "cosine_error": None, "max_abs_over_rms": None, "max_abs": None}
    left_norm, right_norm = float(left.norm()), float(right.norm())
    difference = left - right
    absolute = float(difference.abs().max()) if difference.numel() else 0.
    metrics = {"finite": True, "expected_norm": right_norm, "max_abs": absolute,
               "relative_l2": None, "cosine_error": None, "max_abs_over_rms": None}
    if right_norm:
        cosine = float((left * right).sum()) / (left_norm * right_norm) if left_norm else 0.
        metrics.update(relative_l2=float(difference.norm()) / right_norm,
                       cosine_error=max(0., 1. - min(1., max(-1., cosine))),
                       max_abs_over_rms=absolute / (right_norm / math.sqrt(right.numel())))
    metrics["finite"] = all(value is None or math.isfinite(value) for name, value in metrics.items()
                          if name != "finite")
    return metrics


def fit_reference_budget(samples: Iterable[dict], *, margin: float, zero_atol: float) -> ReferenceErrorBudget:
    """Fit independent reference samples under an explicitly preselected margin.

    Held-out or candidate measurements must not be passed to this function.
    Environment, fixture scope and provenance are supplied by the calibration caller.
    """
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 1:
        raise ValueError("reference margin must be a finite number at least one")
    values = tuple(samples)
    if not values or any(not metric["finite"] for metric in values):
        raise ValueError("reference fitting requires nonempty finite measurements")
    normalized = tuple(metric for metric in values if metric["expected_norm"] > 0)
    # Cosine error is quadratic in angular error, while the other bounds are linear norms.
    limits = tuple(max(metric[name] for metric in normalized) * (margin**2 if name == "cosine_error" else margin)
                   if normalized else None
                   for name in ("relative_l2", "cosine_error", "max_abs_over_rms"))
    return ReferenceErrorBudget(*limits, zero_atol)


def reference_budget_passes(metrics: dict, budget: ReferenceErrorBudget) -> bool:
    """Reject nonfinite values and unseen normalized regimes; use absolute error for a zero oracle."""
    if not metrics["finite"]:
        return False
    if metrics["expected_norm"] == 0:
        return metrics["max_abs"] <= budget.zero_atol
    if budget.relative_l2 is None:
        return False
    return all(metrics[name] <= limit for name, limit in
               (("relative_l2", budget.relative_l2), ("cosine_error", budget.cosine_error),
                ("max_abs_over_rms", budget.max_abs_over_rms)))


def reference_absolute_limit(expected: torch.Tensor, budget: ReferenceErrorBudget) -> float:
    """Convert an independently fitted RMS-scaled maximum into an absolute partial-gradient bound."""
    metric = gradient_error_metrics(expected, expected)
    if not metric["finite"]:
        raise ValueError("owner reference gradients must be finite")
    if metric["expected_norm"] == 0:
        return budget.zero_atol
    if budget.max_abs_over_rms is None:
        raise ValueError("reference profile did not calibrate nonzero owner contributions")
    return budget.max_abs_over_rms * metric["expected_norm"] / math.sqrt(expected.numel())


def ordered_fp32_owner_bound(contributions: Iterable[torch.Tensor], absolute_limits: Iterable[float]) -> torch.Tensor:
    """Bound ordered FP32 peer addition using partial error budgets and the standard gamma(n) rounding bound."""
    parts, limits = tuple(contributions), tuple(absolute_limits)
    if not parts or len(parts) != len(limits):
        raise ValueError("owner bounds require one absolute limit per peer contribution")
    if any(part.device.type != "cpu" or part.dtype != torch.float32 or part.shape != parts[0].shape
           or not bool(torch.isfinite(part).all()) for part in parts):
        raise ValueError("owner contribution evidence must be matching finite FP32 CPU tensors")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in limits):
        raise ValueError("owner partial limits must be finite nonnegative numbers")
    steps = len(parts) - 1
    epsilon = torch.finfo(torch.float32).eps
    if steps * epsilon >= 1:
        raise ValueError("owner reduction exceeds the valid FP32 rounding-bound regime")
    gamma = steps * epsilon / (1 - steps * epsilon)
    magnitude = sum(part.double().abs() for part in parts)
    return magnitude * gamma + sum(limits)


def validate_reference_seed_sets(
    fit_seeds: Iterable[int], held_out_seeds: Iterable[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Freeze nonempty, independent seed sets before reference fitting or device sampling.

    Args:
        fit_seeds: Unique nonnegative Torch RNG seeds used only to fit budgets.
        held_out_seeds: Unique seeds used only to check those frozen budgets.

    Returns:
        Materialized fitting and held-out tuples with no overlap.
    """
    fit, held_out = tuple(fit_seeds), tuple(held_out_seeds)
    if (not fit or not held_out
            or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
                   for seed in fit + held_out)
            or len(set(fit + held_out)) != len(fit + held_out)):
        raise ValueError("reference fitting and held-out seeds must be nonempty, unique, disjoint RNG integers")
    return fit, held_out


def _profile_groups(profile):
    policy = profile["policy"]
    fit, held_out = validate_reference_seed_sets(policy["fit_seeds"], policy["held_out_seeds"])
    buckets = {f"H{heads}/CP{size}/{pattern}/rank{rank}/{mode}" for heads in policy["heads"]
               for size in policy["cp_sizes"] for pattern in policy["patterns"] for rank in range(size)
               for mode in policy["tiles"]}
    groups = {name: [] for name in buckets}
    seen = set()
    for case in profile["cases"]:
        pair = (case["bucket"], case["seed"])
        if pair in seen or pair[0] not in buckets or pair[1] not in fit + held_out:
            raise ValueError("reference profile contains duplicate or unsupported samples")
        seen.add(pair)
        partition = "calibration" if pair[1] in fit else "held_out"
        if (case["partition"] != partition or not case["separate_and_shared_fp64_oracles_agree"]
                or len(case["samples"]) != 2):
            raise ValueError("reference profile lacks independent oracle or repeated sample evidence")
        if any(not metric["passed"] for sample in case["samples"] for metric in sample["raw_merged_vs_autograd"]):
            raise ValueError("stock raw and autograd reference consistency failed")
        groups[pair[0]].append(case)
    expected = {(bucket, seed) for bucket in buckets for seed in fit + held_out}
    if seen != expected or set(profile["budgets"]) != buckets or profile["case_count"] != len(expected):
        raise ValueError("reference profile coverage is incomplete")
    return groups


def _validated_profile_budgets(profile, groups):
    names = ("query", "compressed_kv", "query_rope", "key_rope")
    result = {}
    policy = profile["policy"]
    for bucket, cases in groups.items():
        if set(profile["budgets"][bucket]) != set(names):
            raise ValueError("reference profile must certify all four main gradient fields")
        fields = {}
        for index, name in enumerate(names):
            fitting = [sample["profile_metrics"][index] for case in cases if case["partition"] == "calibration"
                       for sample in case["samples"]]
            validation = [sample["profile_metrics"][index] for case in cases if case["partition"] == "held_out"
                          for sample in case["samples"]]
            fitted = fit_reference_budget(fitting, margin=policy["margin"], zero_atol=policy["zero_atol"])
            recorded = profile["budgets"][bucket][name]
            if (recorded["limits"] != asdict(fitted) or recorded["held_out_samples"] != len(validation)
                    or recorded["held_out_passed"] is not True
                    or not all(reference_budget_passes(metric, fitted) for metric in validation)):
                raise ValueError("reference profile limits or held-out verification were modified")
            fields[name] = fitted
        result[bucket] = fields
    return result


def validate_reference_profile(profile: dict, *, expected_versions: dict,
                               expected_library_hashes: dict, expected_policy_sha256: str) -> dict:
    """Recompute frozen budgets and held-out evidence before allowing a scoped validator to use them.

    Version and primary-reference library identities must be supplied from the
    current environment. The returned budgets never certify an unsupported bucket.
    """
    try:
        if (profile["status"] != "validated" or profile["accepted_tolerance_profile"] is not True
                or profile["candidate_measurements"] is not False
                or profile["policy"]["candidate_measurements"] is not False):
            raise ValueError("a reference profile requires validated independent evidence")
        if (any(profile[key] != value for key, value in expected_versions.items())
                or not expected_library_hashes or profile["loaded_library_sha256"] != expected_library_hashes):
            raise ValueError("reference profile environment or library identity mismatch")
        policy_text = json.dumps(profile["policy"], sort_keys=True, indent=2) + "\n"
        if (hashlib.sha256(policy_text.encode()).hexdigest() != profile["policy_sha256"]
                or profile["policy_sha256"] != expected_policy_sha256):
            raise ValueError("reference profile policy was modified after freezing")
        return _validated_profile_budgets(profile, _profile_groups(profile))
    except (KeyError, TypeError) as error:
        raise ValueError("reference profile schema is incomplete") from error
