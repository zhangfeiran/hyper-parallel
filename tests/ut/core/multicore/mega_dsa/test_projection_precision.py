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
"""Reject stale, incomplete or candidate-contaminated projection reference profiles on CPU."""

import copy
import hashlib
import json
import unittest
from itertools import product

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import projection_precision as precision


class TestProjectionReferenceIntegrity(unittest.TestCase):
    """Independent budgets remain frozen when applying them to an implementation candidate."""

    def setUp(self) -> None:
        """Build complete synthetic reference evidence with independently known metric limits."""
        policy = copy.deepcopy(precision._POLICY)
        policy.update(fit_seeds=list(range(40, 56)), held_out_seeds=list(range(60, 68)))
        digest = hashlib.sha256((json.dumps(policy, sort_keys=True, indent=2)+"\n").encode()).hexdigest()
        self.profile = {"status": "validated", "accepted_tolerance_profile": True, "candidate_measurements": False,
                        "policy": policy, "policy_sha256": digest, "loaded_library_sha256": {"reference.so": "abc"},
                        "torch_version": "reference", "samples": [], "budgets": {}}
        metric = {"finite": True, "expected_norm": 1., "relative_l2": .001,
                  "cosine_error": .000001, "max_abs_over_rms": .003, "max_abs": .0001}
        limits = {"relative_l2": .002, "cosine_error": .000004, "max_abs_over_rms": .006, "zero_atol": 2e-5}
        for heads, dtype, mode, seed, repeat in product(
                (32, 64), policy["weight_dtypes"], policy["modes"],
                policy["fit_seeds"]+policy["held_out_seeds"], (0, 1)):
            fields = precision._FIELDS[mode]
            self.profile["samples"].append({"heads": heads, "weight_dtype": dtype, "mode": mode,
                                            "seed": seed, "repeat": repeat,
                                            "metrics": {name: dict(metric) for name in fields},
                                            "original_pointwise": {name: {"passed": False} for name in fields},
                                            "reference_native_post_bits_exact": True,
                                            "raw_gradients_stock_consistency": [{"passed": True}]*8})
            bucket = f"H{heads}/{dtype}/{mode}"
            self.profile["budgets"][bucket] = {name: {"limits": dict(limits), "held_out_samples": 16,
                                                     "held_out_passed": True} for name in fields}
        self.arguments = {"expected_versions": {"torch_version": "reference"},
                          "expected_library_hashes": {"reference.so": "abc"}, "expected_policy_sha256": digest}

    def test_complete_profile_preserves_original_pointwise_failure(self):
        """A recomputed reference envelope does not rewrite a separate failed pointwise gate."""
        budgets = precision.validate_projection_reference_profile(self.profile, **self.arguments)
        self.assertEqual(len(budgets), 12)
        self.assertFalse(self.profile["samples"][0]["original_pointwise"]["hidden"]["passed"])
        self.assertEqual(budgets["H32/torch.bfloat16/lm_only"]["hidden"].relative_l2, .002)

    def test_rejects_modified_budget_held_out_coverage_and_reference_origin(self):
        """Missing cases, inflated limits, failed held-out values and candidate data cannot authorize a budget."""
        mutations = (
            lambda value: value["samples"].pop(),
            lambda value: value["samples"].append(copy.deepcopy(value["samples"][0])),
            lambda value: value.update(candidate_measurements=True),
            lambda value: value["budgets"]["H32/torch.bfloat16/lm_only"]["hidden"]["limits"].update(relative_l2=.5),
            lambda value: value["samples"][32]["metrics"]["hidden"].update(relative_l2=.5),
            lambda value: value["samples"][0].update(reference_native_post_bits_exact=False),
            lambda value: value["policy"].update(coefficient=.8),
            lambda value: value["samples"][0]["metrics"].pop("hidden"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                profile = copy.deepcopy(self.profile)
                mutate(profile)
                with self.assertRaises(ValueError):
                    precision.validate_projection_reference_profile(profile, **self.arguments)

    def test_rejects_stale_policy_and_library_identity(self):
        """The caller supplies independent environment and policy commitments."""
        for key, value in (("expected_policy_sha256", "stale"),
                           ("expected_versions", {"torch_version": "different"}),
                           ("expected_library_hashes", {"reference.so": "different"})):
            with self.subTest(identity=key):
                arguments = {**self.arguments, key: value}
                with self.assertRaises(ValueError):
                    precision.validate_projection_reference_profile(self.profile, **arguments)

    def test_loaded_reference_allows_additional_cp_libraries(self):
        """CP loads SHMEM in addition to the standalone reference's required primitive libraries."""
        precision.validate_loaded_projection_reference(self.profile, {"reference.so": "abc", "shmem.so": "def"})
        for loaded in ({"shmem.so": "def"}, {"reference.so": "changed", "shmem.so": "def"}, {}):
            with self.subTest(loaded=loaded), self.assertRaises(ValueError):
                precision.validate_loaded_projection_reference(self.profile, loaded)


class TestCoupledProjectionObjective(unittest.TestCase):
    """Scalar cancellation uses component error propagation while gradient envelopes remain fixed."""

    def setUp(self) -> None:
        """Declare independent component budgets before observing coupled values."""
        self.budget = precision.ReferenceErrorBudget(1e-7, 4e-14, 1e-7, 2e-5)
        self.arguments = {"expected_objective": torch.tensor(1e-10, dtype=torch.float64),
                          "expected_auxiliary": torch.tensor(.4, dtype=torch.float64),
                          "upstream": -2.5, "divisor": 1, "main_budget": self.budget,
                          "auxiliary_budget": self.budget}

    def test_cancellation_bound_preserves_independent_budgets(self):
        """A small coupled value can change sign within a proved FP32 error bound."""
        before = copy.deepcopy(self.budget)
        result = precision.coupled_projection_objective_check(
            (torch.tensor(-1e-8),), (torch.tensor(.4),), **self.arguments)
        self.assertTrue(result["passed"])
        self.assertGreater(result["absolute_limit"], result["absolute_error"])
        self.assertEqual(self.budget, before)

    def test_component_corruption_and_nonfinite_values_fail(self):
        """Neither cancellation nor a peer sum hides an inaccurate auxiliary or main component."""
        for objective, loss in ((.001, .4), (-1e-8, .4001)):
            with self.subTest(objective=objective, loss=loss):
                result = precision.coupled_projection_objective_check(
                    (torch.tensor(objective),), (torch.tensor(loss),), **self.arguments)
                self.assertFalse(result["passed"])
        with self.assertRaises(ValueError):
            precision.coupled_projection_objective_check(
                (torch.tensor(float("nan")),), (torch.tensor(.4),), **self.arguments)
