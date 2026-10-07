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
"""CPU acceptance checks prevent a broken stock baseline from masking device errors."""

import unittest

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_cann_matrix import (
    _calibrate,
    _empty_contract,
)


class TestCannMatrixAcceptance(unittest.TestCase):
    """Verify oracle sanity and exact empty-row contracts without device mocks."""

    def test_identical_candidate_and_broken_stock_cannot_pass(self):
        """Enhance/stock agreement is insufficient when both differ greatly from the oracle."""
        reference = {"kl_loss": torch.tensor(0.00002)}
        broken = {"kl_loss": torch.tensor(0.18)}
        checks = _calibrate(broken, broken, reference, torch.ones(1, dtype=torch.bool))
        self.assertFalse(checks["kl_loss"]["stock_calibration_usable"])
        self.assertFalse(checks["kl_loss"]["passed"])

    def test_finite_calibration_preserves_zero_and_nonzero_acceptance(self):
        """Exact zero derivatives pass without cosine/relative-error singularities."""
        expected = {"output": torch.tensor([1.0, 2.0]), "index_q": torch.zeros(2)}
        candidate = {"output": torch.tensor([1.001, 2.001]), "index_q": torch.zeros(2)}
        baseline = {"output": torch.tensor([1.002, 2.002]), "index_q": torch.zeros(2)}
        checks = _calibrate(candidate, baseline, expected, torch.ones(1, dtype=torch.bool))
        self.assertTrue(all(check["passed"] for check in checks.values()))
        candidate["index_q"][0] = torch.nan
        checks = _calibrate(candidate, baseline, expected, torch.ones(1, dtype=torch.bool))
        self.assertFalse(checks["index_q"]["passed"])

    def test_empty_query_checks_keep_shared_key_gradients_separate(self):
        """Empty Q has zero Q gradients while the same K may be used by another Q."""
        values = {name: torch.zeros(2, 1) for name in
                  ("output", "q_nope", "q_rope", "index_q", "merge_weight")}
        values["lse"] = torch.tensor([[-torch.inf], [0.0]])
        values["compressed_kv"] = torch.ones(2, 1)
        stats = {"maximum": torch.tensor([[-torch.inf], [0.0]]), "sum": torch.tensor([[0.0], [1.0]])}
        empty = torch.tensor([True, False])
        result = _empty_contract(values, stats, empty)
        self.assertTrue(result["output_zero"])
        self.assertTrue(result["query_gradients_zero"])
        self.assertTrue(result["sum_zero"])
        self.assertTrue(result["lse_is_negative_infinity"])
        stats["sum"][0] = 1
        values["q_nope"][0] = torch.nan
        result = _empty_contract(values, stats, empty)
        self.assertFalse(result["sum_zero"])
        self.assertFalse(result["query_gradients_zero"])
