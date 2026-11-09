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
"""CPU regressions for strict LongCat acceptance and useful failure diagnostics."""

import json
import unittest

import torch

from tests.torch.models.longcat_flash._precision import compare_tensors, require_strict_pass


class TestLongcatPrecision(unittest.TestCase):
    """Reject near-zero false passes, metadata errors and nonfinite operands."""

    def test_component_success_cannot_mask_strict_failure(self) -> None:
        """A small global L2 and legacy component success still fail near zero."""
        expected = torch.ones(1024)
        expected[0] = 0
        actual = expected.clone()
        actual[0] = 1e-3
        result = compare_tensors(actual, expected)
        self.assertTrue(result["component_pass"])
        self.assertFalse(result["strict_pass"])
        self.assertEqual(result["strict_bad"], 1)
        self.assertEqual(result["first_strict_failures"][0]["flat_index"], 0)
        with self.assertRaisesRegex(RuntimeError, "strict elementwise precision failed"):
            require_strict_pass([{**result, "all_ranks_strict_pass": False}])

    def test_remote_failure_rejects_a_passing_local_rank(self) -> None:
        """A rank with matching local values must fail when another rank failed."""
        result = compare_tensors(torch.ones(8), torch.ones(8))
        self.assertTrue(result["strict_pass"])
        with self.assertRaises(RuntimeError):
            require_strict_pass([{**result, "all_ranks_strict_pass": False}])
        require_strict_pass([{**result, "all_ranks_strict_pass": True}])

    def test_no_completed_records_is_not_a_pass(self) -> None:
        """An empty report cannot establish numerical acceptance."""
        with self.assertRaises(RuntimeError):
            require_strict_pass([])

    def test_same_elements_with_wrong_shape_fail(self) -> None:
        """Flattening must not hide a shape or ownership mismatch."""
        result = compare_tensors(torch.ones(2, 3), torch.ones(3, 2))
        self.assertFalse(result["strict_pass"])
        self.assertEqual(result["reason"], "shape mismatch")

    def test_dtype_mismatch_fails_even_with_identical_values(self) -> None:
        """Silent BF16/FP32 substitution is reported as a metadata failure."""
        result = compare_tensors(torch.ones(4, dtype=torch.bfloat16), torch.ones(4))
        self.assertFalse(result["strict_pass"])
        self.assertFalse(result["metadata_pass"])

    def test_nonfinite_values_fail_and_serialize_as_valid_json(self) -> None:
        """Matching NaNs or infinities cannot produce success or a zero error."""
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                operand = torch.tensor([1.0, value])
                result = compare_tensors(operand, operand)
                self.assertFalse(result["strict_pass"])
                self.assertEqual(result["actual_nonfinite"], 1)
                self.assertEqual(result["reference_nonfinite"], 1)
                self.assertIsNone(result["relative_l2"])
                json.dumps(result, allow_nan=False)

    def test_empty_and_zero_tensors_have_finite_exact_metrics(self) -> None:
        """Empty assignments and zero identity tails are legitimate exact values."""
        for shape in ((0, 4), (4, 4)):
            with self.subTest(shape=shape):
                operand = torch.zeros(shape, dtype=torch.bfloat16)
                result = compare_tensors(operand, operand)
                self.assertTrue(result["strict_pass"])
                self.assertEqual(result["relative_l2"], 0)
                self.assertEqual(result["max_abs"], 0)


if __name__ == "__main__":
    unittest.main()
