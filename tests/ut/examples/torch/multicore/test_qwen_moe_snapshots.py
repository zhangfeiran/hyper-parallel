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
"""CPU contracts for bounded, backend-neutral Qwen precision snapshots."""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from examples.torch.multicore.mega_moe.qwen_moe_snapshots import compare_snapshot, save_snapshot


class TestQwenMoeSnapshots(unittest.TestCase):
    """Keep original FP32 allclose semantics without holding a second device model."""

    def setUp(self) -> None:
        """Give each test an isolated artifact and a rank-specific comparison contract."""
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.path = self.directory / 'initial_parameters.pt'
        self.contract = {'rank': 0, 'world_size': 8, 'seed': 42, 'optimizer': {'lr': 0.001}}

    def test_snapshot_owns_cpu_values_and_keeps_only_the_view(self) -> None:
        """A small CPU view must not serialize an unrelated oversized backing storage."""
        source = torch.arange(10000, dtype=torch.bfloat16)[17:25]
        expected = source.clone()
        save_snapshot(self.path, self.contract, [('weight', source)])
        source.zero_()
        payload = torch.load(self.path, weights_only=True, map_location='cpu')
        actual = payload['tensors']['weight']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.device.type, 'cpu')
        self.assertEqual(actual.untyped_storage().nbytes(), actual.numel() * actual.element_size())

    def test_chunked_comparison_matches_original_asymmetric_oracle(self) -> None:
        """Chunk size must not change max errors or reverse the reference-relative tolerance."""
        reference = torch.tensor([0.0, 0.02, 1.0, -2.0, 0.001], dtype=torch.float32)
        candidate = reference + torch.tensor([0.001, 0.001, 0.019, -0.03, -0.001])
        save_snapshot(self.path, self.contract, [('weight', reference)])
        difference = (candidate - reference).abs()
        for chunk_size in (1, 2, 100):
            with self.subTest(chunk_size=chunk_size):
                result = compare_snapshot(self.path, self.contract, [('weight', candidate)],
                                          rtol=0.02, atol=0.002, chunk_size=chunk_size)
                self.assertTrue(result['passed'])
                self.assertEqual(result['tensor_count'], 1)
                self.assertEqual(result['max_absolute_error'], float(difference.max()))
                self.assertEqual(result['max_normalized_error'],
                                 float((difference / (0.002 + 0.02 * reference.abs())).max()))

    def test_failed_precision_is_reported_without_loosening_tolerances(self) -> None:
        """A near-zero mismatch and a non-finite candidate must remain failures."""
        save_snapshot(self.path, self.contract, [('weight', torch.zeros(3))])
        for candidate in (torch.tensor([0.0, 0.003, 0.0]), torch.tensor([0.0, float('nan'), 0.0])):
            with self.subTest(candidate=candidate):
                result = compare_snapshot(self.path, self.contract, [('weight', candidate)], rtol=0.02, atol=0.002)
                self.assertFalse(result['passed'])

    def test_exact_comparison_does_not_round_large_integer_inputs(self) -> None:
        """The exact initialization/input gate may not lose precision via an FP32 cast."""
        save_snapshot(self.path, self.contract, [('input', torch.tensor([2**30], dtype=torch.int64))])
        result = compare_snapshot(self.path, self.contract, [('input', torch.tensor([2**30 + 1]))], rtol=0, atol=0)
        self.assertFalse(result['passed'])

    def test_contract_shape_dtype_and_names_are_strict(self) -> None:
        """A stale rank/config or changed tensor interface must not look like a numerical pass."""
        save_snapshot(self.path, self.contract, [('weight', torch.ones(2))])
        cases = [({'rank': 1}, [('weight', torch.ones(2))]),
                 (self.contract, [('weight', torch.ones(3))]),
                 (self.contract, [('weight', torch.ones(2, dtype=torch.bfloat16))]),
                 (self.contract, [('renamed', torch.ones(2))]),
                 (self.contract, []),
                 (self.contract, [('weight', torch.ones(2)), ('weight', torch.ones(2))])]
        for contract, tensors in cases:
            with self.subTest(contract=contract, tensors=tensors), self.assertRaises(ValueError):
                compare_snapshot(self.path, contract, tensors, rtol=0.02, atol=0.002)

    def test_existing_snapshot_is_not_overwritten(self) -> None:
        """Reusing an artifact path must not silently replace the baseline."""
        save_snapshot(self.path, self.contract, [('weight', torch.ones(2))])
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_snapshot(self.path, self.contract, [('weight', torch.zeros(2))])
        self.assertEqual(self.path.read_bytes(), before)
