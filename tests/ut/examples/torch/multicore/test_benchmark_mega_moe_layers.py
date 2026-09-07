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
"""CPU contracts for the checkpointed multi-layer benchmark protocol."""

import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from examples.torch.multicore.benchmark_mega_moe_layers import (
    ADAMW_OPTIONS,
    _configuration,
    _make_optimizer,
    _optimizer_state_ledger,
    _python_source_identity,
    _route_pairs,
)


class TestBenchmarkMegaMoeLayers(unittest.TestCase):
    """Keep AdamW execution, state accounting and reported metadata consistent."""

    def test_optimizer_is_adamw_with_reported_options(self):
        """The benchmark must execute AdamW, not SGD or coupled-decay Adam."""
        parameter = torch.nn.Parameter(torch.ones(2, device="cpu"))
        optimizer = _make_optimizer(torch, [parameter])
        args = argparse.Namespace(variant="native", layers=16, warmup=3, measured=7, shuffle_seed=1)
        config = _configuration(args)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(config["optimizer"], {"name": "AdamW", **ADAMW_OPTIONS})
        for name, expected in ADAMW_OPTIONS.items():
            with self.subTest(option=name):
                self.assertEqual(optimizer.param_groups[0][name], expected)

    def test_state_ledger_counts_actual_bfloat16_moments(self):
        """Count both moments and the CPU step scalar, without assuming FP32 moments."""
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16, device="cpu"))
        optimizer = _make_optimizer(torch, [parameter])
        self.assertEqual(_optimizer_state_ledger(torch, optimizer)["tensor_bytes"], 0)
        parameter.square().sum().backward()
        optimizer.step()
        ledger = _optimizer_state_ledger(torch, optimizer)
        moment_bytes = 2 * parameter.numel() * parameter.element_size()
        step_bytes = optimizer.state[parameter]["step"].element_size()
        self.assertEqual(ledger["parameter_state_count"], 1)
        self.assertEqual(ledger["tensor_bytes"], moment_bytes + step_bytes)
        self.assertEqual(ledger["tensor_bytes_by_device"], {"cpu": moment_bytes + step_bytes})
        self.assertEqual(ledger["tensor_bytes_by_dtype"], {"torch.bfloat16": moment_bytes, "torch.float32": step_bytes})

    def test_nonfinite_adamw_state_is_rejected(self):
        """A finite forward alone must not hide a corrupted optimizer state."""
        parameter = torch.nn.Parameter(torch.ones(2, device="cpu"))
        optimizer = _make_optimizer(torch, [parameter])
        parameter.square().sum().backward()
        optimizer.step()
        optimizer.state[parameter]["exp_avg_sq"].fill_(float("inf"))
        with self.assertRaisesRegex(AssertionError, "nonfinite AdamW state: exp_avg_sq"):
            _optimizer_state_ledger(torch, optimizer)

    def test_limit_preserves_full_sweep_prefix(self):
        """Stability smoke must replay the full sweep's route order exactly."""
        args = argparse.Namespace(pairs=None, shuffle_seed=20260903, limit=None)
        full = _route_pairs(args)
        args.limit = 4
        self.assertEqual(len(full), 55)
        self.assertEqual(_route_pairs(args), full[:4])

    def test_source_identity_detects_dirty_runtime_changes(self):
        """Include uncommitted Python implementation edits in benchmark provenance."""
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "hyper_parallel/core/multicore/runtime.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            first = _python_source_identity(repo)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            second = _python_source_identity(repo)
        self.assertEqual(first["file_count"], 1)
        self.assertEqual(second["file_count"], 1)
        self.assertNotEqual(first["sha256"], second["sha256"])
