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
"""CPU oracles for dense canonical states and independent master-parameter trajectories."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.optim import Float16OptimizerWithFloat16Params
from hyper_parallel.core.multicore.examples.mega_ffn import qwen_dense_benchmark as benchmark
from hyper_parallel.core.multicore.examples.mega_ffn.qwen_dense_model import QwenDenseConfig
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from tests.common.mark_utils import arg_mark


class _ScalarModel(nn.Module):
    def __init__(self, backend: str) -> None:
        """Create one separately owned BF16 scalar parameter.

        Args:
            backend: Physical-layout label supplied to the acceptance runner.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
        self.backend, self.layers, self.closed = backend, [], False

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a small trainable loss through the benchmark's actual optimizer loop.

        Args:
            tokens: Independent fixed inputs.
            labels: Fixed loss targets.
        """
        logits = self.weight.float() * tokens
        return {"logits": logits, "loss": (logits - labels * 0.25).square().mean()}

    def close(self) -> None:
        """Record cleanup without changing optimizer evidence."""
        self.closed = True


def _workload(backend):
    model = _ScalarModel(backend)
    inner = torch.optim.AdamW(model.parameters(), lr=0.05, foreach=False)
    optimizer = Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": inner}), model)
    return benchmark.Workload(model, optimizer)


class TestQwenDenseBenchmark(unittest.TestCase):
    """Keep failure reporting strict while using separately owned training state."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_canonical_projection_layouts_and_step_counters(self):
        """Feature: Dense state canonicalization.
        Description: Compare separate, packed-Linear and AST weight/optimizer layouts.
        Expectation: Logical tensors agree and incompatible gate/up step counters fail.
        """
        gate, up, down = torch.randn(6, 4), torch.randn(6, 4), torch.randn(4, 6)
        prefix = "layers.0.mlp."
        common = {prefix + "gate_proj.weight": gate, prefix + "up_proj.weight": up,
                  prefix + "down_proj.weight": down}
        packed = {prefix + "linear_fc1.weight": torch.cat((gate, up)), prefix + "linear_fc2.weight": down}
        ast = {prefix + "gate_up": torch.cat((gate, up)).t(), prefix + "down": down.t()}
        expected = benchmark.canonical_tensors(common, "common")
        for values, backend in ((packed, "packed"), (ast, "mega_ffn")):
            result = benchmark.compare_tensors(expected, benchmark.canonical_tensors(values, backend), 0, rtol=0)
            self.assertTrue(result["passed"])
        steps = {name: torch.tensor(2.0) for name in common}
        canonical = benchmark.canonical_tensors(steps, "common")
        self.assertEqual(len(canonical), 2)
        steps[prefix + "up_proj.weight"] += 1
        with self.assertRaisesRegex(ValueError, "step counters"):
            benchmark.canonical_tensors(steps, "common")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_nonfinite_state_is_rejected_and_json_preserved(self):
        """Feature: Finite numerical acceptance.
        Description: Compare equal NaN/infinity state and serialize the resulting diagnostic.
        Expectation: Invalid states fail and diagnostics use JSON null for undefined errors.
        """
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                tensors = {"weight": torch.tensor([value])}
                result = benchmark.compare_tensors(tensors, tensors, 1e-8)
                self.assertFalse(result["passed"])
                self.assertIsNone(result["max_abs"])
                self.assertIsNone(benchmark._scalar(tensors["weight"]))
                json.dumps(result, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "identical logical"):
            benchmark.compare_tensors({"x": torch.zeros(1)}, {}, 0)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            benchmark.compare_tensors({"x": torch.zeros(1)}, {"x": torch.zeros(2)}, 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_independent_trajectories_advance_fp32_main_weights_and_moments(self):
        """Feature: Independent mixed-precision trajectories.
        Description: Train two separately initialized workloads through the actual acceptance runner.
        Expectation: Five steps agree without shared main weights or moment storage, and both close.
        """
        baseline, candidate = _workload("packed"), _workload("mega_ffn")
        config = QwenDenseConfig()
        with tempfile.TemporaryDirectory() as directory:
            args = benchmark.parse_args(["--mode", "validate", "--steps", "5", "--output",
                                         str(Path(directory) / "trajectory.json")])
            with (patch.object(benchmark, "_build", side_effect=[baseline, candidate]),
                  patch.object(benchmark, "_identity", return_value={"device": "cpu oracle"})):
                benchmark._validate(args, config, torch.ones(2, 2))
            record = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertTrue(record["complete"])
        self.assertTrue(record["numerical_passed"])
        self.assertEqual(len(record["steps"]), 5)
        self.assertLess(record["steps"][-1]["candidate_loss"], record["steps"][0]["candidate_loss"])
        for kind in ("main", "exp_avg", "exp_avg_sq"):
            first, second = benchmark._state(baseline, kind)["weight"], benchmark._state(candidate, kind)["weight"]
            self.assertEqual(first.dtype, torch.float32)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(float(benchmark._state(candidate, "step")["weight"]), 5)
        self.assertTrue(baseline.model.closed and candidate.model.closed)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_hidden_optimizer_divergence_fails_with_equal_model_weights(self):
        """Feature: Optimizer history acceptance.
        Description: Perturb FP32 AdamW momentum while preserving the visible BF16 parameters.
        Expectation: The actual trajectory check identifies the hidden divergence.
        """
        baseline, candidate = _workload("packed"), _workload("mega_ffn")
        reference = benchmark._step(baseline, torch.ones(2), True)
        actual = benchmark._step(candidate, torch.ones(2), True)
        main = candidate.optimizer.optimizer_param_by_model_param[candidate.model.weight]
        candidate.optimizer.chained_optimizers[0].state[main]["exp_avg"].add_(1)
        checks = benchmark._trajectory_checks(baseline, candidate, reference, actual)
        self.assertTrue(checks["checks"]["parameters"]["passed"])
        self.assertFalse(checks["checks"]["exp_avg"]["passed"])
        self.assertFalse(checks["passed"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_initial_main_parameter_mismatch_retains_failure_evidence(self):
        """Feature: Exact initial state gate.
        Description: Perturb only the candidate main parameter before the first training step.
        Expectation: The runner records failure, rejects the run and cleans up both models.
        """
        baseline, candidate = _workload("packed"), _workload("mega_ffn")
        with torch.no_grad():
            candidate.optimizer.optimizer_param_by_model_param[candidate.model.weight].add_(0.01)
        with tempfile.TemporaryDirectory() as directory:
            args = benchmark.parse_args(["--mode", "validate", "--output", str(Path(directory) / "failure.json")])
            with (patch.object(benchmark, "_build", side_effect=[baseline, candidate]),
                  patch.object(benchmark, "_identity", return_value={"device": "cpu oracle"}),
                  self.assertRaisesRegex(RuntimeError, "initial logical model/optimizer")):
                benchmark._validate(args, QwenDenseConfig(), torch.ones(2, 2))
            record = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertFalse(record["numerical_passed"])
        self.assertFalse(record["complete"])
        self.assertTrue(record["initial"]["parameters"]["passed"])
        self.assertFalse(record["initial"]["main"]["passed"])
        self.assertEqual(record["steps"], [])
        self.assertTrue(baseline.model.closed and candidate.model.closed)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_invalid_acceptance_arguments(self):
        """Feature: Acceptance input validation.
        Description: Request nonpositive steps, invalid loss sequence length or nonfinite learning rate.
        Expectation: The CLI rejects unusable settings before device initialization.
        """
        for extra in (("--steps", "0"), ("--seq-len", "1"), ("--learning-rate", "nan")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                benchmark.parse_args(["--mode", "validate", "--output", "unused.json", *extra])
