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
"""CPU oracle tests for independent Qwen training-trajectory validation."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.examples.mega_moe import qwen_moe_benchmark as benchmark


class _ToyModel(torch.nn.Module):
    """Tiny differentiable oracle using the benchmark's actual optimizer loop."""

    def __init__(self) -> None:
        """Initialize identical scalar weights for each independent model."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.layers = []

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        """Return a quadratic loss and its prediction."""
        prediction = self.weight * input_ids
        return {"loss": (prediction - labels).square().mean(), "logits": prediction}

    @staticmethod
    def expert_parameters() -> tuple:
        """This oracle has only dense parameters."""
        return ()


class TestQwenMoeBenchmark(unittest.TestCase):
    """Reject hidden optimizer divergence and preserve independent progress."""

    @staticmethod
    def _workloads() -> dict:
        workloads = {}
        for name in ("mega_moe", "ast"):
            model = _ToyModel()
            workloads[name] = benchmark._Workload(
                model, torch.optim.AdamW(model.parameters(), lr=0.05),
                torch.tensor([1.0]), torch.tensor([0.0]), 1, torch.device("cpu"))
        return workloads

    def test_advances_independent_optimizer_trajectories(self) -> None:
        """Feature: independent optimizer trajectories.

        Description: Advance two separately owned AdamW instances five times.
        Expectation: Loss decreases and equal moments have distinct storage.
        """
        workloads = self._workloads()
        routes = {name: {"ids": torch.tensor([0]), "weights": torch.tensor([1.0])} for name in workloads}
        losses = []
        with patch.object(benchmark.dist, "all_reduce"), patch.object(benchmark, "_synchronize_dense_gradients"):
            for _ in range(5):
                result = benchmark._trajectory_step(workloads, torch.device("cpu"), routes)
                losses.append(result["losses"]["ast"])
                self.assertTrue(result["optimizer_state"]["passed"])
        self.assertLess(losses[-1], losses[0])
        states = [next(iter(workload.optimizer.state.values())) for workload in workloads.values()]
        self.assertEqual(states[0]["step"].item(), 5)
        self.assertEqual(states[1]["step"].item(), 5)
        self.assertNotEqual(states[0]["exp_avg"].data_ptr(), states[1]["exp_avg"].data_ptr())

    def test_rejects_optimizer_divergence_when_model_weights_match(self) -> None:
        """Feature: optimizer-state validation.

        Description: Perturb moments while retaining equal model weights.
        Expectation: The numerical gate rejects the hidden divergence.
        """
        workloads = self._workloads()
        for workload in workloads.values():
            workload.model.weight.grad = torch.ones(1)
            workload.optimizer.step()
        state = next(iter(workloads["ast"].optimizer.state.values()))
        state["exp_avg"].add_(1.0)
        self.assertTrue(torch.equal(workloads["ast"].model.weight, workloads["mega_moe"].model.weight))
        with patch.object(benchmark.dist, "all_reduce"):
            with self.assertRaisesRegex(RuntimeError, "optimizer state.*failed"):
                benchmark._compare_step_state(workloads, torch.device("cpu"))

    def test_modes_and_invalid_trajectory_arguments(self) -> None:
        """Feature: benchmark mode selection.

        Description: Parse legacy, isolated AST and trajectory arguments.
        Expectation: Each selects the correct backends and rejects invalid modes.
        """
        self.assertEqual(benchmark._selected_backends(benchmark.parse_args([])), ("common", "mega_moe"))
        self.assertEqual(benchmark._selected_backends(benchmark.parse_args(["--backend", "ast"])), ("ast",))
        self.assertEqual(benchmark._selected_backends(benchmark.parse_args(["--validation-steps", "100"])),
                         ("mega_moe", "ast"))
        for arguments in (["--validation-steps", "-1"], ["--validation-steps", "2", "--backend", "ast"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                benchmark.parse_args(arguments)

    def test_route_mismatch_fails_even_with_equal_outputs(self) -> None:
        """Feature: learned-route validation.

        Description: Supply different expert IDs with equal model outputs.
        Expectation: Route failure is retained alongside optimizer diagnostics.
        """
        workloads = self._workloads()
        routes = {"mega_moe": {"ids": torch.tensor([0])}, "ast": {"ids": torch.tensor([1])}}
        with patch.object(benchmark.dist, "all_reduce"), patch.object(benchmark, "_synchronize_dense_gradients"):
            result = benchmark._trajectory_step(workloads, torch.device("cpu"), routes)
        self.assertFalse(result["routes"]["passed"])
        self.assertTrue(result["optimizer_state"]["passed"])

    def test_rejects_equal_nonfinite_state(self) -> None:
        """Feature: finite-state validation.

        Description: Compare corresponding state tensors with equal infinities.
        Expectation: Matching nonfinite values fail numerical acceptance.
        """
        with patch.object(benchmark.dist, "all_reduce"):
            with self.assertRaisesRegex(RuntimeError, "nonfinite.*failed"):
                benchmark._compare_tensor_maps("nonfinite", {"state": torch.tensor([float("inf")])},
                                               {"state": torch.tensor([float("inf")])}, torch.device("cpu"),
                                               rtol=0.02, atol=1e-8)
