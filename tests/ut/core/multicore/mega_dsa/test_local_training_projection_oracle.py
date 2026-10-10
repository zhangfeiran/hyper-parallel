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
"""Independent closed-form linear VJPs for the fixed-Native-value projection oracle."""

import unittest

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_local_training_oracle import (
    ProjectionGeometry, project_local_states, projection_fixture, projection_vjp_metrics, projection_vjp_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.reference import selected_kl_reference, sparse_attention_reference


class TestProjectionVjpOracle(unittest.TestCase):
    """Check the oracle's independent chain rule, dtype, global normalization and hidden isolation."""

    def setUp(self) -> None:
        """Use a small packed fixture with padded, reordered selected slots."""
        self.geometry = ProjectionGeometry(heads=2, compressed=3, rope=2, index_heads=2, index_dim=3, output=2)
        self.meta = DsaBatchMeta.packed((3, 4))
        self.hidden, self.weights, self.targets = projection_fixture(7, self.geometry)
        self.indices = torch.tensor([[0, -1, -1, -1], [1, 0, -1, -1], [2, 0, 1, -1],
                                     [3, -1, -1, -1], [4, 3, -1, -1], [5, 3, 4, -1], [6, 3, 4, 5]],
                                    dtype=torch.int32)
        rows = torch.arange(7)
        self.states = project_local_states(self.hidden, self.weights, self.geometry, rows, rows, torch.float32)
        self.output = torch.arange(42).reshape(7, 2, 3).double()/1000-.01

    def _closed_form(self, mode, coefficient, auxiliary):
        hidden = self.hidden.double()
        weights = tuple(weight.double() for weight in self.weights)
        states = tuple(tensor.double().detach().requires_grad_() for tensor in self.states)
        predicted = self.output.flatten(1) @ weights[7].T
        main_loss = (predicted-self.targets.double()).square().sum()/7
        raw = [None]*7
        output_weight_gradient, hidden_gradient = None, None
        if mode != "kl_only":
            predicted_gradient = 2*(predicted-self.targets.double())/7
            output_weight_gradient = predicted_gradient.T @ self.output.flatten(1)
            cotangent = (predicted_gradient @ weights[7]).reshape(self.output.shape)
            main, _ = sparse_attention_reference(*states[:4], self.indices, self.meta, attention_scale=.37)
            raw[:4] = torch.autograd.grad(main, states[:4], grad_outputs=cotangent)
            hidden_gradient = sum(gradient.flatten(1) @ weight for gradient, weight in zip(raw[:4], weights[:4]))
        kl_loss = torch.zeros((), dtype=torch.float64)
        if mode != "lm_only":
            kl_loss = selected_kl_reference(*states[4:], *states[:4], self.indices, self.meta,
                                            attention_scale=.37, normalization=DsaLossNormalization(7),
                                            loss_coeff=coefficient)
            raw[4:] = torch.autograd.grad(kl_loss*auxiliary, states[4:])
        parameter_gradients = tuple(None if gradient is None else gradient.flatten(1).T @ hidden for gradient in raw)
        objective = kl_loss*auxiliary if mode == "kl_only" else main_loss
        if mode == "lm_plus_kl":
            objective = objective+kl_loss*auxiliary
        return objective, kl_loss, (hidden_gradient, *parameter_gradients, output_weight_gradient)

    def test_every_projection_gradient_matches_closed_form_chain_rule(self):
        """Frozen activation/output values never replace their independently differentiated Jacobians."""
        for mode in ("lm_only", "kl_only", "lm_plus_kl"):
            for coefficient in (0., .3):
                for auxiliary in (0., 1., -2.5):
                    with self.subTest(mode=mode, coefficient=coefficient, auxiliary=auxiliary):
                        actual = projection_vjp_reference(
                            self.hidden, self.weights, self.targets, self.states, self.output,
                            self.indices, self.meta, self.geometry, mode=mode, coefficient=coefficient,
                            auxiliary=auxiliary, attention_scale=.37)
                        expected = self._closed_form(mode, coefficient, auxiliary)
                        for value, target in zip(actual[:2], expected[:2]):
                            torch.testing.assert_close(value, target.detach(), rtol=1e-11, atol=1e-12)
                        for value, target in zip(actual[2], expected[2]):
                            if target is None:
                                self.assertIsNone(value)
                            else:
                                torch.testing.assert_close(value, target, rtol=1e-10, atol=1e-12)
                                self.assertEqual(value.dtype, torch.float64)

    def test_actual_query_and_key_projections_preserve_orders_and_detach_hidden(self):
        """The producer projects only caller queries/owner keys and isolates all index projection inputs."""
        hidden = self.hidden.clone().requires_grad_()
        weights = tuple(weight.clone().requires_grad_() for weight in self.weights)
        queries, keys = torch.tensor([6, 0]), torch.tensor([5, 0, 6, 2, 4, 1, 3])
        states = project_local_states(hidden, weights, self.geometry, queries, keys, torch.float32)
        full = project_local_states(hidden, weights, self.geometry, torch.arange(7), torch.arange(7), torch.float32)
        for field, (actual, expected) in enumerate(zip(states, full)):
            torch.testing.assert_close(actual, expected[keys if field in (1, 3, 5) else queries], rtol=0, atol=0)
        auxiliary = sum(tensor.float().square().sum() for tensor in states[4:])
        gradients = torch.autograd.grad(auxiliary, (hidden, *weights), allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertTrue(all(tensor is None for tensor in gradients[1:5]))
        self.assertTrue(all(float(tensor.norm()) > 0 for tensor in gradients[5:8]))
        self.assertIsNone(gradients[8])

    def test_explicit_native_output_values_affect_output_vjp_but_not_auxiliary(self):
        """An observed output override exposes the oracle's identical-state scope rather than model acceptance."""
        results = [projection_vjp_reference(
            self.hidden, self.weights, self.targets, self.states, output, self.indices, self.meta, self.geometry,
            mode="lm_plus_kl", coefficient=.3, auxiliary=-2.5, attention_scale=.37)
                   for output in (self.output, self.output+.125)]
        self.assertFalse(torch.equal(results[0][2][-1], results[1][2][-1]))
        for first, second in zip(results[0][2][5:8], results[1][2][5:8]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_tiny_nonzero_gradients_cannot_pass_only_an_absolute_tolerance(self):
        """Omitting a small projection gradient must fail even when pointwise allclose alone succeeds."""
        expected = torch.full((2, 3), 1e-7, dtype=torch.float64)
        actual = torch.zeros_like(expected)
        metric = projection_vjp_metrics(actual, expected)
        self.assertTrue(metric["pointwise_pass"])
        self.assertFalse(metric["passed"])
        self.assertEqual(metric["relative_l2"], 1.)
        self.assertTrue(projection_vjp_metrics(expected*1.01, expected)["passed"])
        self.assertFalse(projection_vjp_metrics(expected*.5, expected)["passed"])

    def test_zero_reference_has_an_absolute_rule_and_nonfinite_values_fail(self):
        """Zero-oracle fields keep their own rule rather than an invented normalized denominator."""
        expected = torch.zeros(2, 3, dtype=torch.float64)
        self.assertTrue(projection_vjp_metrics(torch.full_like(expected, 1e-5), expected)["passed"])
        self.assertFalse(projection_vjp_metrics(torch.full_like(expected, 3e-5), expected)["passed"])
        self.assertFalse(projection_vjp_metrics(torch.full_like(expected, float("nan")), expected)["passed"])

    def test_nonpositive_geometry_and_unknown_modes_reject(self):
        """Malformed fixture declarations cannot masquerade as a valid projection reference."""
        for width in (0, -1, True, 1.5):
            with self.subTest(width=width), self.assertRaises(ValueError):
                ProjectionGeometry(hidden=width)
        with self.assertRaisesRegex(ValueError, "objective mode"):
            projection_vjp_reference(self.hidden, self.weights, self.targets, self.states, self.output,
                                      self.indices, self.meta, self.geometry, mode="unknown", coefficient=.3,
                                      auxiliary=1., attention_scale=.37)
