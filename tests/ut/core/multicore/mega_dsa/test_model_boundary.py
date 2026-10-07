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
"""CPU model-boundary contracts; NPU RoPE and the backend are explicitly mocked."""

import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _OracleDsaReference,
    _rotary_mul_oracle,
    build_model_fixture,
    model_inputs,
    unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_validate import (
    _apply_calibration,
    _calibration_limits,
    _compare,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.model_boundary import (
    CannDsaReferenceAttention,
)
from hyper_parallel.models.replacement import (
    apply_module_replacements,
    compile_module_replacements,
)
from hyper_parallel.trainer.config import (
    PlanOverride,
    Target,
    entries_to_module_replacements,
)

_BOUNDARY = "hyper_parallel.core.multicore.modules.mega_dsa.model_boundary.CannDsaReferenceAttention"


class TestModelBoundary(unittest.TestCase):
    """Exercise actual HP projections and generic replacements with CPU test doubles."""

    def setUp(self) -> None:
        """Create independent model/metadata and scoped optional-device/global-scale mocks."""
        self.source = build_model_fixture()
        self.meta = DsaBatchMeta.packed((3, 5))
        self.backend = _OracleDsaReference(self.meta, attention_scale=self.source.scaling)
        self.hidden, self.embeddings = model_inputs(self.meta, dtype=torch.float64)
        rotary = patch("hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul",
                       side_effect=_rotary_mul_oracle)
        scale = patch("hyper_parallel.components.functional.aux_loss._AuxLossAutoScaler.main_loss_backward_scale",
                      torch.tensor(1.0))
        rotary.start()
        scale.start()
        self.addCleanup(rotary.stop)
        self.addCleanup(scale.stop)

    def _replace(self):
        root = nn.Module()
        root.attention = self.source
        root.attention_alias = self.source
        entries = [PlanOverride(match=["attention", "attention_alias"], exact_type=True,
                                module_type="hyper_parallel.components.modules.dsa_attention.DeepseekV32DSAAttention",
                                replace_module=Target(CannDsaReferenceAttention, target_path=_BOUNDARY))]
        plan = compile_module_replacements(root, entries_to_module_replacements(entries))
        apply_module_replacements(root, plan)
        return root

    def _forward(self, model, hidden=None):
        return model(self.hidden if hidden is None else hidden, position_embeddings=self.embeddings,
                     dsa_reference=self.backend, actual_seq_len=(3, 8))[0]

    def test_declarative_replacement_preserves_aliases_parameters_and_state_dict(self):
        """The existing Target transport retains source weights and checkpoint keys without conversions."""
        self.source.eval()
        before = dict(self.source.named_parameters())
        state = self.source.state_dict()
        root = self._replace()
        self.assertIs(root.attention, root.attention_alias)
        self.assertIsInstance(root.attention, CannDsaReferenceAttention)
        self.assertFalse(root.attention.training)
        self.assertIsNot(root.attention, self.source)
        self.assertEqual(list(root.attention.state_dict()), list(state))
        self.assertEqual(root.attention.make_transforms(), [])
        self.assertEqual(type(self.source), DeepseekV32DSAAttention)
        for name, parameter in root.attention.named_parameters():
            self.assertIs(parameter, before[name])
            torch.testing.assert_close(parameter, state[name], rtol=0, atol=0)

    def test_absorbed_model_matches_dense_unabsorbed_output_and_all_gradients(self):
        """Input, norm, key/value up-projection, output and all index parameters align in FP64."""
        dense = copy.deepcopy(self.source)
        model = self._replace().attention
        set_aux_loss_scale(torch.tensor(7.0))
        hidden = self.hidden.clone().requires_grad_()
        expected_hidden = self.hidden.clone().requires_grad_()
        output = self._forward(model, hidden)
        indices = self.backend.last_selection.to_global_indices()
        expected, loss = unabsorbed_model_reference(dense, expected_hidden, self.embeddings, self.meta, indices)
        torch.testing.assert_close(output, expected, rtol=1e-10, atol=1e-12)
        actual_grad = torch.autograd.grad(output.square().mean() * 13, (hidden, *model.parameters()))
        expected_grad = torch.autograd.grad(expected.square().mean() * 13 + loss * 7,
                                            (expected_hidden, *dense.parameters()))
        for name, actual, reference in zip(("hidden", *dict(model.named_parameters())), actual_grad, expected_grad):
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-11)

    def test_kl_only_does_not_update_hidden_or_main_parameters(self):
        """Projection input detach and the detached teacher isolate the full model trunk."""
        model = self._replace().attention
        hidden = self.hidden.clone().requires_grad_()
        self._forward(model, hidden)
        parameters = dict(model.named_parameters())
        gradients = torch.autograd.grad(self.backend.last_loss * 7, (hidden, *parameters.values()), allow_unused=True)
        self.assertIsNone(gradients[0])
        for name, gradient in zip(parameters, gradients[1:]):
            if name.startswith("indexer."):
                self.assertIsNotNone(gradient, name)
            else:
                self.assertIsNone(gradient, name)

    def test_lm_only_freeze_zero_coefficient_and_eval_leave_index_parameters_unused(self):
        """Existing training/freeze/coefficient gates continue to control only KL attachment."""
        model = self._replace().attention
        for mode in ("freeze", "zero", "eval"):
            with self.subTest(mode=mode):
                model.freeze_dsa = mode == "freeze"
                model.dsa_loss_coeff = 0 if mode == "zero" else 0.3
                model.train(mode != "eval")
                self.backend.last_loss = None
                output = self._forward(model)
                parameters = dict(model.named_parameters())
                gradients = torch.autograd.grad(output.square().mean(), tuple(parameters.values()), allow_unused=True)
                self.assertIsNone(self.backend.last_loss)
                for name, gradient in zip(parameters, gradients):
                    if name.startswith("indexer."):
                        self.assertIsNone(gradient, name)
                    else:
                        self.assertIsNotNone(gradient, name)

    def test_non_reentrant_checkpoint_preserves_parameter_and_input_gradients(self):
        """Recomputation calls the boundary explicitly without losing the auxiliary edge."""
        model = self._replace().attention
        set_aux_loss_scale(torch.tensor(7.0))
        hidden = self.hidden.clone().requires_grad_()
        output = self._forward(model, hidden)
        expected = torch.autograd.grad(output.square().mean(), (hidden, *model.parameters()))
        hidden = self.hidden.clone().requires_grad_()
        output = checkpoint(lambda value: self._forward(model, value), hidden, use_reentrant=False)
        actual = torch.autograd.grad(output.square().mean(), (hidden, *model.parameters()))
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=1e-10, atol=1e-12)

    def test_reference_metadata_scale_and_parallel_contracts_fail_early(self):
        """An explicit invocation rejects missing/conflicting references before indexer execution."""
        model = self._replace().attention
        with self.assertRaisesRegex(TypeError, "dsa_reference"):
            model(self.hidden)
        cases = (
            ({"actual_seq_len": (8,)}, "conflicts"),
            ({"actual_q_len": (3, 8)}, "aliases"),
            ({"actual_seq_len": torch.empty(2, device="meta", dtype=torch.int32)}, "CPU integer"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                model(self.hidden, dsa_reference=self.backend, **kwargs)
        other = _OracleDsaReference(replace(self.meta, layer=1), attention_scale=model.scaling)
        with self.assertRaisesRegex(ValueError, "layer"):
            model(self.hidden, dsa_reference=other)
        other = _OracleDsaReference(self.meta, attention_scale=0.1)
        with self.assertRaisesRegex(ValueError, "original model scaling"):
            model(self.hidden, dsa_reference=other)
        for context in ({"cp": True}, {"tp": True}):
            with self.assertRaisesRegex(ValueError, "TP=CP=1"):
                CannDsaReferenceAttention(module=self.source, context=context)

    def test_prepared_length_tensor_does_not_require_host_value_reads(self):
        """Device-style prepared lengths are accepted by identity without an invocation-time tolist."""
        model = self._replace().attention
        with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host values")):
            self.assertIs(model._validate_reference(self.hidden, self.backend.layout.length_tensor,
                                                    {"dsa_reference": self.backend}), self.backend)

    def test_unequal_microbatches_recover_global_input_and_parameter_objective(self):
        """Token-weighted LM and auxiliary injection match the packed full-batch mean."""
        model = self._replace().attention
        set_aux_loss_scale(torch.tensor(7.0))
        hidden = self.hidden.clone().requires_grad_()
        full = self._forward(model, hidden)
        expected = torch.autograd.grad(full.square().mean() * 13, (hidden, *model.parameters()))
        model.zero_grad(set_to_none=True)
        hidden = self.hidden.clone().requires_grad_()
        for microbatch, (begin, end) in enumerate(((0, 3), (3, 8))):
            meta = replace(DsaBatchMeta.packed((end - begin,)), microbatch=microbatch, invocation=microbatch)
            backend = _OracleDsaReference(meta, attention_scale=model.scaling)
            fraction = (end - begin) / self.meta.global_valid_queries
            set_aux_loss_scale(torch.tensor(7 * fraction, dtype=torch.float64))
            embeddings = tuple(tensor[:, begin:end] for tensor in self.embeddings)
            output = model(hidden[:, begin:end], position_embeddings=embeddings, dsa_reference=backend)[0]
            (output.square().mean() * 13 * fraction).backward()
        for value, reference in zip((hidden.grad, *(parameter.grad for parameter in model.parameters())), expected):
            torch.testing.assert_close(value, reference, rtol=1e-9, atol=1e-11)

    def test_model_angular_calibration_uses_normalized_distance_and_rejects_bad_baselines(self):
        """A relative angular bound follows the same L2 multiplier; large stock errors cannot calibrate."""
        oracle = {"weight": torch.tensor([1.0, 0.0]), "unused": None}
        stock = {"weight": torch.tensor([1.0, 0.01]), "unused": None}
        candidate = {"weight": torch.tensor([1.0, 0.011]), "unused": None}
        report = _compare(candidate, stock, oracle)
        self.assertTrue(report["accepted"])
        self.assertAlmostEqual(report["checks"]["weight"]["limits"]["angular_l2"],
                               report["checks"]["weight"]["stock_vs_oracle"]["angular_l2"] * 1.25)
        candidate["weight"][1] = 0.02
        self.assertFalse(_compare(candidate, stock, oracle)["accepted"])
        stock["weight"] = torch.tensor([1.0, 5.0])
        self.assertFalse(_compare(stock, stock, oracle)["accepted"])
        stock["weight"][0] = torch.nan
        self.assertFalse(_compare(stock, stock, oracle)["accepted"])
        candidate["unused"] = torch.zeros(1)
        self.assertFalse(_compare(candidate, stock, oracle)["gradient_availability"]["unused"])

    def test_calibration_envelope_uses_stock_only_and_remains_frozen_for_holdouts(self):
        """Candidate errors cannot inflate limits, and a held-out failure cannot update them."""
        oracle = {"weight": torch.tensor([1.0, 0.0])}
        cases = {}
        for seed, error in ((1, 0.01), (2, 0.02)):
            stock = {"weight": torch.tensor([1.0, error])}
            bad_candidate = {"weight": torch.tensor([1.0, 10.0])}
            cases[f"seed={seed},joint,grad_aux=7"] = _compare(bad_candidate, stock, oracle)
        limits = _calibration_limits(cases)["joint,grad_aux=7"]
        frozen = copy.deepcopy(limits)
        self.assertAlmostEqual(limits["weight"]["relative_l2"], 0.025)
        stock = {"weight": torch.tensor([1.0, 0.005])}
        candidate = {"weight": torch.tensor([1.0, 0.02])}
        held_out = _apply_calibration(_compare(candidate, stock, oracle), limits)
        self.assertFalse(held_out["pointwise_accepted"])
        self.assertTrue(held_out["accepted"])
        candidate["weight"][1] = 0.03
        self.assertFalse(_apply_calibration(_compare(candidate, stock, oracle), limits)["accepted"])
        self.assertEqual(limits, frozen)
