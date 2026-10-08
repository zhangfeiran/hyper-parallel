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
"""CPU checks that layer diagnosis preserves model and gradient-isolation semantics."""

import copy
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.core.multicore.examples.mega_dsa_layer_diagnose import (
    _hidden_error_controls,
    _measure_projection,
    _operator_replay,
    _relu_crossings,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _OracleDsaReference,
    _rotary_mul_oracle,
    build_model_fixture,
    model_inputs,
)
from hyper_parallel.core.multicore.examples.mega_dsa_restore_diagnose import (
    _restoration_replay,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.model_boundary import (
    CannDsaReferenceAttention,
)


class TestLayerDiagnose(unittest.TestCase):
    """Use explicitly selected CPU oracles and mock only the optional NPU RoPE primitive."""

    def setUp(self) -> None:
        """Prepare independent metadata, models and scoped auxiliary-scale state."""
        self.meta = DsaBatchMeta.packed((3, 5))
        self.source = build_model_fixture()
        self.hidden, self.embeddings = model_inputs(self.meta, dtype=torch.float64)
        rotary = patch("hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul",
                       side_effect=_rotary_mul_oracle)
        scale = patch("hyper_parallel.components.functional.aux_loss._AuxLossAutoScaler.main_loss_backward_scale",
                      torch.tensor(7.0))
        rotary.start()
        scale.start()
        self.addCleanup(rotary.stop)
        self.addCleanup(scale.stop)

    def test_staged_diagnosis_matches_original_boundary_all_parameters(self):
        """Manual capture retains the original output and every parameter/input derivative."""
        source = copy.deepcopy(self.source)
        hidden = self.hidden.clone().requires_grad_()
        backend = _OracleDsaReference(self.meta, attention_scale=source.scaling)
        measured = _measure_projection(source, hidden, self.embeddings, backend, auxiliary_scale=7)
        model = CannDsaReferenceAttention(module=self.source)
        expected_hidden = self.hidden.clone().requires_grad_()
        backend = _OracleDsaReference(self.meta, attention_scale=model.scaling, indices=measured["indices"])
        set_aux_loss_scale(torch.tensor(7.0))
        output = model(expected_hidden, position_embeddings=self.embeddings, dsa_reference=backend)[0]
        gradients = torch.autograd.grad(output.float().square().mean() * 13,
                                        (expected_hidden, *model.parameters()))
        torch.testing.assert_close(measured["full"]["output"], output.float())
        for name, gradient in zip(("hidden", *dict(model.named_parameters())), gradients):
            with self.subTest(parameter=name):
                torch.testing.assert_close(measured["full"][name], gradient.float(), rtol=1e-6, atol=1e-9)
        for name, value in measured["index_projection_vjp"].items():
            self.assertEqual(value is None, not name.startswith("indexer."), name)
        for name, value in measured["main_projection_vjp"].items():
            if name.startswith(("indexer.", "o_proj.")):
                self.assertIsNone(value, name)

    def test_operator_replay_uses_fixed_output_cotangent_and_auxiliary_scale(self):
        """Changing only the sparse cotangent affects only main gradients; KL scale stays separate."""
        backend = _OracleDsaReference(self.meta, attention_scale=self.source.scaling)
        measured = _measure_projection(self.source, self.hidden.clone().requires_grad_(), self.embeddings,
                                       backend, auxiliary_scale=7)
        values, indices, cotangent = measured["states"], measured["indices"], measured["sparse_cotangent"]
        first = _operator_replay(values, cotangent, backend, indices, auxiliary_scale=2)
        # Power-of-two multipliers isolate scaling from FP32 reduction rounding.
        second = _operator_replay(values, cotangent * 2, backend, indices, auxiliary_scale=1)
        torch.testing.assert_close(first["output"], second["output"], rtol=0, atol=0)
        for name in ("q_nope", "compressed_kv", "q_rope", "k_rope"):
            torch.testing.assert_close(second[name], first[name] * 2, rtol=1e-6, atol=1e-9)
        for name in ("index_q", "index_k", "merge_weight"):
            torch.testing.assert_close(second[name] * 2, first[name], rtol=1e-6, atol=1e-9)
        self.assertTrue(all(not value.requires_grad for value in values))

    def test_relu_crossing_report_excludes_future_and_other_packed_sequences(self):
        """The crossing detector follows packed causal IDs, including the ReLU zero convention."""
        native = [None] * 4 + [torch.ones(8, 1, 1), torch.ones(8, 1)]
        fp32 = [None] * 4 + [torch.ones(8, 1, 1), -torch.ones(8, 1)]
        crossings = _relu_crossings(native, fp32, self.meta)
        self.assertEqual(len(crossings), 21)
        for crossing in crossings:
            query, key = crossing["query"], crossing["key"]
            self.assertLessEqual(key, query)
            self.assertEqual(self.meta.sequence_position(query)[0], self.meta.sequence_position(key)[0])

    def test_restoration_replay_matches_expanded_value_formula_and_parameter_scope(self):
        """An independent per-head value formula verifies both transposes and direct parameter VJPs."""
        backend = _OracleDsaReference(self.meta, attention_scale=self.source.scaling)
        capture = _measure_projection(self.source, self.hidden.clone().requires_grad_(), self.embeddings,
                                      backend, auxiliary_scale=7)["restoration"]
        actual = _restoration_replay(self.source, capture, "combined")
        sparse = capture["sparse_input"].double().requires_grad_()
        weight = self.source.kv_b_proj.weight.view(32, 144, 512)[:, 128:]
        restored = torch.einsum("thc,hvc->thv", sparse, weight).reshape(1, 8, 512)
        output = self.source.o_proj(restored)
        gradients = torch.autograd.grad(output, (sparse, *self.source.parameters()),
                                        grad_outputs=capture["output_cotangent"].double(), allow_unused=True)
        torch.testing.assert_close(actual["output"], output.float(), rtol=1e-6, atol=1e-9)
        for name, value in zip(("input_cotangent", *dict(self.source.named_parameters())), gradients):
            with self.subTest(parameter=name):
                if value is None:
                    self.assertIsNone(actual[name])
                else:
                    torch.testing.assert_close(actual[name], value.float(), rtol=1e-6, atol=1e-9)
        self.assertTrue(bool((actual["W_UK"] == 0).all()))
        for name in ("linear_qkv.weight", "q_a_layernorm.weight", "indexer.wq_b.weight"):
            self.assertIsNone(actual[name])

    def test_separate_restoration_phases_preserve_the_composed_cotangent(self):
        """o_proj followed by the value VJP matches their composition and uses detached captures."""
        self.source.float()
        backend = _OracleDsaReference(self.meta, attention_scale=self.source.scaling)
        capture = _measure_projection(self.source, self.hidden.float().requires_grad_(),
                                      tuple(tensor.float() for tensor in self.embeddings),
                                      backend, auxiliary_scale=7)["restoration"]
        combined = _restoration_replay(self.source, capture, "combined")
        output = _restoration_replay(self.source, capture, "output")
        value_capture = {**capture, "restored_cotangent": output["input_cotangent"]}
        value = _restoration_replay(self.source, value_capture, "value")
        torch.testing.assert_close(value["input_cotangent"], combined["input_cotangent"], rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(output["o_proj.weight"], combined["o_proj.weight"], rtol=1e-6, atol=1e-9)
        self.assertIsNone(value["o_proj.weight"])
        self.assertIsNone(output["kv_b_proj.weight"])
        self.assertTrue(all(not tensor.requires_grad for name, tensor in capture.items() if name != "hidden_shape"))
        with self.assertRaisesRegex(ValueError, "phase"):
            _restoration_replay(self.source, capture, "unknown")

    def test_hidden_precision_controls_recover_the_same_fp32_objective(self):
        """With no precision change, all main-path replay controls recover the full hidden gradient."""
        self.source.float()
        hidden = self.hidden.float().requires_grad_()
        embeddings = tuple(value.float() for value in self.embeddings)
        backend = _OracleDsaReference(self.meta, attention_scale=self.source.scaling)
        native = _measure_projection(self.source, hidden, embeddings, backend, auxiliary_scale=7)
        oracle = _operator_replay(native["states"], native["sparse_cotangent"], backend,
                                  native["indices"], auxiliary_scale=7)
        controls = _hidden_error_controls(self.source, hidden, embeddings, self.meta, native, native, oracle)
        for name, measured in controls.items():
            with self.subTest(control=name):
                self.assertTrue(measured["finite"])
                self.assertLess(measured["relative_l2"], 1e-5)
