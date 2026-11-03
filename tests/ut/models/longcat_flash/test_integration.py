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
"""LongCat complete-model, routing, layout, gradient, and checkpoint contracts."""

import copy
from pathlib import Path
import unittest

import torch
from transformers import LongcatFlashConfig
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashExperts

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts
from hyper_parallel.models.longcat_flash.adapter.layout import ExpertOwnershipPacking
from hyper_parallel.models.longcat_flash.adapter.routing import OrderedInputGather
from hyper_parallel.models.longcat_flash.modeling_longcat_flash import build_longcat_flash
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.components.optim import Float16OptimizerWithFloat16Params
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.models.longcat_flash.adapter.materialization import register_materialization
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements,
)

from tests.common.mark_utils import arg_mark


def small_config(layers: int = 1, zero: int = 3) -> dict:
    """Use small dimensions while exercising mixed real/identity Top-K slots."""
    return {"vocab_size": 32, "hidden_size": 16, "ffn_hidden_size": 32, "num_layers": layers,
                "num_attention_heads": 2, "num_key_value_heads": 2, "q_lora_rank": 8, "kv_lora_rank": 8,
                "qk_nope_head_dim": 4, "qk_rope_head_dim": 4, "v_head_dim": 4, "head_dim": 4,
                "expert_ffn_hidden_size": 8, "n_routed_experts": 4, "zero_expert_num": zero,
                "moe_topk": 3, "max_position_embeddings": 32}


class TestLongcatIntegration(unittest.TestCase):
    """Use the independent, unmodified HF model as the numerical oracle."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_lazy_registration(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: The family resolves without registering a second architecture.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        spec = get_model_adapter("longcat_flash")
        self.assertEqual(spec.architecture, "LongcatFlashForCausalLM")
        self.assertIsNotNone(spec.materialization)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_complete_model_parity_and_checkpoint(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: Both attention/dense sublayers and shortcut MoE match full HF gradients.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        torch.manual_seed(71)
        reference = build_longcat_flash(small_config(layers=2))
        torch.manual_seed(71)
        candidate = build_longcat_flash(small_config(layers=2), expert_backend="native", optimized_attention=True)
        tokens = torch.tensor([[1, 3, 8, 2], [1, 5, 9, 2]])
        expected = reference(tokens, labels=tokens, use_cache=False)
        actual = candidate(tokens, labels=tokens, use_cache=False)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-6)
        expected.loss.backward()
        actual.loss.backward()
        reference_grads = {name: value.grad for name, value in reference.named_parameters()}
        for name, value in candidate.named_parameters():
            if "linear_qkv" in name:
                expected_grad = torch.cat((reference_grads[name.replace("linear_qkv", "q_a_proj")],
                                           reference_grads[name.replace("linear_qkv", "kv_a_proj_with_mqa")]))
            else:
                expected_grad = reference_grads[name]
            if expected_grad is None:
                self.assertIsNone(value.grad)
            else:
                torch.testing.assert_close(value.grad, expected_grad, rtol=1e-4, atol=2e-6)
        restored = revert_weight_conversion(candidate, dict(candidate.state_dict()))
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
        self.assertEqual(len(candidate.model.layers), 2)
        self.assertTrue(all(len(layer.self_attn) == len(layer.mlps) == 2 for layer in candidate.model.layers))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_fixed_routes_outputs_and_all_gradients(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: Check mixed, all-real, and multiple-identity routes independently of Top-K ties.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        for ids in (torch.tensor([[0, 4, 6], [5, 1, 3]]),
                    torch.tensor([[0, 1, 2], [3, 0, 1]]),
                    torch.tensor([[4, 5, 6], [6, 4, 5]])):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(ids=ids.tolist(), dtype=dtype):
                    torch.manual_seed(31)
                    source = LongcatFlashExperts(LongcatFlashConfig.from_dict(small_config())).to(dtype=dtype)
                    for parameter in source.parameters():
                        torch.nn.init.normal_(parameter, std=0.1)
                    target = LongcatFlashMegaMoeExperts(module=copy.deepcopy(source), backend="native")
                    x = torch.randn(2, 16, dtype=dtype, requires_grad=True)
                    z = x.detach().clone().requires_grad_()
                    p = torch.randn(2, 3).softmax(-1).detach().requires_grad_()
                    q = p.detach().clone().requires_grad_()
                    expected, actual = source(x, ids, p), target(z, ids, q)
                    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-6
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    expected.float().square().sum().backward()
                    actual.float().square().sum().backward()
                    torch.testing.assert_close(z.grad, x.grad, rtol=tolerance, atol=tolerance)
                    torch.testing.assert_close(q.grad, p.grad, rtol=tolerance, atol=tolerance)
                    for name, parameter in target.named_parameters():
                        gradient = dict(source.named_parameters())[name].grad
                        if gradient is None:
                            self.assertIsNone(parameter.grad)
                        else:
                            torch.testing.assert_close(parameter.grad, gradient, rtol=tolerance, atol=tolerance)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_all_identity_preserves_unused_parameters_and_adamw(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: Identity-only steps must leave expert parameters unused, avoiding unintended weight decay.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        torch.manual_seed(13)
        model = build_longcat_flash(small_config(), expert_backend="native")
        experts = model.model.layers[0].mlp.experts
        before = {name: value.detach().clone() for name, value in experts.named_parameters()}
        optimizer = torch.optim.AdamW(experts.parameters(), lr=0.01, weight_decay=0.1)
        x = torch.ones(2, 16, requires_grad=True)
        p = torch.ones(2, 3, requires_grad=True)
        experts(x, torch.tensor([[4, 5, 6], [6, 5, 4]]), p).sum().backward()
        optimizer.step()
        for name, parameter in experts.named_parameters():
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        torch.testing.assert_close(p.grad, torch.full_like(p, 16))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_packing_retains_zero_rows_and_local_real_ownership(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: EP shards own the right real experts while HF export is exactly reversible.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        original = torch.arange(12 * 4 * 3).reshape(12, 4, 3)
        for ep in (1, 2, 4):
            with self.subTest(ep=ep):
                packing = ExpertOwnershipPacking(8, 4, ep)
                packed = packing.apply(original)
                for rank, local in enumerate(packed.chunk(ep)):
                    torch.testing.assert_close(local[:8 // ep], original[rank * (8 // ep):(rank + 1) * (8 // ep)])
                torch.testing.assert_close(packing.reverse_op.apply(packed), original, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "divisible"):
            ExpertOwnershipPacking(8, 4, 3)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_backend_and_topology_rejections(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: Unsupported topology and CPU native-kernel requests fail explicitly.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        source = LongcatFlashExperts(LongcatFlashConfig.from_dict(small_config()))
        for axis in ("cp", "pp"):
            with self.assertRaisesRegex(ValueError, "CP=PP=1"):
                LongcatFlashMegaMoeExperts(module=source, context={axis: True})
        with self.assertRaisesRegex(ValueError, "TP=1"):
            LongcatFlashMegaMoeExperts(module=source, context={"tp": True})
        target = LongcatFlashMegaMoeExperts(module=source)
        try:
            with self.assertRaisesRegex(TypeError, "BF16 NPU"):
                target(torch.ones(1, 16), torch.tensor([[0, 4, 5]]), torch.ones(1, 3))
        finally:
            target.close_runtime()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_zero_real_experts(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: A fully identity pool remains a complete differentiable module.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        config = small_config(zero=4)
        config["n_routed_experts"] = 0
        source = LongcatFlashExperts(LongcatFlashConfig.from_dict(config))
        target = LongcatFlashMegaMoeExperts(module=source, backend="native")
        x = torch.randn(2, 16, requires_grad=True)
        p = torch.randn(2, 3).softmax(-1)
        ids = torch.tensor([[0, 1, 2], [1, 2, 3]])
        torch.testing.assert_close(target(x, ids, p), source(x, ids, p))
        target(x, ids, p).sum().backward()
        self.assertIsNotNone(x.grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_kv_cache_matches_full_sequence(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: Optimized MLA retains scaled compressed KV and both sublayer cache indices.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        torch.manual_seed(81)
        model = build_longcat_flash(small_config(), expert_backend="native", optimized_attention=True).eval()
        tokens = torch.tensor([[1, 4, 8, 2]])
        with torch.no_grad():
            expected = model(tokens, use_cache=False).logits[:, -1]
            prefix = model(tokens[:, :-1], use_cache=True)
            actual = model(tokens[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_ep_checkpoint_roundtrip_and_mismatched_layout_rejection(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: Retain every unused row and reject a DCP layout from another EP size.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        model = build_longcat_flash(small_config(zero=4))
        original = copy.deepcopy(model.state_dict())
        conversions = []
        rules = [ModuleReplacementSpec(match=("model.layers.*.mlp.experts",),
                                       factory=LongcatFlashMegaMoeExperts)]
        apply_module_replacements(model, compile_module_replacements(model, rules),
                                  context={"ep_size": 2, "ep": True}, weights_mapping=conversions)
        model._weight_conversions = conversions
        restored = revert_weight_conversion(model, dict(model.state_dict()))
        self.assertEqual(set(restored), set(original))
        for name, value in original.items():
            torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
        experts = model.model.layers[0].mlp.experts
        experts._checkpoint_ep_size.fill_(4)
        with self.assertRaisesRegex(ValueError, "checkpoint EP packing differs"):
            experts(torch.ones(1, 16), torch.tensor([[0, 4, 5]]), torch.ones(1, 3))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_meta_rebuild_keeps_rotary_nonpersistent_and_loaded_state_unchanged(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: to_empty rebuilds only derived frequencies and preserves checkpoint-owned bias.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        model = build_longcat_flash(small_config(), expert_backend="native")
        register_materialization(model)
        frequencies = model.model.rotary_emb.inv_freq.clone()
        model.to_empty(device="cpu")
        router = model.model.layers[0].mlp.router
        router.e_score_correction_bias.fill_(3)
        bias_id = id(router.e_score_correction_bias)
        rebuild_materialized_state(model, MaterializationContext(reason="checkpoint_load", device=torch.device("cpu")))
        torch.testing.assert_close(model.model.rotary_emb.inv_freq, frequencies, rtol=0, atol=0)
        torch.testing.assert_close(router.e_score_correction_bias, torch.full_like(router.e_score_correction_bias, 3))
        self.assertEqual(id(router.e_score_correction_bias), bias_id)
        self.assertNotIn("model.rotary_emb.inv_freq", model.state_dict())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_scratch_materialization_initializes_replaced_experts(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: HF initialization must also initialize the replacement's rank-three weights.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        model = build_longcat_flash(small_config(), expert_backend="native")
        model.config.initializer_range = 0.01
        register_materialization(model)
        model.to_empty(device="cpu")
        _initialize_model_weights(model)
        experts = model.model.layers[0].mlp.experts
        self.assertTrue(torch.isfinite(experts.gate_up_proj).all())
        self.assertLess(abs(float(experts.gate_up_proj.detach().std()) - 0.01), 0.002)
        self.assertLess(abs(float(experts.down_proj.detach().std()) - 0.01), 0.002)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_recipe_optimizer_builds_fp32_main_parameters(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: The shipped optimizer target must build the runtime's chained optimizer.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        root = Path(__file__).resolve().parents[4]
        config = parse_training_args([str(root / "hyper_parallel/models/longcat_flash/recipes/train.yaml")])
        model = build_longcat_flash(small_config(), expert_backend="native")
        optimizer = Float16OptimizerWithFloat16Params(config.optimizer.target.build(model=model).get_optimizer(), model)
        tokens = torch.tensor([[1, 3, 8, 2]])
        model(tokens, labels=tokens, use_cache=False).loss.backward()
        optimizer.step()
        self.assertTrue(all(parameter.main_param.dtype == torch.float32 for parameter in model.parameters()))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_identity_aliases_and_ordered_bf16_source_gradient(self) -> None:
        """Feature: Expert-ordered BF16 input gradients.

        Description: Combine cancelling real and identity derivatives through storage-sharing aliases.
        Expectation: Descending expert accumulation preserves the HF result without identity expansion.
        """
        hidden = torch.ones(1, 1, dtype=torch.bfloat16, requires_grad=True)
        gather = OrderedInputGather(3)
        real = gather(hidden, torch.tensor([0, 0]), torch.tensor([0, 1]))
        self.assertTrue(all(value.data_ptr() == hidden.data_ptr() for value in gather.identity_inputs))
        torch.autograd.backward((real, *gather.identity_inputs),
                                (torch.tensor([[256.], [1.]], dtype=torch.bfloat16),
                                 torch.zeros_like(hidden), torch.zeros_like(hidden),
                                 torch.full_like(hidden, -256.)))
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)
