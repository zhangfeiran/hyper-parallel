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
"""CPU contracts for the adapter over the upstream MegaMoe parameter API."""
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn import functional as F

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements,
)
from hyper_parallel.trainer.config.manager import parse_training_args

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.models.deepseek_v41.adapter.replacements import (
    DeepseekV41TrainingExperts,
)

from hyper_parallel.models.deepseek_v41.adapter.runtime import DeepseekV41Runtime, DeepseekV41TextBatch
from hyper_parallel.models.deepseek_v41.adapter.expert_parallel import configure_megamoe, deepseek_v41_ep_compute_fn
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from examples.training_demo.train_text import main
from scripts.train_vl import main as vlm_main


def _source(limit=10.0):
    source = nn.Module()
    source.num_experts, source.hidden_dim, source.intermediate_dim = 4, 8, 4
    source.limit, source.act_fn = limit, nn.SiLU()
    source.gate_up_proj = nn.Parameter(torch.randn(4, 8, 8))
    source.down_proj = nn.Parameter(torch.randn(4, 8, 4))
    return source


def _native_reference(module, inputs, indices, weights, *, expert_weights=None):
    gate_up, down_weight = expert_weights or (module.gate_up_weight, module.down_weight)
    original_shape = inputs.shape
    inputs = inputs.reshape(-1, inputs.shape[-1])
    result = torch.zeros_like(inputs)
    for slot in range(indices.shape[1]):
        up = torch.bmm(inputs.unsqueeze(1), gate_up[indices[:, slot]]).squeeze(1)
        gate, value = up.chunk(2, dim=-1)
        if module.swiglu_limit is not None:
            gate = gate.clamp(max=module.swiglu_limit)
            value = value.clamp(-module.swiglu_limit, module.swiglu_limit)
        act = F.silu(gate) * value
        down = torch.bmm(act.unsqueeze(1), down_weight[indices[:, slot]]).squeeze(1)
        result = result + down * weights[:, slot, None]
    return result.reshape(original_shape)


class TestDeepseekV41TrainingExperts(unittest.TestCase):
    """Check FSDP-facing ownership, differentiability, and source semantics."""

    def test_external_weights_preserve_gradients_and_state(self):
        """Use current tensors across calls and never register placeholder weights."""
        torch.manual_seed(9)
        source = _source()
        experts = DeepseekV41TrainingExperts(module=source)
        experts.configure(None, 1, 128, 2, 2.0)
        self.addCleanup(experts.close)
        self.assertEqual(set(experts.state_dict()), {"gate_up_proj", "down_proj"})
        self.assertEqual(len(list(experts.parameters())), 2)
        kernel = experts._executor
        self.assertIsNone(kernel.gate_up_weight)
        for step in range(2):
            with self.subTest(step=step):
                inputs = (torch.randn(128, 8) * 5).requires_grad_()
                weights = torch.randn(128, 2, requires_grad=True)
                ids = torch.arange(256).reshape(128, 2) % 4
                reference = SimpleNamespace(gate_up_weight=source.gate_up_proj.transpose(1, 2),
                                            down_weight=source.down_proj.transpose(1, 2), swiglu_limit=10.0)
                expected = _native_reference(reference, inputs, ids, weights)
                with patch.object(MegaMoeExperts, "forward", _native_reference):
                    actual = experts(inputs, ids, weights)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                derivative = torch.randn_like(actual)
                grads = torch.autograd.grad(actual, (inputs, weights, experts.gate_up_proj, experts.down_proj),
                                            derivative, retain_graph=True)
                ref_grads = torch.autograd.grad(expected, (inputs, weights, source.gate_up_proj, source.down_proj),
                                                derivative)
                for left, right in zip(grads[:2], ref_grads[:2]):
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
                for left, right in zip(grads[2:], ref_grads[2:]):
                    torch.testing.assert_close(left, right.transpose(1, 2), rtol=1e-5, atol=1e-5)
                self.assertIsNone(kernel.gate_up_weight)
                with torch.no_grad():
                    source.gate_up_proj.add_(0.1)
                    experts.gate_up_proj.copy_(source.gate_up_proj.transpose(1, 2))
        self.assertEqual(len(experts.make_transforms()), 2)

    def test_exception_does_not_retain_external_parameters(self):
        """An execution failure must not retain FSDP unsharded tensor views."""
        experts = DeepseekV41TrainingExperts(module=_source())
        experts.configure(None, 1, 128, 2)
        self.addCleanup(experts.close)
        with patch.object(MegaMoeExperts, "forward", side_effect=RuntimeError("native failure")):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                experts(torch.zeros(128, 8), torch.zeros(128, 2, dtype=torch.int32), torch.ones(128, 2))
        self.assertIsNone(experts._executor.gate_up_weight)
        self.assertEqual(len(list(experts.parameters())), 2)

    def test_configuration_preserves_limit_and_capacity(self):
        """Preserve source clamp values and reject semantic changes to unclipped models."""
        for limit, expected in ((10.0, 10.0), (1.0, 1.0)):
            with self.subTest(limit=limit):
                experts = DeepseekV41TrainingExperts(module=_source(limit))
                experts.configure(None, 2, 128, 2, 2.0)
                self.addCleanup(experts.close)
                self.assertEqual(experts._executor.swiglu_limit, expected)
                self.assertEqual(experts._executor.expert_capacity_factor, 2.0)
                self.assertEqual(experts._executor.local_experts, 2)
                with self.assertRaisesRegex(RuntimeError, "reconfigure"):
                    experts.configure(None, 2, 128, 2)
        for value in (0.0, None, -1.0, float("inf"), float("nan"), True):
            with self.subTest(invalid=value), self.assertRaises(ValueError):
                DeepseekV41TrainingExperts(module=_source(value))

    def test_reject_mismatched_shared_limit_and_parallel_axes(self):
        """Fail before native initialization when source or topology differs."""
        experts = DeepseekV41TrainingExperts(module=_source())
        self.addCleanup(experts.close)
        module = SimpleNamespace(is_hash=False, gate=Mock(), experts=experts, shared_experts=SimpleNamespace(limit=1.0))
        args = {"module": module, "mesh": None, "tp_mesh": None, "cp_mesh": None, "ep_mesh": None}
        with self.assertRaisesRegex(ValueError, "same swiglu_limit"):
            deepseek_v41_ep_compute_fn(**args, megamoe=True)
        module.shared_experts.limit = 10.0
        args["tp_mesh"] = SimpleNamespace(size=lambda: 2)
        with self.assertRaisesRegex(ValueError, "TP=CP=1"):
            deepseek_v41_ep_compute_fn(**args, megamoe=True)

    def test_checkpoint_round_trip_and_recipe(self):
        """Round-trip native expert layouts using the real checkpoint conversion API."""
        config = DeepseekV4Config.from_dict({"hidden_size": 32, "moe_intermediate_size": 16,
                                             "num_local_experts": 4, "num_experts_per_tok": 2, "swiglu_limit": 10.0})
        model = nn.Module()
        model.config = config
        model.mlp = nn.Module()
        model.mlp.experts = DeepseekV4Experts(config)
        for parameter in model.parameters():
            nn.init.normal_(parameter, std=0.02)
        original = {key: value.clone() for key, value in model.state_dict().items()}
        rules = [ModuleReplacementSpec(match=("mlp.experts",), factory=DeepseekV41TrainingExperts,
                                       module_type=DeepseekV4Experts)]
        conversions = []
        apply_module_replacements(model, compile_module_replacements(model, rules), weights_mapping=conversions)
        model._weight_conversions = conversions
        restored = revert_weight_conversion(model, dict(model.state_dict()))
        for name, value in original.items():
            torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
        recipe = parse_training_args([str(Path(__file__).resolve().parents[5]
                                         / "examples/training_demo/train_deepseek_v41_online.yaml")])
        self.assertEqual(recipe.accelerator.ep_size, 16)
        self.assertFalse(recipe.megamoe)
        self.assertEqual(recipe.accelerator.tp_size, 1)
        self.assertEqual(recipe.accelerator.cp_size, 1)
        experts = model.mlp.experts
        experts.to("meta").to_empty(device="cpu")
        experts.reset_parameters()
        self.assertTrue(torch.isfinite(experts.gate_up_proj).all())
        self.assertGreater(experts.gate_up_proj.std().item(), 0.01)

    def test_router_shared_branch_and_nested_hooks(self):
        """Route once and retain the FSDP-facing expert module invocation."""
        experts = DeepseekV41TrainingExperts(module=_source())
        self.addCleanup(experts.close)
        hidden = torch.randn(128, 8)
        indices = torch.arange(256).reshape(128, 2) % 4
        weights = torch.randn(128, 2)
        gate = Mock(return_value=(None, weights, indices))
        gate.top_k = 2
        shared = nn.Linear(8, 8)
        shared.limit = 10.0
        module = SimpleNamespace(experts=experts, shared_experts=shared, gate=gate, is_hash=False,
                                 forward=lambda hidden_states, input_ids=None: hidden_states)
        compute = deepseek_v41_ep_compute_fn(megamoe=True, module=module, mesh=None, tp_mesh=None, cp_mesh=None,
                                                  ep_mesh=None, max_local_num_tokens=128)
        calls = []
        handle = experts.register_forward_pre_hook(lambda *_: calls.append(True))
        self.addCleanup(handle.remove)
        with patch.object(MegaMoeExperts, "forward", _native_reference):
            expected = experts(hidden, indices, weights) + shared(hidden)
            actual = compute(module, hidden)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gate.assert_called_once_with(hidden)
        self.assertEqual(len(calls), 2)

    def test_packed_sequence_metadata(self):
        """Keep two packed examples separate without materializing a dense mask."""
        batcher = object.__new__(DeepseekV41TextBatch)
        batcher.parallel_context = SimpleNamespace(tp_rank=0, cp_rank=0, tp_world_size=1, cp_world_size=1)
        boundaries = torch.tensor([0, 64, 128], dtype=torch.int32)
        result = batcher._build_runtime_inputs({"input_ids": torch.zeros(1, 128, dtype=torch.long),
                                                "cu_seq_lens": boundaries})
        torch.testing.assert_close(result["packed_seq_params"].cu_seq_lens, boundaries)
        self.assertEqual(result["packed_seq_params"].global_sequence_length, 128)
        self.assertEqual(set(result), {"packed_seq_params"})

    def test_text_batch_preserves_framework_positions_through_input_split(self):
        """Packed resets and CP offsets survive the complete text batch call without collisions."""
        for cp_rank, reset, expected in ((0, False, [0, 1, 2, 3]), (1, False, [4, 5, 6, 7]),
                                        (0, True, [0, 1, 2, 0]), (1, True, [1, 2, 3, 4])):
            with self.subTest(cp_rank=cp_rank, reset=reset):
                batcher = object.__new__(DeepseekV41TextBatch)
                batcher.parallel_context = SimpleNamespace(tp_rank=0, cp_rank=cp_rank,
                                                           tp_world_size=1, cp_world_size=2)
                batcher.reset_position_ids = reset
                batcher.eod_mask_loss = False
                batcher.labels_are_shifted = True
                boundaries = torch.tensor([0, 3, 8], dtype=torch.int32)
                batch = {"input_ids": torch.ones(1, 4, dtype=torch.long),
                         "labels": torch.tensor([[1, 2, -100, 4]]), "cu_seq_lens": boundaries}
                batcher.cp_sharder = SimpleNamespace(shard=lambda value: value)
                batcher.tp_broadcaster = SimpleNamespace(broadcast=lambda value, _: value)
                with (patch.object(batcher, "_read_source_batch", return_value=batch),
                      patch.object(batcher, "_normalize_source_batch", side_effect=lambda value: value),
                      patch.object(batcher, "_resolve_sequence_boundaries", return_value=boundaries),
                      patch.object(batcher, "_log_batch_flow")):
                    model_inputs, loss_inputs = batcher(iter(()))
                torch.testing.assert_close(model_inputs["position_ids"], torch.tensor([expected]))
                torch.testing.assert_close(loss_inputs["loss_mask"], torch.tensor([[1, 1, 0, 1]]))
                self.assertIs(model_inputs["shift_labels"], batch["labels"])
                self.assertEqual(model_inputs["packed_seq_params"].local_query_start, cp_rank * 4)
                self.assertNotIn("image_sequence_start", model_inputs)

                visual = DeepseekV41Runtime().build(batch=batch, parallel_context=batcher.parallel_context)
                torch.testing.assert_close(visual["position_ids"], torch.arange(cp_rank * 4, cp_rank * 4 + 4)[None])
                self.assertEqual(visual["image_sequence_start"], cp_rank * 4)

    def test_disabled_recipe_and_entrypoint_keep_original_behavior(self):
        """Default and explicit false preserve all config values and use TextTrainer."""
        recipe = str(Path(__file__).resolve().parents[5]
                     / "examples/training_demo/train_deepseek_v41_online.yaml")
        default = parse_training_args([recipe])
        explicit = parse_training_args([recipe, "--megamoe=false"])
        self.assertEqual(default.to_dict(), explicit.to_dict())
        for config in (default, explicit):
            original = deepcopy(config.to_dict())
            configure_megamoe(config)
            self.assertEqual(config.to_dict(), original)
            with (patch("examples.training_demo.train_text.parse_training_args", return_value=config),
                  patch("examples.training_demo.train_text.TextTrainer") as trainer):
                main()
                trainer.assert_called_once_with(config)
                trainer.return_value.train.assert_called_once_with()

    def test_enabled_recipe_uses_resolved_tokens_and_keeps_model_dimensions(self):
        """Enable only expert execution and derive tokens from the resolved batch size."""
        recipe = str(Path(__file__).resolve().parents[5]
                     / "examples/training_demo/train_deepseek_v41_online.yaml")
        config = parse_training_args([recipe, "--megamoe=true", "--accelerator.ep_size=8",
                                      "--dataset.data_transform.max_seq_len=128"])
        model_before = deepcopy(config.model.to_dict())
        configure_megamoe(config)
        self.assertEqual(config.model.to_dict(), model_before)
        self.assertIs(config.plan_overrides[0].replace_module._target_, DeepseekV41TrainingExperts)
        entry = next(rule for rule in config.plan_overrides
                         if getattr(rule.local_compute_fn, "megamoe", False))
        self.assertTrue(entry.local_compute_fn.megamoe)
        self.assertEqual(entry.local_compute_fn.max_local_num_tokens, 128)
        self.assertEqual(entry.local_compute_fn.expert_capacity_factor, 2.0)
        self.assertIsNone(entry.when)
        self.assertIs(config.dataloader.get_batch._target_, DeepseekV41TextBatch)

    def test_disabled_ep_retains_native_dispatch_and_router(self):
        """Explicit false and omitted MegaMoe route through the existing EP helpers."""
        experts = SimpleNamespace(_apply_gate=object())
        module = SimpleNamespace(is_hash=False, gate=Mock(), experts=experts,
                                 shared_experts=lambda value: value + 1,
                                 forward=lambda hidden_states, input_ids=None: hidden_states)
        mesh = Mock()
        mesh.get_group.return_value = "ep-group"
        mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 2))
        values = torch.randn(3, 4)
        for options in ({}, {"megamoe": False}):
            with (patch("hyper_parallel.models.deepseek_v41.adapter.expert_parallel.bind_local_expert_forward") as bind,
                  patch("hyper_parallel.models.deepseek_v41.adapter.expert_parallel.ep_routed_forward",
                        return_value=values * 2) as route):
                compute = deepseek_v41_ep_compute_fn(module=module, mesh=None, tp_mesh=None,
                                                     cp_mesh=None, ep_mesh=mesh, **options)
                result = compute(module, values)
                torch.testing.assert_close(result, values * 3 + 1)
                bind.assert_called_once_with(module, 2, apply_gate=experts._apply_gate)
                self.assertEqual(route.call_args.kwargs["ep_group"], "ep-group")

    def test_runtime_shutdown_closes_after_callbacks(self):
        """Close MegaMoe while the process groups are still owned by the base Trainer."""
        experts = DeepseekV41TrainingExperts(module=_source())
        trainer = object.__new__(BaseTrainer)
        events = []
        trainer._megamoe_experts = [experts]
        trainer.state = object()
        trainer._callbacks = [SimpleNamespace(on_train_end=lambda _: events.append("callbacks"))]
        with patch.object(experts, "close", side_effect=lambda: events.append("close")):
            trainer.on_train_end()
        self.assertEqual(events, ["callbacks", "close"])


    def test_vlm_entrypoint_preserves_disabled_recipe(self):
        """The original VLM entrypoint keeps its trainer and configuration with MegaMoe disabled."""
        recipe = str(Path(__file__).resolve().parents[5]
                     / "examples/training_demo/train_deepseek_v41_vlm_online.yaml")
        default = parse_training_args([recipe])
        explicit = parse_training_args([recipe, "--megamoe=false"])
        self.assertEqual(default.to_dict(), explicit.to_dict())
        for config in (default, explicit):
            before = deepcopy(config.to_dict())
            configure_megamoe(config)
            self.assertEqual(config.to_dict(), before)
            with (patch("scripts.train_vl.parse_training_args", return_value=config),
                  patch("scripts.train_vl.VLMTrainer") as trainer):
                vlm_main()
                trainer.assert_called_once_with(config)
                trainer.return_value.train.assert_called_once_with()

    def test_both_trainers_configure_before_distributed_setup(self):
        """Text and VLM constructors reach the common opt-in stage before plan normalization."""
        for trainer_type, recipe_name in ((TextTrainer, "train_deepseek_v41_online.yaml"),
                                           (VLMTrainer, "train_deepseek_v41_vlm_online.yaml")):
            recipe = str(Path(__file__).resolve().parents[5] / "examples/training_demo" / recipe_name)
            for enabled in (False, True):
                with self.subTest(trainer=trainer_type.__name__, enabled=enabled):
                    config = parse_training_args([recipe, f"--megamoe={str(enabled).lower()}"])
                    before = deepcopy(config.to_dict())
                    with (patch("hyper_parallel.trainer.base.setup_logging"),
                          patch("hyper_parallel.trainer.base.initialize_distributed",
                                side_effect=RuntimeError("stop before device initialization"))):
                        with self.assertRaisesRegex(RuntimeError, "stop before device"):
                            trainer_type(config)
                    if enabled:
                        self.assertIs(config.plan_overrides[0].replace_module._target_, DeepseekV41TrainingExperts)
                        entry = next(rule for rule in config.plan_overrides
                                     if getattr(rule.local_compute_fn, "megamoe", False))
                        self.assertTrue(entry.local_compute_fn.megamoe)
                        self.assertFalse(hasattr(entry.local_compute_fn, "pad_to_capacity"))
                    else:
                        self.assertEqual(config.to_dict(), before)

    def test_vlm_capacity_covers_full_samples_and_packing_budget(self):
        """Resolve a common token bound without altering image or batching metadata."""
        recipe = str(Path(__file__).resolve().parents[5]
                     / "examples/training_demo/train_deepseek_v41_vlm_online.yaml")
        for budget, expected in ((128, 250), (300, 300)):
            config = parse_training_args([recipe, "--megamoe=true", "--dataset.data_transform.max_seq_len=250",
                                          f"--dataloader.token_budget={budget}"])
            before = deepcopy(config.to_dict())
            configure_megamoe(config)
            for field in ("model", "dataset", "dataloader", "training"):
                self.assertEqual(config.to_dict()[field], before[field])
            entry = next(rule for rule in config.plan_overrides
                         if getattr(rule.local_compute_fn, "megamoe", False))
            self.assertEqual(entry.local_compute_fn.max_local_num_tokens, expected)

    def test_variable_length_experts_preserve_output_and_all_gradients(self):
        """Real-token execution preserves output and gradients without synthetic routes."""
        torch.manual_seed(74)
        experts = DeepseekV41TrainingExperts(module=_source())
        experts.configure(None, 1, 128, 2)
        self.addCleanup(experts.close)
        for tokens in (0, 1, 17, 126, 128):
            with self.subTest(tokens=tokens):
                hidden = (torch.randn(1, tokens, 8) * 5).requires_grad_()
                ids = torch.arange(tokens * 2).reshape(tokens, 2) % 4
                routing = torch.randn(tokens, 2, requires_grad=True)
                parameters = (experts.gate_up_proj, experts.down_proj)
                expected = _native_reference(experts._executor, hidden.reshape(-1, 8), ids, routing,
                                             expert_weights=parameters).reshape_as(hidden)
                with patch.object(MegaMoeExperts, "forward", autospec=True, side_effect=_native_reference) as call:
                    actual = experts(hidden, ids, routing)
                self.assertIs(call.call_args.args[1], hidden)
                self.assertIs(call.call_args.args[2], ids)
                self.assertIs(call.call_args.args[3], routing)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                gradient = torch.randn_like(actual)
                targets = (hidden, routing, *parameters)
                actual_grads = torch.autograd.grad(actual, targets, gradient)
                expected_grads = torch.autograd.grad(expected, targets, gradient)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)

    def test_multimodal_router_and_shared_expert_gradients(self):
        """Use real image/text router biases and propagate image-token gradients through variable-length experts."""
        torch.manual_seed(81)
        gate = DeepseekV41TopKRouter(SimpleNamespace(
            hidden_size=8, num_local_experts=4, num_experts_per_tok=2, scoring_func="sigmoid",
            routed_scaling_factor=1.0, v41_vision_enabled=True,
        ))
        with torch.no_grad():
            gate.weight.normal_(std=0.1)
            gate.bias.copy_(torch.tensor([2., 2., 0., 0.]))
            gate.bias_vl.copy_(torch.tensor([0., 0., 2., 2.]))
        experts = DeepseekV41TrainingExperts(module=_source())
        self.addCleanup(experts.close)
        shared = nn.Linear(8, 8)
        shared.limit = 10.0
        module = SimpleNamespace(experts=experts, shared_experts=shared, gate=gate, is_hash=False,
                                 forward=lambda hidden_states, input_ids=None, image_mask=None: hidden_states)
        compute = deepseek_v41_ep_compute_fn(megamoe=True, module=module, mesh=None, tp_mesh=None, cp_mesh=None,
                                             ep_mesh=None, max_local_num_tokens=128)
        hidden = (torch.randn(1, 18, 8) * 5).requires_grad_()
        image_mask = torch.zeros(1, 18, dtype=torch.bool)
        image_mask[:, 3:8] = True
        _, weights, ids = gate(hidden, image_mask)
        self.assertTrue(torch.all(ids[image_mask.flatten()] >= 2))
        self.assertTrue(torch.all(ids[~image_mask.flatten()] < 2))
        expected = _native_reference(experts._executor, hidden.reshape(-1, 8), ids, weights,
                                     expert_weights=(experts.gate_up_proj, experts.down_proj)).reshape_as(hidden)
        expected = expected + shared(hidden)
        with patch.object(MegaMoeExperts, "forward", _native_reference):
            actual = compute(module, hidden, image_mask=image_mask)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        targets = (hidden, gate.weight, experts.gate_up_proj, experts.down_proj, shared.weight, shared.bias)
        gradient = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, targets, gradient)
        expected_grads = torch.autograd.grad(expected, targets, gradient)
        for left, right in zip(actual_grads, expected_grads):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
        self.assertGreater(actual_grads[0][image_mask].abs().sum().item(), 0)

    def test_common_trainer_resource_sharing(self):
        """Both training paths share configured resources and disabled models remain untouched."""
        experts = DeepseekV41TrainingExperts(module=_source())
        experts.configure(None, 1, 128, 2)
        self.addCleanup(experts.close)
        trainer = object.__new__(BaseTrainer)
        trainer.model = nn.Sequential(experts)
        for enabled in (False, True):
            trainer.config = SimpleNamespace(megamoe=enabled)
            with patch.object(DeepseekV41TrainingExperts, "share_execution_resources") as share:
                trainer._share_megamoe_resources()
                if enabled:
                    share.assert_called_once_with([experts])
                else:
                    share.assert_not_called()
                    self.assertEqual(trainer._megamoe_experts, [])
