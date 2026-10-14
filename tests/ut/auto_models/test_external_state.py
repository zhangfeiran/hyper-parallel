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
"""Generic external-state build contracts and Host ownership boundaries."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.models._transformers.model_builder import _create_external_state
from hyper_parallel.models.deepseek_v41.adapter.engram.host_state import DeepseekV41HostState
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.deepseek_v41.adapter.policies.sharding import (
    get_fsdp_wrap_modules, get_fsdp_excluded_subtrees, get_fsdp_execution_order,
)


class TestExternalBuildContract(unittest.TestCase):
    """A synthetic provider must work without model-family logic in the builder."""

    def test_optional_provider_and_validation(self) -> None:
        """Only participating adapters receive the generic lifecycle context."""
        model = torch.nn.Module()
        model.config = SimpleNamespace(model_type="synthetic", architectures=[])
        setup, fsdp = object(), object()
        kwargs = {"model": model, "mesh": None, "source_shard_info": {}, "device": torch.device("cpu"),
                  "is_meta_device": True, "model_init_dtype": "float32", "validate_placement": False,
                  "compile_for_execution": False, "distributed_setup": setup, "fsdp2_manager": fsdp}
        with patch("hyper_parallel.models._transformers.model_builder.get_model_adapter", return_value=None):
            self.assertIsNone(_create_external_state(**kwargs))
        self.assertFalse(hasattr(model, "_hp_external_state"))
        external = Mock()
        provider = Mock(return_value=external)
        adapter = SimpleNamespace(external_state=provider)
        with patch("hyper_parallel.models._transformers.model_builder.get_model_adapter", return_value=adapter):
            self.assertIs(_create_external_state(**kwargs), external)
        self.assertIs(model._hp_external_state, external)
        self.assertIs(provider.call_args.args[0], model)
        self.assertEqual(provider.call_args.args[1].model_init_dtype, torch.float32)
        external.validate_build.assert_called_once_with(compile_for_execution=False,
                                                       distributed_setup=setup, fsdp2_manager=fsdp)

    def test_checkpoint_validation_delegates_to_the_provider(self) -> None:
        """Synthetic external state controls its own checkpoint requirements."""
        external = Mock()
        checkpoint = CheckpointingConfig(save_ckpt=False)
        mesh = SimpleNamespace(pp_size=1)
        trainer = SimpleNamespace(model=SimpleNamespace(_hp_external_state=external),
                                  config=SimpleNamespace(checkpoint=checkpoint), mesh=mesh)
        with patch("hyper_parallel.trainer.callbacks.checkpoint_callback.build_checkpointer"):
            CheckpointerCallback(trainer)
        external.validate_checkpoint_config.assert_called_once_with(checkpoint, mesh)

    def test_host_checkpoint_scope_is_validated_before_saving(self) -> None:
        """Async saves and incomplete optimizer scopes cannot bypass Host preflight."""
        checkpoint = CheckpointingConfig(save_ckpt=True, is_async=False,
                                         save_optimizer=True, save_train_state=True)
        mesh = SimpleNamespace(pp_size=1)
        DeepseekV41HostState.validate_checkpoint_config(checkpoint, mesh)
        checkpoint.is_async = True
        with self.assertRaisesRegex(ValueError, "synchronous"):
            DeepseekV41HostState.validate_checkpoint_config(checkpoint, mesh)
        checkpoint.is_async = False
        checkpoint.save_optimizer = False
        with self.assertRaisesRegex(ValueError, "optimizer and train-state"):
            DeepseekV41HostState.validate_checkpoint_config(checkpoint, mesh)

    def test_host_precision_validation_is_model_owned(self) -> None:
        """Unsupported precision and compilation fail before materialization."""
        precision = SimpleNamespace(param_dtype="bfloat16", reduce_dtype="float32",
                                    cast_forward_inputs=False, output_dtype=None)
        setup = SimpleNamespace(fp32_main_params=True)
        manager = SimpleNamespace(config=SimpleNamespace(mix_precision=precision))
        kwargs = {"compile_for_execution": False, "distributed_setup": setup, "fsdp2_manager": manager}
        DeepseekV41HostState.validate_build(**kwargs)
        with self.assertRaisesRegex(ValueError, "torch.compile"):
            DeepseekV41HostState.validate_build(**{**kwargs, "compile_for_execution": True})
        precision.reduce_dtype = "bfloat16"
        with self.assertRaisesRegex(ValueError, "FP32 reduction"):
            DeepseekV41HostState.validate_build(**kwargs)

    def test_fsdp_excludes_host_table_and_keeps_dense_projection(self) -> None:
        """The owner table stays CPU-owned while the Engram projection remains an FSDP unit."""
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        layer = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([layer])
        layer.engram = torch.nn.Module()
        layer.engram.embed = HostEngramTable(
            source_weight=torch.nn.Parameter(torch.empty((16, 2), device="meta")), logical_rows=15,
        )
        layer.engram.wkv = torch.nn.Linear(2, 2)
        self.assertEqual(get_fsdp_wrap_modules(model), ("model.layers.0.engram.wkv",))
        layer.engram.embed = torch.nn.Embedding(16, 2)
        self.assertEqual(get_fsdp_wrap_modules(model),
                         ("model.layers.0.engram.embed", "model.layers.0.engram.wkv"))

    def test_variable_image_recipe_keeps_visual_collectives_outside_image_loop(self) -> None:
        """Selecting root ownership removes every per-image FSDP unit and prefetch target."""
        model = torch.nn.Module()
        model.config = SimpleNamespace(v41_root_owned_vision=False)
        model.model = torch.nn.Module()
        model.model.vision = torch.nn.Module()
        model.model.vision.patch_embed = torch.nn.Linear(2, 2)
        model.model.vision.norm = torch.nn.LayerNorm(2)
        model.model.vision.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        model.model.aligner = torch.nn.Linear(2, 2)
        wrapped = get_fsdp_wrap_modules(model)
        self.assertIn("model.vision.blocks.0", wrapped)
        self.assertIn("model.aligner", wrapped)
        model.config.v41_root_owned_vision = True
        self.assertEqual(get_fsdp_wrap_modules(model), ())
        self.assertEqual(get_fsdp_excluded_subtrees(model), ("model.vision", "model.aligner"))
        self.assertEqual(get_fsdp_execution_order(model, ()), ())
