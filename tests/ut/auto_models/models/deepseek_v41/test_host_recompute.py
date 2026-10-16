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
"""Compose Host rows, packed multi-image VLM and decoder attention replay."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from hyper_parallel.components.modules.shared_compressed_dsa_attention import SharedCompressedPackedSequence
from hyper_parallel.distributed.activation_checkpoint import _apply_activation_checkpointing
from hyper_parallel.models.deepseek_v41.adapter.engram.host_replacement import HostEngramModule
from hyper_parallel.models.deepseek_v41.adapter.engram.host_state import build_deepseek_v41_external_state
from hyper_parallel.models.deepseek_v41.adapter.registration import DEEPSEEK_V41_ADAPTER_SPEC
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM
from hyper_parallel.models.external_state import ExternalBuildContext


def _tiny_model(directory: str) -> DeepseekV41ForCausalLM:
    """Build a four-layer language and one-layer vision composition fixture."""
    assets = {
        "layer_ids": [1], "max_ngram_size": 3, "num_heads": 2, "head_dim": 4,
        "primes": [[[17, 19], [23, 29]]], "num_embeddings": [88],
        "multipliers": [[101, 103, 107]], "token_map": list(range(32)), "pad_token_id": 0,
    }
    path = Path(directory) / "engram.json"
    path.write_text(json.dumps(assets), encoding="utf-8")
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32, hidden_size=16, moe_intermediate_size=24, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=1, head_dim=8, q_lora_rank=8,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        layer_types=["sliding_attention"] * 2 + ["compressed_sparse_attention"] * 2,
        mlp_layer_types=["moe"] * 4, swiglu_limit=1.5, sliding_window=4,
        o_groups=2, o_lora_rank=4, index_n_heads=2, index_head_dim=4, index_topk=2,
        partial_rotary_factor=0.5, pad_token_id=0,
    )
    config.architectures = ["DeepseekV41ForCausalLM"]
    config.v41_model_mode = "validation_crop"
    config.v41_compress_ratios = [0, 0, 2, 2]
    config.v41_kv_source_layer_ids = [2]
    config.v41_index_source_layer_ids = [2]
    config.v41_candidate_source_layer_id = -1
    config.v41_indexer_loss_coeff = 0.001
    config.v41_engram_assets_path = str(path)
    config.v41_engram_table_pad_multiple = 1
    config.v41_vision_enabled = True
    config.v41_vision_hidden_size = 16
    config.v41_vision_intermediate_size = 24
    config.v41_vision_patch_size = 2
    config.v41_vision_num_attention_heads = 2
    config.v41_vision_rope_theta = 10000.0
    config.v41_vision_num_hidden_layers = 1
    config.v41_vision_downsample_ratio = 1
    return DeepseekV41ForCausalLM(config).train()


class TestHostRecompute(unittest.TestCase):
    """Exercise the real model with BF16 rows and sparse backward replay."""

    def test_bfloat16_rows_match_packed_vlm_without_recompute(self) -> None:
        """The production Host binding must stage BF16 rows exactly once in backward."""
        self.assertEqual(DEEPSEEK_V41_ADAPTER_SPEC.architecture, "DeepseekV41ForCausalLM")
        torch.manual_seed(19)
        with TemporaryDirectory() as directory, patch.object(DefaultDeviceType, "get_device_type", return_value="cpu"):
            reference = _tiny_model(directory)
            candidate = deepcopy(reference)
            source = candidate.model.layers[1].engram
            values = source.embed.weight.detach().clone()
            source.embed.weight = torch.nn.Parameter(torch.empty(values.shape, device="meta"))
            candidate.model.layers[1].engram = HostEngramModule(module=source)
            candidate.config.engram_storage_backend = "host"
            external = build_deepseek_v41_external_state(
                candidate, ExternalBuildContext(None, {}, torch.device("cpu"), torch.float32, False),
            )
            self.assertIsNotNone(external)
            table = candidate.model.layers[1].engram.embed
            self.assertEqual(table.lookup_compute_dtype, torch.bfloat16)
            torch.utils.swap_tensors(table.weight, torch.nn.Parameter(values))
            _apply_activation_checkpointing(
                candidate, "full", selection=SimpleNamespace(
                    source="model_adapter_safe_regions", layer_count=4, layer_indices=None,
                ),
            )
            reference.bfloat16()
            candidate.bfloat16()
            self.assertEqual(table.weight.dtype, torch.float32)
            token_types = torch.full((1, 32), -1, dtype=torch.long)
            for start in (2, 18):
                token_types[0, start:start + 8] = torch.tensor([0, 1, 1, 2, 1, 1, 2, 3])
            inputs = {
                "input_ids": torch.randint(1, 32, (1, 32)), "token_types": token_types,
                "pixel_values": torch.randn(8, 3, 2, 2).bfloat16(),
                "image_patch_offsets": torch.tensor([0, 4, 8]),
                "image_vit_grid_hw": torch.tensor([[2, 2], [2, 2]]),
                "image_llm_grid_hw": torch.tensor([[2, 2], [2, 2]]),
                "image_batch_indices": torch.tensor([0, 0]), "image_token_starts": torch.tensor([2, 18]),
                "use_cache": False,
                "packed_seq_params": SharedCompressedPackedSequence(torch.tensor([0, 16, 32]), 0, 32, 32),
            }
            expected = reference(**inputs).logits
            actual = candidate(**inputs).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.float().square().mean().backward()
            actual.float().square().mean().backward()
            self.assertEqual(len(table.pending_chunks), 1)
            gradients = {}
            for name, parameter in reference.named_parameters():
                gradient = candidate.get_parameter(name).grad
                if name == "model.layers.1.engram.embed.weight":
                    gradient = table.coalesce_pending().to_dense()
                if parameter.grad is None:
                    self.assertIsNone(gradient)
                    continue
                self.assertIsNotNone(gradient, name)
                torch.testing.assert_close(
                    gradient.float(), parameter.grad.float(), rtol=2.0e-2, atol=1.0e-6, msg=name,
                )
                gradients[name] = gradient
            self.assertTrue(any(".vision." in name for name in gradients))
            self.assertTrue(any(".aligner." in name for name in gradients))
