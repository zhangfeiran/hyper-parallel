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
"""Complete dense decoder/optimizer graph with the device attention provider mocked."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.core.multicore.examples.mega_ffn.qwen_dense_model import QwenDenseConfig, QwenDenseModel
from tests.common.mark_utils import arg_mark


def _cpu_attention(module, query, key, value, **kwargs):
    del module
    repeats = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)
    scores = (query.float() @ key.float().transpose(-1, -2)) * kwargs["scaling"]
    causal = torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool).tril()
    output = (scores.masked_fill(~causal, float("-inf")).softmax(-1) @ value.float()).to(query.dtype)
    return output.transpose(1, 2).contiguous(), None


def _cpu_norm(value, weight, epsilon):
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + epsilon)
    return normalized.to(value.dtype) * weight


def _cpu_rotary(query, key, cosine, sine):
    cosine, sine = cosine[None, None], sine[None, None]

    def _rotate(value):
        left, right = value.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    return query * cosine + _rotate(query) * sine, key * cosine + _rotate(key) * sine


class TestQwenDenseModel(unittest.TestCase):
    """Check standard dense training integration without invoking device libraries."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    @patch.object(DefaultDeviceType, "_default_device_type", "cpu")
    def test_complete_attention_ffn_loss_and_optimizer_step(self):
        """Feature: Full dense training graph.
        Description: Train all three FFN backends through attention, residuals and next-token loss.
        Expectation: Every trainable parameter receives finite gradients and the optimizer changes the head.
        """
        config = QwenDenseConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, max_seq_len=8)
        path = "hyper_parallel.core.multicore.examples.mega_moe.qwen_moe_model.run_qwen3_moe_flash_attention"
        with (patch(path, side_effect=_cpu_attention),
              patch("hyper_parallel.components.modules.rms_norm.rms_norm", side_effect=_cpu_norm),
              patch("hyper_parallel.core.multicore.examples.mega_moe.qwen_moe_model.apply_rotary_pos_emb",
                    side_effect=_cpu_rotary)):
            for backend in ("common", "packed", "mega_ffn"):
                with self.subTest(backend=backend):
                    torch.manual_seed(42)
                    model = QwenDenseModel(config, backend, recompute=True).to(dtype=torch.bfloat16)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
                    ids = torch.randint(0, config.vocab_size, (2, 8))
                    head = model.lm_head.weight.detach().clone()
                    result = model(ids, ids)
                    self.assertEqual(result["logits"].shape, (2, 8, 64))
                    self.assertTrue(torch.isfinite(result["loss"]))
                    result["loss"].backward()
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    optimizer.step()
                    self.assertFalse(torch.equal(head, model.lm_head.weight))
                    model.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_replacement_keeps_initial_logical_weights_equal(self):
        """Feature: Independent training initialization.
        Description: Independently initialize source and AST graphs from one seed before optimization.
        Expectation: Canonical FFN and ordinary model weights match without resynchronization.
        """
        config = QwenDenseConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, max_seq_len=8)
        torch.manual_seed(42)
        common = QwenDenseModel(config, "common")
        torch.manual_seed(42)
        candidate = QwenDenseModel(config, "mega_ffn")
        torch.testing.assert_close(common.embed_tokens.weight, candidate.embed_tokens.weight, rtol=0, atol=0)
        torch.testing.assert_close(common.lm_head.weight, candidate.lm_head.weight, rtol=0, atol=0)
        for source, actual in zip(common.layers, candidate.layers):
            torch.testing.assert_close(torch.cat((source.mlp.gate_proj.weight, source.mlp.up_proj.weight)).t(),
                                       actual.mlp.gate_up, rtol=0, atol=0)
            torch.testing.assert_close(source.mlp.down_proj.weight.t(), actual.mlp.down, rtol=0, atol=0)
            torch.testing.assert_close(source.attention.q_proj.weight, actual.attention.q_proj.weight, rtol=0, atol=0)
