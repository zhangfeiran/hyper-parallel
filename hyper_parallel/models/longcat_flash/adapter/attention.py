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
"""LongCat scaling and interleaved RoPE on the generic MLA implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Native tensor APIs are the project backend contract.
# pylint: disable=forbidden-backend-import
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.longcat_flash.modeling_longcat_flash import (
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)

from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward
from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.models.replacement import module_replacement


def longcat_attention_interface(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                               value: torch.Tensor, attention_mask: torch.Tensor | None,
                               dropout: float = 0.0, scaling: float | None = None,
                               **kwargs: Any) -> tuple:
    """Select the accelerator attention primitive without changing LongCat QKV.

    Args:
        module: Source module supplied by the replacement executor.
        query: Query states in BNSD order.
        key: Key states in BNSD order.
        value: Value states in BNSD order.
        attention_mask: Mask in the source attention convention.
        dropout: Attention dropout probability.
        scaling: Source attention score scale.
    """
    if query.device.type == "npu" and module.config._attn_implementation != "eager":
        return npu_fusion_attention_forward(module, query, key, value, attention_mask,
                                           dropout=dropout, scaling=scaling, **kwargs)
    if kwargs.get("actual_seq_len") is not None:
        raise ValueError("CPU LongCat attention requires an explicit packed attention mask")
    return eager_attention_forward(module, query, key, value, attention_mask,
                                   dropout=dropout, scaling=scaling)


@module_replacement
class LongcatFlashMLAAttention(MLAAttention):
    """Keep LongCat's Q/KV LoRA scales and rotary convention with fused latents."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Construct MLA from the complete source attention and retain its state."""
        if (context or {}).get("cp") or (context or {}).get("pp"):
            raise ValueError("LongCat optimized attention currently requires CP=PP=1")
        super().__init__(module=module, module_fqn=module_fqn, context=context,
                         attention_interface=longcat_attention_interface,
                         rotary_interface=apply_rotary_pos_emb_interleave)
        self.mla_scale_q_lora = module.mla_scale_q_lora
        self.mla_scale_kv_lora = module.mla_scale_kv_lora
        with torch.no_grad():
            self.linear_qkv.weight.copy_(torch.cat((module.q_a_proj.weight, module.kv_a_proj_with_mqa.weight)))
            if self.linear_qkv.bias is not None:
                self.linear_qkv.bias.copy_(torch.cat((module.q_a_proj.bias, module.kv_a_proj_with_mqa.bias)))

    def project_latent_inputs(self, hidden_states: torch.Tensor) -> tuple:
        """Retain separate BF16 projection gradients when eager attention is requested.

        Args:
            hidden_states: Original token states before latent normalization.
        """
        if self.config._attn_implementation != "eager":
            return super().project_latent_inputs(hidden_states)
        weight = self.linear_qkv.weight
        bias = self.linear_qkv.bias
        query = F.linear(  # pylint: disable=not-callable
                         hidden_states, weight[:self.q_lora_rank],
                         None if bias is None else bias[:self.q_lora_rank])
        compressed = F.linear(  # pylint: disable=not-callable
                              hidden_states, weight[self.q_lora_rank:],
                              None if bias is None else bias[self.q_lora_rank:])
        kv_nope, key_rope = compressed.split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        return self.q_a_layernorm(query), self.kv_a_layernorm(kv_nope), key_rope

    def _project_latents(self, hidden_states: torch.Tensor) -> Any:
        """Apply source scales after the projections and before RoPE/cache writes.

        Args:
            hidden_states: Token states in original source order.
        """
        latents = super()._project_latents(hidden_states)
        latents.query_pass = latents.query_pass * self.mla_scale_q_lora
        latents.query_rope = latents.query_rope * self.mla_scale_q_lora
        latents.kv_nope = latents.kv_nope * self.mla_scale_kv_lora
        return latents
