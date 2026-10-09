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
"""Complete random-initialized Qwen dense decoder training graph for FFN acceptance."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.modules import RMSNorm, SwiGLUMLP
from hyper_parallel.core.multicore.examples.mega_moe.qwen_moe_model import QwenAttention
from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)


@dataclass(frozen=True)
class QwenDenseConfig:
    """Dense shape/attention contract without router or expert topology."""

    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_layers: int = 2
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_seq_len: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    hidden_act: str = "silu"

    def __post_init__(self) -> None:
        sizes = (self.vocab_size, self.hidden_size, self.intermediate_size, self.num_layers,
                 self.num_attention_heads, self.num_key_value_heads, self.max_seq_len)
        if any(type(size) not in (int,) or size <= 0 for size in sizes):
            raise ValueError("Qwen dense dimensions must be positive integers")
        if (self.hidden_size % self.num_attention_heads
                or self.num_attention_heads % self.num_key_value_heads
                or (self.hidden_size // self.num_attention_heads) % 2):
            raise ValueError("Qwen dense attention heads must divide dimensions and have even rotary width")
        if self.hidden_act != "silu":
            raise ValueError("Qwen dense FFN acceptance currently supports SiLU only")


class DenseMLP(nn.Module):
    """Original separate gate/up/down projections for the independent baseline."""

    def __init__(self, config: QwenDenseConfig) -> None:
        """Create a bias-free standard dense SwiGLU.

        Args:
            config: Dense decoder shape and activation contract.
        """
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the conventional three-Linear FFN.

        Args:
            x: Attention/normalization output tokens.
        """
        return self.down_proj(functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DenseDecoder(nn.Module):
    """Residual GQA attention followed by residual dense FFN."""

    def __init__(self, config: QwenDenseConfig, recompute: bool) -> None:
        """Reuse the existing Qwen attention/normalization path.

        Args:
            config: Dense model dimensions.
            recompute: Apply standard non-reentrant activation checkpointing.
        """
        super().__init__()
        self.attention = QwenAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DenseMLP(config)
        self.recompute = recompute

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compose attention and FFN in the normal decoder order.

        Args:
            x: Decoder residual stream.
            positions: Rotary token positions.
        """
        hidden = x + self.attention(self.input_layernorm(x), positions)
        normalized = self.post_attention_layernorm(hidden)
        if self.recompute and self.training:
            return hidden + checkpoint(self.mlp, normalized, use_reentrant=False)
        return hidden + self.mlp(normalized)


class QwenDenseModel(nn.Module):
    """Complete dense LM with module replacement before optimizer/sharding."""

    def __init__(self, config: QwenDenseConfig, backend: str = "mega_ffn", recompute: bool = False,
                 execution: DenseExecutionConfig | None = None) -> None:
        """Construct one independently initialized model and its checkpoint mapping.

        Args:
            config: Dense decoder architecture.
            backend: common, packed or mega_ffn; packed uses the existing SwiGLUMLP.
            recompute: Checkpoint the FFN via the standard Torch API.
            execution: MegaFFN backend selected through the existing replacement context.
        """
        super().__init__()
        if backend not in ("common", "packed", "mega_ffn"):
            raise ValueError("Qwen dense backend must be common, packed or mega_ffn")
        self.config, self.backend = config, backend
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DenseDecoder(config, recompute) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.weights_mapping = []
        if backend != "common":
            factory = MegaFFNAdapter if backend == "mega_ffn" else SwiGLUMLP
            rules = [ModuleReplacementSpec(("layers.*.mlp",), factory, DenseMLP)]
            apply_module_replacements(self, compile_module_replacements(self, rules),
                                      weights_mapping=self.weights_mapping,
                                      context={"mega_ffn_execution": execution})

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        """Run embeddings, all decoder layers, LM head and shifted-token loss.

        Args:
            input_ids: Batch of token identifiers.
            labels: Optional labels, shifted internally for causal LM training.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Qwen dense tokens must fit the configured sequence length")
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        for layer in self.layers:
            hidden = layer(hidden, positions)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            shifted = functional.pad(labels, (0, 1), value=-100)[..., 1:]
            loss = functional.cross_entropy(logits.float().flatten(0, 1), shifted.flatten(), ignore_index=-100)
        return {"logits": logits, "loss": loss}

    def close(self) -> None:
        """Close each FFN while preserving outstanding backward state."""
        for layer in self.layers:
            if hasattr(layer.mlp, "close"):
                layer.mlp.close()
