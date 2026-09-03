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

"""Compact random-weight Qwen MoE model for optimizer-step benchmarking."""

from __future__ import annotations

__all__ = [
    "QwenMoeBlock",
    "QwenMoeConfig",
    "QwenMoeModel",
    "RoutingMode",
]

from dataclasses import dataclass
import math
from typing import Any, Literal

# This is a Torch-only executable example, not a platform-agnostic library module.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.auto_models.components.models.qwen3_moe_attention_common import (
    run_qwen3_moe_flash_attention,
)
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.models.modules import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLUMLP,
    apply_rotary_pos_emb,
)

RoutingMode = Literal["balanced", "topk", "dirichlet"]


@dataclass(frozen=True)
class QwenMoeConfig:
    """Fixed-shape synthetic Qwen model configuration.

    The example intentionally models one TP1/EP8 optimizer-step workload. Its
    deterministic balanced default gives every destination the same load;
    optional Dirichlet routing reproduces load-skew and fallback behavior.
    """

    vocab_size: int = 32000
    hidden_size: int = 5120
    num_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    num_experts: int = 48
    top_k: int = 8
    intermediate_size: int = 1792
    shared_expert_intermediate_size: int = 512
    routed_scaling_factor: float = 1.0
    routing: RoutingMode = "balanced"
    router_load_alpha: float = 1.0
    router_load_seed: int = 42
    local_num_tokens: int = 4096
    expert_capacity_factor: float | None = None
    ep_size: int = 8

    def __post_init__(self) -> None:
        """Validate the fixed optimizer-step topology."""
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_seq_len": self.max_seq_len,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "intermediate_size": self.intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "local_num_tokens": self.local_num_tokens,
            "ep_size": self.ep_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if self.num_experts % self.ep_size:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by ep_size ({self.ep_size})."
            )
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})."
            )
        if self.local_num_tokens % 128:
            raise ValueError(
                f"local_num_tokens ({self.local_num_tokens}) must be divisible by 128."
            )
        if self.max_seq_len > self.local_num_tokens:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) cannot exceed local_num_tokens "
                f"({self.local_num_tokens}) in the fixed batch-one workload."
            )
        self._validate_routing()

    def _validate_routing(self) -> None:
        """Validate learned and synthetic router parameters."""
        if self.routing not in ("balanced", "topk", "dirichlet"):
            raise ValueError(f"unsupported routing mode: {self.routing!r}.")
        if self.routing == "balanced" and self.top_k != self.ep_size:
            raise ValueError(
                "balanced routing requires top_k to equal ep_size so every token "
                f"selects one expert per destination, got top_k={self.top_k} and "
                f"ep_size={self.ep_size}."
            )
        if not math.isfinite(self.router_load_alpha) or self.router_load_alpha < 0.01:
            raise ValueError(
                "router_load_alpha must be finite and at least 0.01, got "
                f"{self.router_load_alpha}."
            )
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError(
                "routed_scaling_factor must be finite and positive, got "
                f"{self.routed_scaling_factor}."
            )


class QwenAttention(nn.Module):
    """Thin Qwen GQA using the library's fused Ascend attention kernel."""

    def __init__(self, config: QwenMoeConfig) -> None:
        """Initialize QKV, output, and rotary projections."""
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta,
        )
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused causal grouped-query attention."""
        batch_size, sequence_length, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        output, _ = run_qwen3_moe_flash_attention(
            self,
            query,
            key,
            value,
            attention_mask=None,
            dropout=0.0,
            scaling=self.scaling,
        )
        return self.o_proj(output.reshape(batch_size, sequence_length, -1))


def _topk_route(
    hidden_states: torch.Tensor,
    gate: nn.Linear,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized learned TopK scores and expert IDs."""
    logits = F.linear(  # pylint: disable=not-callable
        hidden_states.float(),
        gate.weight.float(),
    )
    top_logits, topk_ids = torch.topk(logits, top_k, dim=-1)
    topk_weights = F.softmax(top_logits, dim=-1, dtype=torch.float32)
    return topk_weights * routed_scaling_factor, topk_ids


def _dirichlet_expert_counts(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    alpha: float,
    seed: int,
) -> torch.Tensor:
    """Draw reproducible expert counts with exact, distinct Top-K slots."""
    total_slots = num_tokens * top_k
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        concentration = torch.full((num_experts,), alpha, dtype=torch.float64)
        shares = torch.distributions.Dirichlet(concentration).sample()

    # Cap every expert at one slot per token. This makes it possible to turn
    # the expert-major histogram into token-major Top-K selections without a
    # repeated expert for any token.
    target = torch.zeros(num_experts, dtype=torch.float64)
    remaining = torch.ones(num_experts, dtype=torch.bool)
    remaining_slots = total_slots
    while remaining_slots:
        active = remaining.nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            raise RuntimeError("Dirichlet route apportionment exhausted all experts.")
        active_shares = shares[active]
        share_sum = active_shares.sum()
        if not bool(share_sum > 0):
            active_shares = torch.ones_like(active_shares)
            share_sum = active_shares.sum()
        proposed = active_shares / share_sum * remaining_slots
        saturated = proposed > num_tokens
        if not bool(saturated.any()):
            target[active] = proposed
            break
        saturated_experts = active[saturated]
        target[saturated_experts] = num_tokens
        remaining[saturated_experts] = False
        remaining_slots -= num_tokens * saturated_experts.numel()

    counts = target.floor().to(torch.int64)
    residual = total_slots - int(counts.sum())
    if residual:
        fractions = target - counts
        fractions[counts >= num_tokens] = -1.0
        order = torch.argsort(fractions, descending=True, stable=True)
        counts[order[:residual]] += 1
    if int(counts.sum()) != total_slots or bool((counts > num_tokens).any()):
        raise RuntimeError("Dirichlet route apportionment violated its slot constraints.")
    return counts


def _fixed_token_major_route(
    num_tokens: int,
    top_k: int,
    counts: torch.Tensor,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert exact expert counts into deterministic token-major selections."""
    expert_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.long, device=counts.device),
        counts,
    )
    if expert_ids.numel() != num_tokens * top_k:
        raise ValueError(
            f"route counts contain {expert_ids.numel()} slots, expected "
            f"{num_tokens * top_k}."
        )
    topk_ids = expert_ids.view(top_k, num_tokens).transpose(0, 1).contiguous()
    topk_weights = torch.full(
        (num_tokens, top_k),
        routed_scaling_factor / top_k,
        dtype=torch.float32,
        device=counts.device,
    )
    return topk_weights, topk_ids


def _balanced_token_major_route(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    ep_size: int,
    source_rank: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an exact destination-balanced route with near-even experts."""
    if top_k != ep_size:
        raise ValueError("balanced routing requires top_k to equal ep_size.")
    local_experts = num_experts // ep_size
    token_index = torch.arange(num_tokens, dtype=torch.long).unsqueeze(1)
    destination = torch.arange(ep_size, dtype=torch.long).unsqueeze(0)
    global_token_index = token_index * ep_size + source_rank
    local_expert = (global_token_index + destination) % local_experts
    topk_ids = destination * local_experts + local_expert
    topk_weights = torch.full(
        (num_tokens, top_k),
        routed_scaling_factor / top_k,
        dtype=torch.float32,
    )
    counts = torch.bincount(topk_ids.reshape(-1), minlength=num_experts).to(torch.int32)
    return topk_weights, topk_ids, counts


class QwenMoeBlock(nn.Module):
    """One Qwen router, managed routed-expert, and shared-expert block."""

    def __init__(
        self,
        config: QwenMoeConfig,
        ep_group: Any | None,
        *,
        layer_index: int = 0,
    ) -> None:
        """Initialize the router, routed experts, and shared expert."""
        super().__init__()
        self.config = config
        self.gate = (
            nn.Linear(config.hidden_size, config.num_experts, bias=False)
            if config.routing == "topk"
            else None
        )
        fixed_counts = None
        fixed_weights = None
        fixed_ids = None
        if config.routing == "balanced":
            source_rank = dist.get_rank(ep_group) if dist.is_initialized() else 0
            fixed_weights, fixed_ids, fixed_counts = _balanced_token_major_route(
                config.local_num_tokens,
                config.top_k,
                config.num_experts,
                config.ep_size,
                source_rank,
                config.routed_scaling_factor,
            )
        elif config.routing == "dirichlet":
            fixed_counts = _dirichlet_expert_counts(
                config.local_num_tokens,
                config.top_k,
                config.num_experts,
                config.router_load_alpha,
                config.router_load_seed + layer_index,
            ).to(torch.int32)
            fixed_weights, fixed_ids = _fixed_token_major_route(
                config.local_num_tokens,
                config.top_k,
                fixed_counts,
                config.routed_scaling_factor,
            )
        self.register_buffer("_fixed_route_counts", fixed_counts, persistent=False)
        self.register_buffer("_fixed_topk_weights", fixed_weights, persistent=False)
        self.register_buffer("_fixed_topk_ids", fixed_ids, persistent=False)
        self._last_route_counts: torch.Tensor | None = None
        self.experts = MegaMoeExperts(
            local_num_tokens=config.local_num_tokens,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            expert_capacity_factor=config.expert_capacity_factor,
            ep_size=config.ep_size,
            ep_group=ep_group,
        )
        self.shared_expert = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run routed experts and the always-active shared expert."""
        hidden_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_shape[-1])
        if self.config.routing in ("balanced", "dirichlet"):
            if (
                self._fixed_topk_weights is None
                or self._fixed_topk_ids is None
                or self._fixed_route_counts is None
            ):
                raise RuntimeError("Dirichlet route buffers were not initialized.")
            topk_weights = self._fixed_topk_weights
            topk_ids = self._fixed_topk_ids
            tokens_per_expert = self._fixed_route_counts
        else:
            if self.gate is None:
                raise RuntimeError("learned Top-K routing requires a gate projection.")
            topk_weights, topk_ids = _topk_route(
                hidden_flat,
                self.gate,
                self.config.top_k,
                self.config.routed_scaling_factor,
            )
            tokens_per_expert = torch.bincount(
                topk_ids.reshape(-1).to(torch.int64),
                minlength=self.config.num_experts,
            ).to(torch.int32)
        self._last_route_counts = tokens_per_expert.detach()
        routed_output = self.experts(
            hidden_flat,
            topk_ids,
            topk_weights,
            tokens_per_expert=tokens_per_expert,
        )
        shared_output = self.shared_expert(hidden_flat)
        return (routed_output + shared_output).view(hidden_shape)

    def route_counts(self) -> torch.Tensor:
        """Return the most recently executed local expert histogram."""
        if self._last_route_counts is None:
            raise RuntimeError("route counts are unavailable before the first forward pass.")
        return self._last_route_counts


class QwenDecoderLayer(nn.Module):
    """One residual attention and MoE decoder layer."""

    def __init__(
        self,
        config: QwenMoeConfig,
        ep_group: Any | None,
        *,
        layer_index: int = 0,
    ) -> None:
        """Initialize one attention and MoE decoder layer."""
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMoeBlock(
            config,
            ep_group,
            layer_index=layer_index,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run residual attention followed by residual MoE."""
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class QwenMoeModel(nn.Module):
    """Random-weight text-only Qwen MoE language model."""

    def __init__(
        self,
        config: QwenMoeConfig,
        *,
        ep_group: Any | None = None,
    ) -> None:
        """Initialize all decoder layers and their shared execution group."""
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            QwenDecoderLayer(
                config,
                ep_group,
                layer_index=layer_index,
            )
            for layer_index in range(config.num_layers)
        )
        MegaMoeExperts.share_execution_resources(
            layer.mlp.experts for layer in self.layers
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return token logits and optional next-token loss."""
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(
            hidden_states.shape[1],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            ignored_label = torch.full(
                (*labels.shape[:-1], 1),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            shift_labels = torch.cat((labels, ignored_label), dim=-1)[..., 1:]
            loss = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}

    def expert_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return local routed-expert parameters excluded from dense all-reduce."""
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.mlp.experts.parameters())
        return tuple(parameters)

    def close(self) -> None:
        """Release the shared MegaMoe runtime after all queued work finishes."""
        for layer in self.layers:
            layer.mlp.experts.close()
