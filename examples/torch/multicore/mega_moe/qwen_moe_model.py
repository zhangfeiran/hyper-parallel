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
    "CompactLayoutStrategy",
    "QwenMoeBlock",
    "QwenMoeConfig",
    "QwenMoeModel",
    "RoutingMode",
]

from typing import Any

# This is a Torch-only executable example, not a platform-agnostic library module.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeMLP,
    Qwen3MoeRotaryEmbedding,
)

from examples.torch.multicore.mega_moe.qwen_moe_config import (
    CompactLayoutStrategy,
    QwenMoeConfig,
    RoutingMode,
)
from hyper_parallel.components.functional.rotary_embedding import apply_rotary_pos_emb
from hyper_parallel.components.modules.rms_norm import RMSNorm
from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.route import count_topk_experts
from hyper_parallel.models.qwen3_moe.adapter.attention import (
    run_qwen3_moe_flash_attention,
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
        self.rotary_emb = Qwen3MoeRotaryEmbedding(
            Qwen3MoeConfig.from_dict({
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "max_position_embeddings": config.max_seq_len,
                "rope_theta": config.rope_theta,
            })
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
        """Run fused causal grouped-query attention.

        Args:
            hidden_states: Token states shaped ``[batch, sequence, hidden]``.
            position_ids: Sequence positions used by rotary embeddings.

        Returns:
            Attention output with the same shape as ``hidden_states``.
        """
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
            replica_slots_per_rank=config.replica_slots_per_rank,
            compact_physical_weights=config.compact_physical_weights,
            shared_physical_weights=config.shared_physical_weights,
            compact_layout_strategy=config.compact_layout_strategy,
            ep_size=config.ep_size,
            ep_group=ep_group,
        )
        self.shared_expert = SwiGLUMLP(
            module=Qwen3MoeMLP(
                Qwen3MoeConfig.from_dict({
                    "hidden_size": config.hidden_size,
                    "intermediate_size": config.shared_expert_intermediate_size,
                })
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run routed experts and the always-active shared expert.

        Args:
            hidden_states: Token states shaped ``[..., hidden_size]``.

        Returns:
            Sum of routed- and shared-expert outputs in the input shape.
        """
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
            tokens_per_expert = count_topk_experts(topk_ids, self.config.num_experts)
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
        """Run residual attention followed by residual MoE.

        Args:
            hidden_states: Decoder input shaped ``[batch, sequence, hidden]``.
            position_ids: Sequence positions used by rotary embeddings.

        Returns:
            Decoder output with both residual updates applied.
        """
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
        """Return token logits and optional next-token loss.

        Args:
            input_ids: Token IDs shaped ``[batch, sequence]``.
            labels: Optional next-token training targets in the same shape.

        Returns:
            Token logits and an optional cross-entropy loss.
        """
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(
            hidden_states.shape[1],
            dtype=torch.int64,
            device=hidden_states.device,
        ).unsqueeze(0).expand(hidden_states.shape[0], -1)
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
