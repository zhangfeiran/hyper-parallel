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
"""Small real HP module fixtures and explicitly selected CPU validation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
    CannDsaSelection,
    CannDsaStats,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    indexer_reference,
    selected_kl_reference,
    sparse_attention_reference,
)


def build_model_fixture(*, dtype: torch.dtype = torch.float64, seed: int = 20261007) -> DeepseekV32DSAAttention:
    """Build the actual HP attention class with smaller random projection dimensions.

    Returns:
        A CPU module, H=32, Hi=8, C=512, Dr=64, Di=128, K=2048; hidden/Q-LoRA/
        Q-nope/value dimensions are 32/64/128/16. This is a randomly initialized
        source-protocol fixture, not a pretrained Transformers checkpoint.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        source = nn.Module()
        source.config = SimpleNamespace(
            num_attention_heads=32, q_lora_rank=64, kv_lora_rank=512,
            qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=16,
            index_head_dim=128, index_n_heads=8, index_topk=2048,
            dsa_loss_coeff=0.3, freeze_dsa=False, attention_dropout=0.0,
        )
        source.layer_idx = 0
        source.q_a_proj = nn.Linear(32, 64, bias=False)
        source.kv_a_proj_with_mqa = nn.Linear(32, 576, bias=False)
        source.q_a_layernorm = nn.RMSNorm(64, eps=1e-6)
        source.kv_a_layernorm = nn.RMSNorm(512, eps=1e-6)
        source.q_b_proj = nn.Linear(64, 32 * 192, bias=False)
        source.kv_b_proj = nn.Linear(512, 32 * 144, bias=False)
        source.o_proj = nn.Linear(32 * 16, 32, bias=False)
        source.indexer = nn.Module()
        source.indexer.wq_b = nn.Linear(64, 8 * 128, bias=False)
        source.indexer.wk = nn.Linear(32, 128, bias=False)
        source.indexer.k_norm = nn.LayerNorm(128)
        source.indexer.weights_proj = nn.Linear(32, 8, bias=False)
        model = DeepseekV32DSAAttention(module=source)
        with torch.no_grad():
            model.linear_qkv.weight.copy_(torch.cat((source.q_a_proj.weight, source.kv_a_proj_with_mqa.weight)))
        return model.to(dtype=dtype)


def model_inputs(
    meta: DsaBatchMeta, *, dtype: torch.dtype, seed: int = 20261008,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic CPU hidden states and packed-reset RoPE frequencies."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, meta.global_valid_queries, 32, generator=generator).to(dtype)
    positions = torch.tensor([meta.sequence_position(token)[1] for token in meta.q_global_ids], dtype=torch.float64)
    inverse = 10000 ** (-torch.arange(0, 64, 2, dtype=torch.float64) / 64)
    frequencies = positions[:, None] * inverse
    phases = torch.cat((frequencies, frequencies), dim=-1)[None]
    return hidden, (phases.cos().to(dtype), phases.sin().to(dtype))


def _rotary_mul_oracle(
    tensor: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, *, rotary_mode: str,
) -> torch.Tensor:
    """CPU-only replacement for the optional NPU RoPE primitive during validation."""
    if tensor.device.type != "cpu":
        raise ValueError("the mocked RoPE primitive is CPU-only")
    if rotary_mode == "half":
        first, second = tensor.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    elif rotary_mode == "interleave":
        rotated = torch.stack((-tensor[..., 1::2], tensor[..., 0::2]), dim=-1).flatten(-2)
    else:
        raise ValueError("unknown rotary mode")
    return tensor * cosine + rotated * sine


class _OracleDsaReference(CannDsaReference):
    """Explicit CPU test double, never constructed by the model boundary itself."""

    def __init__(self, meta: DsaBatchMeta, *, attention_scale: float, indices: torch.Tensor | None = None) -> None:
        """Bind CPU metadata and optionally fix the selection for an independent comparison."""
        super().__init__(CannDsaLayout(meta, "cpu"), attention_scale=attention_scale)
        self.fixed_selection = None if indices is None else self.prepare_selection(indices)
        self.last_loss = None
        self.last_selection = None

    def indexer(
        self, index_query: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
    ) -> CannDsaSelection:
        """Use an explicitly fixed selection or the CPU oracle's deterministic Top-K."""
        if self.fixed_selection is not None:
            selection = self.fixed_selection
        else:
            indices = indexer_reference(index_query, index_key, merge_weight, self.layout.batch_meta, sparse_count=2048)
            selection = self.prepare_selection(indices)
        self.last_selection = selection
        return selection

    def attention(
        self, query: torch.Tensor, compressed_kv: torch.Tensor, query_rope: torch.Tensor,
        key_rope: torch.Tensor, topk_indices: CannDsaSelection,
    ) -> tuple[torch.Tensor, CannDsaStats]:
        """Evaluate the compressed attention with ordinary CPU autograd."""
        output, stats = sparse_attention_reference(
            query, compressed_kv, query_rope, key_rope, topk_indices.to_global_indices(), self.layout.batch_meta,
            attention_scale=self.attention_scale,
        )
        return output, CannDsaStats(stats.maximum[None], stats.denominator[None])

    def kl_loss(
        self, index_query: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
        main_inputs: tuple[torch.Tensor, ...], topk_indices: CannDsaSelection, stats: CannDsaStats,
        *, normalization: DsaLossNormalization, loss_coeff: float = 1.0,
    ) -> torch.Tensor:
        """Keep the CPU selected teacher detached, recording the loss for KL-only tests."""
        del stats
        self.last_loss = selected_kl_reference(
            index_query, index_key, merge_weight, *main_inputs,
            topk_indices.to_global_indices(), self.layout.batch_meta,
            attention_scale=self.attention_scale, normalization=normalization, loss_coeff=loss_coeff,
        )
        return self.last_loss


def unabsorbed_model_reference(
    model: DeepseekV32DSAAttention, hidden: torch.Tensor, embeddings: tuple[torch.Tensor, torch.Tensor],
    meta: DsaBatchMeta, indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate full-history dense MLA with explicit per-head K/V up-projection.

    This independent main formula is for short CPU fixtures where K covers
    every legal causal key. The teacher/indexer use the same explicit selected
    set. Optional NPU RoPE must be explicitly mocked by the caller on CPU.
    """
    total, heads = meta.global_valid_queries, model.num_heads
    if meta.q_global_ids != tuple(range(total)) or meta.kv_global_ids != tuple(range(total)):
        raise ValueError("the unabsorbed model oracle requires complete ordered Q/K")
    for token, row in enumerate(indices.tolist()):
        start = meta.global_cu_seqlens[meta.sequence_position(token)[0]]
        if {key for key in row if key >= 0} != set(range(start, token + 1)):
            raise ValueError("the dense model oracle requires complete legal histories")
    q_resid, main, _ = model._prepare_attention_states(hidden, embeddings)
    query, compressed, qr, kr = main
    q_pass = model.q_b_proj(q_resid).reshape(total, heads, model.qk_head_dim)[..., :model.qk_nope_head_dim]
    expanded = compressed.reshape(total, 512) @ model.kv_b_proj.weight.T
    key, value = expanded.reshape(total, heads, -1).split((model.qk_nope_head_dim, model.v_head_dim), dim=-1)
    logits = torch.einsum("thd,shd->ths", q_pass, key)
    logits = logits + torch.einsum("thd,sd->ths", qr.reshape(total, heads, 64), kr.reshape(total, 64))
    tokens = torch.arange(total)
    starts = torch.tensor([meta.global_cu_seqlens[meta.sequence_position(token)[0]] for token in range(total)])
    valid = (tokens[None] >= starts[:, None]) & (tokens[None] <= tokens[:, None])
    probability = (logits * model.scaling).masked_fill(~valid[:, None], -torch.inf).softmax(-1)
    output = torch.einsum("ths,shv->thv", probability, value)
    output = model.o_proj(output.reshape(*hidden.shape[:2], heads * model.v_head_dim))
    if not (model.training and not model.freeze_dsa and model.dsa_loss_coeff):
        return output, None
    iq, ik, weight = model._project_index_states(hidden, q_resid, embeddings)
    loss = selected_kl_reference(
        iq.reshape(total, model.num_index_heads, model.index_head_dim), ik.reshape(total, model.index_head_dim),
        weight.reshape(total, model.num_index_heads), query.reshape(total, heads, 512), compressed.reshape(total, 512),
        qr.reshape(total, heads, 64), kr.reshape(total, 64), indices, meta, attention_scale=model.scaling,
        normalization=DsaLossNormalization(total), loss_coeff=model.dsa_loss_coeff,
    )
    return output, loss
