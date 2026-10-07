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
"""Explicit single-card enhance reference boundary for P0 backend validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch
from torch.nn import functional

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)


def _load_custom_ops() -> Any:
    # Omni registration is optional and must not be loaded by CPU metadata/oracle imports.
    import_module("omni_training_custom_ops")
    return torch.ops.custom


@dataclass(frozen=True)
class CannDsaStats:
    """Opaque enhance softmax max/sum in native [1,T,H] layout."""

    maximum: torch.Tensor
    denominator: torch.Tensor

    @property
    def lse(self) -> torch.Tensor:
        """Convert native statistics to natural-log FP32 LSE for validation."""
        return self.maximum.float() + self.denominator.float().log()


class CannDsaLayout:
    """Prepare full CP=1 packed TND metadata and sequence-local CANN indices.

    The locked SFA kernel adds each sequence's storage base to sparse indices,
    whereas the core contract uses global packed IDs. Construction is outside
    the hot path and supports CPU for contract tests. It rejects partial,
    reordered or CP-sharded Q/K: right-down masking cannot express their global
    query offsets. Backend calls remain NPU-only.
    """

    def __init__(self, batch_meta: DsaBatchMeta, device: torch.device | str) -> None:
        """Bind complete logical metadata and preallocate backend length/index offsets."""
        expected = tuple(range(batch_meta.global_valid_queries))
        if len(batch_meta.cp_ranks) != 1 or batch_meta.q_global_ids != expected or batch_meta.kv_global_ids != expected:
            raise ValueError("CANN reference requires complete, ordered CP=1 packed Q/K")
        if batch_meta.global_valid_queries > torch.iinfo(torch.int32).max:
            raise ValueError("CANN int32 lengths cannot address this packed batch")
        self.batch_meta = batch_meta
        self.device = torch.device(device)
        self.cumulative_lengths = batch_meta.global_cu_seqlens[1:]
        self.length_tensor = torch.tensor(self.cumulative_lengths, dtype=torch.int32, device=self.device)
        starts = [batch_meta.global_cu_seqlens[batch_meta.sequence_position(token)[0]] for token in expected]
        self.sequence_starts = torch.tensor(starts, dtype=torch.int32, device=self.device)[:, None]
        self.query_ids = torch.arange(len(expected), dtype=torch.int32, device=self.device)[:, None]

    def _validate_indices(self, indices: torch.Tensor, native: bool = False) -> None:
        expected_dims = 3 if native else 2
        if indices.dtype != torch.int32 or indices.device != self.device or indices.ndim != expected_dims:
            raise ValueError("indices must be int32 on the prepared layout device with the declared rank")
        if indices.shape[0] != self.batch_meta.global_valid_queries or indices.shape[-1] <= 0:
            raise ValueError("indices must cover all queries and have a positive K")
        if native and indices.shape[1] != 1:
            raise ValueError("the absorbed MQA backend requires Nkv=1 indices")

    def global_to_sequence_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Filter and compact global [T,K] into sequence-local int32 [T,1,K].

        CANN terminates reads at the first -1, so masked slots must follow all
        legal entries. Stable compaction retains selected-slot order. Padding,
        cross-sequence and future/unknown IDs are masked without a host sync.
        Final Top-K selections must be unique; this conversion does not dedupe.
        """
        self._validate_indices(indices)
        valid = (indices >= self.sequence_starts) & (indices <= self.query_ids)
        local = torch.where(valid, indices - self.sequence_starts, -1)
        order = valid.to(torch.int32).argsort(dim=-1, descending=True, stable=True)
        return local.gather(-1, order)[:, None, :].contiguous()

    def sequence_to_global_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert native [T,1,K] sequence-local results to global packed [T,K]."""
        self._validate_indices(indices, native=True)
        local = indices[:, 0, :]
        valid = (local >= 0) & (local <= self.query_ids - self.sequence_starts)
        return torch.where(valid, local + self.sequence_starts, -1).contiguous()


class _SelectedKlFunction(torch.autograd.Function):
    """Save forward-computed indexer gradients once per invocation."""

    @staticmethod
    def forward(
        ctx: Any, index_query: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
        main_inputs: tuple[torch.Tensor, ...], indices: torch.Tensor, stats: CannDsaStats,
        layout: CannDsaLayout, attention_scale: float, loss_scale: float,
    ) -> torch.Tensor:
        """Compute selected KL and save its three indexer derivatives once."""
        query, compressed, query_rope, key_rope = main_inputs
        grad_query, grad_key, grad_weight, loss = (
            _load_custom_ops().npu_sparse_lightning_indexer_grad_kl_loss_enhance(
                query=query.detach(), key=compressed.detach()[:, None, :],
                query_index=index_query, key_index=index_key[:, None, :], weights=merge_weight,
                sparse_indices=indices, softmax_max=stats.maximum.detach(), softmax_sum=stats.denominator.detach(),
                scale_value=attention_scale, query_rope=query_rope.detach(), key_rope=key_rope.detach()[:, None, :],
                actual_seq_qlen=list(layout.cumulative_lengths), actual_seq_klen=list(layout.cumulative_lengths),
                layout="TND", sparse_mode=3, sparse_block_size=1,
                deterministic=torch.are_deterministic_algorithms_enabled(),
            )
        )
        if grad_query.shape != index_query.shape or grad_key.shape != (index_key.shape[0], 1, index_key.shape[1]):
            raise RuntimeError("enhance KL returned incompatible index Q/K gradient shapes")
        if grad_weight.shape != merge_weight.shape or loss.numel() != 1:
            raise RuntimeError("enhance KL returned incompatible weight gradient or loss shape")
        ctx.save_for_backward(grad_query * loss_scale, grad_key[:, 0, :] * loss_scale, grad_weight * loss_scale)
        return loss.reshape(()) * loss_scale

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> tuple:
        """Apply the upstream auxiliary scale to this invocation's saved derivatives."""
        return (*(gradient * grad_loss for gradient in ctx.saved_tensors), *((None,) * 6))


class CannDsaReference:
    """P0 host-dispatched enhance path with canonical core tensor layouts.

    Initial support is BF16, C=512, Dr=64, K=2048, main H=32/64/128,
    index H=8/16/32/64 and Di=128. This is an explicit development reference,
    never selected implicitly as a runtime fallback. Model projections,
    projection-input detach and aux_loss_auto_scale remain outside this class.
    """

    def __init__(self, layout: CannDsaLayout, *, attention_scale: float) -> None:
        """Bind the prepared packed layout and the original model's attention scale."""
        if not math.isfinite(attention_scale) or attention_scale <= 0:
            raise ValueError("attention_scale must be the finite positive model scale")
        self.layout = layout
        self.attention_scale = attention_scale

    def _require_npu(self) -> None:
        if self.layout.device.type != "npu":
            raise ValueError("CANN backend calls require an NPU layout")

    def _validate_tensors(self, tensors: tuple[torch.Tensor, ...]) -> None:
        self._require_npu()
        if any(tensor.dtype != torch.bfloat16 or tensor.device != self.layout.device for tensor in tensors):
            raise ValueError("CANN reference inputs must be BF16 on the prepared NPU device")

    def _validate_main(self, tensors: tuple[torch.Tensor, ...]) -> None:
        self._validate_tensors(tensors)
        query, compressed, query_rope, key_rope = tensors
        total = self.layout.batch_meta.global_valid_queries
        if query.ndim != 3 or query.shape[0] != total or query.shape[1] not in (32, 64, 128):
            raise ValueError("CANN reference query must be [T,32/64/128,512]")
        if query.shape[-1] != 512 or compressed.shape != (total, 512):
            raise ValueError("CANN reference absorbed query and compressed KV require C=512")
        if query_rope.shape != (*query.shape[:2], 64) or key_rope.shape != (total, 64):
            raise ValueError("CANN reference RoPE states require Dr=64")

    def _validate_index(self, tensors: tuple[torch.Tensor, ...]) -> None:
        self._validate_tensors(tensors)
        query, key, weight = tensors
        total = self.layout.batch_meta.global_valid_queries
        if query.ndim != 3 or query.shape[0] != total or query.shape[1] not in (8, 16, 32, 64):
            raise ValueError("CANN reference index query must be [T,8/16/32/64,128]")
        if query.shape[-1] != 128 or key.shape != (total, 128) or weight.shape != query.shape[:2]:
            raise ValueError("CANN reference index key/weights must match the fixed Di=128 layout")

    def _native_indices(self, indices: torch.Tensor) -> torch.Tensor:
        native = self.layout.global_to_sequence_indices(indices)
        if indices.shape[-1] != 2048:
            raise ValueError("the initial CANN reference supports K=2048 only")
        return native

    def indexer(
        self, index_query: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Return global packed int32 Top-K, preserving already-scaled signed weights."""
        self._validate_index((index_query, index_key, merge_weight))
        with torch.no_grad():
            indices, _ = _load_custom_ops().npu_lightning_indexer_enhance(
                index_query.contiguous(), index_key[:, None, :].contiguous(), merge_weight.contiguous(),
                actual_seq_lengths_query=self.layout.length_tensor,
                actual_seq_lengths_key=self.layout.length_tensor, block_table=None,
                layout_query="TND", layout_key="TND", sparse_count=2048, sparse_mode=3, return_value=False,
            )
        return self.layout.sequence_to_global_indices(indices)

    def attention(
        self, query: torch.Tensor, compressed_kv: torch.Tensor, query_rope: torch.Tensor,
        key_rope: torch.Tensor, topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, CannDsaStats]:
        """Return compressed output and native statistics through enhance autograd.

        The same compressed tensor is passed as key and value. The registered
        enhance backward owns their separate gradients; Torch adds both once.
        No second sparse backward bridge is installed by this adapter.
        """
        self._validate_main((query, compressed_kv, query_rope, key_rope))
        indices = self._native_indices(topk_indices)
        key = compressed_kv[:, None, :].contiguous()
        output, maximum, denominator = _load_custom_ops().npu_sparse_flash_attention_enhance(
            query.contiguous(), key, key, indices, self.attention_scale, block_table=None,
            actual_seq_lengths_query=self.layout.length_tensor, actual_seq_lengths_kv=self.layout.length_tensor,
            query_rope=query_rope.contiguous(), key_rope=key_rope[:, None, :].contiguous(),
            sparse_block_size=1, layout_query="TND", layout_kv="TND", sparse_mode=3,
            attention_mode=2, return_softmax_lse=True,
        )
        stats_shape = (1, query.shape[0], query.shape[1])
        if output.shape != query.shape or maximum.shape != stats_shape or denominator.shape != stats_shape:
            raise RuntimeError("enhance attention returned incompatible output/statistics layouts")
        return output, CannDsaStats(maximum.detach(), denominator.detach())

    def kl_loss(
        self, index_query: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
        main_inputs: tuple[torch.Tensor, ...], topk_indices: torch.Tensor, stats: CannDsaStats,
        *, normalization: DsaLossNormalization, loss_coeff: float = 1.0,
    ) -> torch.Tensor:
        """Compute one normalized selected-set KL with forward-saved index gradients."""
        self._validate_index((index_query, index_key, merge_weight))
        self._validate_main(main_inputs)
        if normalization.global_valid_queries != self.layout.batch_meta.global_valid_queries:
            raise ValueError("KL normalization must use the complete packed query count")
        if not math.isfinite(loss_coeff) or loss_coeff < 0:
            raise ValueError("loss_coeff must be finite and nonnegative")
        expected = (1, main_inputs[0].shape[0], main_inputs[0].shape[1])
        if stats.maximum.shape != expected or stats.denominator.shape != expected:
            raise ValueError("KL requires native [1,T,H] attention statistics")
        if any(tensor.dtype != torch.float32 or tensor.device != self.layout.device
               for tensor in (stats.maximum, stats.denominator)):
            raise ValueError("KL statistics must be native FP32 tensors on the layout device")
        return _SelectedKlFunction.apply(
            index_query.contiguous(), index_key.contiguous(), merge_weight.contiguous(),
            main_inputs, self._native_indices(topk_indices), stats, self.layout, self.attention_scale,
            loss_coeff * normalization.local_sum_scale,
        )

    def attention_bsnd(
        self, query: torch.Tensor, compressed_kv: torch.Tensor, query_rope: torch.Tensor,
        key_rope: torch.Tensor, topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, CannDsaStats]:
        """Adapt the existing model's BSND tensors and restore RoPE output padding.

        Q/RoPE have H heads, KV/RoPE have one head. The externally supplied
        indices retain the core global [T,K] namespace. Projections and model
        parameters remain owned by the existing attention module.
        """
        if any(tensor.ndim != 4 for tensor in (query, compressed_kv, query_rope, key_rope)):
            raise ValueError("model boundary tensors must use BSND layout")
        if query.shape[:2] != compressed_kv.shape[:2] or query.shape[:2] != query_rope.shape[:2]:
            raise ValueError("model boundary requires matching batch and token dimensions")
        if key_rope.shape[:2] != query.shape[:2] or compressed_kv.shape[2] != 1 or key_rope.shape[2] != 1:
            raise ValueError("model boundary KV and key RoPE must use one MQA head")
        output, stats = self.attention(
            query.flatten(0, 1), compressed_kv.flatten(0, 1)[:, 0, :],
            query_rope.flatten(0, 1), key_rope.flatten(0, 1)[:, 0, :], topk_indices,
        )
        return functional.pad(output, (0, 64)).reshape(*query.shape[:3], 576), stats
