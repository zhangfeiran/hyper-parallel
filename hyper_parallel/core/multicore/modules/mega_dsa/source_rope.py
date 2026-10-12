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
"""Exact source RoPE values with FP32 first-order activation VJPs."""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel.components.modules.dsa_attention import _apply_attention_rope


def _transpose_rotation(gradient: torch.Tensor, *, interleaved: bool) -> torch.Tensor:
    if interleaved:
        return torch.stack((gradient[..., 1::2], -gradient[..., 0::2]), -1).flatten(-2)
    first, second = gradient.chunk(2, -1)
    return torch.cat((second, -first), -1)


class _SourceRope(torch.autograd.Function):
    """Use the source kernel for values and the transpose rotary map for derivatives."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor, interleaved: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Call the original BF16 kernel and represent its results in FP32."""
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(query, key, cos, sin)
        ctx.interleaved = interleaved
        with torch.autocast(device_type=query.device.type, enabled=False):
            values = _apply_attention_rope(query.bfloat16(), key.bfloat16(), (cos, sin), interleaved=interleaved)
        return tuple(value.float() for value in values)

    @staticmethod
    def backward(ctx: Any, grad_query: torch.Tensor | None, grad_key: torch.Tensor | None) -> tuple:
        """Apply the transpose rotary map in FP32 with original frequency values."""
        if torch.is_grad_enabled():
            raise ValueError('source RoPE supports first-order backward only')
        _query, _key, cos, sin = ctx.saved_tensors
        with torch.autocast(device_type=cos.device.type, enabled=False):
            cos, sin = cos.float(), sin.float()
            if cos.ndim == 2:
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            if ctx.interleaved:
                cos = cos[..., :cos.shape[-1]//2].repeat_interleave(2, -1)
                sin = sin[..., :sin.shape[-1]//2].repeat_interleave(2, -1)
            cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
            gradients = []
            for gradient in (grad_query, grad_key):
                if gradient is None:
                    gradients.append(None)
                    continue
                gradient = gradient.float()
                if ctx.interleaved:
                    first, second = gradient.chunk(2, -1)
                    gradient = torch.stack((first, second), -1).flatten(-2)
                gradients.append(gradient*cos+_transpose_rotation(gradient*sin, interleaved=ctx.interleaved))
        return *gradients, None, None, None


def source_attention_rope_with_fp32_vjp(
        query: torch.Tensor, key: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None, *,
        interleaved: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain the BF16 source forward while avoiding BF16 rotary activation gradients.

    Args:
        query: FP32 representation of source-rounded [B,T,H,Dr] queries.
        key: FP32 representation of source-rounded [B,T,1,Dr] keys.
        position_embeddings: Non-trainable source cosine/sine [T,Dr] or [B,T,Dr].
            Singleton batch frequencies may broadcast over the input batch.
        interleaved: Whether the source uses pair rotation and then half-layout output.

    Returns:
        Source BF16 values represented in FP32. With no embeddings, returns
        the original inputs. Only first-order input VJPs are supported.

    Raises:
        ValueError: Precision, geometry or trainable frequencies are unsupported.
    """
    if (not isinstance(interleaved, bool) or query.dtype != torch.float32 or key.dtype != torch.float32
            or query.ndim != 4 or key.ndim != 4 or query.shape[:2] != key.shape[:2]
            or query.shape[-1] != key.shape[-1] or query.shape[-1] <= 0 or query.shape[-1] % 2
            or query.device != key.device or query.layout != torch.strided or key.layout != torch.strided):
        raise ValueError('source RoPE requires dense FP32 Q/K with matching even rotary geometry')
    if position_embeddings is None:
        return query, key
    if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
        raise ValueError('source RoPE requires a cosine/sine pair')
    cos, sin = position_embeddings
    if (cos.shape != sin.shape or cos.ndim not in (2, 3) or cos.device != query.device or sin.device != query.device
            or cos.dtype != sin.dtype or cos.dtype not in (torch.bfloat16, torch.float32)
            or cos.requires_grad or sin.requires_grad or cos.layout != torch.strided or sin.layout != torch.strided
            or cos.shape[-2:] != (query.shape[1], query.shape[-1])
            or (cos.ndim == 3 and cos.shape[0] not in (1, query.shape[0]))):
        raise ValueError('source RoPE requires non-trainable matching source frequencies')
    return _SourceRope.apply(query, key, cos, sin, interleaved)
