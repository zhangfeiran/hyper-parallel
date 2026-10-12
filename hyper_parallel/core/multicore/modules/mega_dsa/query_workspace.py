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
"""Owned FP32 query/RoPE VJPs from the locked arch22 ND gradient workspace."""

from __future__ import annotations

import math

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.mixed_attention import (
    local_sfa_grad_workspace_bytes,
)


@torch.no_grad()
def scaled_query_workspace_vjps(retained: torch.Tensor, *, query_tokens: int, key_tokens: int,
                                 heads: int, attention_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode producer-local ND dQ and apply the original post scale exactly once.

    Args:
        retained: Producer-owned contiguous uint8 workspace or an immutable
            prefix snapshot through the complete dQ plane. Producer completion
            must precede this call on the same stream or via an explicit event.
        query_tokens: Producer's packed local query count, including zero.
        key_tokens: Producer's complete packed key count.
        heads: Producer's local H32/H64 geometry.
        attention_scale: Original positive finite scale, converted to FP32 as
            in the locked native ND post implementation.

    Returns:
        Owned FP32 dQ [T,H,512] and dQ-RoPE [T,H,64], independent of scratch.
        The locked tiler stores ND [T,H,576] after per-logical-core scratch.
        Empty queries return empty owned tensors without interpreting the
        special empty-query workspace layout. No host synchronization occurs.

    Raises:
        ValueError: Geometry, scale, alignment or retained capacity is invalid.

    Note:
        This decoder is specific to arch22 gradient ABI 2/local CP ABI 1,
        TND C512/Dr64, layout3/input_format0. It must not decode a generic
        workspace or a different tiler. It installs no autograd edge.
    """
    local_sfa_grad_workspace_bytes(query_tokens, key_tokens, heads)
    if (isinstance(attention_scale, bool) or not isinstance(attention_scale, (int, float))
            or not math.isfinite(attention_scale) or attention_scale <= 2**-150
            or attention_scale > torch.finfo(torch.float32).max):
        raise ValueError('query workspace requires a finite positive attention scale')
    if (retained.dtype != torch.uint8 or retained.ndim != 1 or not retained.is_contiguous()
            or retained.storage_offset() % 4):
        raise ValueError('query workspace requires aligned contiguous uint8 storage')
    if query_tokens == 0:
        return tuple(torch.empty((0, heads, width), dtype=torch.float32, device=retained.device)
                     for width in (512, 64))
    scratch_per_core = 128*576*2*4+128*512*2*2+heads*128*4*2*4
    offset = min(query_tokens, 20)*scratch_per_core
    elements = query_tokens*heads*576
    if retained.numel() < offset+elements*4:
        raise ValueError('query workspace does not contain a complete ND dQ plane')
    with torch.autocast(device_type=retained.device.type, enabled=False):
        values = retained.narrow(0, offset, elements*4).view(torch.float32).view(query_tokens, heads, 576)
        scaled = values*float(attention_scale)
        return scaled[..., :512].contiguous(), scaled[..., 512:].contiguous()
