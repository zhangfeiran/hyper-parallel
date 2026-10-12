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
"""Bounded device gather followed by explicit group-rank-order parameter SUM."""

from __future__ import annotations

import torch
import torch.distributed as dist


def ordered_parameter_sum(gradient: torch.Tensor, group: dist.ProcessGroup, *,
                          chunk_elements: int = 262144) -> torch.Tensor:
    """SUM detached parameter contributions in explicit group-rank order.

    This first-order caller-owned reducer performs no mean or leaf dtype cast.
    It provides an experimental control for collective reduction-order effects;
    its communication and performance must be measured before default adoption.

    Args:
        gradient: Dense detached FP32/FP64 local contribution, including empty
            tensors. Noncontiguous inputs are copied before communication.
        group: Explicit CP process group containing this caller. Every member
            must use the same shape, dtype, call order and chunk size.
        chunk_elements: Positive maximum number of elements per gathered chunk.
            Communication scratch is bounded by group size times this value.

    Returns:
        Fresh contiguous full-precision SUM in the input shape and device. The
        input is unchanged; a single-member group performs no communication.

    Raises:
        ValueError: The input is not a detached full-precision dense tensor,
            the chunk size is invalid, or the group is absent/nonparticipating.
    """
    if (gradient.layout != torch.strided or gradient.dtype not in (torch.float32, torch.float64)
            or gradient.requires_grad):
        raise ValueError('ordered parameter SUM requires detached dense FP32/FP64 contributions')
    if isinstance(chunk_elements, bool) or not isinstance(chunk_elements, int) or chunk_elements <= 0:
        raise ValueError('ordered parameter SUM requires a positive integer chunk size')
    if group is None or not dist.is_initialized():
        raise ValueError('ordered parameter SUM requires an explicit initialized process group')
    size = dist.get_world_size(group)
    if size not in (1, 2, 4, 8):
        raise ValueError('ordered parameter SUM requires a participating group of size 1, 2, 4 or 8')
    flat = gradient.contiguous().view(-1)
    result = torch.empty_like(flat)
    if size == 1 or flat.numel() == 0:
        result.copy_(flat)
        return result.view(gradient.shape)
    width = min(chunk_elements, flat.numel())
    scratch = torch.empty(size*width, dtype=gradient.dtype, device=gradient.device)
    for start in range(0, flat.numel(), chunk_elements):
        count = min(chunk_elements, flat.numel()-start)
        gathered = scratch[:size*count]
        handle = dist.all_gather_into_tensor(gathered, flat[start:start+count], group=group, async_op=True)
        handle.wait()
        rows = gathered.view(size, count)
        target = result[start:start+count]
        target.copy_(rows[0])
        for rank in range(1, size):
            target.add_(rows[rank])
    return result.view(gradient.shape)
