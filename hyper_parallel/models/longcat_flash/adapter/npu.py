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
"""Native NPU EP experts using permute, AllToAll, grouped GEMM and SwiGLU."""

from typing import Any

# Native tensor APIs are the model adapter contract; torch_npu is loaded only on NPU selection.
# pylint: disable=forbidden-backend-import
import torch
import torch.distributed as dist
import torch_npu
from torch.nn import functional as F

from hyper_parallel.components.functional.grouped_matmul import grouped_matmul
from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all


def _permute(hidden: torch.Tensor, ids: torch.Tensor) -> tuple:
    """Retain an autograd-connected empty input when no source rows exist."""
    if hidden.shape[0] == 0:
        return hidden, ids.new_empty((0,), dtype=torch.int32)
    return torch_npu.npu_moe_token_permute(hidden, ids.to(torch.int32).contiguous())


def _unpermute(hidden: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Restore K=1 rows without manufacturing probabilities or doing a reduction."""
    if mapping.numel() == 0:
        return hidden[:0]
    return torch_npu.npu_moe_token_unpermute(hidden, mapping)


def _exchange_counts(counts: torch.Tensor, group: Any, ep_size: int) -> tuple:
    """Exchange the small expert histogram instead of a route ID for every row."""
    received = torch.empty_like(counts)
    if ep_size > 1:
        dist.all_to_all_single(received, counts.contiguous(), group=group)
    else:
        received.copy_(counts)
    local_experts = counts.numel() // ep_size
    segments = torch.stack((counts.view(ep_size, local_experts).sum(1),
                            received.view(ep_size, local_experts).sum(1))).tolist()
    return received, segments[0], segments[1]


def _grouped_experts(hidden: torch.Tensor, weights: tuple, counts: torch.Tensor,
                     activation: str) -> torch.Tensor:
    """Run two grouped GEMMs with either source-exact casts or native SwiGLU."""
    gate, down = weights
    if hidden.shape[0] == 0:
        return hidden + (gate.reshape(-1)[:1].sum() + down.reshape(-1)[:1].sum()) * 0
    group_list = counts.to(torch.int64).cumsum(0)
    projected = grouped_matmul(hidden, gate.transpose(1, 2), group_list=group_list)
    if activation == "fused":
        intermediate = torch_npu.npu_swiglu(projected)
    elif activation == "hf":
        gate_state, up_state = projected.chunk(2, -1)
        intermediate = F.silu(gate_state) * up_state
    else:
        raise ValueError("LongCat NPU activation must be fused or hf")
    return grouped_matmul(intermediate, down.transpose(1, 2), group_list=group_list)


def npu_expert_forward(hidden: torch.Tensor, ids: torch.Tensor, weights: tuple,
                       group: Any, ep_size: int, num_experts: int, layout: str,
                       activation: str = "fused") -> torch.Tensor:
    """Execute real K=1 assignments with native NPU operators and differentiable EP.

    Args:
        hidden: Compact real-assignment matrix [R, H].
        ids: Global real-expert IDs [R, 1].
        weights: Current unsharded local Gate/Up and Down parameters.
        group: Explicit expert process group.
        ep_size: Expert group size.
        num_experts: Global real-expert count.
        layout: EOI or EIO parameter layout.
        activation: Native fused SwiGLU or HF's separate BF16 activation casts.

    Returns:
        Unweighted expert outputs in the original assignment order.
    """
    if hidden.device.type != "npu":
        raise ValueError("native NPU experts require NPU inputs")
    if layout == "EIO":
        weights = tuple(weight.transpose(1, 2) for weight in weights)
    counts = torch.bincount(ids.reshape(-1).long(), minlength=num_experts)
    received_counts, send_sizes, receive_sizes = _exchange_counts(counts, group, ep_size)
    source, source_mapping = _permute(hidden, ids)
    received = ep_all_to_all(source, send_sizes, receive_sizes, group) if ep_size > 1 else source
    local_experts = num_experts // ep_size
    local_ids = torch.arange(local_experts, device=hidden.device).repeat(ep_size)
    local_ids = local_ids.repeat_interleave(received_counts).reshape(-1, 1)
    expert_input, owner_mapping = _permute(received, local_ids)
    local_counts = received_counts.view(ep_size, local_experts).sum(0)
    expert_output = _grouped_experts(expert_input, weights, local_counts, activation)
    owner_output = _unpermute(expert_output, owner_mapping)
    returned = ep_all_to_all(owner_output, receive_sizes, send_sizes, group) if ep_size > 1 else owner_output
    return _unpermute(returned, source_mapping)
