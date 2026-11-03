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
"""Portable real-expert reference using framework differentiable EP collectives."""

from typing import Any

# Native tensor APIs are the project backend contract.
# pylint: disable=forbidden-backend-import
import torch
from torch.nn import functional as F

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all, ep_exchange_counts, ep_split_sizes


def native_expert_forward(hidden: torch.Tensor, ids: torch.Tensor, weights: tuple,
                          group: Any, ep_size: int, num_experts: int, layout: str) -> torch.Tensor:
    """Route K=1 assignments to their owner and return unweighted expert outputs.

    Args:
        hidden: Token states in original source order.
        ids: Selected global expert IDs.
        weights: Current differentiable routing or expert weights.
        group: Explicit expert process group.
        ep_size: Number of ranks in the expert group.
        num_experts: Global real-expert count.
        layout: Expert parameter axis order.
    """
    gate, down = weights
    if layout == "EIO":
        gate, down = gate.transpose(1, 2), down.transpose(1, 2)
    flat_ids = ids.flatten().long()
    order = flat_ids.float().argsort(stable=True)
    sorted_hidden, sorted_ids = hidden[order], flat_ids[order]
    local_experts = num_experts // ep_size
    send = torch.bincount(sorted_ids // local_experts, minlength=ep_size)
    if ep_size > 1:
        receive = ep_exchange_counts(send, group)
        send_sizes, receive_sizes = ep_split_sizes(send), ep_split_sizes(receive)
        sorted_hidden = ep_all_to_all(sorted_hidden, send_sizes, receive_sizes, group)
        sorted_ids = ep_all_to_all(sorted_ids[:, None], send_sizes, receive_sizes, group).flatten()
    output = sorted_hidden * 0 + (gate.reshape(-1)[:1].sum() + down.reshape(-1)[:1].sum()) * 0
    for expert in range(local_experts):
        positions = (sorted_ids % local_experts == expert).nonzero(as_tuple=True)[0]
        if positions.numel():
            state = sorted_hidden[positions]
            gate_state, up_state = F.linear(state, gate[expert]).chunk(2, dim=-1)  # pylint: disable=not-callable
            result = F.linear(F.silu(gate_state) * up_state, down[expert])  # pylint: disable=not-callable
            output = output.index_copy(0, positions, result)
    if ep_size > 1:
        output = ep_all_to_all(output, receive_sizes, send_sizes, group)
    return torch.empty_like(output).index_copy(0, order, output)
