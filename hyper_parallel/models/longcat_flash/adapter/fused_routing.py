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
"""Compact real routes and fuse ordered local identity arithmetic on Ascend."""

from typing import Any

# Native tensor APIs are the model adapter contract.
# pylint: disable=forbidden-backend-import
import torch
import torch.distributed as dist

from hyper_parallel.models.longcat_flash.adapter.route_kernels import OrderedCombine, OrderedGather


def _validate_fused_inputs(module: Any, hidden: torch.Tensor, probabilities: torch.Tensor) -> None:
    """Reject unsupported casts or ID ranges before selecting optional device kernels."""
    if hidden.device.type != "npu" or hidden.dtype != torch.bfloat16 or probabilities.dtype != torch.float32:
        raise ValueError("LongCat Triton routing requires NPU BF16 states and FP32 probabilities")
    if module.total_experts > 2**24:
        raise ValueError("LongCat Triton routing requires exactly float32-representable expert IDs")


def fused_expert_forward(module: Any, hidden: torch.Tensor, ids: torch.Tensor,
                         probabilities: torch.Tensor) -> torch.Tensor:
    """Execute the same real/identity split with explicitly ordered BF16 kernels.

    Args:
        module: Parameter-owning LongCat expert adapter.
        hidden: Contiguous token states [tokens, hidden_size].
        ids: Original global Top-K expert IDs.
        probabilities: Differentiable FP32 router probabilities.

    Returns:
        Combined BF16 expert output in source token order.
    """
    _validate_fused_inputs(module, hidden, probabilities)
    hidden = hidden.contiguous()
    # Every allowed ID is exactly representable, avoiding Ascend's int64 AICPU sort.
    order = ids.float().argsort(dim=-1)
    ids, probabilities = ids.gather(1, order), probabilities.gather(1, order)
    token, slot = (ids < module.num_experts).nonzero(as_tuple=True)
    token, slot = token.contiguous(), slot.contiguous()
    invalid = ((ids < 0) | (ids >= module.total_experts)).any().to(torch.int64)
    agreement = torch.stack((invalid.new_tensor(token.numel()), invalid))
    if module._ep_group is not None:  # pylint: disable=protected-access
        dist.all_reduce(agreement, group=module._ep_group)  # pylint: disable=protected-access
    total, error = agreement.tolist()
    if error:
        raise ValueError("LongCat route contains an invalid expert ID")
    inverse = torch.full(ids.shape, -1, device=ids.device, dtype=torch.int32)
    if token.numel():
        positions = token * ids.shape[1] + slot
        inverse.view(-1).index_copy_(0, positions, torch.arange(token.numel(), device=ids.device, dtype=torch.int32))
    states, identity = OrderedGather.apply(hidden, token, inverse)
    if not total:
        unweighted = states
    else:
        if not module._configured:  # pylint: disable=protected-access
            if module.ep_size != 1:
                raise RuntimeError("Configure LongCat EP experts before execution")
            module.configure()
        weights = module._real_weights()  # pylint: disable=protected-access
        inputs = (states, ids[token, slot].reshape(-1, 1).contiguous(),
                  probabilities.new_ones((token.numel(), 1)))
        if module.backend == "megamoe":
            unweighted = module._executor(*inputs, expert_weights=weights)  # pylint: disable=protected-access
        else:
            unweighted = module._native_forward(*inputs, expert_weights=weights)  # pylint: disable=protected-access
    return OrderedCombine.apply(identity, unweighted, probabilities, inverse, token, slot)
