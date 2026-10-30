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
"""Torch EP-compatible row order without exchanging per-token route identities."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from .native_combine import _is_supported, _kernel

if TYPE_CHECKING:
    from .spec import MegaMoeSpec

_SORT_FP32_ENABLED = os.environ.get("HP_EP_SORT_FP32", "0") == "1"
_SORT_KEY_FP32_LIMIT = 2 ** 24


def _sort_keys(keys: torch.Tensor, bound: int) -> torch.Tensor:
    """Use the same integer-key sorting contract as Torch EP."""
    if _SORT_FP32_ENABLED and 0 < bound <= _SORT_KEY_FP32_LIMIT:
        return keys.to(torch.float32).argsort()
    return keys.argsort()


def permute_torch_input(
    hidden: torch.Tensor, ids: torch.Tensor, spec: MegaMoeSpec, output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather in Torch dispatch order, optionally directly into leased SHMEM."""
    expert_ids = ids.reshape(-1).long()
    destinations = expert_ids.div(spec.local_experts, rounding_mode="floor")
    keys = destinations * spec.num_experts + expert_ids
    if keys.device.type == "npu" and not _SORT_FP32_ENABLED:
        # Integer Sort on this backend already falls back to the host. Sorting
        # the small key vector directly preserves its unstable tie order while
        # avoiding the exposed AiCPU launch. Rebuild it for every dispatch.
        order = keys.cpu().argsort().to(device=keys.device)
    else:
        order = _sort_keys(keys, spec.ep_size * spec.num_experts)
    token_ids = order.div(spec.top_k, rounding_mode="floor")
    if output is None:
        routed = torch.index_select(hidden, 0, token_ids)
    else:
        routed = output[:order.numel()]
        torch.index_select(hidden, 0, token_ids, out=routed)
    return routed, order


def received_torch_order(counts: torch.Tensor, capacity: int) -> torch.Tensor:
    """Map native receive-sort rows to expert-major/source-major transport rows.

    Source dispatch already preserves native order within each expert. The
    remaining permutation depends only on the count matrix exchanged for SHMEM.
    """
    peers, experts = counts.shape
    flat_counts = counts.reshape(-1).long()
    raw_offsets = flat_counts.cumsum(0) - flat_counts
    expert_counts = counts.T.contiguous().reshape(-1).long()
    expert_offsets = (expert_counts.cumsum(0) - expert_counts).reshape(experts, peers).T.reshape(-1)
    segments = torch.arange(peers * experts, device=counts.device)
    # Expand the unchanged sort key and each segment's constant displacement together.
    metadata = torch.stack((segments.remainder(experts), expert_offsets - raw_offsets))
    expanded = torch.repeat_interleave(metadata, flat_counts, dim=1,
                                      output_size=capacity if capacity > 1 else None)
    order = _sort_keys(expanded[0], experts)
    return expanded[1][order] + order


def _restore_torch_output_with_state(
    expert_output: torch.Tensor, source_order: torch.Tensor, weights: torch.Tensor,
    *, inverse_order: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the literal combine and operands retained by its native autograd graph."""
    scores = weights.reshape(-1).to(expert_output.dtype)[source_order].unsqueeze(-1)
    token_ids = source_order.div(weights.shape[1], rounding_mode="floor")
    if _is_supported(expert_output, source_order, weights):
        result = _kernel(expert_output.shape[-1], weights.shape[1], expert_output.device.index)(
            expert_output, scores, source_order, inverse_order=inverse_order)
        return result, scores, token_ids
    weighted = expert_output * scores
    result = expert_output.new_zeros((weights.shape[0], expert_output.shape[-1]))
    result.index_add_(0, token_ids, weighted)
    return result, scores, token_ids


def restore_torch_output(
    expert_output: torch.Tensor, source_order: torch.Tensor, weights: torch.Tensor,
    *, inverse_order: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match native BF16 product rounding, accumulation order and its adjoint."""
    if _is_supported(expert_output, source_order, weights):
        return _NativeCombineAutograd.apply(expert_output, source_order, weights, inverse_order)
    return _restore_torch_output_with_state(expert_output, source_order, weights)[0]


def _restore_torch_output_grad_out(
    expert_output: torch.Tensor, source_order: torch.Tensor, weights: torch.Tensor,
    grad_output: torch.Tensor, out: torch.Tensor, scores: torch.Tensor, token_ids: torch.Tensor,
    *, need_weight_grad: bool,
) -> torch.Tensor | None:
    """Write the literal Torch combine adjoint directly into the leased source buffer."""
    if source_order.numel() == 0:
        return torch.zeros_like(weights) if need_weight_grad else None
    if _is_supported(expert_output, source_order, weights):
        product = _kernel(expert_output.shape[-1], weights.shape[1], expert_output.device.index).gradients(
            expert_output, scores, source_order, grad_output, out, need_weight_grad)
        if not need_weight_grad:
            return None
        grad_scores = product.sum_to_size(scores.shape).reshape(-1)
    else:
        grad_weighted = torch.index_select(grad_output, 0, token_ids)
        torch.mul(grad_weighted, scores, out=out)
        if not need_weight_grad:
            return None
        # Keep the BF16 product and sum-to-size boundary used by MulBackward0.
        grad_scores = (grad_weighted * expert_output).sum_to_size(scores.shape).reshape(-1)
    grad_weights = torch.zeros(weights.numel(), dtype=expert_output.dtype, device=weights.device)
    # IndexBackward0 accumulates into zero even for unique indices, including signed zero.
    grad_weights.index_put_((source_order,), grad_scores, accumulate=True)
    return grad_weights.reshape(weights.shape).to(weights.dtype)


class _NativeCombineAutograd(torch.autograd.Function):
    """Keep the literal adjoint when the push path uses the fused forward kernel."""

    @staticmethod
    def forward(ctx: Any, expert: torch.Tensor, source_order: torch.Tensor, weights: torch.Tensor,
                inverse_order: torch.Tensor | None = None) -> torch.Tensor:
        """Retain the same BF16 scores and source identities consumed by the native adjoint."""
        output, scores, token_ids = _restore_torch_output_with_state(
            expert, source_order, weights, inverse_order=inverse_order)
        ctx.save_for_backward(expert, source_order, weights, scores, token_ids)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, None, Any, None]:
        """Produce the source gradient and preserve the existing router reduction boundary."""
        expert, source_order, weights, scores, token_ids = ctx.saved_tensors
        grad_expert = torch.empty_like(expert)
        grad_weights = _restore_torch_output_grad_out(
            expert, source_order, weights, grad_output, grad_expert, scores, token_ids,
            need_weight_grad=ctx.needs_input_grad[2])
        return grad_expert if ctx.needs_input_grad[0] else None, None, grad_weights, None
