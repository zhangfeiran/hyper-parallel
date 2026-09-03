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
"""Top-K route preparation for external and fused MegaMoe endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._framework import get_torch, get_torch_distributed
from .config import MegaMoeConfig
from .runtime import MegaMoeRuntime

if TYPE_CHECKING:
    from .workspace import MegaMoeWorkspace


@dataclass(frozen=True)
class RouteMetadata:
    """Exact row offsets and counts consumed by the low-level ABI."""

    dispatch_src_off: Any
    dispatch_target_off: Any
    dispatch_size: Any
    combine_src_off: Any
    combine_target_off: Any
    combine_size: Any
    group_list: Any
    expert_capacity: int


@dataclass(frozen=True)
class PreparedTopKRoute:
    """Permuted payload, count metadata and the capacity decision for one route."""

    routed_tokens: Any | None
    unpermute_mapping: Any
    permute_src_index: Any | None
    expert_route_weights: Any | None
    tokens_per_expert: Any
    received_counts: Any
    receive_overflow: bool
    metadata: RouteMetadata | None


def _validate_topk_inputs(
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
) -> Any:
    """Validate route tensors and return flattened expert IDs."""
    torch = get_torch()

    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_ids must be int32 or int64, got {topk_ids.dtype}.")
    if not topk_weights.is_floating_point():
        raise TypeError(
            f"topk_weights must be floating point, got {topk_weights.dtype}."
        )
    if (
        topk_ids.device != hidden_states.device
        or topk_weights.device != hidden_states.device
    ):
        raise ValueError(
            "hidden_states, topk_ids and topk_weights must be on one device."
        )
    return topk_ids.reshape(-1)


def _resolve_counts(
    flat_ids: Any, supplied_counts: Any | None, config: MegaMoeConfig
) -> Any:
    """Use a trusted supplied histogram or compute it when omitted."""
    torch = get_torch()

    if supplied_counts is None:
        return (
            torch.bincount(flat_ids, minlength=config.num_experts)
            .to(torch.int32)
            .contiguous()
        )
    if tuple(supplied_counts.shape) != (config.num_experts,):
        raise ValueError(
            f"tokens_per_expert must have shape ({config.num_experts},), "
            f"got {tuple(supplied_counts.shape)}."
        )
    if supplied_counts.device != flat_ids.device:
        raise ValueError("tokens_per_expert must be on the route tensor device.")
    if supplied_counts.dtype not in (torch.int32, torch.int64):
        raise TypeError("tokens_per_expert must be int32 or int64.")
    return supplied_counts.to(torch.int32).contiguous()


def _start_count_gather(counts: Any, config: MegaMoeConfig) -> tuple[Any, Any | None]:
    """Start the rank-major INT32 count gather used by route construction."""
    torch = get_torch()
    dist = get_torch_distributed()

    if config.ep_size == 1:
        return counts.reshape(1, config.num_experts), None
    gathered_counts = torch.empty(
        config.ep_size * config.num_experts,
        dtype=torch.int32,
        device=counts.device,
    )
    work = dist.all_gather_into_tensor(
        gathered_counts,
        counts,
        group=config.ep_group,
        async_op=True,
    )
    return gathered_counts.reshape(config.ep_size, config.num_experts), work


def _finish_count_gather(
    counts_by_source: Any,
    work: Any | None,
    config: MegaMoeConfig,
) -> tuple[Any, int]:
    """Wait for gathered counts and return this rank's receive matrix and global maximum."""
    torch = get_torch()

    if work is not None:
        work.wait()
    local_start = config.rank_id * config.local_experts
    local_end = local_start + config.local_experts
    received_counts = counts_by_source[:, local_start:local_end].contiguous()
    destination_loads = counts_by_source.reshape(
        config.ep_size,
        config.ep_size,
        config.local_experts,
    ).sum(dim=(0, 2), dtype=torch.int32)
    return received_counts, int(destination_loads.max().item())


def _permute_topk_input(hidden_states: Any, topk_ids: Any) -> tuple[Any, Any]:
    """Expand hidden rows with the native NPU MoE permutation operator."""
    # pylint: disable=C0415
    import torch_npu

    torch = get_torch()
    routed_tokens, unpermute_mapping = torch_npu.npu_moe_token_permute(
        hidden_states,
        topk_ids,
    )
    return (
        routed_tokens,
        unpermute_mapping.reshape(-1).to(torch.int32).contiguous(),
    )


def _build_route_indices(
    token_rows: Any,
    topk_ids: Any,
) -> tuple[Any, Any]:
    """Build expert-major source rows and the inverse token route map."""
    # pylint: disable=C0415
    import torch_npu

    torch = get_torch()

    permuted_rows, inverse_order = torch_npu.npu_moe_token_permute(
        token_rows,
        topk_ids,
    )
    permute_src_index = permuted_rows.reshape(-1).to(torch.int32).contiguous()
    inverse_order = inverse_order.reshape(-1).to(torch.int32).contiguous()
    return permute_src_index, inverse_order.reshape_as(topk_ids).contiguous()


def _build_expert_route_weights(
    inverse_order: Any,
    topk_weights: Any,
) -> Any:
    """Scatter token-major weights into expert-major dispatch order."""
    torch = get_torch()

    expert_route_weights = torch.empty(
        topk_weights.numel(),
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )
    expert_route_weights.scatter_(
        0,
        inverse_order.reshape(-1),
        topk_weights.detach().reshape(-1),
    )
    return expert_route_weights


def _compute_route_metadata(
    counts: Any,
    counts_by_source: Any,
    received_counts: Any,
    runtime: MegaMoeRuntime,
    maximum_received_slots: int,
) -> RouteMetadata:
    """Build exact row offsets without padding or route truncation."""
    torch = get_torch()

    config = runtime.config
    counts_i32 = counts.to(dtype=torch.int32).contiguous()
    split = max(config.dispatch_split, config.combine_split)
    rounded_capacity = (maximum_received_slots + split - 1) // split * split
    expert_capacity = max(config.routed_slots, rounded_capacity)

    source_prefix = counts_by_source.cumsum(dim=1, dtype=torch.int32) - counts_by_source
    destination_layout = (
        counts_by_source.reshape(
            config.ep_size,
            config.ep_size,
            config.local_experts,
        )
        .permute(1, 2, 0)
        .contiguous()
    )
    destination_flat = destination_layout.reshape(config.ep_size, -1)
    destination_prefix = (
        destination_flat.cumsum(dim=1, dtype=torch.int32) - destination_flat
    ).reshape(config.ep_size, config.local_experts, config.ep_size)
    local_destination_prefix = destination_prefix[config.rank_id]
    local_destination_counts = destination_layout[config.rank_id]
    local_start = config.rank_id * config.local_experts
    local_end = local_start + config.local_experts
    return RouteMetadata(
        dispatch_src_off=source_prefix[config.rank_id].to(torch.int64),
        dispatch_target_off=(
            destination_prefix[:, :, config.rank_id].reshape(-1).to(torch.int64)
        ),
        dispatch_size=counts_i32,
        combine_src_off=local_destination_prefix.T.reshape(-1).to(torch.int64),
        combine_target_off=(
            source_prefix[:, local_start:local_end].reshape(-1).to(torch.int64)
        ),
        combine_size=received_counts.reshape(-1),
        group_list=(
            local_destination_prefix[:, -1] + local_destination_counts[:, -1]
        ).to(torch.int64),
        expert_capacity=expert_capacity,
    )


def prepare_topk_route(
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
    runtime: MegaMoeRuntime,
    tokens_per_expert: Any | None,
    workspace: MegaMoeWorkspace | None = None,
) -> PreparedTopKRoute:
    """Overlap native TopK permutation with count gathering and build route metadata."""
    config = runtime.config
    flat_ids = _validate_topk_inputs(hidden_states, topk_ids, topk_weights)
    counts = _resolve_counts(flat_ids, tokens_per_expert, config)
    if workspace is not None:
        # Waiting before the mandatory count exchange turns its completion into
        # the cross-rank rendezvous for the next symmetric-buffer generation.
        workspace.wait_for_reuse()
    counts_by_source, count_work = _start_count_gather(counts, config)
    if config.forward_permute_fused:
        token_rows = runtime.route_token_rows
        if token_rows is None:
            raise RuntimeError("route-fused MegaMoe is missing token-row metadata.")
        permute_src_index, unpermute_mapping = _build_route_indices(
            token_rows,
            topk_ids,
        )
        expert_route_weights = (
            _build_expert_route_weights(unpermute_mapping, topk_weights)
            if config.backward_permute_fused
            else None
        )
        routed_tokens = None
    else:
        routed_tokens, unpermute_mapping = _permute_topk_input(
            hidden_states,
            topk_ids,
        )
        permute_src_index = None
        expert_route_weights = None
    received_counts, maximum_received_slots = _finish_count_gather(
        counts_by_source,
        count_work,
        config,
    )
    receive_overflow = maximum_received_slots > config.receive_capacity
    if receive_overflow:
        if routed_tokens is None:
            routed_tokens, unpermute_mapping = _permute_topk_input(
                hidden_states,
                topk_ids,
            )
        return PreparedTopKRoute(
            routed_tokens=routed_tokens,
            unpermute_mapping=unpermute_mapping,
            permute_src_index=permute_src_index,
            expert_route_weights=expert_route_weights,
            tokens_per_expert=counts,
            received_counts=received_counts,
            receive_overflow=True,
            metadata=None,
        )
    return PreparedTopKRoute(
        routed_tokens=routed_tokens,
        unpermute_mapping=unpermute_mapping,
        permute_src_index=permute_src_index,
        expert_route_weights=expert_route_weights,
        tokens_per_expert=counts,
        received_counts=received_counts,
        receive_overflow=False,
        metadata=_compute_route_metadata(
            counts,
            counts_by_source,
            received_counts,
            runtime,
            maximum_received_slots,
        ),
    )


def restore_topk_output(
    expert_output: Any,
    unpermute_mapping: Any,
    topk_weights: Any,
) -> Any:
    """Restore token order and route weights with the native NPU MoE operator."""
    # pylint: disable=C0415
    import torch_npu

    torch = get_torch()
    mapping = unpermute_mapping
    if mapping.ndim != 1:
        mapping = mapping.reshape(-1)
    if mapping.dtype != torch.int32:
        mapping = mapping.to(torch.int32)
    if not mapping.is_contiguous():
        mapping = mapping.contiguous()
    return torch_npu.npu_moe_token_unpermute(
        expert_output,
        mapping,
        probs=topk_weights.float().contiguous(),
    )
