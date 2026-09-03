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

from importlib import import_module

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch_npu

from .spec import MegaMoeSpec
from .plan import MegaMoePlan

torch = import_module("torch")
dist = import_module("torch.distributed")

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
    flat_ids: Any, supplied_counts: Any | None, spec: MegaMoeSpec
) -> Any:
    """Use a trusted supplied histogram or compute it when omitted."""

    if supplied_counts is None:
        return (
            torch.bincount(flat_ids, minlength=spec.num_experts)
            .to(torch.int32)
            .contiguous()
        )
    if tuple(supplied_counts.shape) != (spec.num_experts,):
        raise ValueError(
            f"tokens_per_expert must have shape ({spec.num_experts},), "
            f"got {tuple(supplied_counts.shape)}."
        )
    if supplied_counts.device != flat_ids.device:
        raise ValueError("tokens_per_expert must be on the route tensor device.")
    if supplied_counts.dtype not in (torch.int32, torch.int64):
        raise TypeError("tokens_per_expert must be int32 or int64.")
    return supplied_counts.to(torch.int32).contiguous()


def _start_count_gather(counts: Any, spec: MegaMoeSpec) -> tuple[Any, Any | None]:
    """Start the rank-major INT32 count gather used by route construction."""

    if spec.ep_size == 1:
        return counts.reshape(1, spec.num_experts), None
    gathered_counts = torch.empty(
        spec.ep_size * spec.num_experts,
        dtype=torch.int32,
        device=counts.device,
    )
    work = dist.all_gather_into_tensor(
        gathered_counts,
        counts,
        group=spec.ep_group,
        async_op=True,
    )
    return gathered_counts.reshape(spec.ep_size, spec.num_experts), work


def _finish_count_gather(
    counts_by_source: Any,
    work: Any | None,
    spec: MegaMoeSpec,
) -> tuple[Any, int]:
    """Wait for gathered counts and return this rank's receive matrix and global maximum."""

    if work is not None:
        work.wait()
    local_start = spec.rank_id * spec.local_experts
    local_end = local_start + spec.local_experts
    received_counts = counts_by_source[:, local_start:local_end].contiguous()
    if spec.capacity_is_lossless:
        return received_counts, spec.receive_capacity
    destination_loads = counts_by_source.reshape(
        spec.ep_size,
        spec.ep_size,
        spec.local_experts,
    ).sum(dim=(0, 2), dtype=torch.int32)
    return received_counts, int(destination_loads.max().item())


def _permute_topk_input(hidden_states: Any, topk_ids: Any) -> tuple[Any, Any]:
    """Expand hidden rows with the native NPU MoE permutation operator."""

    routed_tokens, unpermute_mapping = torch_npu.npu_moe_token_permute(
        hidden_states,
        topk_ids,
    )
    return (
        routed_tokens,
        unpermute_mapping.reshape(-1).to(torch.int32).contiguous(),
    )


def _build_route_indices(
    route_slots: Any,
    topk_ids: Any,
) -> tuple[Any, Any]:
    """Build expert-major route slots and the inverse token route map."""


    permuted_slots, inverse_order = torch_npu.npu_moe_token_permute(
        route_slots,
        topk_ids.reshape(-1, 1),
    )
    permute_src_index = permuted_slots.reshape(-1).to(torch.int32).contiguous()
    inverse_order = inverse_order.reshape(-1).to(torch.int32).contiguous()
    return permute_src_index, inverse_order.reshape_as(topk_ids).contiguous()


def _compute_route_metadata(
    counts: Any,
    counts_by_source: Any,
    received_counts: Any,
    plan: MegaMoePlan,
    maximum_received_slots: int,
) -> RouteMetadata:
    """Build exact row offsets without padding or route truncation."""

    spec = plan.spec
    counts_i32 = counts.to(dtype=torch.int32).contiguous()
    split = max(spec.dispatch_split, spec.combine_split)
    rounded_capacity = (maximum_received_slots + split - 1) // split * split
    expert_capacity = max(spec.routed_slots, rounded_capacity)

    source_prefix = counts_by_source.cumsum(dim=1, dtype=torch.int32) - counts_by_source
    destination_layout = (
        counts_by_source.reshape(
            spec.ep_size,
            spec.ep_size,
            spec.local_experts,
        )
        .permute(1, 2, 0)
        .contiguous()
    )
    destination_flat = destination_layout.reshape(spec.ep_size, -1)
    destination_prefix = (
        destination_flat.cumsum(dim=1, dtype=torch.int32) - destination_flat
    ).reshape(spec.ep_size, spec.local_experts, spec.ep_size)
    local_destination_prefix = destination_prefix[spec.rank_id]
    local_destination_counts = destination_layout[spec.rank_id]
    local_start = spec.rank_id * spec.local_experts
    local_end = local_start + spec.local_experts
    return RouteMetadata(
        dispatch_src_off=source_prefix[spec.rank_id].to(torch.int64),
        dispatch_target_off=(
            destination_prefix[:, :, spec.rank_id].reshape(-1).to(torch.int64)
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
    plan: MegaMoePlan,
    tokens_per_expert: Any | None,
    workspace: MegaMoeWorkspace | None = None,
) -> PreparedTopKRoute:
    """Overlap native TopK permutation with count gathering and build route metadata."""
    spec = plan.spec
    flat_ids = _validate_topk_inputs(hidden_states, topk_ids, topk_weights)
    counts = _resolve_counts(flat_ids, tokens_per_expert, spec)
    if workspace is not None:
        # Waiting before the mandatory count exchange turns its completion into
        # the cross-rank rendezvous for the next symmetric-buffer generation.
        workspace.wait_for_reuse()
    counts_by_source, count_work = _start_count_gather(counts, spec)
    if spec.forward_permute_fused:
        route_slots = plan.route_slots
        if route_slots is None:
            raise RuntimeError("route-fused MegaMoe is missing route-slot metadata.")
        permute_src_index, unpermute_mapping = _build_route_indices(
            route_slots,
            topk_ids,
        )
        routed_tokens = None
    else:
        routed_tokens, unpermute_mapping = _permute_topk_input(
            hidden_states,
            topk_ids,
        )
        permute_src_index = None
    received_counts, maximum_received_slots = _finish_count_gather(
        counts_by_source,
        count_work,
        spec,
    )
    receive_overflow = maximum_received_slots > spec.receive_capacity
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
            tokens_per_expert=counts,
            received_counts=received_counts,
            receive_overflow=True,
            metadata=None,
        )
    return PreparedTopKRoute(
        routed_tokens=routed_tokens,
        unpermute_mapping=unpermute_mapping,
        permute_src_index=permute_src_index,
        tokens_per_expert=counts,
        received_counts=received_counts,
        receive_overflow=False,
        metadata=_compute_route_metadata(
            counts,
            counts_by_source,
            received_counts,
            plan,
            maximum_received_slots,
        ),
    )


def restore_topk_output(
    expert_output: Any,
    unpermute_mapping: Any,
    topk_weights: Any,
) -> Any:
    """Restore token order and route weights with the native NPU MoE operator."""

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
