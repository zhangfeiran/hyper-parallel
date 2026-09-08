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

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch_npu

from hyper_parallel.core.multicore.modules.mega_moe.placement import (
    ExpertPlacementPlan,
    build_constrained_expert_placement,
)

from .plan import MegaMoePlan
from .spec import MegaMoeSpec

if TYPE_CHECKING:
    from .workspace import MegaMoeWorkspace


torch = import_module("torch")
dist = import_module("torch.distributed")


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
    retained_capacity: int
    receive_capacity: int
    wave_count: int


@dataclass(frozen=True)
class PreparedTopKRoute:
    """Permuted payload, count metadata and the capacity decision for one route."""

    plan: MegaMoePlan
    routed_tokens: Any | None
    physical_topk_ids: Any
    unpermute_mapping: Any
    permute_src_index: Any | None
    tokens_per_expert: Any
    received_counts: Any
    receive_overflow: bool
    metadata: RouteMetadata | None
    placement: ExpertPlacementPlan | None
    placement_payload: Any | None


@dataclass(frozen=True)
class _PreparedRouteLayout:
    """Physical layout selected before exact route metadata is constructed."""

    plan: MegaMoePlan
    routed_tokens: Any | None
    physical_topk_ids: Any
    unpermute_mapping: Any
    permute_src_index: Any | None
    physical_counts: Any
    physical_counts_by_source: Any
    received_counts: Any
    maximum_received_slots: int
    placement: ExpertPlacementPlan | None = None
    placement_payload: Any | None = None


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


def count_topk_experts(topk_ids: Any, num_experts: int) -> Any:
    """Count known-valid router assignments without data-dependent output sizing.

    Args:
        topk_ids: Integer expert IDs produced by TopK over ``num_experts``
            logits. Every ID must be in ``[0, num_experts)``; this trusted
            helper does not perform a synchronizing range validation.
        num_experts: Fixed number of bins. Assignment counts must fit INT32.

    Returns:
        A device-local INT32 histogram, including zero-count experts.
    """
    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    return counts.scatter_add_(
        0, flat_ids, torch.ones_like(flat_ids, dtype=torch.int32)
    )


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


def _finish_static_count_gather(
    counts_by_source: Any,
    work: Any | None,
    spec: MegaMoeSpec,
) -> tuple[Any, int]:
    """Finish a home-only gather and return local receives and global maximum."""

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


def _wait_count_gather(work: Any | None) -> None:
    """Wait before consuming an asynchronously gathered count matrix."""
    if work is not None:
        work.wait()


def _physical_copy_runs(
    plan: ExpertPlacementPlan,
    source_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return physical IDs and row counts ordered by logical expert."""
    runs_by_expert = [[] for _ in range(plan.logical_num_experts)]
    # One pass retains ascending physical-slot order within each logical
    # expert, which is the ordering used by the device permutation's quotas.
    for physical_expert, (logical_expert, count) in enumerate(zip(
            plan.physical_to_logical_expert, plan.dispatch_counts[source_rank])):
        if count:
            runs_by_expert[logical_expert].append((physical_expert, count))
    physical_ids = []
    copy_counts = []
    for runs in runs_by_expert:
        for physical_expert, count in runs:
            physical_ids.append(physical_expert)
            copy_counts.append(count)
    return tuple(physical_ids), tuple(copy_counts)


def _remap_topk_ids_to_physical(
    topk_ids: Any,
    plan: ExpertPlacementPlan,
    source_rank: int,
) -> Any:
    """Remap logical IDs on device while preserving every Top-K route slot."""
    physical_ids, copy_counts = _physical_copy_runs(plan, source_rank)
    route_count = topk_ids.numel()
    if sum(copy_counts) != route_count:
        raise ValueError("physical copy quota total does not match the number of route slots")
    device = topk_ids.device
    # The host plan already determines the length. Avoid a device reduction
    # and scalar readback solely to size this fixed-shape metadata tensor.
    expert_major_physical_ids = torch.repeat_interleave(
        torch.tensor(physical_ids, dtype=torch.int32, device=device),
        torch.tensor(copy_counts, dtype=torch.int64, device=device),
        output_size=route_count,
    )
    flat_ids = topk_ids.reshape(-1)
    route_slots = torch.arange(
        flat_ids.numel(),
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    expert_major_slots, _ = _build_route_indices(route_slots, topk_ids)
    # Building from shape avoids inheriting an unsupported private format from
    # the logical-ID tensor on NPU; route metadata must stay in base format.
    physical_ids_by_route = torch.empty(
        flat_ids.shape,
        dtype=torch.int32,
        device=device,
    )
    physical_ids_by_route.scatter_(
        0,
        expert_major_slots.to(torch.int64),
        expert_major_physical_ids,
    )
    return physical_ids_by_route.reshape_as(topk_ids).contiguous()


@lru_cache(maxsize=128)
def _cached_replica_plan(
    logical_counts: tuple[tuple[int, ...], ...],
    replica_slots_per_rank: int,
    receive_capacity: int,
) -> ExpertPlacementPlan:
    """Return a bounded cached placement for an immutable count matrix."""
    return build_constrained_expert_placement(
        logical_counts,
        replica_slots_per_rank,
        receive_capacity=receive_capacity,
    )


def _build_replica_plan(
    counts_by_source: Any,
    spec: MegaMoeSpec,
) -> ExpertPlacementPlan:
    """Build or reuse one bounded host placement from gathered counts."""
    # The first Ascend integration uses a bounded host reference planner. The
    # explicit opt-in keeps this synchronization out of the default path until
    # the planner is moved into the device route kernel.
    logical_counts = tuple(
        tuple(int(count) for count in source_counts)
        for source_counts in counts_by_source.tolist()
    )
    global_counts = tuple(map(sum, zip(*logical_counts)))
    destination_loads = tuple(
        sum(global_counts[start:start + spec.local_experts])
        for start in range(0, spec.num_experts, spec.local_experts)
    )
    planning_capacity = (
        spec.routed_slots
        if max(destination_loads) > spec.receive_capacity
        else spec.receive_capacity
    )
    return _cached_replica_plan(
        logical_counts,
        spec.replica_slots_per_rank,
        # Once overflow requires another wave, target the exact mean so the
        # slowest expert rank does not retain a long GMM tail. A route already
        # inside the one-wave window stops without needless replica traffic.
        planning_capacity,
    )


def _materialize_replica_counts(
    plan: ExpertPlacementPlan,
    counts_by_source: Any,
    spec: MegaMoeSpec,
) -> tuple[Any, Any, int]:
    """Copy planned physical counts to device and select local receives."""
    physical_counts = torch.tensor(
        plan.dispatch_counts,
        dtype=torch.int32,
        device=counts_by_source.device,
    )
    local_start = spec.rank_id * spec.physical_local_experts
    local_end = local_start + spec.physical_local_experts
    received_counts = physical_counts[:, local_start:local_end].contiguous()
    return physical_counts, received_counts, max(plan.destination_loads)


def _drain_placement_payload(payload: Any | None) -> None:
    """Complete an optional asynchronous placement payload after route failure."""
    if payload is None:
        return
    drain = getattr(payload, "drain", None)
    if callable(drain):
        drain()


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


def _prepare_route_endpoints(
    hidden_states: Any,
    topk_ids: Any,
    plan: MegaMoePlan,
) -> tuple[Any | None, Any, Any | None]:
    """Prepare payload and route indices for one physical expert-ID layout."""
    spec = plan.spec
    if spec.forward_permute_fused:
        route_slots = plan.route_slots
        if route_slots is None:
            raise RuntimeError("route-fused MegaMoe is missing route-slot metadata.")
        permute_src_index, unpermute_mapping = _build_route_indices(
            route_slots,
            topk_ids,
        )
        return None, unpermute_mapping, permute_src_index
    routed_tokens, unpermute_mapping = _permute_topk_input(
        hidden_states,
        topk_ids,
    )
    return routed_tokens, unpermute_mapping, None


def _prepare_home_layout(
    hidden_states: Any,
    topk_ids: Any,
    counts: Any,
    counts_by_source: Any,
    count_work: Any | None,
    plan: MegaMoePlan,
) -> _PreparedRouteLayout:
    """Finish counts and prepare endpoints for the static home layout."""
    routed_tokens, unpermute_mapping, permute_src_index = _prepare_route_endpoints(
        hidden_states,
        topk_ids,
        plan,
    )
    received_counts, maximum_received_slots = _finish_static_count_gather(
        counts_by_source,
        count_work,
        plan.spec,
    )
    return _PreparedRouteLayout(
        plan=plan,
        routed_tokens=routed_tokens,
        physical_topk_ids=topk_ids,
        unpermute_mapping=unpermute_mapping,
        permute_src_index=permute_src_index,
        physical_counts=counts,
        physical_counts_by_source=counts_by_source,
        received_counts=received_counts,
        maximum_received_slots=maximum_received_slots,
    )


def _prepare_bounded_layout(
    hidden_states: Any,
    topk_ids: Any,
    counts: Any,
    counts_by_source: Any,
    plan: MegaMoePlan,
    placement_callback: Callable[[ExpertPlacementPlan], Any] | None,
    home_runtime: MegaMoePlan | None,
) -> _PreparedRouteLayout:
    """Select a replica-free layout or materialize one bounded placement."""
    spec = plan.spec
    received_counts, maximum_received_slots = _finish_static_count_gather(
        counts_by_source,
        None,
        spec,
    )
    if home_runtime is not None and maximum_received_slots <= spec.receive_capacity:
        routed_tokens, unpermute_mapping, permute_src_index = _prepare_route_endpoints(
            hidden_states,
            topk_ids,
            home_runtime,
        )
        return _PreparedRouteLayout(
            plan=home_runtime,
            routed_tokens=routed_tokens,
            physical_topk_ids=topk_ids,
            unpermute_mapping=unpermute_mapping,
            permute_src_index=permute_src_index,
            physical_counts=counts,
            physical_counts_by_source=counts_by_source,
            received_counts=received_counts,
            maximum_received_slots=maximum_received_slots,
        )

    placement = _build_replica_plan(counts_by_source, spec)
    placement_payload = (
        placement_callback(placement) if placement_callback is not None else None
    )
    try:
        physical_counts_by_source, received_counts, maximum_received_slots = (
            _materialize_replica_counts(
                placement,
                counts_by_source,
                spec,
            )
        )
        physical_topk_ids = _remap_topk_ids_to_physical(
            topk_ids,
            placement,
            spec.rank_id,
        )
        routed_tokens, unpermute_mapping, permute_src_index = (
            _prepare_route_endpoints(
                hidden_states,
                physical_topk_ids,
                plan,
            )
        )
    except Exception:
        _drain_placement_payload(placement_payload)
        raise
    return _PreparedRouteLayout(
        plan=plan,
        routed_tokens=routed_tokens,
        physical_topk_ids=physical_topk_ids,
        unpermute_mapping=unpermute_mapping,
        permute_src_index=permute_src_index,
        physical_counts=physical_counts_by_source[spec.rank_id],
        physical_counts_by_source=physical_counts_by_source,
        received_counts=received_counts,
        maximum_received_slots=maximum_received_slots,
        placement=placement,
        placement_payload=placement_payload,
    )


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
    rounded_capacity = (
        (maximum_received_slots + split - 1) // split * split
    )
    wave_count = max(
        1,
        math.ceil(maximum_received_slots / spec.receive_capacity),
    )
    # A single wave keeps the existing exact-capacity allocation. Overflow is
    # split into fixed receive-capacity windows so every peer reuses the same
    # symmetric address range for each kernel-internal wave.
    expert_capacity = (
        rounded_capacity if wave_count == 1 else spec.receive_capacity
    )

    source_prefix = counts_by_source.cumsum(dim=1, dtype=torch.int32) - counts_by_source
    destination_layout = (
        counts_by_source.reshape(
            spec.ep_size,
            spec.ep_size,
            spec.physical_local_experts,
        )
        .permute(1, 2, 0)
        .contiguous()
    )
    destination_flat = destination_layout.reshape(spec.ep_size, -1)
    destination_prefix = (
        destination_flat.cumsum(dim=1, dtype=torch.int32) - destination_flat
    ).reshape(spec.ep_size, spec.physical_local_experts, spec.ep_size)
    local_destination_prefix = destination_prefix[spec.rank_id]
    local_destination_counts = destination_layout[spec.rank_id]
    local_start = spec.rank_id * spec.physical_local_experts
    local_end = local_start + spec.physical_local_experts
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
        retained_capacity=(
            expert_capacity if wave_count == 1 else maximum_received_slots
        ),
        receive_capacity=spec.receive_capacity,
        wave_count=wave_count,
    )


def prepare_topk_route(
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
    plan: MegaMoePlan,
    tokens_per_expert: Any | None,
    workspace: MegaMoeWorkspace | None = None,
    placement_callback: Callable[[ExpertPlacementPlan], Any] | None = None,
    home_runtime: MegaMoePlan | None = None,
) -> PreparedTopKRoute:
    """Prepare physical Top-K routing with count and weight-transfer overlap.

    Args:
        hidden_states: Token-major local activations.
        topk_ids: Logical expert IDs selected for every token.
        topk_weights: Router weights paired with ``topk_ids``.
        plan: Static MegaMoe plan and topology.
        tokens_per_expert: Optional trusted local route histogram.
        workspace: Optional workspace whose prior generation must be complete.
        placement_callback: Optional callback launched after placement but
            before device remap, allowing weight prefetch to overlap route work.
        home_runtime: Optional replica-free plan used when static home
            placement already fits one receive window.

    Returns:
        Prepared payload, route mappings, counts, and multi-wave metadata.
    """
    spec = plan.spec
    flat_ids = _validate_topk_inputs(hidden_states, topk_ids, topk_weights)
    counts = _resolve_counts(flat_ids, tokens_per_expert, spec)
    if workspace is not None:
        # Waiting before the mandatory count exchange turns its completion into
        # the cross-rank rendezvous for the next symmetric-buffer generation.
        workspace.wait_for_reuse()
    counts_by_source, count_work = _start_count_gather(counts, spec)
    if spec.replica_slots_per_rank:
        _wait_count_gather(count_work)
        layout = _prepare_bounded_layout(
            hidden_states,
            topk_ids,
            counts,
            counts_by_source,
            plan,
            placement_callback,
            home_runtime,
        )
    else:
        layout = _prepare_home_layout(
            hidden_states,
            topk_ids,
            counts,
            counts_by_source,
            count_work,
            plan,
        )
    try:
        metadata = _compute_route_metadata(
            layout.physical_counts,
            layout.physical_counts_by_source,
            layout.received_counts,
            layout.plan,
            layout.maximum_received_slots,
        )
    except Exception:
        _drain_placement_payload(layout.placement_payload)
        raise
    return PreparedTopKRoute(
        plan=layout.plan,
        routed_tokens=layout.routed_tokens,
        physical_topk_ids=layout.physical_topk_ids,
        unpermute_mapping=layout.unpermute_mapping,
        permute_src_index=layout.permute_src_index,
        tokens_per_expert=layout.physical_counts,
        received_counts=layout.received_counts,
        receive_overflow=False,
        metadata=metadata,
        placement=layout.placement,
        placement_payload=layout.placement_payload,
    )


def restore_topk_output(
    expert_output: Any,
    unpermute_mapping: Any,
    topk_weights: Any,
) -> Any:
    """Restore token order and route weights with the native NPU MoE operator.

    Args:
        expert_output: Expert-major local route outputs.
        unpermute_mapping: Inverse permutation from routes to token slots.
        topk_weights: Token-major routing probabilities.

    Returns:
        Weighted token-major output.
    """

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
