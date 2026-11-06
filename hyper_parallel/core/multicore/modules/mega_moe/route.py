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
"""Prepare trusted Router output for managed Torch MegaMoe execution."""

from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from hyper_parallel.core.expert_parallel.hot_replica.routing import (
    ReplicaRoute, _has_home_recompute_hint, _replica_count_payload,
)

from hyper_parallel.core.multicore._loader import load_torch_npu
from hyper_parallel.core.multicore.torch import ops as multicore_ops

from .spec import MegaMoeSpec, _align_capacity

if TYPE_CHECKING:
    from .workspace import MegaMoeWorkspace


# Avoid large Host transfers and Python boxing when the gathered count table grows.
_HOST_ROUTE_MAX_COUNTS = 2048
# Preserve the smaller descriptor budget when counts change on every call.
_HOST_DESCRIPTOR_MAX_COUNTS = 1024
_ROUTE_CACHE_REPROBE_STEPS = 8
_METADATA_ALIGNMENT = 512
_PULL_STAGING_MAX_BYTES = 64 * 1024 * 1024

ReplicaRouteSelector = Callable[
    [torch.Tensor, list[list[int]], Optional[list[list[int]]]], Optional[tuple[MegaMoeSpec, ReplicaRoute]]]


@dataclass(frozen=True)
class RouteSchedule:
    """Exact global task extents and rank-local empty expert matrices."""

    expert_rows: int
    source_expert_rows: int
    empty_experts: tuple[int, ...]


@dataclass(frozen=True)
class RouteMetadata:
    """Exact offsets and counts consumed by the low-level native ABI."""

    dispatch_src_off: torch.Tensor
    dispatch_target_off: torch.Tensor
    dispatch_size: torch.Tensor
    combine_src_off: torch.Tensor
    combine_target_off: torch.Tensor
    combine_size: torch.Tensor
    group_list: torch.Tensor
    expert_capacity: int
    local_num_tokens: int | None = None
    plan_tokens: int | None = None
    replica_route: ReplicaRoute | None = None
    schedule: RouteSchedule | None = None
    empty_owner_experts: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PreparedTopKRoute:
    """Permuted payload and exact metadata for one Router decision."""

    routed_tokens: torch.Tensor
    unpermute_mapping: torch.Tensor
    tokens_per_expert: torch.Tensor
    received_counts: torch.Tensor
    metadata: RouteMetadata
    maximum_received_slots: int = 0


@dataclass(frozen=True)
class _HostRouteCache:
    """One validated wire table and immutable route descriptors for a workspace."""

    signature: tuple[Any, ...]
    raw: list[list[int]]
    summary: tuple[int, int, int]
    metadata: RouteMetadata | None
    stream: Any = None
    ready: Any = None

    def wait(self, device: Any) -> None:
        """Order a different consumer stream after the descriptor upload.

        Args:
            device: Device whose current stream consumes the cached descriptors.
        """
        if self.ready is not None and self.metadata is not None:
            current = torch.npu.current_stream(device)
            if current != self.stream:
                current.wait_event(self.ready)
                for tensor in vars(self.metadata).values():
                    if isinstance(tensor, torch.Tensor):
                        tensor.record_stream(current)


@dataclass(frozen=True)
class _RouteHeaderCache:
    """One immutable control header, retained across identical local invocations."""

    key: tuple[Any, ...]
    tensor: torch.Tensor | None
    stream: Any = None
    ready: Any = None

    def wait(self, device: Any) -> None:
        """Preserve upload ordering and allocation lifetime on another stream.

        Args:
            device: Device whose current stream consumes the cached header.
        """
        if self.ready is not None and self.tensor is not None:
            current = torch.npu.current_stream(device)
            if current != self.stream:
                current.wait_event(self.ready)
                self.tensor.record_stream(current)


def _validate_topk_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Validate route tensor metadata without synchronizing tensor contents."""
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
    flat_ids: torch.Tensor,
    supplied_counts: torch.Tensor | None,
    spec: MegaMoeSpec,
) -> torch.Tensor:
    """Use trusted Router counts or compute them only when omitted."""
    if supplied_counts is None:
        return (
            torch.bincount(flat_ids.to(torch.int64), minlength=spec.num_experts)
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
        raise TypeError(
            f"tokens_per_expert must be int32 or int64, got {supplied_counts.dtype}."
        )
    if supplied_counts.dtype == torch.int32 and supplied_counts.is_contiguous():
        return supplied_counts
    return supplied_counts.to(torch.int32).contiguous()


def _start_count_gather(
    counts: torch.Tensor,
    spec: MegaMoeSpec,
) -> tuple[torch.Tensor, Any | None]:
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
    counts_by_source: torch.Tensor,
    work: Any | None,
    spec: MegaMoeSpec,
) -> torch.Tensor:
    """Wait for gathered counts before reading route metadata."""
    if work is not None:
        work.wait()
    local_start = spec.rank_id * spec.local_experts
    local_end = local_start + spec.local_experts
    return counts_by_source[:, local_start:local_end].contiguous()


def _expert_capacity(
    counts_by_source: torch.Tensor,
    spec: MegaMoeSpec,
    loads: tuple[int, ...] | None = None,
) -> tuple[int, int]:
    """Size rank-local tensors, reusing exact host loads from a replica plan."""
    if loads is None:
        destination_loads = counts_by_source.reshape(
            spec.ep_size, spec.ep_size, spec.local_experts,
        ).sum(dim=(0, 2), dtype=torch.int32)
        # Ordinary routes need one readback for local allocation and global growth.
        loads = destination_loads.tolist()
    # Keep a non-null ABI argument on ranks whose experts receive no tokens.
    # Source outputs and symmetric communication buffers retain their own sizes.
    return max(1, loads[spec.rank_id]), max(loads)


def _permute_topk_input(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand hidden rows with the native NPU MoE permutation operator."""
    if hidden_states.shape[0] == 0:
        return hidden_states, torch.empty(0, dtype=torch.int32, device=hidden_states.device)
    routed_tokens, unpermute_mapping = load_torch_npu().npu_moe_token_permute(
        hidden_states,
        topk_ids,
    )
    return (
        routed_tokens,
        unpermute_mapping.reshape(-1).to(torch.int32).contiguous(),
    )


def _permute_topk_input_out(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the inverse mapping owned while writing rows into leased SHMEM."""
    mapping = torch.empty(topk_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    multicore_ops.moe_token_permute_out(hidden_states.contiguous(), topk_ids.contiguous(), output, mapping)
    return output, mapping


def _compute_route_metadata(
    counts: torch.Tensor,
    counts_by_source: torch.Tensor,
    received_counts: torch.Tensor,
    spec: MegaMoeSpec,
    expert_capacity: int,
    *,
    route_counts: list[list[int]] | None = None,
) -> RouteMetadata:
    """Build exact element offsets without padding or route truncation."""
    counts_i32 = counts.to(dtype=torch.int32).contiguous()
    source_prefix = counts_by_source.cumsum(dim=1, dtype=torch.int64) - counts_by_source
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
        destination_flat.cumsum(dim=1, dtype=torch.int64) - destination_flat
    ).reshape(spec.ep_size, spec.local_experts, spec.ep_size)
    local_destination_prefix = destination_prefix[spec.rank_id]
    local_destination_counts = destination_layout[spec.rank_id]
    local_start = spec.rank_id * spec.local_experts
    local_end = local_start + spec.local_experts
    return RouteMetadata(
        dispatch_src_off=(source_prefix[:, local_start:local_end].reshape(-1).to(torch.int64)
                          if spec.dispatch_mode == "pull" else source_prefix[spec.rank_id].to(torch.int64)),
        dispatch_target_off=(
            local_destination_prefix.T.reshape(-1).to(torch.int64) if spec.dispatch_mode == "pull"
            else destination_prefix[:, :, spec.rank_id].reshape(-1).to(torch.int64)
        ),
        dispatch_size=(received_counts.reshape(-1).to(torch.int32) if spec.dispatch_mode == "pull" else counts_i32),
        combine_src_off=local_destination_prefix.T.reshape(-1).to(torch.int64),
        combine_target_off=(
            source_prefix[:, local_start:local_end].reshape(-1).to(torch.int64)
        ),
        combine_size=received_counts.reshape(-1).to(torch.int32),
        group_list=(
            local_destination_prefix[:, -1] + local_destination_counts[:, -1]
        ).to(torch.int64),
        expert_capacity=expert_capacity,
        schedule=_route_schedule(counts_by_source, spec, route_counts),
    )


def _route_schedule(counts: torch.Tensor | None, spec: MegaMoeSpec,
                    host_counts: list[list[int]] | None = None) -> RouteSchedule | None:
    """Bound tasks by current global counts while keeping allocation bounds intact."""
    if not spec.auto_tokens or not spec.route_task_bounds or spec.replica_slots_per_rank:
        return None
    if host_counts is not None:
        totals = [sum(values) for values in zip(*host_counts)]
        source_maximum = max(max(row) for row in host_counts)
    else:
        totals_tensor = counts.sum(0, dtype=torch.int64)
        values = torch.cat((totals_tensor, counts.amax().reshape(1).to(torch.int64))).tolist()
        totals, source_maximum = values[:-1], values[-1]
    start = spec.rank_id * spec.local_experts
    empty = tuple(index for index, rows in enumerate(totals[start:start + spec.local_experts]) if not rows)
    return RouteSchedule(max(totals), source_maximum, empty)


def _upload_route_tensor(tensor: torch.Tensor, device: Any) -> torch.Tensor:
    """Keep Host metadata alive through the pinned allocator during asynchronous upload."""
    if torch.device(device).type == "npu":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def _host_route_metadata(
    raw: list[list[int]],
    spec: MegaMoeSpec,
    capacity: int,
    device: Any,
) -> RouteMetadata:
    """Pack the native INT64 offsets and INT32 route sizes in one retained allocation."""
    counts = [row[3:] for row in raw]
    local, rank = spec.local_experts, spec.rank_id
    start = rank * local
    prefix, destinations, destination_prefix = _host_route_prefixes(counts, spec)
    source_offsets = [
        prefix[source][expert] for source in range(spec.ep_size) for expert in range(start, start + local)
    ]
    target_offsets = [
        destination_prefix[rank][expert * spec.ep_size + source]
        for source in range(spec.ep_size)
        for expert in range(local)
    ]
    received = [counts[source][expert] for source in range(spec.ep_size) for expert in range(start, start + local)]
    if spec.dispatch_mode == "pull":
        dispatch_source, dispatch_target, dispatch_sizes = source_offsets, target_offsets, received
    else:
        dispatch_source = prefix[rank]
        dispatch_target = [
            destination_prefix[target][expert * spec.ep_size + rank]
            for target in range(spec.ep_size)
            for expert in range(local)
        ]
        dispatch_sizes = counts[rank]
    groups = [
        destination_prefix[rank][(expert + 1) * spec.ep_size - 1] + destinations[rank][(expert + 1) * spec.ep_size - 1]
        for expert in range(local)
    ]
    wide = (dispatch_source, dispatch_target, target_offsets, source_offsets, groups)
    metadata = _pack_host_route(wide, (dispatch_sizes, received), capacity, device)
    return replace(metadata, schedule=_route_schedule(None, spec, counts))


def _host_route_prefixes(
    counts: list[list[int]], spec: MegaMoeSpec,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Scan each source and each destination independently in native layout order."""
    local = spec.local_experts
    prefix = [list(accumulate(row, initial=0))[:-1] for row in counts]
    destinations = [
        [
            counts[source][expert]
            for expert in range(target * local, (target + 1) * local)
            for source in range(spec.ep_size)
        ]
        for target in range(spec.ep_size)
    ]
    destination_prefix = [list(accumulate(row, initial=0))[:-1] for row in destinations]
    return prefix, destinations, destination_prefix


def _pack_host_route(
    wide: tuple[list[int], ...],
    size_fields: tuple[list[int], ...],
    capacity: int,
    device: Any,
) -> RouteMetadata:
    """Transfer aligned descriptor fields together and retain storage through their views."""
    fields = [*wide, *size_fields]
    dtypes = [torch.int64] * len(wide) + [torch.int32] * 2
    storage = bytearray()
    slices = []
    for values, dtype in zip(fields, dtypes):
        encoded = array("q" if dtype == torch.int64 else "i", values).tobytes()
        start = len(storage)
        storage.extend(encoded)
        slices.append((start, len(storage), dtype))
        storage.extend(bytes((-len(storage)) % _METADATA_ALIGNMENT))
    buffer = _upload_route_tensor(torch.frombuffer(storage, dtype=torch.uint8), device)
    tensors = [buffer[start:end].view(dtype) for start, end, dtype in slices]
    offsets, sizes = tensors[:5], tensors[5:]
    return RouteMetadata(
        dispatch_src_off=offsets[0],
        dispatch_target_off=offsets[1],
        dispatch_size=sizes[0],
        combine_src_off=offsets[2],
        combine_target_off=offsets[3],
        combine_size=sizes[1],
        group_list=offsets[4],
        expert_capacity=capacity,
    )


def prepare_topk_route(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    spec: MegaMoeSpec,
    tokens_per_expert: torch.Tensor | None,
    workspace: MegaMoeWorkspace | None = None,
    replica_route: ReplicaRoute | None = None,
    *,
    autograd_mask: int = 0,
    prepare_workspace: Callable[[int, int], None] | None = None,
    _select_replica_route: ReplicaRouteSelector | None = None,
) -> PreparedTopKRoute:
    """Overlap Top-K permutation with count exchange and build route metadata.

    Args:
        hidden_states: Flattened token states to route.
        topk_ids: Global experts selected for every token.
        topk_weights: Router weights paired with ``topk_ids``.
        spec: Bound shape and expert-parallel specification.
        tokens_per_expert: Optional trusted Router histogram.
        workspace: Optional workspace ordered before the count exchange. Pull
            requires a lease acquired by the caller or the automatic-capacity callback.
        replica_route: Optional shared placement plan and completed dispatch counts.
        autograd_mask: Group-consistent gradient participation flags in dynamic mode.
        prepare_workspace: Automatic-capacity callback, run after count agreement and before pull permutation.
        _select_replica_route: Internal placement using the existing dynamic count agreement.

    Returns:
        Permuted tokens and exact native route metadata.
    """
    flat_ids = _validate_topk_inputs(hidden_states, topk_ids, topk_weights)
    if spec.dynamic_tokens:
        return _prepare_dynamic_route(hidden_states, topk_ids, flat_ids, spec, tokens_per_expert,
                                      workspace, autograd_mask, replica_route=replica_route,
                                      prepare_workspace=prepare_workspace,
                                      select_replica_route=_select_replica_route)
    counts = _resolve_counts(flat_ids, tokens_per_expert, spec)
    permutation_output = None
    if workspace is not None:
        if spec.dispatch_mode == "pull":
            if not workspace.in_use or workspace.source_buffer is None:
                raise RuntimeError("Pull permutation requires an initialized, claimed workspace.")
            permutation_output = workspace.source_buffer
        else:
            workspace.wait_for_reuse()
    if replica_route is None:
        counts_by_source, count_work = _start_count_gather(counts, spec)
    else:
        counts_by_source, count_work = replica_route.counts_by_source, None
    if permutation_output is None:
        routed_tokens, unpermute_mapping = _permute_topk_input(hidden_states, topk_ids)
    else:
        routed_tokens, unpermute_mapping = _permute_topk_input_out(hidden_states, topk_ids, permutation_output)
    received_counts = _finish_count_gather(counts_by_source, count_work, spec)
    loads = None if replica_route is None else replica_route.plan.destination_loads
    expert_capacity, maximum_received_slots = _expert_capacity(counts_by_source, spec, loads)
    return PreparedTopKRoute(
        routed_tokens=routed_tokens,
        unpermute_mapping=unpermute_mapping,
        tokens_per_expert=counts,
        received_counts=received_counts,
        maximum_received_slots=maximum_received_slots,
        metadata=replace(_compute_route_metadata(
            counts,
            counts_by_source,
            received_counts,
            spec,
            expert_capacity,
        ), replica_route=None if replica_route is not None and replica_route.home_only else replica_route),
    )


def _valid_route_counts(row: list[int], top_k: int) -> bool:
    """Validate one source histogram against its actual distinct Top-K rows."""
    return row[3] == row[1] * top_k and row[4] >= 0 and row[5] <= row[1]


def _validate_dynamic_sizes(rows: list[list[int]], spec: MegaMoeSpec) -> None:
    """Keep native size and legacy bound rejection ahead of route publication."""
    flexible = spec.auto_tokens
    if flexible and any(row[0] or row[1] < 0 for row in rows):
        raise RuntimeError("MegaMoe dynamic route exceeds native INT32 size limits")
    if not flexible and any(
            row[0] or not 0 <= row[1] <= spec.max_local_num_tokens for row in rows):
        raise RuntimeError(f"MegaMoe token capacity overflow: limit={spec.max_local_num_tokens}, "
                           f"actual={[row[1] for row in rows]}")


def _validate_dynamic_rows(rows: list[list[int]], spec: MegaMoeSpec) -> None:
    """Validate every rank before any automatic allocation or SHMEM write."""
    _validate_dynamic_sizes(rows, spec)
    if any(row[2] != rows[0][2] for row in rows):
        raise RuntimeError("MegaMoe autograd participation differs across EP ranks")
    if any(not _valid_route_counts(row, spec.top_k) for row in rows):
        raise RuntimeError("MegaMoe counts must describe the actual distinct Top-K routes")


def _dynamic_summary(payload: torch.Tensor, spec: MegaMoeSpec) -> tuple[int, int, int]:
    """Use one Host summary for shape, participation, receive limits, and schedule."""
    return _summarize_dynamic_rows(_device_route_rows(payload, spec), spec)


def _device_route_rows(payload: torch.Tensor, spec: MegaMoeSpec) -> list[list[int]]:
    """Read the compact agreement once for validation and replica admission."""
    counts = payload[:, 3:]
    loads = counts.reshape(spec.ep_size, spec.ep_size, spec.local_experts).sum(dim=(0, 2))
    summary = torch.cat((payload[:, :3], counts.sum(1, keepdim=True),
                         counts.amin(1, keepdim=True), counts.amax(1, keepdim=True), loads[:, None]), dim=1)
    return summary.tolist()


def _summarize_dynamic_rows(rows: list[list[int]], spec: MegaMoeSpec,
                            *, defer_receive_limit: bool = False) -> tuple[int, int, int]:
    """Keep overflow and participation checks common to both metadata policies."""
    _validate_dynamic_rows(rows, spec)
    maximum = max(row[6] for row in rows)
    maximum_tokens = max(row[1] for row in rows) if spec.auto_tokens else spec.local_num_tokens
    maximum_capacity = _align_capacity(spec.ep_size * maximum_tokens * spec.top_k)
    if maximum > maximum_capacity:
        raise RuntimeError(
            f"MegaMoe receive capacity overflow: configured_capacity={maximum_capacity}, actual_maximum={maximum}"
        )
    plan_tokens = max(128, _align_capacity(max(row[1] for row in rows)))
    if (plan_tokens * max(spec.hidden_size, spec.top_k) > 2**31 - 1
            or maximum > 2**31 - 1 and not defer_receive_limit):
        raise RuntimeError("MegaMoe dynamic route exceeds native INT32 size limits")
    return max(1, rows[spec.rank_id][6]), plan_tokens, maximum


def _owner_empty_experts(spec: MegaMoeSpec, agreement: list[list[int]] | None,
                         host_rows: list[list[int]] | None) -> tuple[int, ...] | None:
    """Elide full initialization only when every owner gradient is overwritten."""
    if not spec._selective_owner_gradient_init or spec.replica_slots_per_rank:
        return None
    if agreement is not None and any(row[4] > 0 for row in agreement):
        # One source covers every logical expert, so every group has positive K.
        return ()
    if host_rows is None:
        return None
    first = 3 + spec.rank_id * spec.local_experts
    if all(any(row[first + expert] for row in host_rows) for expert in range(spec.local_experts)):
        return ()
    return None


def _read_dynamic_route(
    payload: torch.Tensor,
    spec: MegaMoeSpec,
    *, defer_receive_limit: bool = False,
) -> tuple[tuple[int, int, int], list[list[int]] | None, list[list[int]]]:
    """Reuse the required readback for small descriptors; reduce large tables on device."""
    host_limit = _HOST_DESCRIPTOR_MAX_COUNTS if defer_receive_limit else _HOST_ROUTE_MAX_COUNTS
    if spec.ep_size * spec.num_experts > host_limit:
        rows = _device_route_rows(payload, spec)
        return _summarize_dynamic_rows(rows, spec, defer_receive_limit=defer_receive_limit), None, rows
    raw = payload.tolist()
    counts = [row[3:] for row in raw]
    local = spec.local_experts
    loads = [
        sum(sum(values[start : start + local]) for values in counts) for start in range(0, spec.num_experts, local)
    ]
    rows = [row[:3] + [sum(values), min(values), max(values), load] for row, values, load in zip(raw, counts, loads)]
    return _summarize_dynamic_rows(rows, spec, defer_receive_limit=defer_receive_limit), raw, rows


def _summarize_host_route(raw: list[list[int]], spec: MegaMoeSpec) -> tuple[int, int, int]:
    """Validate a bounded Host table before allocating or publishing source rows."""
    counts = [row[3:] for row in raw]
    local = spec.local_experts
    loads = [
        sum(sum(values[start : start + local]) for values in counts) for start in range(0, spec.num_experts, local)
    ]
    rows = [row[:3] + [sum(values), min(values), max(values), load] for row, values, load in zip(raw, counts, loads)]
    return _summarize_dynamic_rows(rows, spec)


def _route_cache_signature(spec: MegaMoeSpec, device: Any) -> tuple[Any, ...]:
    """Keep descriptor reuse specific to the route topology and size contract."""
    return (spec.ep_size, id(spec.ep_group), spec.rank_id, spec.num_experts, spec.top_k, spec.hidden_size,
            spec.dispatch_mode, spec.auto_tokens, spec.max_local_num_tokens, torch.device(device))


def _read_cached_route(
    payload: torch.Tensor, spec: MegaMoeSpec, workspace: MegaMoeWorkspace | None, device: Any,
) -> tuple[tuple[int, int, int], list[list[int]] | None, RouteMetadata | None]:
    """Reuse only an identical, previously validated all-rank wire table."""
    if workspace is not None:
        workspace.empty_owner_experts = None
    if workspace is None or not spec.auto_tokens or spec.ep_size * spec.num_experts > _HOST_ROUTE_MAX_COUNTS:
        summary, raw, agreement = _read_dynamic_route(payload, spec)
        if workspace is not None:
            workspace.empty_owner_experts = _owner_empty_experts(spec, agreement, raw)
        return summary, raw, None
    cached = getattr(workspace, "route_cache", None)
    same_topology = isinstance(cached, _HostRouteCache) and cached.signature == _route_cache_signature(spec, device)
    remaining = getattr(workspace, "route_cache_probe_after", 0)
    if same_topology and remaining > 0:
        workspace.route_cache_probe_after = remaining - 1
        agreement = _device_route_rows(payload, spec)
        workspace.empty_owner_experts = _owner_empty_experts(spec, agreement, None)
        return _summarize_dynamic_rows(agreement, spec), None, None
    workspace.route_cache_probe_after = 0
    raw = payload.tolist()
    if same_topology and cached.raw == raw:
        cached.wait(device)
        return cached.summary, raw, cached.metadata
    summary = _summarize_host_route(raw, spec)
    if same_topology and spec.ep_size * spec.num_experts > _HOST_DESCRIPTOR_MAX_COUNTS:
        workspace.route_cache_probe_after = _ROUTE_CACHE_REPROBE_STEPS
    return summary, raw, None


def _cache_host_route(
    workspace: MegaMoeWorkspace | None, spec: MegaMoeSpec, raw: list[list[int]],
    summary: tuple[int, int, int], metadata: RouteMetadata, device: Any,
) -> None:
    """Retain descriptors only after confirming a repeated validated wire table."""
    if workspace is None or not spec.auto_tokens:
        return
    signature = _route_cache_signature(spec, device)
    previous = getattr(workspace, "route_cache", None)
    repeated = isinstance(previous, _HostRouteCache) and previous.signature == signature and previous.raw == raw
    retained = metadata if repeated else None
    stream, ready = None, None
    if repeated and torch.device(device).type == "npu":
        stream = torch.npu.current_stream(device)
        ready = torch.npu.Event()
        ready.record(stream)
    workspace.route_cache = _HostRouteCache(signature, raw, summary, retained, stream, ready)


def _dynamic_route_metadata(
    counts: torch.Tensor, gathered: torch.Tensor, host_rows: list[list[int]] | None,
    spec: MegaMoeSpec, capacity: int,
) -> RouteMetadata:
    """Keep wider cache misses on device instead of rebuilding large Host descriptors."""
    if host_rows is not None and spec.ep_size * spec.num_experts <= _HOST_DESCRIPTOR_MAX_COUNTS:
        return _host_route_metadata(host_rows, spec, capacity, counts.device)
    counts_by_source = gathered[:, 3:].contiguous()
    start = spec.rank_id * spec.local_experts
    received_counts = counts_by_source[:, start:start + spec.local_experts].contiguous()
    # Retained dispatch sizes must not alias a caller-owned Router count buffer.
    local_counts = counts_by_source[spec.rank_id] if spec.auto_tokens else counts
    route_counts = None if host_rows is None else [row[3:] for row in host_rows]
    return _compute_route_metadata(local_counts, counts_by_source, received_counts, spec, capacity,
                                   route_counts=route_counts)


def _permute_dynamic_pull(
    hidden_states: torch.Tensor, topk_ids: torch.Tensor, spec: MegaMoeSpec, workspace: MegaMoeWorkspace | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the source view only after automatic growth and lease acquisition."""
    if workspace is None or not workspace.in_use or workspace.source_buffer is None:
        raise RuntimeError("Pull permutation requires an initialized, claimed workspace.")
    tokens = hidden_states.shape[0]
    output = workspace.source_buffer[:tokens * spec.top_k]
    return (_permute_topk_input_out(hidden_states, topk_ids, output) if tokens
            else (output, topk_ids.reshape(-1)))


def _can_stage_pull(hidden_states: torch.Tensor, spec: MegaMoeSpec) -> bool:
    """Bound ordinary-memory permutation staging while capacity agreement is pending."""
    return (spec.auto_tokens and spec.dispatch_mode == "pull" and hidden_states.device.type == "npu"
            and hidden_states.numel() * spec.top_k * hidden_states.element_size() <= _PULL_STAGING_MAX_BYTES)


def _stage_dynamic_permutation(
    hidden_states: torch.Tensor, topk_ids: torch.Tensor, spec: MegaMoeSpec, invalid: bool,
    workspace: MegaMoeWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Stage only valid automatic routes whose ordinary storage survives heap growth."""
    if invalid or not spec.auto_tokens:
        return None
    if (spec.dispatch_mode == "push"
            or _can_stage_pull(hidden_states, spec) and not getattr(workspace, "route_cache_probe_after", 0)):
        return _permute_topk_input(hidden_states, topk_ids)
    return None


def _publish_staged_pull(
    permuted: tuple[torch.Tensor, torch.Tensor], hidden_states: torch.Tensor,
    spec: MegaMoeSpec, workspace: MegaMoeWorkspace | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish staged rows only after every rank validated and committed its workspace."""
    if workspace is None or not workspace.in_use or workspace.source_buffer is None:
        raise RuntimeError("Pull publication requires an initialized, claimed workspace.")
    output = workspace.source_buffer[:hidden_states.shape[0] * spec.top_k]
    output.copy_(permuted[0])
    return output, permuted[1]


def _dynamic_header(
    values: tuple[int, int, int], dtype: torch.dtype, device: Any, workspace: MegaMoeWorkspace | None,
) -> torch.Tensor:
    """Avoid repeated Host allocation and upload of an unchanged control header."""
    key = (values, dtype, torch.device(device))
    cached = getattr(workspace, "route_header_cache", None)
    repeated = isinstance(cached, _RouteHeaderCache) and cached.key == key
    if repeated and cached.tensor is not None:
        cached.wait(device)
        return cached.tensor
    header = _upload_route_tensor(torch.tensor(values, dtype=dtype), device)
    if workspace is not None:
        stream, ready = None, None
        if repeated and torch.device(device).type == "npu":
            stream = torch.npu.current_stream(device)
            ready = torch.npu.Event()
            ready.record(stream)
        workspace.route_header_cache = _RouteHeaderCache(key, header if repeated else None, stream, ready)
    return header


def _dynamic_route_payload(
    hidden_states: torch.Tensor, flat_ids: torch.Tensor, spec: MegaMoeSpec,
    supplied_counts: torch.Tensor | None, autograd_mask: int, workspace: MegaMoeWorkspace | None = None,
    validate_replica_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Pack one group-consistent wire format, including coordinated overflow flags."""
    tokens = hidden_states.shape[0]
    flexible = spec.auto_tokens
    invalid = (tokens * max(spec.hidden_size, spec.top_k) > 2**31 - 1 if flexible
               else tokens > spec.max_local_num_tokens)
    if validate_replica_ids and supplied_counts is None and not invalid:
        checked = _replica_count_payload(flat_ids.reshape(tokens, spec.top_k).to(torch.int64), spec.num_experts)
        counts = checked[:spec.num_experts].to(torch.int32)
        # A negative count carries invalid IDs through the ordinary wire format;
        # every rank rejects it before placement, growth or remote publication.
        counts[:1].masked_fill_(checked[-1:].bool(), -1)
    else:
        counts = (torch.zeros(spec.num_experts, dtype=torch.int32, device=hidden_states.device) if invalid
                  else _resolve_counts(flat_ids, supplied_counts, spec))
    payload_dtype = torch.int32 if flexible else torch.int64
    # An invalid source must still join the same collective before all ranks reject it.
    header_tokens = min(tokens, 2**31 - 1) if flexible else tokens
    header = _dynamic_header((int(invalid), header_tokens, autograd_mask), payload_dtype,
                             hidden_states.device, workspace if flexible else None)
    payload = torch.cat((header, counts.to(payload_dtype)))
    return counts, payload, invalid


def _start_dynamic_gather(payload: torch.Tensor, spec: MegaMoeSpec) -> tuple[torch.Tensor, Any | None]:
    """Match receive storage to the dynamic count wire dtype."""
    if spec.ep_size == 1:
        gathered, work = payload.unsqueeze(0), None
    else:
        gathered = torch.empty((spec.ep_size, payload.numel()), dtype=payload.dtype, device=payload.device)
        work = dist.all_gather_into_tensor(gathered.reshape(-1), payload, group=spec.ep_group, async_op=True)
    return gathered, work


def _prepare_ordinary_dynamic_route(
    hidden_states: torch.Tensor, topk_ids: torch.Tensor, flat_ids: torch.Tensor, spec: MegaMoeSpec,
    supplied_counts: torch.Tensor | None, workspace: MegaMoeWorkspace | None, autograd_mask: int,
    prepare_workspace: Callable[[int, int], None] | None = None,
) -> PreparedTopKRoute:
    """Exchange variable-length metadata without adding a token-count collective."""
    tokens = hidden_states.shape[0]
    counts, payload, invalid = _dynamic_route_payload(
        hidden_states, flat_ids, spec, supplied_counts, autograd_mask, workspace)
    if workspace is not None and spec.dispatch_mode != "pull":
        workspace.wait_for_reuse()
    gathered, work = _start_dynamic_gather(payload, spec)
    # Ordinary storage remains valid if count agreement triggers a SHMEM restart.
    permuted = _stage_dynamic_permutation(hidden_states, topk_ids, spec, invalid, workspace)
    summary, host_rows, metadata = None, None, None
    if spec.auto_tokens:
        if work is not None:
            work.wait()
            work = None
        summary, host_rows, metadata = _read_cached_route(gathered, spec, workspace, hidden_states.device)
        if prepare_workspace is None:
            raise RuntimeError("Automatic MegaMoe routes require workspace capacity preparation")
        prepare_workspace(summary[1], summary[2])
    if invalid:
        permuted = None
    elif spec.dispatch_mode == "pull":
        permuted = (_permute_dynamic_pull(hidden_states, topk_ids, spec, workspace) if permuted is None else
                    _publish_staged_pull(permuted, hidden_states, spec, workspace))
    elif permuted is None:
        permuted = _permute_topk_input(hidden_states, topk_ids)
    if work is not None:
        work.wait()
    if summary is None:
        summary, host_rows, _ = _read_dynamic_route(gathered, spec)
    capacity, plan_tokens, maximum = summary
    if metadata is None:
        metadata = _dynamic_route_metadata(counts, gathered, host_rows, spec, capacity)
        if host_rows is not None:
            _cache_host_route(workspace, spec, host_rows, summary, metadata, hidden_states.device)
    received_counts = metadata.combine_size.reshape(spec.ep_size, spec.local_experts)
    return PreparedTopKRoute(
        routed_tokens=permuted[0], unpermute_mapping=permuted[1], tokens_per_expert=counts,
        received_counts=received_counts,
        maximum_received_slots=maximum,
        metadata=replace(
            metadata, local_num_tokens=tokens, plan_tokens=plan_tokens,
            empty_owner_experts=(_owner_empty_experts(spec, None, host_rows) if host_rows is not None
                                 else getattr(workspace, "empty_owner_experts", None)),
        ),
    )


def _overlap_replica_permutation(hidden: torch.Tensor, ids: torch.Tensor, spec: MegaMoeSpec,
                                 invalid: bool, selector: ReplicaRouteSelector | None,
                                 supplied_counts: torch.Tensor | None = None) -> tuple | None:
    """Overlap only a validated push route whose physical IDs cannot change."""
    if (spec.auto_tokens and spec.dispatch_mode == "push" and not invalid
            and (selector is None or supplied_counts is not None and _has_home_recompute_hint())):
        return _permute_topk_input(hidden, ids)
    return None


def _complete_replica_permutation(hidden: torch.Tensor, ids: torch.Tensor, spec: MegaMoeSpec,
                                  workspace: MegaMoeWorkspace | None, invalid: bool,
                                  permuted: tuple | None) -> tuple | None:
    """Keep permutation after replica admission for pull and rewritten push routes."""
    if invalid:
        return None
    if spec.dispatch_mode == "pull":
        return _permute_dynamic_pull(hidden, ids, spec, workspace)
    if permuted is None:
        return _permute_topk_input(hidden, ids)
    return permuted


def _finish_replica_dynamic_route(hidden_states: torch.Tensor, counts: torch.Tensor,
                                  gathered: torch.Tensor, spec: MegaMoeSpec, permuted: tuple,
                                  summary: tuple | None, host_rows: list | None, source_summary: list | None,
                                  replica_route: ReplicaRoute | None) -> PreparedTopKRoute:
    """Pack the selected replica route after permutation and count exchange complete."""
    agreement = source_summary
    if (source_summary is not None and replica_route._host_counts is not None
            and spec.ep_size * spec.num_experts <= _HOST_ROUTE_MAX_COUNTS):
        # The CPU planner already owns these counts. Match the ordinary small-route
        # packing path instead of rebuilding the same offsets with device kernels.
        host_rows = [list(header[:3]) + list(counts)
                     for header, counts in zip(source_summary, replica_route._host_counts)]
    if summary is None:
        if source_summary is None:
            summary, host_rows, agreement = _read_dynamic_route(gathered, spec)
        else:
            summary = _summarize_dynamic_rows(source_summary, spec)
    capacity, plan_tokens, maximum = summary
    if host_rows is None:
        counts_by_source = gathered[:, 3:].contiguous() if source_summary is None else gathered
        start = spec.rank_id * spec.local_experts
        received_counts = counts_by_source[:, start:start + spec.local_experts].contiguous()
        metadata = _compute_route_metadata(counts, counts_by_source, received_counts, spec, capacity)
    else:
        metadata = _host_route_metadata(host_rows, spec, capacity, hidden_states.device)
        received_counts = metadata.combine_size.reshape(spec.ep_size, spec.local_experts)
    return PreparedTopKRoute(
        routed_tokens=permuted[0], unpermute_mapping=permuted[1], tokens_per_expert=counts,
        received_counts=received_counts,
        maximum_received_slots=maximum,
        metadata=replace(
            metadata, local_num_tokens=hidden_states.shape[0], plan_tokens=plan_tokens,
            replica_route=None if replica_route is not None and replica_route.home_only else replica_route,
            empty_owner_experts=_owner_empty_experts(spec, agreement, host_rows),
        ),
    )


def _prepare_replica_auto_capacity(hidden_states: torch.Tensor, gathered: torch.Tensor, spec: MegaMoeSpec,
                                   source_summary: list | None, replica_route: ReplicaRoute | None,
                                   select_replica_route: ReplicaRouteSelector | None,
                                   workspace: MegaMoeWorkspace | None,
                                   autograd_mask: int, prepare_workspace: Callable | None) -> tuple:
    """Select one replica placement and prepare its exact lossless receive reservation."""
    summary, host_rows = None, None
    if source_summary is None:
        summary, host_rows, agreement = _read_dynamic_route(
            gathered, spec, defer_receive_limit=select_replica_route is not None)
        if select_replica_route is not None:
            proposal = select_replica_route(
                gathered[:, 3:], agreement, None if host_rows is None else [row[3:] for row in host_rows])
            if proposal is not None:
                selected_spec, replica_route = proposal
                if not replica_route.home_only:
                    permuted = None
                    ids = replica_route.physical_ids
                    rewritten = _prepare_dynamic_route(
                        hidden_states, ids, ids.reshape(-1), selected_spec,
                        replica_route.counts_by_source[selected_spec.rank_id], workspace, autograd_mask,
                        replica_route=replica_route, prepare_workspace=prepare_workspace)
                    return summary, host_rows, replica_route, rewritten
            if summary[2] > 2**31 - 1:
                raise RuntimeError("MegaMoe selected home route exceeds native INT32 receive limits")
    else:
        summary = _summarize_dynamic_rows(source_summary, spec)
    if prepare_workspace is None:
        raise RuntimeError("Automatic MegaMoe routes require workspace capacity preparation")
    prepare_workspace(summary[1], summary[2])
    return summary, host_rows, replica_route, None


def _prepare_dynamic_route(
    hidden_states: torch.Tensor, topk_ids: torch.Tensor, flat_ids: torch.Tensor, spec: MegaMoeSpec,
    supplied_counts: torch.Tensor | None, workspace: MegaMoeWorkspace | None, autograd_mask: int,
    replica_route: ReplicaRoute | None = None,
    prepare_workspace: Callable[[int, int], None] | None = None,
    select_replica_route: ReplicaRouteSelector | None = None,
) -> PreparedTopKRoute:
    """Exchange variable-length metadata without adding a token-count collective."""
    if replica_route is None and select_replica_route is None:
        return _prepare_ordinary_dynamic_route(
            hidden_states, topk_ids, flat_ids, spec, supplied_counts, workspace,
            autograd_mask, prepare_workspace,
        )
    source_summary = None if replica_route is None else replica_route.dynamic_summary
    payload = None
    if source_summary is None:
        counts, payload, invalid = _dynamic_route_payload(
            hidden_states, flat_ids, spec, supplied_counts, autograd_mask,
            validate_replica_ids=select_replica_route is not None)
    else:
        counts = _resolve_counts(flat_ids, supplied_counts, spec)
        invalid = bool(source_summary[spec.rank_id][0])
    if workspace is not None and spec.dispatch_mode != "pull":
        workspace.wait_for_reuse()
    gathered, work = (_start_dynamic_gather(payload, spec) if payload is not None
                      else (replica_route.counts_by_source, None))
    permuted = _overlap_replica_permutation(hidden_states, topk_ids, spec, invalid,
                                            select_replica_route, supplied_counts)
    summary, host_rows = None, None
    if spec.auto_tokens:
        if work is not None:
            work.wait()
            work = None
        summary, host_rows, replica_route, rewritten = _prepare_replica_auto_capacity(
            hidden_states, gathered, spec, source_summary, replica_route, select_replica_route,
            workspace, autograd_mask, prepare_workspace)
        if rewritten is not None:
            return rewritten
    permuted = _complete_replica_permutation(hidden_states, topk_ids, spec, workspace, invalid, permuted)
    if work is not None:
        work.wait()
    return _finish_replica_dynamic_route(
        hidden_states, counts, gathered, spec, permuted, summary, host_rows, source_summary, replica_route)


class _LegacyTopKRestore(torch.autograd.Function):
    """Keep Torch-NPU 2.9.0 unpermute gradients on the explicit ACLNN adapter."""

    @staticmethod
    def forward(
        ctx: Any, expert_output: torch.Tensor, mapping: torch.Tensor, probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Retain owned forward inputs for the precompiled backward entry.

        Args:
            ctx: Autograd context for this output restoration.
            expert_output: Owned expert-major forward output rows.
            mapping: INT32 inverse permutation mapping.
            probabilities: Contiguous FP32 Router weights.
        """
        ctx.save_for_backward(expert_output, mapping, probabilities)
        return load_torch_npu().npu_moe_token_unpermute(expert_output, mapping, probs=probabilities)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor | None, None, torch.Tensor | None]:
        """Return correctly ordered expert and routing gradients in owned tensors.

        Args:
            ctx: Context retaining the expert outputs and Router metadata.
            gradient: Token-major output gradient.
        """
        expert_output, mapping, probabilities = ctx.saved_tensors
        grad_experts = torch.empty_like(expert_output)
        grad_probabilities = torch.empty_like(probabilities)
        multicore_ops.mega_moe_unpermute_grad_out(
            expert_output, gradient, mapping, probabilities, grad_experts, grad_probabilities,
        )
        return (
            grad_experts if ctx.needs_input_grad[0] else None,
            None,
            grad_probabilities if ctx.needs_input_grad[2] else None,
        )


def restore_topk_output(
    expert_output: torch.Tensor,
    unpermute_mapping: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Restore token order and apply Router weights with the native NPU op.

    Args:
        expert_output: Expert outputs in the expanded routing order.
        unpermute_mapping: Inverse mapping retained by the source permutation.
        topk_weights: Router weights for the original token rows.

    Returns:
        Weighted expert outputs in the original token order.
    """
    if topk_weights.shape[0] == 0:
        # Keep both the expert backward and the empty routing-weight gradient connected.
        return expert_output[:0] + topk_weights.sum().to(expert_output.dtype)
    probabilities = topk_weights.float().contiguous()
    npu_version = getattr(load_torch_npu(), "__version__", "").split("+", 1)[0].split(".")[:3]
    if npu_version == ["2", "9", "0"] and torch.is_grad_enabled():
        return _LegacyTopKRestore.apply(expert_output, unpermute_mapping, probabilities)
    return load_torch_npu().npu_moe_token_unpermute(
        expert_output,
        unpermute_mapping,
        probs=probabilities,
    )
