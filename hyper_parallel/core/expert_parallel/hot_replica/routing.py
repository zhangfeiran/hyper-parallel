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
"""Logical routing to stable physical expert slots."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state

from .capacity import ExpertReplicaConfig, _integer
from .cost import ExpertReplicaCostModel
from .count_kernel import try_count_payload
from .device import DeviceExpertExecutionPlan, ReplicaPlanSummary, build_device_expert_replica_plan
from .device import validate_planner_backend
from .plan import ExpertExecutionPlan
from .planner import build_expert_replica_plan


@dataclass(frozen=True)
class ReplicaRoute:
    """Invocation-owned logical plan and device execution metadata.

    The optional dynamic summary stores invalid/token/autograd headers, logical
    count sum/min/max, and destination load for every EP rank on the host.
    """

    plan: ExpertExecutionPlan | ReplicaPlanSummary
    physical_ids: torch.Tensor
    counts_by_source: torch.Tensor
    rank: int
    group: object
    transport: object | None = None
    device_plan: DeviceExpertExecutionPlan | None = None
    home_only: bool = False
    dynamic_summary: tuple[tuple[int, ...], ...] | None = None
    plan_reused: bool = False


class _RecomputedReplicaPlans:
    """Retain only immutable CPU plans for one checkpoint invocation."""

    def __init__(self) -> None:
        """Initialize ordered call records without retaining any device tensors."""
        self.entries: list[tuple[Any, ExpertExecutionPlan]] = []
        self.cursor = 0

    def clear(self) -> None:
        """Release all plans after complete or interrupted recomputation."""
        self.entries.clear()
        self.cursor = 0

    def lookup(self, signature: Any, counts: tuple[tuple[int, ...], ...]) -> ExpertExecutionPlan | None:
        """Advance in invocation order and reject changed counts or static policy."""
        if self.cursor >= len(self.entries):
            return None
        expected, plan = self.entries[self.cursor]
        self.cursor += 1
        return plan if signature == expected and counts == plan.logical_counts else None


def _plan_replica_route(counts: Any, config: ExpertReplicaConfig, group: object, rank: int,
                        **options: Any) -> tuple[ExpertExecutionPlan, bool]:
    """Replay the original decision after the usual count exchange validates current inputs."""
    state = get_recompute_state()
    if state is None:
        return build_expert_replica_plan(counts, config.replica_slots_per_rank, **options), False
    frozen = tuple(tuple(row) for row in counts)
    # Heap migration may increase the reservation between phases. Replaying the
    # original decision preserves checkpoint metadata while growth remains legal.
    policy = tuple((key, type(value), value) for key, value in sorted(options.items()) if key != "receive_capacity")
    signature = (config, id(group), rank, policy)
    cache = state.get_resource(_RecomputedReplicaPlans, _RecomputedReplicaPlans)
    if state.is_recomputing:
        plan = cache.lookup(signature, frozen)
        if plan is not None:
            return plan, True
    plan = build_expert_replica_plan(frozen, config.replica_slots_per_rank, **options)
    if not state.is_recomputing:
        cache.entries.append((signature, plan))
    return plan, False


def stable_expert_order(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Sort small expert keys on device while keeping permutation indices integer.

    FP32 represents every expert ID exactly below 2**24. Token indices are never
    converted to floating point, so large token batches keep exact inverses.

    Args:
        ids: Logical expert IDs to order stably.
        num_experts: Logical expert count determining the exact sorting key dtype.

    Returns:
        Integer permutation preserving the order of equal expert IDs.
    """
    keys = ids.float() if num_experts <= 2**24 else ids
    return torch.argsort(keys, stable=True)


def _upload_route_metadata(plan: ExpertExecutionPlan, rank: int,
                           device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upload one owned buffer with aligned int64 runs and int32 dispatch counts."""
    runs = plan.source_runs[rank] if plan.transfers else ()
    slots = tuple(slot for slot, _ in runs)
    lengths = tuple(count for _, count in runs)
    counts = tuple(count for row in plan.dispatch_counts for count in row)
    data = struct.pack(f"<{2 * len(runs)}q{len(counts)}i", *slots, *lengths, *counts)
    # Blocking upload retains the host buffer until DMA completes. Device views
    # own their storage through permutation and every downstream count consumer.
    packed = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)
    run_bytes = 16 * len(runs)
    indices = packed[:run_bytes].view(torch.int64).reshape(2, len(runs))
    dispatch = packed[run_bytes:].view(torch.int32).reshape(plan.config.ep_size, plan.config.physical_experts)
    return indices[0], indices[1], dispatch


def _remap_replica_ids(ids: torch.Tensor, plan: ExpertExecutionPlan,
                       slots: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Keep TopK order, bypassing quota expansion when all experts stay home."""
    if not plan.transfers:
        home = plan.config.home_experts
        return ids + torch.div(ids, home, rounding_mode="floor") * plan.config.replica_slots_per_rank
    destinations = torch.repeat_interleave(slots, lengths, output_size=ids.numel())
    order = stable_expert_order(ids.flatten(), plan.config.num_experts)
    remapped = torch.empty_like(ids.flatten())
    remapped.scatter_(0, order, destinations)
    return remapped.reshape_as(ids)


def _gather_replica_counts(payload: torch.Tensor, size: int, group: object) -> torch.Tensor:
    """Gather rank-major counts into one invocation-owned contiguous buffer."""
    gathered = torch.empty((size, payload.numel()), dtype=payload.dtype, device=payload.device)
    if size > 1:
        work = dist.all_gather_into_tensor(gathered.flatten(), payload, group=group, async_op=True)
        work.wait()
    else:
        gathered[0].copy_(payload)
    return gathered


def _replica_count_payload(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Keep route validation and logical counts in one collective payload."""
    fused = try_count_payload(ids, num_experts)
    if fused is not None:
        return fused
    return _torch_replica_count_payload(ids, num_experts)


def _torch_replica_count_payload(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Validate and count exactly when the optional native counter is unavailable."""
    ordered = (ids.float() if num_experts <= 2**24 else ids).sort(dim=1).values
    invalid = ((ids < 0) | (ids >= num_experts)).any()
    invalid = invalid | (ordered[:, 1:] == ordered[:, :-1]).any()
    # The clamped IDs have a known range. A fixed integer histogram avoids
    # bincount's device min/max scalar reads while retaining exact counts.
    clamped = ids.clamp(0, num_experts - 1).flatten()
    counts = torch.zeros(num_experts, dtype=torch.int64, device=ids.device)
    counts.scatter_add_(0, clamped, torch.ones_like(clamped))
    return torch.cat((counts, invalid.reshape(1).to(counts.dtype)))


def prepare_replica_route(
    topk_ids: torch.Tensor, config: ExpertReplicaConfig, group: object = None,
    *, target_load: int | None = None, minimum_replica_rows: int = 0,
    cost_model: ExpertReplicaCostModel | None = None, planner_backend: str = "cpu",
    compact_home: bool = False,
    minimum_load_reduction: float = 0.0,
    minimum_saved_rows: int = 0,
    dynamic_header: tuple[int, int, int] | None = None,
    max_local_num_tokens: int | None = None,
    receive_capacity: int | None = None,
) -> ReplicaRoute:
    """Gather logical counts once, plan replicas, and preserve every TopK slot.

    Route validation is collective: invalid IDs and repeated experts are reported
    by all ranks before any sparse weight transfers. All ranks must supply the
    same Top-K width and configuration; source token counts may differ.

    Args:
        topk_ids: Integer expert IDs, with distinct experts in each token's row.
        config: Logical expert topology and per-rank replica budget.
        group: EP process group; None selects the default group.
        target_load: Optional preferred destination load, capped by the receive bound.
        minimum_replica_rows: Minimum useful row count for a guest copy.
        cost_model: Optional calibrated model for CPU quota refinement.
        planner_backend: CPU or device placement implementation.
        compact_home: Keep logical IDs and counts when the global plan has no guests.
            The caller must execute that invocation with a home-only kernel topology.
        minimum_load_reduction: Required receive-peak reduction for optional copies.
        minimum_saved_rows: Required absolute receive-peak reduction for optional copies.
        dynamic_header: Optional invalid/token-count/autograd header fused into the count gather.
        max_local_num_tokens: Optional source-row reservation, validated again by the dynamic route summary.
        receive_capacity: Globally agreed resident receive rows for capacity-aware CPU cost refinement.

    Returns:
        Invocation-owned placement, physical IDs and dispatch counts.
    """
    validate_planner_backend(planner_backend, minimum_replica_rows, cost_model)
    if receive_capacity is not None:
        _integer(receive_capacity, "receive_capacity", 0)
    if max_local_num_tokens is not None:
        _integer(max_local_num_tokens, "max_local_num_tokens", 1)
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be a two-dimensional integer tensor")
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    size = dist.get_world_size(group) if dist.is_initialized() else 1
    if size != config.ep_size:
        raise ValueError("replica configuration must match the EP group size")
    tokens, top_k = topk_ids.shape
    config.maximum_receive_rows(max(1, tokens), top_k, alignment=1)
    ids = topk_ids.to(torch.int64)
    payload = _replica_count_payload(ids, config.num_experts)
    if dynamic_header is not None:
        payload = torch.cat((payload, payload.new_tensor(dynamic_header)))
    matrix = _gather_replica_counts(payload, size, group)
    logical_counts = matrix[:, :config.num_experts]
    if planner_backend == "device":
        source_summary = None
        if dynamic_header is not None:
            summary = torch.cat((matrix[:, config.num_experts + 1:], logical_counts.sum(1, keepdim=True),
                                 logical_counts.amin(1, keepdim=True), logical_counts.amax(1, keepdim=True)), dim=1)
            source_summary = summary.cpu().tolist()
        maximum_tokens = (max(row[1] for row in source_summary) if source_summary is not None
                          else int(logical_counts.sum(dim=1).max()) // top_k)
        upper = config.maximum_receive_rows(max(1, maximum_tokens), top_k, alignment=1)
        device_plan = build_device_expert_replica_plan(
            logical_counts, config, capacity_limit=upper,
            target_load=min(upper, target_load) if target_load is not None else None,
            minimum_replica_rows=minimum_replica_rows, minimum_load_reduction=minimum_load_reduction,
            minimum_saved_rows=minimum_saved_rows)
        device_plan.control[:1].bitwise_or_(matrix[:, config.num_experts].any().to(torch.int64))
        plan = device_plan.host_summary()
        dynamic_summary = (None if source_summary is None else
                           tuple(tuple(row) + (load,) for row, load in zip(source_summary, plan.destination_loads)))
        if compact_home and not plan.transfers:
            return ReplicaRoute(plan, topk_ids, logical_counts.to(torch.int32).contiguous(),
                                rank, group, device_plan=device_plan, home_only=True, dynamic_summary=dynamic_summary)
        slots, lengths = device_plan.source_runs(rank)
        remapped = _remap_replica_ids(ids, plan, slots, lengths)
        return ReplicaRoute(plan, remapped.to(topk_ids.dtype), device_plan.dispatch_counts.to(torch.int32),
                            rank, group, device_plan=device_plan, dynamic_summary=dynamic_summary)
    # The deterministic host planner shares the route-count synchronization with
    # split sizing; no per-expert device-to-host reads occur below.
    host = matrix.cpu().tolist()
    if any(row[config.num_experts] for row in host):
        raise ValueError("hot replication requires in-range, distinct expert IDs per token")
    logical_host = [row[:config.num_experts] for row in host]
    # Every planner must use the same bound when source ranks have unequal or zero rows.
    maximum_tokens = max(sum(row) for row in logical_host) // top_k
    upper = config.maximum_receive_rows(max(1, maximum_tokens), top_k, alignment=1)
    plan, reused = _plan_replica_route(
        logical_host, config, group, rank,
        target_load=min(upper, target_load) if target_load is not None else None,
        minimum_replica_rows=minimum_replica_rows, capacity_limit=upper, cost_model=cost_model,
        minimum_load_reduction=minimum_load_reduction,
        minimum_saved_rows=minimum_saved_rows,
        receive_capacity=receive_capacity,
    )
    if max(plan.destination_loads) > upper:
        raise RuntimeError("replica planner exceeded the theoretical receive bound")
    # Distinct logical Top-K counts bound every physical split, including home-only plans.
    dynamic_summary = (None if dynamic_header is None else tuple(
        tuple(row[config.num_experts + 1:]) + (sum(counts), min(counts), max(counts), load)
        for row, counts, load in zip(host, logical_host, plan.destination_loads)))
    if compact_home and not plan.transfers:
        return ReplicaRoute(plan, topk_ids, logical_counts.to(torch.int32).contiguous(),
                            rank, group, home_only=True, dynamic_summary=dynamic_summary, plan_reused=reused)
    slots, lengths, device_counts = _upload_route_metadata(plan, rank, ids.device)
    remapped = _remap_replica_ids(ids, plan, slots, lengths)
    return ReplicaRoute(plan, remapped.to(topk_ids.dtype), device_counts, rank, group,
                        dynamic_summary=dynamic_summary, plan_reused=reused)
