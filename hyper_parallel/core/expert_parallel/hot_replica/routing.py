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

import torch
import torch.distributed as dist

from .capacity import ExpertReplicaConfig
from .cost import ExpertReplicaCostModel
from .device import DeviceExpertExecutionPlan, ReplicaPlanSummary, build_device_expert_replica_plan
from .device import validate_planner_backend
from .plan import ExpertExecutionPlan
from .planner import build_expert_replica_plan


@dataclass(frozen=True)
class ReplicaRoute:
    """Invocation-owned logical plan and device execution metadata."""

    plan: ExpertExecutionPlan | ReplicaPlanSummary
    physical_ids: torch.Tensor
    counts_by_source: torch.Tensor
    rank: int
    group: object
    transport: object | None = None
    device_plan: DeviceExpertExecutionPlan | None = None


def stable_expert_order(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Sort small expert keys on device while keeping permutation indices integer.

    FP32 represents every expert ID exactly below 2**24. Token indices are never
    converted to floating point, so large token batches keep exact inverses.
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


def prepare_replica_route(
    topk_ids: torch.Tensor, config: ExpertReplicaConfig, group: object = None,
    *, target_load: int | None = None, minimum_replica_rows: int = 0,
    cost_model: ExpertReplicaCostModel | None = None, planner_backend: str = "cpu",
) -> ReplicaRoute:
    """Gather logical counts once, plan replicas, and preserve every TopK slot.

    Route validation is collective: invalid IDs and repeated experts are reported
    by all ranks before any sparse weight transfers. All ranks must supply the
    same shape and configuration, as required by the EP module contract.
    """
    validate_planner_backend(planner_backend, minimum_replica_rows, cost_model)
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be a two-dimensional integer tensor")
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    size = dist.get_world_size(group) if dist.is_initialized() else 1
    if size != config.ep_size:
        raise ValueError("replica configuration must match the EP group size")
    tokens, top_k = topk_ids.shape
    upper = config.maximum_receive_rows(tokens, top_k, alignment=1)
    ids = topk_ids.to(torch.int64)
    ordered = (ids.float() if config.num_experts <= 2**24 else ids).sort(dim=1).values
    invalid = ((ids < 0) | (ids >= config.num_experts)).any()
    invalid = invalid | (ordered[:, 1:] == ordered[:, :-1]).any()
    counts = torch.bincount(ids.clamp(0, config.num_experts - 1).flatten(), minlength=config.num_experts)
    payload = torch.cat((counts, invalid.reshape(1).to(counts.dtype)))
    gathered = [torch.empty_like(payload) for _ in range(size)]
    if size > 1:
        work = dist.all_gather(gathered, payload, group=group, async_op=True)
        work.wait()
    else:
        gathered[0].copy_(payload)
    if planner_backend == "device":
        matrix = torch.stack(gathered)
        device_plan = build_device_expert_replica_plan(matrix[:, :-1], config, capacity_limit=upper)
        device_plan.control[:1].bitwise_or_(matrix[:, -1].any().to(torch.int64))
        plan = device_plan.host_summary()
        slots, lengths = device_plan.source_runs(rank)
        remapped = _remap_replica_ids(ids, plan, slots, lengths)
        return ReplicaRoute(plan, remapped.to(topk_ids.dtype), device_plan.dispatch_counts.to(torch.int32),
                            rank, group, device_plan=device_plan)
    # The deterministic host planner shares the route-count synchronization with
    # split sizing; no per-expert device-to-host reads occur below.
    host = torch.stack(gathered).cpu().tolist()
    if any(row[-1] for row in host):
        raise ValueError("hot replication requires in-range, distinct expert IDs per token")
    plan = build_expert_replica_plan(
        [row[:-1] for row in host], config.replica_slots_per_rank,
        target_load=min(upper, target_load) if target_load is not None else None,
        minimum_replica_rows=minimum_replica_rows, capacity_limit=upper, cost_model=cost_model,
    )
    if max(plan.destination_loads) > upper:
        raise RuntimeError("replica planner exceeded the theoretical receive bound")
    slots, lengths, device_counts = _upload_route_metadata(plan, rank, ids.device)
    remapped = _remap_replica_ids(ids, plan, slots, lengths)
    return ReplicaRoute(plan, remapped.to(topk_ids.dtype), device_counts, rank, group)
