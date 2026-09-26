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
"""Device-resident constructive placement and lossless source quota assignment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch

from .capacity import ExpertReplicaConfig, _integer
from .plan import ExpertReplicaTransfer


@dataclass(frozen=True)
class ReplicaPlanSummary:
    """Host-only transfer and allocation controls; source quotas stay on device."""

    config: ExpertReplicaConfig
    slot_to_logical: tuple[tuple[int, ...], ...]
    destination_counts: tuple[tuple[int, ...], ...]
    rank_splits: tuple[tuple[int, ...], ...]

    @cached_property
    def physical_to_logical(self) -> tuple[int, ...]:
        """Return fixed physical ownership, with -1 for unused guest slots."""
        return tuple(expert for row in self.slot_to_logical for expert in row)

    @cached_property
    def destination_loads(self) -> tuple[int, ...]:
        """Provide the dynamic capacity decision without reading device quotas."""
        return tuple(sum(row) for row in self.destination_counts)

    @cached_property
    def transfers(self) -> tuple[ExpertReplicaTransfer, ...]:
        """Provide the globally identical ordinary P2P/SHMEM transfer order."""
        home = self.config.home_experts
        return tuple(ExpertReplicaTransfer(expert, expert // home, expert % home, rank, slot)
                     for rank, row in enumerate(self.slot_to_logical)
                     for slot, expert in enumerate(row) if slot >= home and expert >= 0)


@dataclass(frozen=True)
class DeviceExpertExecutionPlan:
    """Keep per-source routing data on device and expose one compact control copy."""

    config: ExpertReplicaConfig
    slot_to_logical: torch.Tensor
    destination_counts: torch.Tensor
    dispatch_counts: torch.Tensor
    control: torch.Tensor

    def host_summary(self) -> ReplicaPlanSummary:
        """Read placement, destination boundaries and rank splits once, never quotas."""
        values = self.control.cpu().tolist()
        if values[0]:
            raise ValueError("Device replica plan has invalid counts, conservation or capacity")
        ranks, width = self.config.ep_size, self.config.slots_per_rank
        size = ranks * width
        slots = tuple(tuple(values[1 + rank * width:1 + (rank + 1) * width]) for rank in range(ranks))
        counts = tuple(tuple(values[1 + size + rank * width:1 + size + (rank + 1) * width])
                       for rank in range(ranks))
        splits = tuple(tuple(values[1 + 2 * size + rank * ranks:1 + 2 * size + (rank + 1) * ranks])
                       for rank in range(ranks))
        return ReplicaPlanSummary(self.config, slots, counts, splits)

    def source_runs(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return device physical slots in stable logical occurrence order."""
        logical = self.slot_to_logical.flatten()
        keys = torch.where(logical >= 0, logical, self.config.num_experts)
        order = torch.argsort(keys, stable=True)
        return order, self.dispatch_counts[rank].index_select(0, order)


def _bounded_device_copies(counts: torch.Tensor, config: ExpertReplicaConfig) -> torch.Tensor:
    """Execute the constructive capacity proof in a fixed number of device rounds."""
    ranks, home = config.ep_size, config.home_experts
    experts = torch.arange(config.num_experts, device=counts.device)
    owners = torch.div(experts, home, rounding_mode="floor")
    rank_ids = torch.arange(ranks, device=counts.device)
    totals = counts.sum(0)
    copies = totals.unsqueeze(0) * (rank_ids.unsqueeze(1) == owners.unsqueeze(0))
    budget = min(config.replica_slots_per_rank, home)
    if not budget or ranks == 1:
        return copies
    loads = copies.sum(1)
    targets = torch.div(loads.sum(), ranks, rounding_mode="floor") + (rank_ids < loads.sum().remainder(ranks))
    remaining = totals.clone()
    # Each round finalizes a previously unsatisfied receiver. The last rank is
    # determined by conservation, so no device scalar drives Python control.
    for _ in range(ranks - 1):
        difference = loads - targets
        donor = difference.argmax().reshape(1)
        receiver = difference.argmin().reshape(1)
        amount = (-difference.index_select(0, receiver)).clamp_min(0)
        available = remaining * (owners == donor)
        order = torch.argsort(available, descending=True, stable=True)
        sorted_rows = available.index_select(0, order)
        before = sorted_rows.cumsum(0) - sorted_rows
        segments = torch.minimum(sorted_rows, (amount - before).clamp_min(0))
        consumed = torch.zeros_like(remaining).scatter(0, order, segments)
        remaining = remaining - consumed
        # Select top-B balanced-edge segments, using logical ID for equal sizes.
        selected = torch.argsort(consumed, descending=True, stable=True)[:budget]
        sizes = consumed.index_select(0, selected)
        quota = torch.div(amount * budget, home, rounding_mode="floor")
        kept = torch.minimum(sizes, (quota - (sizes.cumsum(0) - sizes)).clamp_min(0))
        move = torch.zeros_like(remaining).scatter(0, selected, kept).unsqueeze(0)
        copies = copies.index_add(0, donor, -move).index_add(0, receiver, move)
        loads = loads.index_add(0, donor, -amount).index_add(0, receiver, amount)
    return copies


def _device_materialize(counts: torch.Tensor, copies: torch.Tensor,
                        config: ExpertReplicaConfig, capacity_limit: int | torch.Tensor) -> DeviceExpertExecutionPlan:
    """Intersect source and destination prefix intervals after retaining local rows."""
    ranks, home, width = config.ep_size, config.home_experts, config.slots_per_rank
    experts = torch.arange(config.num_experts, device=counts.device)
    rank_ids = torch.arange(ranks, device=counts.device)
    local = torch.minimum(counts, copies)
    source_end = (counts - local).cumsum(0)
    target_end = (copies - local).cumsum(0)
    source_begin = source_end - counts + local
    target_begin = target_end - copies + local
    quotas = (torch.minimum(source_end[:, None, :], target_end[None, :, :])
              - torch.maximum(source_begin[:, None, :], target_begin[None, :, :])).clamp_min(0)
    quotas = quotas + torch.eye(ranks, dtype=torch.int64, device=counts.device)[:, :, None] * local[:, None, :]
    is_guest = torch.div(experts, home, rounding_mode="floor").unsqueeze(0) != rank_ids.unsqueeze(1)
    guests = torch.where(is_guest & (copies > 0), experts.unsqueeze(0), config.num_experts).sort(1).values
    budget = config.replica_slots_per_rank
    guests = guests[:, :min(budget, config.num_experts)]
    if budget > config.num_experts:
        guests = torch.cat((guests, torch.full((ranks, budget - config.num_experts), config.num_experts,
                                              device=counts.device, dtype=torch.int64)), dim=1)
    slots = torch.cat((experts.reshape(ranks, home), guests), dim=1)
    valid = slots < config.num_experts
    indices = slots.clamp_max(config.num_experts - 1)
    destination = copies.gather(1, indices) * valid
    dispatch = (quotas.gather(2, indices.unsqueeze(0).expand(ranks, -1, -1)) * valid.unsqueeze(0))
    splits = dispatch.sum(2)
    slots = torch.where(valid, slots, -1)
    status = ((counts < 0).any() | (destination.sum(1) > capacity_limit).any()
              | (quotas.sum(1) != counts).any() | (quotas.sum(0) != copies).any()
              | ((is_guest & (copies > 0)).sum(1) > budget).any()).to(torch.int64).reshape(1)
    control = torch.cat((status, slots.flatten(), destination.flatten(), splits.flatten()))
    return DeviceExpertExecutionPlan(config, slots, destination, dispatch.reshape(ranks, ranks * width), control)


@torch.no_grad()
def build_device_expert_replica_plan(counts_by_source: torch.Tensor, config: ExpertReplicaConfig,
                                     *, capacity_limit: int | None = None) -> DeviceExpertExecutionPlan:
    """Solve constructive bounded placement without a counts-to-host roundtrip.

    Only static topology controls Python loops. Tensor outputs are independent
    of this invocation's host control copy and can feed remap/count consumers
    directly. This entry uses the constructive policy; host quota heuristics
    and offline cost refinement are not applied to its device result.
    """
    if capacity_limit is not None:
        _integer(capacity_limit, "capacity_limit", 0)
    if counts_by_source.shape != (config.ep_size, config.num_experts):
        raise ValueError("Device source counts do not match the replica topology")
    if counts_by_source.dtype not in (torch.int32, torch.int64):
        raise ValueError("Device source counts require int32 or int64")
    counts = counts_by_source.to(torch.int64)
    if capacity_limit is None:
        # Native lacks the original TopK shape. Derive a globally identical
        # histogram bound, including uneven source sizes, without scalar reads.
        home, ranks = config.home_experts, config.ep_size
        budget = min(config.replica_slots_per_rank, home)
        owner_max = counts.sum(0).reshape(ranks, home).sum(1).max()
        average = torch.div(counts.sum() + ranks - 1, ranks, rounding_mode="floor")
        mixed = torch.div((home - budget) * owner_max + budget * average + home - 1,
                          home, rounding_mode="floor")
        capacity_limit = average if budget == home else torch.minimum(owner_max, mixed + ranks - 1)
    return _device_materialize(counts, _bounded_device_copies(counts, config), config, capacity_limit)


def validate_planner_backend(backend: str, minimum_rows: int, cost_model: object) -> None:
    """Reject silently mixing host-only quota heuristics into device execution."""
    if backend not in ("cpu", "device"):
        raise ValueError("replica_planner must be cpu or device")
    if backend == "device" and (minimum_rows or cost_model is not None):
        raise ValueError("Device constructive planning requires replica_min_rows=0 and no host cost model")
