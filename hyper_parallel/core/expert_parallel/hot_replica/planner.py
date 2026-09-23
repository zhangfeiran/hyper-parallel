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
"""Deterministic single-execution placement with a constructive capacity bound."""

from __future__ import annotations

from collections.abc import Sequence

from .capacity import ExpertReplicaConfig, _integer
from .plan import ExpertExecutionPlan


def _counts_matrix(counts: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Freeze and validate integer source histograms."""
    rows = tuple(tuple(row) for row in counts)
    if not rows or not rows[0]:
        raise ValueError("counts_by_source must be a nonempty rectangular matrix")
    for row in rows:
        if len(row) != len(rows[0]):
            raise ValueError("counts_by_source must be rectangular")
        for count in row:
            _integer(count, "expert count", 0)
    return rows


def _balanced_edges(loads: list[int], expert_counts: list[int], home: int) -> list:
    """Fill each receiver once, always sourcing original home experts."""
    rank_count = len(loads)
    quotient, remainder = divmod(sum(loads), rank_count)
    targets = [quotient + int(rank < remainder) for rank in range(rank_count)]
    remaining = list(expert_counts)
    current = list(loads)
    edges = []
    while current != targets:
        donor = max(range(rank_count), key=lambda rank: (current[rank] - targets[rank], -rank))
        receiver = min(range(rank_count), key=lambda rank: (current[rank] - targets[rank], rank))
        amount = targets[receiver] - current[receiver]
        pending = amount
        segments = []
        experts = sorted(range(donor * home, (donor + 1) * home), key=lambda expert: (-remaining[expert], expert))
        for expert in experts:
            moved = min(pending, remaining[expert])
            if moved:
                segments.append((expert, moved))
                remaining[expert] -= moved
                pending -= moved
        if pending:
            raise RuntimeError("balanced transfer exceeds remaining home work")
        edges.append((donor, receiver, segments))
        current[donor] -= amount
        current[receiver] += amount
    return edges


def _bounded_copies(loads: list[int], expert_counts: list[int], config: ExpertReplicaConfig) -> list[dict[int, int]]:
    """Thin balanced edges to fit B distinct guests without changing ownership."""
    home = config.home_experts
    budget = min(config.replica_slots_per_rank, home)
    copies = [{expert: expert_counts[expert] for expert in range(rank * home, (rank + 1) * home)}
              for rank in range(config.ep_size)]
    if not budget:
        return copies
    for donor, receiver, segments in _balanced_edges(loads, expert_counts, home):
        quota = budget * sum(count for _, count in segments) // home
        for expert, count in sorted(segments, key=lambda segment: (-segment[1], segment[0]))[:budget]:
            kept = min(count, quota)
            if kept:
                copies[donor][expert] -= kept
                copies[receiver][expert] = kept
                quota -= kept
        if quota:
            raise RuntimeError("top-B experts cannot satisfy bounded transfer quota")
    return copies


def _improve_copies(copies: list[dict[int, int]], config: ExpertReplicaConfig, target: int) -> None:
    """Use spare guest capacity only for moves that reduce a donor's excess."""
    loads = [sum(row.values()) for row in copies]
    home = config.home_experts
    while True:
        moved = False
        for donor in sorted(range(config.ep_size), key=lambda rank: (-loads[rank], rank)):
            if loads[donor] <= target:
                continue
            best_score = (0, 0, 0, 0)
            best = None
            for receiver in range(config.ep_size):
                if loads[receiver] >= target:
                    continue
                guest_count = sum(expert // home != receiver for expert in copies[receiver])
                for expert in range(donor * home, (donor + 1) * home):
                    existing = expert in copies[receiver]
                    if not existing and guest_count >= config.replica_slots_per_rank:
                        continue
                    rows = min(copies[donor][expert], loads[donor] - target, target - loads[receiver])
                    score = (rows, int(existing), -receiver, -expert)
                    if rows and score > best_score:
                        best_score = score
                        best = (receiver, expert, rows)
            if best is not None:
                receiver, expert, rows = best
                copies[donor][expert] -= rows
                copies[receiver][expert] = copies[receiver].get(expert, 0) + rows
                loads[donor] -= rows
                loads[receiver] += rows
                moved = True
                break
        if not moved:
            return


def _pruning_capacity(counts: tuple[tuple[int, ...], ...], loads: list[int],
                      config: ExpertReplicaConfig) -> int:
    """Use the smallest compatible TopK shape when only histograms are available."""
    totals = [sum(row) for row in counts]
    largest = max(max(row) for row in counts)
    if largest and len(set(totals)) == 1:
        # Distinct TopK implies S >= max(count) and K divides S*K. The largest
        # compatible K gives the smallest receive bound, so it is safe for
        # every original shape represented by these equal-size histograms.
        top_k = min(config.num_experts, totals[0] // largest)
        while totals[0] % top_k:
            top_k -= 1
        return config.maximum_receive_rows(totals[0] // top_k, top_k, alignment=1)
    home = config.home_experts
    budget = min(config.replica_slots_per_rank, home)
    average = (sum(loads) + config.ep_size - 1) // config.ep_size
    mixed = ((home - budget) * max(loads) + budget * average + home - 1) // home
    return average if budget == home else min(max(loads), mixed + config.ep_size - 1)


def _replacement_rows(copies: list[dict[int, int]], loads: list[int], home: int, target: int,
                      expert: int, rows: int, limit: int) -> list[tuple[int, int, int]] | None:
    """Concentrate necessary work in existing copies before removing a small copy."""
    owner = expert // home
    needed = max(loads[owner] + rows - limit, 0)
    available = dict(copies[owner])
    available[expert] += rows
    headroom = [limit - load + (rows if rank == target else 0) for rank, load in enumerate(loads)]
    existing = sorted((-count, rank, logical) for rank, row in enumerate(copies) if rank != owner
                      for logical, count in row.items() if logical // home == owner and count > 0
                      and (rank, logical) != (target, expert))
    replacements = []
    for _, rank, logical in existing:
        if not needed:
            break
        moved = min(needed, available[logical], headroom[rank])
        if moved > 0:
            replacements.append((rank, logical, moved))
            available[logical] -= moved
            headroom[rank] -= moved
            needed -= moved
    return None if needed else replacements


def _prune_small_copies(copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...],
                        loads: list[int], config: ExpertReplicaConfig, minimum_rows: int,
                        capacity_limit: int | None) -> None:
    """Remove or consolidate small replicas without exceeding the receive bound."""
    if not minimum_rows:
        return
    home = config.home_experts
    limit = min(max(loads), _pruning_capacity(counts, loads, config) if capacity_limit is None else capacity_limit)
    current = [sum(row.values()) for row in copies]
    candidates = sorted((rows, target, expert) for target, row in enumerate(copies)
                        for expert, rows in row.items() if expert // home != target and 0 < rows < minimum_rows)
    for _, target, expert in candidates:
        rows = copies[target].get(expert, 0)
        if not 0 < rows < minimum_rows:
            continue
        replacements = _replacement_rows(copies, current, home, target, expert, rows, limit)
        if replacements is None:
            continue
        owner = expert // home
        copies[owner][expert] += rows
        del copies[target][expert]
        current[owner] += rows
        current[target] -= rows
        for rank, logical, moved in replacements:
            copies[owner][logical] -= moved
            copies[rank][logical] += moved
            current[owner] -= moved
            current[rank] += moved


def _materialize(counts: tuple[tuple[int, ...], ...], copies: list[dict[int, int]],
                 config: ExpertReplicaConfig, capacity_limit: int | None = None) -> ExpertExecutionPlan:
    """Assign each logical expert's source occurrences to final copy quotas."""
    home = config.home_experts
    width = config.slots_per_rank
    slot_map = []
    by_expert = [[] for _ in range(config.num_experts)]
    for rank, row in enumerate(copies):
        guests = sorted(expert for expert, count in row.items() if expert // home != rank and count)
        slots = list(range(rank * home, (rank + 1) * home)) + guests
        slots.extend([-1] * (width - len(slots)))
        slot_map.append(tuple(slots))
        for slot, expert in enumerate(slots):
            if expert >= 0 and row.get(expert, 0):
                by_expert[expert].append([rank * width + slot, row[expert]])
    dispatch = [[0] * config.physical_experts for _ in counts]
    for expert, quotas in enumerate(by_expert):
        pending_by_source = [row[expert] for row in counts]
        # Local source rows and destination quotas are independent per rank;
        # saturating their intersections minimizes remote rows for this placement.
        for quota in quotas:
            physical, remaining = quota
            source = physical // width
            local = min(pending_by_source[source], remaining)
            dispatch[source][physical] = local
            pending_by_source[source] -= local
            quota[1] -= local
        index = 0
        for source, pending in enumerate(pending_by_source):
            while pending:
                while quotas[index][1] == 0:
                    index += 1
                physical, remaining = quotas[index]
                rows = min(pending, remaining)
                dispatch[source][physical] += rows
                quotas[index][1] -= rows
                pending -= rows
    plan = ExpertExecutionPlan(config, counts, tuple(slot_map), tuple(tuple(row) for row in dispatch))
    plan.validate()
    if capacity_limit is not None and max(plan.destination_loads) > capacity_limit:
        raise ValueError("replica plan exceeds capacity_limit; use a feasible receive bound")
    return plan


def build_expert_replica_plan(counts_by_source: Sequence[Sequence[int]], replica_slots_per_rank: int,
                              *, target_load: int | None = None, minimum_replica_rows: int = 0,
                              capacity_limit: int | None = None) -> ExpertExecutionPlan:
    """Build a bounded single-execution plan, then reduce residual imbalance.

    Args:
        counts_by_source: Exact integer logical-expert histograms, one per rank.
        replica_slots_per_rank: Resident non-parameter guest slots on every rank.
        target_load: Preferred receive size, normally the current push capacity.
            It is a target, not a drop threshold: residual load remains lossless.
        minimum_replica_rows: Prefer removing copies below this threshold. Zero retains
            token-balancing placement. Mandatory copies remain when capacity requires them.
        capacity_limit: Optional receive bound already derived from the invocation shape.
            Without shape metadata, use a bound safe for every compatible distinct-TopK
            shape, falling back to observed home loads for unequal source sizes.

    Returns:
        Immutable physical placement and exact per-source dispatch quotas.
    """
    _integer(minimum_replica_rows, "minimum_replica_rows", 0)
    if capacity_limit is not None:
        _integer(capacity_limit, "capacity_limit", 0)
    counts = _counts_matrix(counts_by_source)
    config = ExpertReplicaConfig(len(counts[0]), len(counts), replica_slots_per_rank)
    if target_load is not None:
        _integer(target_load, "target_load", 0)
    expert_counts = [sum(row[expert] for row in counts) for expert in range(config.num_experts)]
    home = config.home_experts
    loads = [sum(expert_counts[rank * home:(rank + 1) * home]) for rank in range(config.ep_size)]
    average = (sum(loads) + config.ep_size - 1) // config.ep_size
    target = max(average, target_load or 0)
    baseline = [{expert: expert_counts[expert] for expert in range(rank * home, (rank + 1) * home)}
                for rank in range(config.ep_size)]
    if not replica_slots_per_rank or max(loads) <= target:
        return _materialize(counts, baseline, config, capacity_limit)
    copies = _bounded_copies(loads, expert_counts, config)
    if max(sum(row.values()) for row in copies) > max(loads):
        copies = baseline
    _improve_copies(copies, config, target)
    _prune_small_copies(copies, counts, loads, config, minimum_replica_rows, capacity_limit)
    return _materialize(counts, copies, config, capacity_limit)
