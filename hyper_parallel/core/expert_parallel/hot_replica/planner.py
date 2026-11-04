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
from dataclasses import dataclass, replace
from heapq import heapify, heappop, heappush
from itertools import chain
from math import isfinite
import os

from .admission import reduction_basis_points
from .capacity import ExpertReplicaConfig, _integer
from .plan import ExpertExecutionPlan
from .cost import ExpertReplicaCostModel, _QuotaCostState, refine_replica_quotas
from .topology import _validate_rank_nodes
from .host import _validate_counts

_COMPLETE_GUEST_HOTSPOT_FACTOR = 3


def _counts_matrix(counts: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Freeze and validate integer source histograms."""
    rows = tuple(tuple(row) for row in counts)
    if not rows or not rows[0]:
        raise ValueError("counts_by_source must be a nonempty rectangular matrix")
    if _validate_counts(rows):
        return rows
    for row in rows:
        if len(row) != len(rows[0]):
            raise ValueError("counts_by_source must be rectangular")
        for count in row:
            if type(count) is not int or count < 0:
                _integer(count, "expert count", 0)
    return rows


def _balanced_edges(loads: list[int], expert_counts: list[int], home: int) -> list:
    """Fill each receiver once, always sourcing original home experts."""
    rank_count = len(loads)
    quotient, remainder = divmod(sum(loads), rank_count)
    targets = [quotient + int(rank < remainder) for rank in range(rank_count)]
    remaining = list(expert_counts)
    excess = [load - target for load, target in zip(loads, targets)]
    edges = []
    while any(excess):
        donor = excess.index(max(excess))
        receiver = excess.index(min(excess))
        amount = -excess[receiver]
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
        excess[donor] -= amount
        excess[receiver] = 0
    return edges


def _bounded_copies(loads: list[int], expert_counts: list[int], config: ExpertReplicaConfig,
                    *, edges: list | None = None) -> list[dict[int, int]]:
    """Thin balanced edges to fit B distinct guests without changing ownership."""
    home = config.home_experts
    budget = min(config.replica_slots_per_rank, home)
    copies = [{expert: expert_counts[expert] for expert in range(rank * home, (rank + 1) * home)}
              for rank in range(config.ep_size)]
    if not budget:
        return copies
    for donor, receiver, segments in (_balanced_edges(loads, expert_counts, home) if edges is None else edges):
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
    guest_counts = [sum(expert // home != rank for expert in row) for rank, row in enumerate(copies)]
    existing_by_owner = [[] for _ in range(config.ep_size)]
    for receiver, row in enumerate(copies):
        for expert in row:
            if expert // home != receiver:
                existing_by_owner[expert // home].append((receiver, expert))
    donors = [(-load, rank) for rank, load in enumerate(loads) if load > target]
    heapify(donors)
    while donors:
        _, donor = heappop(donors)
        free = [(rank, target - load) for rank, load in enumerate(loads)
                if load < target and guest_counts[rank] < config.replica_slots_per_rank]
        headroom = max((rows for _, rows in free), default=0)
        best_score = (0, 0, 0, 0)
        best = None
        excess = loads[donor] - target
        experts = range(donor * home, (donor + 1) * home)
        # Over free receivers, max min(quota, excess, headroom) separates.
        # Rank/expert ascending order preserves the original tie-break.
        rows = min(max(copies[donor][expert] for expert in experts), excess, headroom)
        if rows:
            receiver = next(rank for rank, room in free if room >= rows)
            expert = next(expert for expert in experts if copies[donor][expert] >= rows)
            best_score = rows, int(expert in copies[receiver]), -receiver, -expert
            best = receiver, expert, rows
        for receiver, expert in existing_by_owner[donor]:
            rows = min(copies[donor][expert], excess, target - loads[receiver])
            score = rows, 1, -receiver, -expert
            if rows > 0 and score > best_score:
                best_score = score
                best = receiver, expert, rows
        if best is None:
            # Headroom and free slots only shrink; another owner cannot create
            # an existing copy for this donor. An infeasible donor stays so.
            continue
        receiver, expert, rows = best
        copies[donor][expert] -= rows
        if expert not in copies[receiver]:
            guest_counts[receiver] += 1
            existing_by_owner[donor].append((receiver, expert))
        copies[receiver][expert] = copies[receiver].get(expert, 0) + rows
        loads[donor] -= rows
        loads[receiver] += rows
        if loads[donor] > target:
            heappush(donors, (-loads[donor], donor))


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
                      expert: int, rows: int, limit: int,
                      copies_by_owner: list[list[tuple[int, int]]]) -> list[tuple[int, int, int]] | None:
    """Concentrate necessary work in existing copies before removing a small copy."""
    owner = expert // home
    needed = max(loads[owner] + rows - limit, 0)
    if not needed:
        return []
    if not copies_by_owner:
        # Direct removals need no search. Once consolidation is needed, its
        # index remains complete because pruning never creates guest slots.
        copies_by_owner.extend([] for _ in copies)
        for rank, row in enumerate(copies):
            for logical, count in row.items():
                logical_owner = logical // home
                if logical_owner != rank and count > 0:
                    copies_by_owner[logical_owner].append((rank, logical))
    available = dict(copies[owner])
    available[expert] += rows
    existing = sorted((-copies[rank][logical], rank, logical) for rank, logical in copies_by_owner[owner]
                      if copies[rank].get(logical, 0) > 0 and (rank, logical) != (target, expert))
    headroom = {rank: limit - loads[rank] + (rows if rank == target else 0) for _, rank, _ in existing}
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
    copies_by_owner = []
    for _, target, expert in candidates:
        rows = copies[target].get(expert, 0)
        if not 0 < rows < minimum_rows:
            continue
        replacements = _replacement_rows(copies, current, home, target, expert, rows, limit,
                                         copies_by_owner)
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


def _within_transfer_budget(
    copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...],
    changes: dict[tuple[int, int], int], rank_nodes: tuple[int, ...], node_ranks: dict[int, list[int]],
    credits: tuple[int, int],
) -> bool:
    """Bound changed token traffic by the deleted matrix edge's conservative row credits."""
    remote_delta = 0
    for (rank, expert), change in changes.items():
        deficit = counts[rank][expert] - copies[rank].get(expert, 0)
        remote_delta += max(deficit - change, 0) - max(deficit, 0)
    if remote_delta > credits[0]:
        return False
    node_changes = {}
    if node_ranks:
        for (rank, expert), change in changes.items():
            key = rank_nodes[rank], expert
            node_changes[key] = node_changes.get(key, 0) + change
    cross_delta = 0
    for (node, expert), change in node_changes.items():
        if change:
            deficit = sum(counts[rank][expert] - copies[rank].get(expert, 0) for rank in node_ranks[node])
            cross_delta += max(deficit - change, 0) - max(deficit, 0)
    return cross_delta <= credits[1]


def _consolidate_copies(copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...],
                        config: ExpertReplicaConfig, rank_nodes: tuple[int, ...],
                        transfer_row_credit: int = 0) -> None:
    """Delete redundant matrices without increasing receive peak or the traffic budget.

    Existing guests of the same owner may take over returned work. No new edge
    is created, so each owner's fanout and both local and remote matrix payloads
    are nonincreasing. Source and node deficits match the materializer's minimum
    remote-row counts without rebuilding a full dispatch table for every trial.
    Without runtime shape credits, neither remote-row count may increase.
    """
    home = config.home_experts
    loads = [sum(row.values()) for row in copies]
    ceiling = max(loads)
    by_owner = [[] for _ in copies]
    for rank, row in enumerate(copies):
        for expert, rows in row.items():
            if rows and expert // home != rank:
                by_owner[expert // home].append((rank, expert))
    candidates = sorted(
        (-int(bool(rank_nodes) and rank_nodes[expert // home] != rank_nodes[rank]),
         -len(by_owner[expert // home]), copies[rank][expert], rank, expert)
        for rank, expert in chain.from_iterable(by_owner))
    node_ranks = {}
    if 1 < len(set(rank_nodes)) < config.ep_size:
        for rank, node in enumerate(rank_nodes):
            node_ranks.setdefault(node, []).append(rank)
    for negative_remote, _, _, target, expert in candidates:
        rows = copies[target][expert]
        replacements = _replacement_rows(copies, loads, home, target, expert, rows, ceiling, by_owner)
        if replacements is None:
            continue
        owner = expert // home
        changes = {(owner, expert): rows, (target, expert): -rows}
        for rank, logical, moved in replacements:
            key = owner, logical
            changes[key] = changes.get(key, 0) - moved
            key = rank, logical
            changes[key] = changes.get(key, 0) + moved
        cross_credit = -negative_remote * transfer_row_credit
        if not _within_transfer_budget(copies, counts, changes, rank_nodes, node_ranks,
                                       (transfer_row_credit, cross_credit)):
            continue
        for (rank, logical), change in changes.items():
            count = copies[rank].get(logical, 0) + change
            if count or logical // home == rank:
                copies[rank][logical] = count
            else:
                del copies[rank][logical]
            loads[rank] += change


def _rebalance_existing_copies(copies: list[dict[int, int]], config: ExpertReplicaConfig) -> None:
    """Reduce pairwise load peaks using only already retained guest experts.

    Each move transfers original home work, stops before reversing the load
    difference, and preserves every guest slot. Thus the global receive peak
    cannot increase, and the integer sum of squared loads strictly decreases.
    """
    home = config.home_experts
    loads = [sum(row.values()) for row in copies]
    edges = [(target, expert) for target, row in enumerate(copies)
             for expert, rows in sorted(row.items()) if expert // home != target and rows > 0]
    while True:
        best = None
        best_score = (0, 0, 0, 0)
        for target, expert in edges:
            owner = expert // home
            rows = min(copies[owner][expert], (loads[owner] - loads[target]) // 2)
            score = (rows, loads[owner], -target, -expert)
            if rows > 0 and score > best_score:
                best_score = score
                best = (owner, target, expert, rows)
        if best is None:
            return
        owner, target, expert, rows = best
        copies[owner][expert] -= rows
        copies[target][expert] += rows
        loads[owner] -= rows
        loads[target] += rows


def _fill_node_quotas(quotas: list[list[int]], pending: list[int], dispatch: list[list[int]],
                      width: int, rank_nodes: tuple[int, ...], sources: dict[int, list[int]]) -> None:
    """Saturate node-local intersections after preserving all rank-local rows.

    For fixed expert quotas, each node must export at least its positive source
    surplus. Filling every local intersection attains that bound simultaneously
    with the rank-local bound, without changing any compute or replica quota.
    """
    destinations = {}
    for quota in quotas:
        if quota[1]:
            destinations.setdefault(rank_nodes[quota[0] // width], []).append(quota)
    for node, local in destinations.items():
        index = 0
        for source in sources[node]:
            while pending[source] and index < len(local):
                quota = local[index]
                moved = min(pending[source], quota[1])
                dispatch[source][quota[0]] += moved
                pending[source] -= moved
                quota[1] -= moved
                if not quota[1]:
                    index += 1


def _materialize(counts: tuple[tuple[int, ...], ...], copies: list[dict[int, int]],
                 config: ExpertReplicaConfig, capacity_limit: int | None = None,
                 *, rank_nodes: tuple[int, ...] = ()) -> ExpertExecutionPlan:
    """Assign each logical expert's source occurrences to final copy quotas."""
    home = config.home_experts
    width = config.slots_per_rank
    if all(expert // home == rank or not count for rank, row in enumerate(copies)
           for expert, count in row.items()):
        # No placement choice remains. Validated source histograms already
        # provide exact quotas; expanding and redistributing them is redundant.
        slots = tuple(tuple(range(rank * home, (rank + 1) * home)) + (-1,) * config.replica_slots_per_rank
                      for rank in range(config.ep_size))
        padding = (0,) * config.replica_slots_per_rank
        starts = range(0, config.num_experts, home)
        dispatch = tuple(tuple(chain.from_iterable(row[start:start + home] + padding for start in starts))
                         for row in counts)
        plan = ExpertExecutionPlan(config, counts, slots, dispatch)
        if capacity_limit is not None and max(plan.destination_loads) > capacity_limit:
            raise ValueError("replica plan exceeds capacity_limit; use a feasible receive bound")
        return plan
    node_sources = {}
    if rank_nodes and 1 < len(set(rank_nodes)) < config.ep_size:
        for rank, node in enumerate(rank_nodes):
            node_sources.setdefault(node, []).append(rank)
    slot_map = []
    by_expert = [[] for _ in range(config.num_experts)]
    for rank, row in enumerate(copies):
        guests = sorted(expert for expert, count in row.items() if expert // home != rank and count)
        if len(guests) > config.replica_slots_per_rank:
            raise ValueError("replica layout exceeds the physical guest slot cap")
        slots = list(range(rank * home, (rank + 1) * home)) + guests
        slots.extend([-1] * (width - len(slots)))
        slot_map.append(tuple(slots))
        for slot, expert in enumerate(slots):
            if expert >= 0 and row.get(expert, 0):
                by_expert[expert].append([rank * width + slot, row[expert]])
    padding = (0,) * config.replica_slots_per_rank
    starts = range(0, config.num_experts, home)
    dispatch = [list(chain.from_iterable(row[start:start + home] + padding for start in starts)) for row in counts]
    for expert, quotas in enumerate(by_expert):
        # Empty experts already have zero source counts in the owner layout.
        if not quotas:
            continue
        owner_physical = expert // home * width + expert % home
        if len(quotas) == 1 and quotas[0][0] == owner_physical:
            continue
        for output_row in dispatch:
            output_row[owner_physical] = 0
        if len(quotas) == 1:
            physical = quotas[0][0]
            for output_row, input_row in zip(dispatch, counts):
                output_row[physical] = input_row[expert]
            continue
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
        if node_sources:
            _fill_node_quotas(quotas, pending_by_source, dispatch, width, rank_nodes, node_sources)
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
    # Inputs are validated at entry and source quotas are conserved by this
    # construction. Full plan.validate() remains available to diagnostics/tests.
    if capacity_limit is not None and max(plan.destination_loads) > capacity_limit:
        raise ValueError("replica plan exceeds capacity_limit; use a feasible receive bound")
    return plan


def _home_layout(counts, config):
    """Create original owner quotas and rank loads from exact integer counts."""
    totals = list(map(sum, zip(*counts)))
    home = config.home_experts
    copies = [{expert: totals[expert] for expert in range(rank * home, (rank + 1) * home)}
              for rank in range(config.ep_size)]
    return totals, [sum(row.values()) for row in copies], copies


@dataclass(frozen=True)
class _UncalibratedCandidate:
    """Own admitted quotas until full source materialization consumes them."""

    config: ExpertReplicaConfig
    totals: tuple[int, ...]
    copies: list[dict[int, int]]
    limit: int
    resident: int
    target: int
    minimum_rows: int = 0
    admission: int = 0
    saved_rows: int = 0

    def matches(self, totals: list[int], config: ExpertReplicaConfig,
                limit: int, resident: int, target: int, minimum_rows: int = 0,
                admission: int = 0, saved_rows: int = 0) -> bool:
        """Reject reuse if the source totals or agreed allocation policy changed."""
        return (self.config == config and self.totals == tuple(totals)
                and (self.limit, self.resident, self.target, self.minimum_rows, self.admission, self.saved_rows)
                == (limit, resident, target, minimum_rows, admission, saved_rows))


@dataclass
class _QuotaSearchPrefix:
    """Retain the unmodified common budget prefix for one wider search."""

    next_budget: int = 1
    edges: list | None = None
    fallback: tuple[int, list[dict[int, int]] | None] | None = None
    reached_target: bool = False


def _uncalibrated_copies(totals: list[int], loads: list[int], config: ExpertReplicaConfig,
                         limit: int, resident: int, target: int, *,
                         prefix: _QuotaSearchPrefix | None = None) -> list[dict[int, int]] | None:
    """Select source-independent quotas, using None for unchanged owner placement."""
    owner_peak = max(loads)
    cutoff = min(target, resident)
    if owner_peak <= cutoff:
        if prefix is not None:
            prefix.reached_target = True
        return None
    start = prefix.next_budget if prefix is not None else 1
    stop = min(config.replica_slots_per_rank, config.home_experts)
    if start == 1:
        fallback = (owner_peak, None) if owner_peak <= limit else None
        edges = _balanced_edges(loads, totals, config.home_experts)
    else:
        fallback, edges = prefix.fallback, prefix.edges
    for budget in range(start, stop + 1):
        candidate = ExpertReplicaConfig(config.num_experts, config.ep_size, budget)
        copies = _bounded_copies(loads, totals, candidate, edges=edges)
        peak = max(sum(row.values()) for row in copies)
        if peak <= cutoff:
            if prefix is not None:
                prefix.reached_target = True
            return copies
        if peak <= limit and (fallback is None or peak < fallback[0]):
            fallback = peak, copies
    if prefix is not None and start == 1:
        prefix.next_budget = stop + 1
        prefix.edges = edges
        if fallback is not None:
            peak, selected = fallback
            prefix.fallback = (peak, None if selected is None else [dict(row) for row in selected])
    if fallback is None:
        # Thinning can exceed a hard limit before spare guest slots are filled.
        # Complete the bounded candidate once before declaring it infeasible.
        copies = _bounded_copies(loads, totals, config, edges=edges)
        _improve_copies(copies, config, cutoff)
        if max(sum(row.values()) for row in copies) <= limit:
            return copies
        raise ValueError("Adaptive replica cap cannot satisfy capacity_limit")
    selected = fallback[1]
    if selected is not None:
        _improve_copies(selected, config, cutoff)
    return selected


def _admitted_uncalibrated_copies(totals: list[int], loads: list[int], config: ExpertReplicaConfig,
                                  limit: int, resident: int, target: int, minimum_rows: int,
                                  admission: int, saved_rows: int, *,
                                  allow_complete_search: bool = False,
                                  projected_rank_nodes: tuple[int, ...] = (),
                                  prefix: _QuotaSearchPrefix | None = None) -> list[dict[int, int]] | None:
    """Admit feasible quotas using the selected uncalibrated placement policy."""
    wider_cap = config.replica_slots_per_rank == 2 and config.home_experts > 2
    search_wider = wider_cap and (allow_complete_search or projected_rank_nodes)
    average = (sum(loads) + config.ep_size - 1) // config.ep_size if search_wider else target
    project = bool(projected_rank_nodes) and wider_cap and target > average and min(loads) < max(loads) <= limit
    if project:
        prefix = _QuotaSearchPrefix()
    copies = _uncalibrated_copies(totals, loads, config, limit, resident, target, prefix=prefix)
    if copies is not None:
        peak = max(sum(row.values()) for row in copies)
        ceiling = min(limit, max(resident, peak))
        # A validated explicit limit makes fragment pruning source independent.
        _prune_small_copies(copies, (), loads, config, minimum_rows, ceiling)
        peak = max(sum(row.values()) for row in copies)
        if not _optional_peak_is_useful(peak, loads, ceiling, admission, saved_rows):
            copies = None
        elif (allow_complete_search and wider_cap and peak > target > average
              and max(loads) >= _COMPLETE_GUEST_HOTSPOT_FACTOR * average):
            complete = _complete_bounded_copies(totals, loads, config, copies, ceiling, minimum_rows, peak,
                                                edges=prefix.edges if prefix is not None else None)
            if complete is not None:
                copies = complete
    if project and not prefix.reached_target:
        projected = _project_wider_quota(totals, loads, config, copies, limit, resident, target,
                                        minimum_rows, admission, saved_rows, projected_rank_nodes, prefix)
        if projected is not None:
            return projected
    return copies


def _copy_pressure(copies: list[dict[int, int]], home: int, nodes: tuple[int, ...]) -> tuple[int, int]:
    """Bound concurrent matrix fanout and count cross-node copies per source node."""
    weighted_outgoing = [0] * len(copies)
    remote_by_node: dict[int, int] = {}
    for target, row in enumerate(copies):
        for expert, count in row.items():
            owner = expert // home
            if count and owner != target:
                remote = nodes[owner] != nodes[target]
                weighted_outgoing[owner] += 1 + int(remote)
                if remote:
                    remote_by_node[nodes[owner]] = remote_by_node.get(nodes[owner], 0) + 1
    return max(weighted_outgoing), max(remote_by_node.values(), default=0)


def _project_wider_quota(totals: list[int], loads: list[int], config: ExpertReplicaConfig,
                         current: list[dict[int, int]] | None, limit: int, resident: int, target: int,
                         minimum_rows: int, admission: int, saved_rows: int,
                         nodes: tuple[int, ...], prefix: _QuotaSearchPrefix) -> list[dict[int, int]] | None:
    """Try wider quota selection without consuming more physical guest slots."""
    grain = max(1, minimum_rows)
    home = config.home_experts
    current_peak = max(sum(row.values()) for row in current) if current is not None else max(loads)
    average = (sum(loads) + config.ep_size - 1) // config.ep_size
    if current_peak - average < grain:
        return None
    virtual = replace(config, replica_slots_per_rank=home)
    proposal = _admitted_uncalibrated_copies(totals, loads, virtual, limit, resident, target, minimum_rows,
                                            admission, saved_rows, prefix=prefix)
    if proposal is None or any(sum(bool(count) and expert // home != rank for expert, count in row.items())
                               > config.replica_slots_per_rank for rank, row in enumerate(proposal)):
        return None
    saved = current_peak - max(sum(row.values()) for row in proposal)
    current_expert_peak = max(max(row.values()) for row in current) if current is not None else max(totals)
    if saved < grain or max(max(row.values()) for row in proposal) > current_expert_peak:
        return None
    old_rank, old_node = _copy_pressure(current, home, nodes) if current is not None else (0, 0)
    new_rank, new_node = _copy_pressure(proposal, home, nodes)
    # A fixed matrix copy must earn extra peak-row savings. Cross-node fanout
    # costs twice the rank credit; take the larger rank or node increment.
    penalty = max(0, new_rank - old_rank, new_node - old_node)
    return proposal if saved >= grain + penalty * ((grain + 1) // 2) else None


def _projected_rank_nodes(nodes: tuple[int, ...]) -> tuple[int, ...]:
    """Expose the topology only to the isolated no-table projection experiment."""
    return nodes if os.environ.get("HYPER_PARALLEL_REPLICA_PROJECTED_GUEST_SEARCH") == "1" else ()


def _complete_guest_search_enabled() -> bool:
    """Enable the isolated no-table candidate without changing default decisions."""
    return os.environ.get("HYPER_PARALLEL_REPLICA_COMPLETE_GUEST_SEARCH") == "1"


def _complete_bounded_copies(totals: list[int], loads: list[int], config: ExpertReplicaConfig,
                             current: list[dict[int, int]], ceiling: int, minimum_rows: int,
                             peak: int, *, edges: list | None = None) -> list[dict[int, int]] | None:
    """Keep a complete quota only when every receiver already fits the physical cap."""
    home = config.home_experts
    if edges is None:
        edges = _balanced_edges(loads, totals, home)
    # The balancing skeleton emits at most one edge per receiver. There is no
    # need to shrink its rows when the full edge fits the reserved guest slots.
    if any(len(segments) > config.replica_slots_per_rank for _, _, segments in edges):
        return None
    candidate = _bounded_copies(loads, totals, replace(config, replica_slots_per_rank=home), edges=edges)
    _prune_small_copies(candidate, (), loads, config, minimum_rows, ceiling)
    if max(sum(row.values()) for row in candidate) + max(1, minimum_rows) > peak:
        return None
    if max(max(row.values()) for row in candidate) > max(max(row.values()) for row in current):
        return None
    return candidate


def _adaptive_copies(counts: tuple[tuple[int, ...], ...], config: ExpertReplicaConfig,
                     limit: int, receive_capacity: int | None,
                     model: ExpertReplicaCostModel | None, target: int,
                     minimum_replica_rows: int = 0,
                     durations: dict[int, tuple[float, float]] | None = None,
                     admission: int = 0, minimum_saved_rows: int = 0,
                     rank_nodes: tuple[int, ...] = (),
                     ) -> tuple[list[dict[int, int]], tuple[float, float] | None]:
    """Return bounded quotas and their optional (score, optimistic lower bound)."""
    totals, loads, home = _home_layout(counts, config)
    resident = min(limit, receive_capacity) if receive_capacity is not None else limit
    if model is None:
        selected = _admitted_uncalibrated_copies(totals, loads, config, limit, resident, target,
                                                minimum_replica_rows, admission, minimum_saved_rows,
                                                allow_complete_search=_complete_guest_search_enabled(),
                                                projected_rank_nodes=_projected_rank_nodes(rank_nodes))
        return home if selected is None else selected, None
    measured = min(model.forward_ms[-1][0], model.backward_ms[-1][0])
    durations = {} if durations is None else durations

    def score(copies: list[dict[int, int]]) -> float | None:
        """Exclude candidates outside measured compute coverage."""
        coverage = measured
        if model.critical_path is not None and not any(
                rows and expert // config.home_experts != rank
                for rank, row in enumerate(copies) for expert, rows in row.items()):
            coverage = model.critical_path.owner_coverage
        if max(rows for row in copies for rows in row.values()) > coverage:
            return None
        return _QuotaCostState(model, copies, counts, config, durations).score()

    best = home if max(loads) <= resident else None
    best_score = score(home) if best is not None else None
    margin = max(model.minimum_gain_ms, model.minimum_replica_gain_ms or 0)
    baseline_score = best_score
    if model.critical_path is not None:
        margin = max(margin, 2 * model.critical_path.uncertainty_ms)
        if best_score is not None:
            baseline_score = model.critical_path.estimate(model, home, counts, config, baseline=True).total_ms
    average = sum(loads) / config.ep_size
    rates = [min(duration / rows for rows, duration in table if rows)
             for table in (model.forward_ms, model.backward_ms)]
    optimistic = (model.forward_repeats * (rates[0] * average + model.planner_overhead_ms
                                         + model.replica_overhead_ms) + rates[1] * average)
    # The old separable lower bound does not describe shared resource queues.
    if model.critical_path is not None:
        optimistic = float('-inf')

    def improves(value: float | None) -> bool:
        if value is None:
            return False
        reference = best_score
        if baseline_score is not None and model.critical_path is not None:
            reference = min(baseline_score, best_score) if best_score is not None else baseline_score
        return reference is None or value + margin < reference
    if best_score is not None and best_score <= optimistic + margin:
        return best, (best_score, optimistic)
    fallback = (max(loads), 0, home) if max(loads) <= limit else None
    edges = _balanced_edges(loads, totals, config.home_experts)
    completed_copies = None
    completed_score = None
    deferred = []
    for budget in range(1, min(config.replica_slots_per_rank, config.home_experts) + 1):
        candidate_config = ExpertReplicaConfig(config.num_experts, config.ep_size, budget)
        copies = _bounded_copies(loads, totals, candidate_config, edges=edges)
        peak = max(sum(row.values()) for row in copies)
        guests = sum(expert // config.home_experts != rank for rank, row in enumerate(copies) for expert in row)
        overflow = peak > limit and model.critical_path is None
        if peak <= limit and (fallback is None or (peak, guests) < fallback[:2]):
            fallback = peak, guests, copies
        if peak <= resident:
            candidate_score = score(copies)
            if improves(candidate_score):
                best, best_score = copies, candidate_score
        # The bounded skeleton deliberately keeps thin guest quotas. Admission
        # must also compare the completed load policy before rejecting replicas.
        # Complete overflowing seeds independently, preserving the original
        # feasible candidate chain and its gain-margin comparisons.
        previous_completed = None if overflow else completed_copies
        completed = [dict(row) for row in (copies if previous_completed is None else previous_completed)]
        _improve_copies(completed, candidate_config, min(target, resident))
        completed_peak = max(sum(row.values()) for row in completed)
        ceiling = min(limit, max(receive_capacity or 0, completed_peak)) if receive_capacity is not None else limit
        _prune_small_copies(completed, counts, loads, candidate_config, minimum_replica_rows, ceiling)
        if not overflow:
            completed_copies = completed
        completed_peak = max(sum(row.values()) for row in completed)
        if completed_peak <= limit:
            completed_guests = sum(expert // config.home_experts != rank and bool(rows)
                                   for rank, row in enumerate(completed) for expert, rows in row.items())
            if fallback is None or (completed_peak, completed_guests) < fallback[:2]:
                fallback = completed_peak, completed_guests, completed
        if completed_peak <= resident:
            # More slots can leave the completion unchanged. Preserve insertion
            # order as well as quotas because it determines floating-point sums.
            unchanged = (previous_completed is not None and completed == previous_completed
                         and all(tuple(row) == tuple(previous) for row, previous in zip(completed, previous_completed)))
            candidate_score = completed_score if unchanged else score(completed)
            if overflow:
                deferred.append((completed, candidate_score))
            else:
                completed_score = candidate_score
                if improves(candidate_score):
                    best, best_score = completed, candidate_score
        if best_score is not None and best_score <= optimistic + margin:
            break
    for completed, candidate_score in deferred:
        if improves(candidate_score):
            best, best_score = completed, candidate_score
    if (best is home and best_score is not None and isfinite(best_score)
            and baseline_score is not None and isfinite(baseline_score)
            and (model.effective_group_ms or model.critical_path is not None)
            and model.quota_search_budget_ms > 0):
        retry_target = (sum(loads) + config.ep_size - 1) // config.ep_size
        if (retry_target < target
                and _optional_peak_is_useful(retry_target, loads, resident, admission, minimum_saved_rows)):
            # Charge extra search on top of the uncertainty margin, against the
            # feasible owner route that would otherwise execute.
            retry = _admitted_uncalibrated_copies(totals, loads, config, limit, resident, retry_target,
                                                  minimum_replica_rows, admission, minimum_saved_rows)
            if retry is not None and max(sum(row.values()) for row in retry) <= resident:
                retry_score = score(retry)
                if (retry_score is not None and isfinite(retry_score)
                        and improves(retry_score + model.quota_search_budget_ms)):
                    best, best_score = retry, retry_score
    if best_score is not None:
        return best, None if model.critical_path is not None else (best_score, optimistic)
    if fallback is None:
        raise ValueError("Adaptive replica cap cannot satisfy capacity_limit")
    return fallback[2], None


def _locality_copies(selected: list[dict[int, int]], config: ExpertReplicaConfig,
                     rank_nodes: tuple[int, ...]) -> list[dict[int, int]]:
    """Move, merge or swap whole guest fragments without adding replica edges or receive peak."""
    copies = [dict(row) for row in selected]
    home = config.home_experts
    current = [sum(row.values()) for row in copies]
    ceiling = max(current)
    guests = [sum(expert // home != rank and bool(rows) for expert, rows in row.items())
              for rank, row in enumerate(copies)]
    edges = sorted((-rows, rank, expert) for rank, row in enumerate(copies) for expert, rows in row.items()
                   if rows and expert // home != rank and rank_nodes[expert // home] != rank_nodes[rank])
    for negative_rows, source, expert in edges:
        rows = -negative_rows
        if copies[source].get(expert) != rows:
            continue
        owner = expert // home
        local = sorted((rank for rank, node in enumerate(rank_nodes) if node == rank_nodes[owner]),
                       key=lambda rank: (expert not in copies[rank], current[rank], rank))
        moved = False
        for target in local:
            if (current[target] + rows <= ceiling and
                    (expert in copies[target] or guests[target] < config.replica_slots_per_rank)):
                if expert not in copies[target] and owner != target:
                    guests[target] += 1
                copies[target][expert] = copies[target].get(expert, 0) + rows
                del copies[source][expert]
                guests[source] -= 1
                current[source] -= rows
                current[target] += rows
                moved = True
                break
        if moved:
            continue
        best_score = (0, 0, 0, 0)
        best_swap = None
        for target in local:
            if expert in copies[target]:
                continue
            for other, amount in copies[target].items():
                if not amount or other // home == target or other in copies[source]:
                    continue
                saved = (1 + int(rank_nodes[other // home] != rank_nodes[target])
                         - int(rank_nodes[other // home] != rank_nodes[source]))
                left, right = current[source] - rows + amount, current[target] - amount + rows
                if saved > 0 and max(left, right) <= ceiling:
                    score = saved, -max(left, right), -target, -other
                    if score > best_score:
                        best_score = score
                        best_swap = target, other, amount
        if best_swap is not None:
            target, other, amount = best_swap
            del copies[source][expert]
            del copies[target][other]
            copies[source][other], copies[target][expert] = amount, rows
            current[source] += amount - rows
            current[target] += rows - amount
    return selected if copies == selected else copies


def _optional_peak_is_useful(candidate_peak, loads, limit, admission, minimum_saved_rows):
    """Keep required placements and compare optional load reduction using integers."""
    peak = max(loads)
    if (not admission and not minimum_saved_rows) or peak > limit:
        return True
    saved = peak - candidate_peak
    return saved >= minimum_saved_rows and saved * 10000 >= peak * admission


def _build_expert_replica_plan(counts_by_source: Sequence[Sequence[int]], replica_slots_per_rank: int,
                              *, target_load: int | None = None, minimum_replica_rows: int = 0,
                              capacity_limit: int | None = None,
                              cost_model: ExpertReplicaCostModel | None = None,
                              minimum_load_reduction: float = 0.0, minimum_saved_rows: int = 0,
                              receive_capacity: int | None = None,
                              adaptive: bool = False,
                              _rank_nodes: tuple[int, ...] = (),
                              _replica_transfer_row_credit: int = 0,
                              _uncalibrated_candidate: _UncalibratedCandidate | None = None) -> ExpertExecutionPlan:
    """Build a bounded single-execution plan, then reduce residual imbalance.

    Args:
        counts_by_source: Exact integer logical-expert histograms, one per rank.
        replica_slots_per_rank: Resident non-parameter guest slots on every rank.
        target_load: Preferred receive size, normally the current push capacity.
            It is a target, not a drop threshold: residual load remains lossless.
        minimum_replica_rows: Prefer removing copies below this threshold. Zero retains
            token-balancing placement. Mandatory copies remain when capacity requires them;
            without a cost model retained copies may take additional home rows. A cost
            model accepts deletions only without worsening its score or the receive peak.
        capacity_limit: Optional receive bound already derived from the invocation shape.
            Without shape metadata, use a bound safe for every compatible distinct-TopK
            shape, falling back to observed home loads for unequal source sizes.

        cost_model: Optional matching offline calibration for refining existing
            replica quotas. Uncalibrated row ranges retain the original policy.
        minimum_load_reduction: Required reduction of the receive peak for optional
            copies. Capacity-required copies are retained; zero preserves the original policy.
        minimum_saved_rows: Required absolute receive-peak reduction for optional copies.
            It covers fixed communication overhead independently of the fractional threshold.
        receive_capacity: Current globally agreed receive reservation, when known.
            Cost refinement cannot grow beyond it or the uncalibrated plan's peak.
            Necessary growth and the independent lossless capacity bound remain enabled.
        adaptive: Select actual guest quotas within the fixed physical slot cap using
            one shared balancing skeleton. Missing costs use load and capacity constraints.
        _replica_transfer_row_credit: Internal activation-row allowance from mandatory
            gradient bytes. Zero retains strict remote-row bounds during consolidation.
        _uncalibrated_candidate: Invocation-local quotas admitted using compact counts.
            Reuse requires the same histogram, configuration, and capacity policy.

    Returns:
        Immutable physical placement and exact per-source dispatch quotas.
    """
    _integer(minimum_replica_rows, "minimum_replica_rows", 0)
    _integer(minimum_saved_rows, "minimum_saved_rows", 0)
    _integer(_replica_transfer_row_credit, "Replica transfer row credit", 0)
    admission = reduction_basis_points(minimum_load_reduction)
    if capacity_limit is not None:
        _integer(capacity_limit, "capacity_limit", 0)
    if receive_capacity is not None:
        _integer(receive_capacity, "receive_capacity", 0)
    if cost_model is not None and not isinstance(cost_model, ExpertReplicaCostModel):
        raise ValueError("cost_model must be an ExpertReplicaCostModel")
    if type(adaptive) is not bool or (adaptive and not replica_slots_per_rank):
        raise ValueError("Adaptive replicas require a boolean flag and a positive slot cap")
    counts = _counts_matrix(counts_by_source)
    config = ExpertReplicaConfig(len(counts[0]), len(counts), replica_slots_per_rank)
    rank_nodes = _validate_rank_nodes(_rank_nodes, config.ep_size)
    if cost_model is not None:
        if rank_nodes and cost_model._rank_nodes and rank_nodes != cost_model._rank_nodes:
            raise ValueError('Replica topology differs from the cost model topology')
        rank_nodes = rank_nodes or cost_model._rank_nodes
        if rank_nodes and rank_nodes != cost_model._rank_nodes:
            cost_model = replace(cost_model, _rank_nodes=rank_nodes)
    if target_load is not None:
        _integer(target_load, "target_load", 0)
    expert_counts, loads, baseline = _home_layout(counts, config)
    if cost_model is not None and cost_model.ep_size != config.ep_size:
        raise ValueError("Replica cost model EP size does not match the plan")
    average = (sum(loads) + config.ep_size - 1) // config.ep_size
    target = max(average, target_load or 0)
    if adaptive:
        durations = {}
        limit = max(loads) if capacity_limit is None else capacity_limit
        # Equal owner loads produce no balancing edges or donor excess. No
        # calibrated candidate can differ from this already-feasible layout.
        if max(loads) <= limit and min(loads) == max(loads):
            return _materialize(counts, baseline, config, capacity_limit, rank_nodes=rank_nodes)
        if cost_model is None and not _optional_peak_is_useful(
                average, loads, min(limit, receive_capacity) if receive_capacity is not None else limit,
                admission, minimum_saved_rows):
            return _materialize(counts, baseline, config, capacity_limit, rank_nodes=rank_nodes)
        def refine(copies: list[dict[int, int]],
                   bounds: tuple[float, float] | None) -> list[dict[int, int]]:
            """Apply the same pruning and admission before comparing final placements."""
            # Every candidate was scored with the complete device resource graph.
            # Incremental two-rank scores cannot account for shared queue changes.
            if cost_model is not None and cost_model.critical_path is not None:
                return copies
            peak = max(sum(row.values()) for row in copies)
            ceiling = min(limit, max(receive_capacity or 0, peak)) if receive_capacity is not None else limit
            if cost_model is not None:
                small = any(0 < rows < minimum_replica_rows and expert // config.home_experts != rank
                            for rank, row in enumerate(copies) for expert, rows in row.items())
                # Admission already compared owner-only. Reuse its lower bound
                # when no cheap deletion or worthwhile quota search remains.
                search = (bounds is None or bounds[0] - bounds[1]
                          > cost_model.minimum_gain_ms + cost_model.quota_search_budget_ms)
                if small or search:
                    refine_replica_quotas(copies, counts, config, ceiling, cost_model,
                                          minimum_replica_rows=minimum_replica_rows, _durations=durations)
            else:
                _prune_small_copies(copies, counts, loads, config, minimum_replica_rows, ceiling)
            peak = max(sum(row.values()) for row in copies)
            return copies if _optional_peak_is_useful(peak, loads, ceiling, admission, minimum_saved_rows) else baseline

        def materialize(copies: list[dict[int, int]]) -> ExpertExecutionPlan:
            """Consolidate the final uncalibrated layout before assigning source quotas."""
            if cost_model is None:
                _consolidate_copies(copies, counts, config, rank_nodes, _replica_transfer_row_credit)
            return _materialize(counts, copies, config, capacity_limit, rank_nodes=rank_nodes)

        resident = min(limit, receive_capacity) if receive_capacity is not None else limit
        if (_uncalibrated_candidate is not None and cost_model is None
                and _uncalibrated_candidate.matches(expert_counts, config, limit, resident, target,
                                                    minimum_replica_rows, admission, minimum_saved_rows)):
            copies, bounds = _uncalibrated_candidate.copies, None
        else:
            copies, bounds = _adaptive_copies(counts, config, limit, receive_capacity, cost_model, target,
                                              minimum_replica_rows, durations, admission, minimum_saved_rows,
                                              rank_nodes)
        if cost_model is not None:
            copies = refine(copies, bounds)
        has_guests = any(expert // config.home_experts != rank and rows
                         for rank, row in enumerate(copies) for expert, rows in row.items())
        if has_guests and rank_nodes and 1 < len(set(rank_nodes)) < config.ep_size:
            proposal = _locality_copies(copies, config, rank_nodes)
            if proposal is copies:
                return materialize(copies)
            peak = max(sum(row.values()) for row in copies)
            ceiling = min(limit, max(receive_capacity or 0, peak))
            alternative = proposal
            if alternative == copies:
                return materialize(copies)
            alternate_peak = max(sum(row.values()) for row in alternative)

            def key(layout: list[dict[int, int]]) -> tuple[float, int]:
                """Keep calibrated latency first; uncalibrated placements keep receive peak first."""
                remote = sum(bool(rows) and expert // config.home_experts != rank
                             and rank_nodes[expert // config.home_experts] != rank_nodes[rank]
                             for rank, row in enumerate(layout) for expert, rows in row.items())
                value = (max(sum(row.values()) for row in layout) if cost_model is None else
                         _QuotaCostState(cost_model, layout, counts, config, durations).score())
                return value, remote

            covered = (cost_model is None or max(rows for layout in (copies, alternative)
                       for row in layout for rows in row.values()) <=
                       min(cost_model.forward_ms[-1][0], cost_model.backward_ms[-1][0]))
            if (alternate_peak <= ceiling and covered and
                    _optional_peak_is_useful(alternate_peak, loads, ceiling, admission, minimum_saved_rows)):
                after, before = key(alternative), key(copies)
                if after < before:
                    copies = alternative
        return materialize(copies)
    limit = _pruning_capacity(counts, loads, config) if capacity_limit is None else capacity_limit
    if (not replica_slots_per_rank or max(loads) <= target
            or not _optional_peak_is_useful(average, loads, limit, admission, minimum_saved_rows)):
        return _materialize(counts, baseline, config, capacity_limit, rank_nodes=rank_nodes)
    copies = _bounded_copies(loads, expert_counts, config)
    if max(sum(row.values()) for row in copies) > max(loads):
        copies = [dict(row) for row in baseline]
    _improve_copies(copies, config, target)
    if cost_model is None:
        _prune_small_copies(copies, counts, loads, config, minimum_replica_rows, capacity_limit)
        if minimum_replica_rows:
            _rebalance_existing_copies(copies, config)
    if cost_model is not None:
        if receive_capacity is not None:
            peak = max(sum(row.values()) for row in copies)
            limit = min(limit, max(receive_capacity, peak))
        refine_replica_quotas(copies, counts, config, limit, cost_model,
                              minimum_replica_rows=minimum_replica_rows)
    candidate_peak = max(sum(row.values()) for row in copies)
    if not _optional_peak_is_useful(candidate_peak, loads, limit, admission, minimum_saved_rows):
        copies = baseline
    return _materialize(counts, copies, config, capacity_limit, rank_nodes=rank_nodes)


def build_expert_replica_plan(counts_by_source: Sequence[Sequence[int]], replica_slots_per_rank: int,
                              *, target_load: int | None = None, minimum_replica_rows: int = 0,
                              capacity_limit: int | None = None, minimum_load_reduction: float = 0.0,
                              minimum_saved_rows: int = 0, receive_capacity: int | None = None) -> ExpertExecutionPlan:
    """Choose lossless adaptive quotas within a fixed per-rank physical slot cap.

    Counts-only callers use load and capacity constraints. Runtime adapters resolve
    matching execution costs internally; callers need no cost object or policy toggle.

    Args:
        counts_by_source: Exact integer expert counts for each source rank.
        replica_slots_per_rank: Physical guest slot cap; zero keeps owner-only placement.
        target_load: Preferred destination row count, defaulting to the group average.
        minimum_replica_rows: Soft minimum rows per copy; capacity-required copies remain.
        capacity_limit: Optional hard maximum rows received by any rank.
        minimum_load_reduction: Fractional peak reduction required for optional copies.
        minimum_saved_rows: Absolute peak reduction required for optional copies.
        receive_capacity: Current globally agreed receive reservation, when known.

    Returns:
        Immutable physical placement and exact per-source dispatch quotas.
    """
    return _build_expert_replica_plan(counts_by_source, replica_slots_per_rank, target_load=target_load,
        minimum_replica_rows=minimum_replica_rows, capacity_limit=capacity_limit,
        minimum_load_reduction=minimum_load_reduction, minimum_saved_rows=minimum_saved_rows,
        receive_capacity=receive_capacity, adaptive=bool(replica_slots_per_rank))
