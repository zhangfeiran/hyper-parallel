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
"""Reference control plane for bounded hot-expert replication.

The planner deliberately contains no framework imports.  It defines the
logical-expert to physical-slot contract shared by a future device planner and
the MegaMoe runtime: every rank keeps its home experts, owns a bounded replica
pool, and leaves residual imbalance to lossless multi-wave execution.

MoonEP's CUDA implementation exposes ``E + B`` VM groups because every rank
can address every home weight through one virtual mapping. MegaMoe currently
uses rank-local GMM weights, so this Ascend-facing contract instead exposes
``E / R + B`` slots per rank and flattens them into ``E + R * B`` physical
route IDs. The planner semantics stay independent of that storage adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest


def _validate_nonnegative_integer(value: object, name: str) -> int:
    """Return one validated nonnegative integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")
    return value


def _normalize_counts(
    counts_by_source: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Validate and freeze a rank-major logical-expert count matrix."""
    rows = tuple(tuple(row) for row in counts_by_source)
    if not rows:
        raise ValueError("counts_by_source must contain at least one source rank.")
    num_experts = len(rows[0])
    if num_experts == 0:
        raise ValueError("counts_by_source must contain at least one expert.")
    for source_rank, row in enumerate(rows):
        if len(row) != num_experts:
            raise ValueError(
                "counts_by_source must be rectangular: "
                f"row 0 has {num_experts} entries but row {source_rank} has "
                f"{len(row)}."
            )
        for expert, count in enumerate(row):
            _validate_nonnegative_integer(
                count,
                f"counts_by_source[{source_rank}][{expert}]",
            )
    return rows


def _wave_count(maximum_load: int, receive_capacity: int) -> int:
    """Return the number of fixed-capacity windows required by one load."""
    if not isinstance(receive_capacity, int) or isinstance(receive_capacity, bool):
        raise ValueError(
            "receive_capacity must be a positive integer, "
            f"got {receive_capacity!r}."
        )
    if receive_capacity <= 0:
        raise ValueError(
            "receive_capacity must be a positive integer, "
            f"got {receive_capacity!r}."
        )
    return max(1, (maximum_load + receive_capacity - 1) // receive_capacity)


def _validate_slot_map(
    counts: tuple[tuple[int, ...], ...],
    replicas: int,
    slot_map: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Validate physical slots and return their flattened logical IDs."""
    ep_size = len(counts)
    num_experts = len(counts[0])
    if num_experts % ep_size:
        raise ValueError(
            f"logical expert count {num_experts} must be divisible by "
            f"ep_size {ep_size}."
        )
    local_experts = num_experts // ep_size
    slots_per_rank = local_experts + replicas
    if len(slot_map) != ep_size:
        raise ValueError("slot_to_logical_expert must contain one row per EP rank.")
    for rank, row in enumerate(slot_map):
        if len(row) != slots_per_rank:
            raise ValueError(
                f"slot_to_logical_expert[{rank}] must contain "
                f"{slots_per_rank} entries, got {len(row)}."
            )
        expected_home = tuple(
            range(rank * local_experts, (rank + 1) * local_experts)
        )
        if tuple(row[:local_experts]) != expected_home:
            raise ValueError(
                f"rank {rank} home slots must be {expected_home}, got "
                f"{tuple(row[:local_experts])}."
            )
        _validate_replica_slots(row, rank, local_experts, num_experts)
    return tuple(expert for rank_slots in slot_map for expert in rank_slots)


def _validate_replica_slots(
    rank_slots: tuple[int, ...],
    rank: int,
    local_experts: int,
    num_experts: int,
) -> None:
    """Validate one rank's replica suffix."""
    occupied = set(rank_slots[:local_experts])
    for slot, expert in enumerate(rank_slots[local_experts:], local_experts):
        if expert == -1:
            continue
        if not isinstance(expert, int) or isinstance(expert, bool):
            raise ValueError(
                f"physical slot ({rank}, {slot}) has invalid expert {expert!r}."
            )
        if not 0 <= expert < num_experts:
            raise ValueError(
                f"physical slot ({rank}, {slot}) expert {expert} is outside "
                f"[0, {num_experts})."
            )
        if expert in occupied:
            raise ValueError(
                f"rank {rank} contains logical expert {expert} more than once."
            )
        occupied.add(expert)


def _reconstruct_logical_counts(
    source_counts: tuple[int, ...],
    source_rank: int,
    logical_for_physical: tuple[int, ...],
    num_experts: int,
) -> tuple[int, ...]:
    """Validate one physical count row and reconstruct its logical histogram."""
    reconstructed = [0] * num_experts
    for physical_expert, count in enumerate(source_counts):
        count = _validate_nonnegative_integer(
            count,
            f"dispatch_counts[{source_rank}][{physical_expert}]",
        )
        logical_expert = logical_for_physical[physical_expert]
        if logical_expert == -1:
            if count:
                raise ValueError(
                    f"unused physical expert {physical_expert} received {count} routes."
                )
            continue
        reconstructed[logical_expert] += count
    return tuple(reconstructed)


def _validate_dispatch_counts(
    counts: tuple[tuple[int, ...], ...],
    logical_for_physical: tuple[int, ...],
    dispatch_counts: tuple[tuple[int, ...], ...],
) -> None:
    """Validate physical count shapes and per-source conservation."""
    ep_size = len(counts)
    num_experts = len(counts[0])
    physical_experts = len(logical_for_physical)
    if len(dispatch_counts) != ep_size:
        raise ValueError("dispatch_counts must contain one row per source rank.")
    for source_rank, row in enumerate(dispatch_counts):
        if len(row) != physical_experts:
            raise ValueError(
                f"dispatch_counts[{source_rank}] must contain "
                f"{physical_experts} entries, got {len(row)}."
            )
        reconstructed = _reconstruct_logical_counts(
            row,
            source_rank,
            logical_for_physical,
            num_experts,
        )
        if reconstructed != counts[source_rank]:
            raise ValueError(
                f"dispatch_counts[{source_rank}] does not conserve its "
                "logical expert histogram."
            )


@dataclass(frozen=True)
class ExpertReplicaTransfer:
    """One home-to-replica weight transfer for a placement generation.

    Attributes:
        logical_expert: Router-visible expert represented by the replica.
        owner_rank: EP rank that owns the optimizer parameter.
        owner_slot: Rank-local home slot on ``owner_rank``.
        target_rank: EP rank receiving the temporary replica.
        target_slot: Rank-local physical slot on ``target_rank``.
    """

    logical_expert: int
    owner_rank: int
    owner_slot: int
    target_rank: int
    target_slot: int


@dataclass(frozen=True)
class ExpertPlacementPlan:
    """Immutable logical-to-physical expert placement for one routing step.

    Attributes:
        logical_counts_by_source: Logical counts shaped ``[ep_size, E]``.
        replica_slots_per_rank: Bounded hot-expert slots available on each rank.
        slot_to_logical_expert: Logical expert in each physical slot, shaped
            ``[ep_size, E / ep_size + B]``. Unused replica slots contain ``-1``.
        dispatch_counts: Per-source physical counts shaped
            ``[ep_size, ep_size * (E / ep_size + B)]``.
    """

    logical_counts_by_source: tuple[tuple[int, ...], ...]
    replica_slots_per_rank: int
    slot_to_logical_expert: tuple[tuple[int, ...], ...]
    dispatch_counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """Validate the complete placement and count-conservation contract."""
        counts = _normalize_counts(self.logical_counts_by_source)
        replicas = _validate_nonnegative_integer(
            self.replica_slots_per_rank,
            "replica_slots_per_rank",
        )
        slot_map = tuple(tuple(row) for row in self.slot_to_logical_expert)
        dispatch_counts = tuple(tuple(row) for row in self.dispatch_counts)
        logical_for_physical = _validate_slot_map(
            counts,
            replicas,
            slot_map,
        )
        _validate_dispatch_counts(
            counts,
            logical_for_physical,
            dispatch_counts,
        )
        object.__setattr__(self, "logical_counts_by_source", counts)
        object.__setattr__(self, "replica_slots_per_rank", replicas)
        object.__setattr__(self, "slot_to_logical_expert", slot_map)
        object.__setattr__(self, "dispatch_counts", dispatch_counts)

    @property
    def ep_size(self) -> int:
        """Return the number of expert-parallel ranks."""
        return len(self.logical_counts_by_source)

    @property
    def logical_num_experts(self) -> int:
        """Return the router-visible expert count."""
        return len(self.logical_counts_by_source[0])

    @property
    def local_home_experts(self) -> int:
        """Return the number of optimizer-owned experts on each rank."""
        return self.logical_num_experts // self.ep_size

    @property
    def physical_experts_per_rank(self) -> int:
        """Return home plus replica slots participating in local GMM."""
        return self.local_home_experts + self.replica_slots_per_rank

    @property
    def physical_num_experts(self) -> int:
        """Return the flattened physical-ID count in the placement plan."""
        return self.ep_size * self.physical_experts_per_rank

    @property
    def replica_expert_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return the rank-major contents of only the bounded replica pool."""
        local_experts = self.local_home_experts
        return tuple(
            tuple(row[local_experts:]) for row in self.slot_to_logical_expert
        )

    @property
    def physical_to_logical_expert(self) -> tuple[int, ...]:
        """Return the logical expert represented by every physical ID."""
        return tuple(
            expert
            for rank_slots in self.slot_to_logical_expert
            for expert in rank_slots
        )

    @property
    def replica_gradient_pairs(self) -> tuple[tuple[int, int], ...]:
        """Return ``(replica, home)`` physical IDs for gradient reduction."""
        local_experts = self.local_home_experts
        slots_per_rank = self.physical_experts_per_rank
        pairs = []
        for physical_expert, logical_expert in enumerate(
            self.physical_to_logical_expert
        ):
            if logical_expert == -1 or physical_expert % slots_per_rank < local_experts:
                continue
            owner_rank, owner_slot = divmod(logical_expert, local_experts)
            home_physical_expert = owner_rank * slots_per_rank + owner_slot
            pairs.append((physical_expert, home_physical_expert))
        return tuple(pairs)

    @property
    def replica_transfers(self) -> tuple[ExpertReplicaTransfer, ...]:
        """Return deterministic home-to-replica transfer descriptors."""
        local_experts = self.local_home_experts
        transfers = []
        for target_rank, rank_slots in enumerate(self.slot_to_logical_expert):
            for target_slot, logical_expert in enumerate(
                rank_slots[local_experts:],
                local_experts,
            ):
                if logical_expert == -1:
                    continue
                owner_rank, owner_slot = divmod(logical_expert, local_experts)
                transfers.append(
                    ExpertReplicaTransfer(
                        logical_expert=logical_expert,
                        owner_rank=owner_rank,
                        owner_slot=owner_slot,
                        target_rank=target_rank,
                        target_slot=target_slot,
                    )
                )
        return tuple(transfers)

    @property
    def baseline_destination_loads(self) -> tuple[int, ...]:
        """Return per-rank loads before any hot-expert replication."""
        local_experts = self.local_home_experts
        loads = [0] * self.ep_size
        for source_counts in self.logical_counts_by_source:
            for expert, count in enumerate(source_counts):
                loads[expert // local_experts] += count
        return tuple(loads)

    @property
    def destination_loads(self) -> tuple[int, ...]:
        """Return per-rank loads after applying this placement."""
        slots_per_rank = self.physical_experts_per_rank
        loads = [0] * self.ep_size
        for source_counts in self.dispatch_counts:
            for physical_expert, count in enumerate(source_counts):
                loads[physical_expert // slots_per_rank] += count
        return tuple(loads)

    @property
    def physical_expert_loads(self) -> tuple[int, ...]:
        """Return globally flattened loads for every home or replica slot."""
        return tuple(
            sum(source_counts[physical_expert]
                for source_counts in self.dispatch_counts)
            for physical_expert in range(self.physical_num_experts)
        )

    @property
    def group_lists_by_rank(self) -> tuple[tuple[int, ...], ...]:
        """Return cumulative local GMM row ends for each destination rank."""
        slots_per_rank = self.physical_experts_per_rank
        physical_loads = self.physical_expert_loads
        group_lists = []
        for rank in range(self.ep_size):
            start = rank * slots_per_rank
            cumulative = 0
            rank_group_list = []
            for load in physical_loads[start:start + slots_per_rank]:
                cumulative += load
                rank_group_list.append(cumulative)
            group_lists.append(tuple(rank_group_list))
        return tuple(group_lists)

    def baseline_wave_count(self, receive_capacity: int) -> int:
        """Return multi-wave count required by static home placement.

        Args:
            receive_capacity: Maximum rows accepted by one rank in one wave.

        Returns:
            Number of fixed-capacity waves, always at least one.
        """
        return _wave_count(max(self.baseline_destination_loads), receive_capacity)

    def wave_count(self, receive_capacity: int) -> int:
        """Return residual multi-wave count after bounded replication.

        Args:
            receive_capacity: Maximum rows accepted by one rank in one wave.

        Returns:
            Number of fixed-capacity waves, always at least one.
        """
        return _wave_count(max(self.destination_loads), receive_capacity)


def _select_replica_move(
    copy_loads: list[dict[int, int]],
    rank_loads: list[int],
    remaining_slots: list[int],
    load_ceiling: int,
) -> tuple[int, int, int, int, bool] | None:
    """Select ``(expert, source, target, rows, new_copy)`` for one move."""
    best_score: tuple[int, int, int, int, int, int, int, int] | None = None
    best_move: tuple[int, int, int, int, bool] | None = None
    exact_balance = sum(rank_loads) == len(rank_loads) * load_ceiling
    # A move cannot originate at an empty copy or an underfull rank. Build
    # this ordered list once instead of rediscovering it for every receiver.
    active_copies = [
        (expert, loads_by_rank, source, expert_load, rank_loads[source] - load_ceiling)
        for expert, loads_by_rank in enumerate(copy_loads)
        for source, expert_load in loads_by_rank.items()
        if expert_load > 0 and rank_loads[source] > load_ceiling
    ]
    for target, available in enumerate(remaining_slots):
        target_room = load_ceiling - rank_loads[target]
        if target_room <= 0:
            continue
        for expert, loads_by_rank, source, expert_load, source_surplus in active_copies:
            new_copy = target not in loads_by_rank
            if new_copy and available == 0:
                continue
            # Fill the cold rank in one shot when this copy permits it.
            # The sender may become temporarily underfull and receive a
            # different home group later. This chained transfer is what
            # keeps MoonEP's training bound at one remote home group (at
            # most ``local_home_experts`` distinct copies) per rank.
            rows = min(expert_load, target_room)
            if not exact_balance:
                # Capacity headroom does not need a transfer chain. Move
                # only the overflow so prefetch and communication stay
                # proportional to the imbalance being removed.
                rows = min(rows, source_surplus)
            score = (
                rows,
                int(not new_copy),
                source_surplus,
                target_room,
                rank_loads[source],
                expert_load,
                -target,
                -expert,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_move = (expert, source, target, rows, new_copy)
    return best_move


def _build_slot_map(
    logical_counts: tuple[tuple[int, ...], ...],
    replica_experts: list[list[int]],
) -> tuple[tuple[tuple[int, ...], ...], dict[tuple[int, int], int]]:
    """Build uniform rank slots and map each logical copy to a physical ID."""
    ep_size = len(logical_counts)
    num_experts = len(logical_counts[0])
    local_experts = num_experts // ep_size
    replica_slots = max((len(experts) for experts in replica_experts), default=0)
    slots_per_rank = local_experts + replica_slots
    slot_map = []
    physical_by_copy: dict[tuple[int, int], int] = {}
    for rank in range(ep_size):
        home = list(range(rank * local_experts, (rank + 1) * local_experts))
        replicas = replica_experts[rank]
        rank_slots = home + replicas + [-1] * (replica_slots - len(replicas))
        slot_map.append(tuple(rank_slots))
        for slot, expert in enumerate(rank_slots):
            if expert != -1:
                physical_by_copy[(expert, rank)] = rank * slots_per_rank + slot
    return tuple(slot_map), physical_by_copy


def _transport_one_expert(
    expert: int,
    logical_counts: tuple[tuple[int, ...], ...],
    loads_by_rank: dict[int, int],
    physical_by_copy: dict[tuple[int, int], int],
    dispatch: list[list[int]],
) -> None:
    """Assign one expert's per-source rows to its planned physical copies."""
    remaining = [
        [physical_by_copy[(expert, rank)], load]
        for rank, load in loads_by_rank.items()
    ]
    remaining.sort(key=lambda item: item[0])
    copy_index = 0
    for source_rank, source_counts in enumerate(logical_counts):
        rows = source_counts[expert]
        while rows:
            while copy_index < len(remaining) and remaining[copy_index][1] == 0:
                copy_index += 1
            if copy_index == len(remaining):
                raise RuntimeError("replica planner failed to conserve expert rows.")
            physical_expert, available = remaining[copy_index]
            moved = min(rows, available)
            dispatch[source_rank][physical_expert] += moved
            remaining[copy_index][1] -= moved
            rows -= moved
    if any(available for _, available in remaining):
        raise RuntimeError("replica planner left expert rows unassigned.")


def _materialize_dispatch_counts(
    logical_counts: tuple[tuple[int, ...], ...],
    copy_loads: list[dict[int, int]],
    replica_experts: list[list[int]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Build the slot map and an exact source-to-physical transport matrix."""
    ep_size = len(logical_counts)
    slot_map, physical_by_copy = _build_slot_map(
        logical_counts,
        replica_experts,
    )
    physical_experts = sum(len(row) for row in slot_map)
    dispatch = [[0] * physical_experts for _ in range(ep_size)]
    for expert, loads_by_rank in enumerate(copy_loads):
        _transport_one_expert(
            expert,
            logical_counts,
            loads_by_rank,
            physical_by_copy,
            dispatch,
        )
    return slot_map, tuple(tuple(row) for row in dispatch)


def build_constrained_expert_placement(
    counts_by_source: Sequence[Sequence[int]],
    replica_slots_per_rank: int,
    *,
    receive_capacity: int | None = None,
) -> ExpertPlacementPlan:
    """Build a deterministic bounded hot-expert placement.

    The reference heuristic repeatedly splits the most useful expert copy from
    a more-loaded rank onto a less-loaded rank.  Planning stops once every rank
    fits ``receive_capacity``, when the uniform replica pool is exhausted, or
    when no move can further balance two ranks.  Any remaining overflow is
    represented exactly and is therefore safe for MegaMoe multi-wave replay.

    Args:
        counts_by_source: Rank-major logical expert counts shaped
            ``[ep_size, logical_num_experts]``.
        replica_slots_per_rank: Maximum number of non-parameter replica slots
            available on every rank.
        receive_capacity: Optional per-wave receive-row capacity. If supplied,
            stop allocating replicas as soon as a one-wave route is possible.

    Returns:
        Validated immutable placement and physical dispatch counts.
    """
    logical_counts = _normalize_counts(counts_by_source)
    replica_slots = _validate_nonnegative_integer(
        replica_slots_per_rank,
        "replica_slots_per_rank",
    )
    ep_size = len(logical_counts)
    num_experts = len(logical_counts[0])
    if num_experts % ep_size:
        raise ValueError(
            f"logical expert count {num_experts} must be divisible by "
            f"ep_size {ep_size}."
        )
    if receive_capacity is not None:
        _wave_count(0, receive_capacity)

    local_experts = num_experts // ep_size
    expert_loads = [
        sum(source_counts[expert] for source_counts in logical_counts)
        for expert in range(num_experts)
    ]
    copy_loads = [
        {expert // local_experts: expert_load}
        for expert, expert_load in enumerate(expert_loads)
    ]
    rank_loads = [0] * ep_size
    for expert, expert_load in enumerate(expert_loads):
        rank_loads[expert // local_experts] += expert_load
    remaining_slots = [replica_slots] * ep_size
    replica_experts: list[list[int]] = [[] for _ in range(ep_size)]
    balanced_load = (sum(rank_loads) + ep_size - 1) // ep_size
    load_ceiling = max(receive_capacity or 0, balanced_load)

    while True:
        if max(rank_loads) <= load_ceiling:
            break
        move = _select_replica_move(
            copy_loads,
            rank_loads,
            remaining_slots,
            load_ceiling,
        )
        if move is None:
            break
        expert, source, target, rows, new_copy = move
        copy_loads[expert][source] -= rows
        copy_loads[expert][target] = copy_loads[expert].get(target, 0) + rows
        rank_loads[source] -= rows
        rank_loads[target] += rows
        if new_copy:
            remaining_slots[target] -= 1
            replica_experts[target].append(expert)

    # Keep the physical GMM shape uniform even when capacity-aware planning
    # stops early and leaves suffix replica slots empty.
    for rank in range(ep_size):
        replica_experts[rank].extend(
            [-1] * (replica_slots - len(replica_experts[rank]))
        )
    slot_map, dispatch_counts = _materialize_dispatch_counts(
        logical_counts,
        copy_loads,
        replica_experts,
    )
    return ExpertPlacementPlan(
        logical_counts_by_source=logical_counts,
        replica_slots_per_rank=replica_slots,
        slot_to_logical_expert=slot_map,
        dispatch_counts=dispatch_counts,
    )


def _source_copy_quotas(
    source_rank: int,
    plan: ExpertPlacementPlan,
) -> list[list[list[int]]]:
    """Return mutable ``[physical_id, rows]`` queues for one source rank."""
    physical_to_logical = tuple(
        expert for rank_slots in plan.slot_to_logical_expert for expert in rank_slots
    )
    copies: list[list[list[int]]] = [
        [] for _ in range(plan.logical_num_experts)
    ]
    for physical_expert, logical_expert in enumerate(physical_to_logical):
        count = plan.dispatch_counts[source_rank][physical_expert]
        if logical_expert != -1 and count:
            copies[logical_expert].append([physical_expert, count])
    return copies


def _validate_logical_route(
    logical_expert: object,
    route_index: int,
    num_experts: int,
) -> int:
    """Return one validated logical route ID."""
    if (
        not isinstance(logical_expert, int)
        or isinstance(logical_expert, bool)
        or not 0 <= logical_expert < num_experts
    ):
        raise ValueError(
            f"logical_expert_ids[{route_index}] is invalid: {logical_expert!r}."
        )
    return logical_expert


def _consume_physical_copy(
    logical_expert: int,
    source_rank: int,
    copies: list[list[list[int]]],
    copy_indices: list[int],
) -> int:
    """Consume one planned row and return its physical expert ID."""
    expert_copies = copies[logical_expert]
    index = copy_indices[logical_expert]
    while index < len(expert_copies) and expert_copies[index][1] == 0:
        index += 1
    if index == len(expert_copies):
        raise ValueError(
            "logical_expert_ids does not match the plan histogram for "
            f"source rank {source_rank}."
        )
    physical_expert = expert_copies[index][0]
    expert_copies[index][1] -= 1
    copy_indices[logical_expert] = index
    return physical_expert


def remap_logical_routes(
    logical_expert_ids: Sequence[int],
    source_rank: int,
    plan: ExpertPlacementPlan,
) -> tuple[int, ...]:
    """Apply one source row of a placement to flattened logical route IDs.

    This deterministic host reference assigns successive occurrences of each
    logical expert to physical copies according to ``dispatch_counts``. Device
    planners may use a different stable ordering as long as they preserve the
    same count and slot contract.

    Args:
        logical_expert_ids: Flattened logical Top-K expert IDs for one rank.
        source_rank: Source EP rank represented by the route sequence.
        plan: Validated placement returned by
            :func:`build_constrained_expert_placement`.

    Returns:
        Physical expert IDs in the original route order.
    """
    if not isinstance(source_rank, int) or isinstance(source_rank, bool):
        raise ValueError(f"source_rank must be an integer, got {source_rank!r}.")
    if not 0 <= source_rank < plan.ep_size:
        raise ValueError(
            f"source_rank must be in [0, {plan.ep_size}), got {source_rank}."
        )
    routes = tuple(logical_expert_ids)
    expected_rows = sum(plan.logical_counts_by_source[source_rank])
    if len(routes) != expected_rows:
        raise ValueError(
            f"source rank {source_rank} expects {expected_rows} routes, got "
            f"{len(routes)}."
        )

    copies = _source_copy_quotas(source_rank, plan)
    copy_indices = [0] * plan.logical_num_experts
    physical_routes = []
    for route_index, logical_expert in enumerate(routes):
        logical_expert = _validate_logical_route(
            logical_expert,
            route_index,
            plan.logical_num_experts,
        )
        physical_routes.append(
            _consume_physical_copy(
                logical_expert,
                source_rank,
                copies,
                copy_indices,
            )
        )
    if any(remaining for expert_copies in copies for _, remaining in expert_copies):
        raise ValueError(
            "logical_expert_ids does not match the plan histogram for "
            f"source rank {source_rank}."
        )
    return tuple(physical_routes)


def _gradient_transfer_groups(
    plan: ExpertPlacementPlan,
    receive_slots: int,
    *,
    group_by_slot: bool = False,
) -> tuple[tuple[int | None, tuple[tuple[ExpertReplicaTransfer, ...], ...]], ...]:
    """Bound owner fan-in, optionally isolating low-precision accumulators."""
    by_slot: dict[int | None, dict[int, list[ExpertReplicaTransfer]]] = {}
    for transfer in plan.replica_transfers:
        owner_slot = transfer.owner_slot if group_by_slot else None
        owners = by_slot.setdefault(owner_slot, {})
        owners.setdefault(transfer.owner_rank, []).append(transfer)
    groups = []
    for owner_slot, owners in sorted(by_slot.items()):
        # Spread a window across replica peers instead of filling it with one
        # peer's different experts. Stable sorting preserves each expert's
        # original contribution order, including floating-point cancellation.
        for transfers in owners.values():
            transfers.sort(key=lambda transfer: transfer.owner_slot)
        owner_batches = []
        for transfers in owners.values():
            peer_count = len({transfer.target_rank for transfer in transfers})
            # A partial extra round of peers makes its few repeated senders
            # bottlenecks. Leave that tail unused when a full peer round fits.
            width = receive_slots
            if peer_count <= width:
                width -= width % peer_count
            owner_batches.append(tuple(
                tuple(transfers[start:start + width]) for start in range(0, len(transfers), width)
            ))
        batches = tuple(
            tuple(transfer for batch in owner_round for transfer in batch)
            for owner_round in zip_longest(*owner_batches, fillvalue=())
        )
        groups.append((owner_slot, batches))
    return tuple(groups)
