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
"""A3 CPU port of MoonEP's allocation and rank dedup reference.

Algorithm reference: https://gitcode.com/Ascend/MoonEP/blob/dev_server_dedup/test_npu/planing_reference.py
The A5 SIMT planner and UDMA server dedup are not executed by this port.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MoonEPPlacement:
    """Exact per-expert quotas, physical offsets and selected remote weights."""

    allocation: np.ndarray
    offsets: np.ndarray
    experts_to_copy: np.ndarray
    compact_counts: np.ndarray
    remote_stats: np.ndarray


def plan_moonep(counts: np.ndarray, capacity: int) -> MoonEPPlacement:
    """Port upstream greedy perfect balancing with one home group per receiver.

    Args:
        counts: Nonnegative per-source expert counts, shape ``[R, E]``.
        capacity: Routed rows per destination, equal to ``S*K``.

    Returns:
        Placement using ``E/R`` home and ``E/R`` remote slots per rank.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 2 or np.any(counts < 0) or capacity <= 0:
        raise ValueError("MoonEP counts must be nonnegative [R,E] and capacity positive")
    ranks, experts = counts.shape
    if not ranks or not experts or experts % ranks or not np.all(counts.sum(1) == capacity):
        raise ValueError("MoonEP requires E divisible by R and exactly S*K rows per source")
    home = experts // ranks
    totals = counts.sum(0)
    balance = totals.reshape(ranks, home).sum(1) - capacity
    migration = np.zeros((ranks, ranks), dtype=np.int64)
    allocation = np.zeros((experts, ranks), dtype=np.int64)
    allocation[np.arange(experts), np.arange(experts) // home] = totals
    while balance.max() > 0:
        owner, destination = int(balance.argmax()), int(balance.argmin())
        move = -balance[destination]
        migration[owner, destination] = move
        balance[owner] -= move
        balance[destination] = 0
    for owner in range(ranks):
        start = owner * home
        remaining = totals[start:start + home].copy()
        quotas = migration[owner].copy()
        while quotas.max() > 0:
            destination, local = int(quotas.argmax()), int(remaining.argmax())
            take = min(remaining[local], quotas[destination])
            allocation[start + local, destination] += take
            allocation[start + local, owner] -= take
            remaining[local] -= take
            quotas[destination] -= take
    if np.any(allocation < 0) or not np.array_equal(allocation.sum(1), totals):
        raise ValueError("MoonEP allocation did not conserve expert counts")
    if not np.all(allocation.sum(0) == capacity):
        raise ValueError("MoonEP allocation did not balance every destination")
    slots = np.full((ranks, home), -1, dtype=np.int32)
    offsets = np.zeros((ranks, experts), dtype=np.int64)
    compact = np.zeros((ranks, 2 * home), dtype=np.int64)
    stats = np.zeros((ranks, 2), dtype=np.int32)
    for destination in range(ranks):
        start = destination * home
        remote = [expert for expert in range(experts)
                  if allocation[expert, destination] > 0 and expert // home != destination]
        remote.sort(key=lambda expert: (allocation[expert, destination], expert), reverse=True)
        if len(remote) > home:
            raise ValueError("MoonEP receiver requires more than E/R remote slots")
        slots[destination, :len(remote)] = remote
        stats[destination, 0] = len(remote)
        for expert in remote:
            stats[expert // home, 1] += 1
        logical = list(range(start, start + home)) + remote
        cursor = 0
        for slot, expert in enumerate(logical):
            compact[destination, slot] = allocation[expert, destination]
            offsets[destination, expert] = cursor
            cursor += allocation[expert, destination]
    return MoonEPPlacement(allocation, offsets, slots, compact, stats)


def encode_moonep_routes(ids: np.ndarray, counts: np.ndarray, placement: MoonEPPlacement,
                         rank: int, capacity: int) -> np.ndarray:
    """Encode local source rows into rank-major packed destination addresses."""
    flat = np.asarray(ids, dtype=np.int64).reshape(-1)
    if flat.size != capacity or not 0 <= rank < counts.shape[0]:
        raise ValueError("MoonEP route size or source rank is invalid")
    if np.any(flat < 0) or np.any(flat >= counts.shape[1]):
        raise ValueError("MoonEP expert ids are out of range")
    if not np.array_equal(np.bincount(flat, minlength=counts.shape[1]), counts[rank]):
        raise ValueError("MoonEP local expert histogram does not match route ids")
    order = np.argsort(flat, kind="stable")
    starts = np.cumsum(counts[rank]) - counts[rank]
    ordinal = np.empty(capacity, dtype=np.int64)
    ordinal[order] = np.arange(capacity) - starts[flat[order]]
    ordinal += counts[:rank].sum(0)[flat]
    cumulative = placement.allocation.cumsum(1)
    destinations = (ordinal[:, None] >= cumulative[flat]).sum(1)
    previous = cumulative[flat, np.maximum(destinations - 1, 0)]
    previous[destinations == 0] = 0
    result = destinations * capacity + placement.offsets[destinations, flat] + ordinal - previous
    if result.max() > np.iinfo(np.int32).max:
        raise ValueError("MoonEP packed destination exceeds int32")
    return result.astype(np.int32)


def deduplicate_moonep(raw: np.ndarray, rank: int, top_k: int, capacity: int) -> tuple:
    """Build upstream rank dedup tables while preserving every routing weight.

    Args:
        raw: All sources' packed destinations, shape ``[R, S*K]``.
        rank: Destination rank whose duplicate tables are returned.
        top_k: Routes per input token.
        capacity: Rows in each destination communication buffer.

    Returns:
        Local encoded destinations, duplicate groups, offsets and two counts.
    """
    raw = np.asarray(raw, dtype=np.int32)
    if raw.ndim != 2 or raw.shape[1] != capacity or capacity % top_k:
        raise ValueError("MoonEP packed routes must be [R,S*K]")
    routed = raw.reshape(raw.shape[0], -1, top_k)
    destinations = routed // capacity
    local = routed[rank].copy()
    groups, loffs = [], []
    group_count, loff_count = 0, 0
    for route in range(top_k):
        duplicate = (destinations[:, :, :route] == destinations[:, :, route:route + 1]).any(2)
        local[:, route][duplicate[rank]] = -local[:, route][duplicate[rank]] - 1
        primary = (destinations[:, :, route] == rank) & ~duplicate
        source_indices, token_indices = np.nonzero(primary)
        selected = routed[source_indices, token_indices]
        duplicate_mask = (selected // capacity == rank) & (np.arange(top_k) > route)
        duplicate_counts = duplicate_mask.sum(1)
        active = duplicate_counts > 0
        selected, duplicate_mask, duplicate_counts = selected[active], duplicate_mask[active], duplicate_counts[active]
        if selected.size:
            starts = duplicate_counts.cumsum() - duplicate_counts + loff_count
            groups.append(np.stack((selected[:, route] % capacity, starts, duplicate_counts), axis=1))
            loffs.append(selected[duplicate_mask] % capacity)
            group_count += selected.shape[0]
            loff_count += int(duplicate_counts.sum())
    group_array = np.zeros((capacity, 3), dtype=np.int32)
    loff_array = np.zeros(capacity, dtype=np.int32)
    if groups:
        group_array[:group_count] = np.concatenate(groups)
        loff_array[:loff_count] = np.concatenate(loffs)
    return local.reshape(-1), group_array, loff_array, np.array([group_count, loff_count], dtype=np.int32)
