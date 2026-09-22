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
"""Immutable logical ownership and physical dispatch contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .capacity import ExpertReplicaConfig


@dataclass(frozen=True)
class ExpertReplicaTransfer:
    """One original owner supplying one temporary physical expert slot."""

    logical_expert: int
    owner_rank: int
    owner_slot: int
    target_rank: int
    target_slot: int


@dataclass(frozen=True)
class ExpertExecutionPlan:
    """Exact source quotas and bounded physical slots for one forward graph."""

    config: ExpertReplicaConfig
    logical_counts: tuple[tuple[int, ...], ...]
    slot_to_logical: tuple[tuple[int, ...], ...]
    dispatch_counts: tuple[tuple[int, ...], ...]

    @property
    def destination_loads(self) -> tuple[int, ...]:
        """Return total real work assigned to each execution rank."""
        width = self.config.slots_per_rank
        return tuple(sum(sum(row[rank * width:(rank + 1) * width]) for row in self.dispatch_counts)
                     for rank in range(self.config.ep_size))

    @property
    def physical_to_logical(self) -> tuple[int, ...]:
        """Return flattened physical-slot ownership, using -1 for empty slots."""
        return tuple(expert for row in self.slot_to_logical for expert in row)

    @property
    def transfers(self) -> tuple[ExpertReplicaTransfer, ...]:
        """Return transfers in globally identical target/slot order."""
        home = self.config.home_experts
        return tuple(ExpertReplicaTransfer(expert, expert // home, expert % home, rank, slot)
                     for rank, row in enumerate(self.slot_to_logical)
                     for slot, expert in enumerate(row) if slot >= home and expert >= 0)

    def validate(self) -> None:
        """Check per-source conservation and physical ownership invariants."""
        config = self.config
        if len(self.logical_counts) != config.ep_size or len(self.slot_to_logical) != config.ep_size:
            raise ValueError("plan must contain one logical and physical row per rank")
        for rank, row in enumerate(self.slot_to_logical):
            expected = tuple(range(rank * config.home_experts, (rank + 1) * config.home_experts))
            active = [expert for expert in row if expert != -1]
            if (len(row) != config.slots_per_rank or row[:config.home_experts] != expected
                    or len(active) != len(set(active)) or any(not 0 <= e < config.num_experts for e in active)):
                raise ValueError("plan contains invalid or duplicate physical expert ownership")
        if len(self.dispatch_counts) != config.ep_size:
            raise ValueError("plan must contain one dispatch row per source rank")
        for source, row in enumerate(self.dispatch_counts):
            if len(row) != config.physical_experts:
                raise ValueError("dispatch row does not match physical expert count")
            reconstructed = [0] * config.num_experts
            for expert, count in zip(self.physical_to_logical, row):
                if not isinstance(count, int) or isinstance(count, bool) or count < 0 or (expert == -1 and count):
                    raise ValueError("invalid physical dispatch count")
                if expert >= 0:
                    reconstructed[expert] += count
            if tuple(reconstructed) != self.logical_counts[source]:
                raise ValueError("physical dispatch must conserve every source/logical-expert count")
