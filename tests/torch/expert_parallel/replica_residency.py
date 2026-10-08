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
"""Host-only shadow ledger for potential weight reuse, never a transport cache."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable


@dataclass(frozen=True)
class ResidentWeight:
    """Identity supplied by a controlled diagnostic's parameter-owning caller."""

    layer: str
    expert: int
    generation: int | None
    layout: tuple
    heap_generation: int


class ResidencyLedger:
    """Track successful ordered leases without retaining parameters or tensors."""

    def __init__(self) -> None:
        """Start with no known guest content."""
        self.entries: dict[Hashable, ResidentWeight] = {}

    def inspect(self, slot: Hashable, current: ResidentWeight) -> str:
        """Classify a potential reuse before the unchanged real transfer.

        Args:
            slot: Shared resource, guest slot and projection identity.
            current: Current caller-owned identity and explicit weight generation.

        Returns:
            ordered_hit or the first invalidation reason. Completion is ordered by
            the actual serial lease, rather than inferred from pointer equality.
        """
        previous = self.entries.get(slot)
        if current.generation is None:
            return "unknown_generation"
        if previous is None:
            return "empty"
        fields = (("heap_generation", "heap_changed"), ("layer", "layer_changed"),
                  ("expert", "expert_changed"), ("generation", "weight_updated"), ("layout", "layout_changed"))
        for field, reason in fields:
            if getattr(previous, field) != getattr(current, field):
                return reason
        return "ordered_hit"

    def commit(self, slot: Hashable, current: ResidentWeight) -> None:
        """Publish shadow content after a successful lease has established ordering.

        Args:
            slot: Resource, guest slot and projection identity.
            current: Identity of the successfully ordered content.
        """
        if current.generation is None:
            self.entries.pop(slot, None)
        else:
            self.entries[slot] = current

    def invalidate(self, slots: tuple[Hashable, ...]) -> None:
        """Forget potentially partially written slots after an observed failure.

        Args:
            slots: Slot identities whose real transfer may not have completed.
        """
        for slot in slots:
            self.entries.pop(slot, None)
