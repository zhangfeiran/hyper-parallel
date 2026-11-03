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
"""Immutable owner-local residency observations without touching transfer state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from hyper_parallel.core.expert_parallel.hot_replica.residency import _WeightSnapshot


POLICY_VERSION = 'owner-authorized-fixed-physical-slot-v1-materialization-sensitive'


@dataclass(frozen=True)
class ResidentFact:
    """A point-in-time observation, not a reservation or a global certificate."""

    owner: int
    owner_expert: int
    target: int
    physical_slot: int
    status: str
    reasons: tuple[str, ...]
    copy_bytes: int | None
    gradient_return_bytes_planned: int
    parameter_versions: tuple[int, ...]
    storage_ids: tuple[int, ...]
    materialization_ids: tuple[int | None, ...]


@dataclass(frozen=True)
class ResidentView:
    """Keep the policy domain and local evidence separate from remote unknowns."""

    policy_version: str
    scope: str
    provider_id: int
    protocol_epoch: int
    facts: tuple[ResidentFact, ...]


def invalidation_reasons(previous: Any, current: _WeightSnapshot | None, expert: int) -> tuple[str, ...]:
    """Explain every observable mismatch without changing either snapshot."""
    if current is None:
        return ('unversioned',)
    if previous is None:
        return ('missing_evidence',)
    old_expert, old = previous
    reasons = []
    if old_expert != expert:
        reasons.append('expert_changed')
    if any(left() is not right() or left() is None for left, right in zip(old.tensors, current.tensors)):
        reasons.append('tensor_identity')
    if any(left() is not right() or left() is None for left, right in zip(old.storages, current.storages)):
        reasons.append('storage_identity')
    if len(old.metadata) != len(current.metadata):
        reasons.append('matrix_layout')
        return tuple(reasons)
    if any(a[-1] is not b[-1] for a, b in zip(old.metadata, current.metadata)):
        reasons.append('materialization')
    if any(a[0] != b[0] for a, b in zip(old.metadata, current.metadata)):
        reasons.append('parameter_version')
    if any(a[1:7] != b[1:7] for a, b in zip(old.metadata, current.metadata)):
        reasons.append('storage_layout')
    return tuple(reasons)


def readonly_view(provider: Any, weights: tuple[torch.Tensor, ...], route: Any) -> ResidentView:
    """Read current local owner evidence; never call prepare or prefetch.

    A different owner is unknown, even when its old placement is locally known.
    Callers must revalidate at execution: this view does not pin any slot.
    """
    current = _WeightSnapshot.capture(weights)
    elements = sum(math.prod(tuple(weight.shape[1:])) for weight in weights)
    weight_bytes = sum(math.prod(tuple(weight.shape[1:])) * weight.element_size() for weight in weights)
    versions = tuple(item[0] for item in current.metadata) if current is not None else ()
    storages = tuple(id(ref()) for ref in current.storages) if current is not None else ()
    materializations = (tuple(None if item[-1] is None else id(item[-1]) for item in current.metadata)
                        if current is not None else ())
    facts = []
    home = route.plan.config.home_experts
    for edge in route.plan.transfers:
        key = edge.target_rank, edge.target_slot - home
        owned = edge.owner_rank == route.rank
        reasons = (invalidation_reasons(provider._weight_cache._entries.get(key), current, edge.owner_slot)
                   if owned else ('non_owner_unknown',))
        status = 'unknown' if not owned else ('invalid' if reasons else 'valid')
        facts.append(ResidentFact(edge.owner_rank, edge.owner_slot, *key, status, reasons,
                                  None if not owned else (weight_bytes if reasons else 0), elements * 4,
                                  versions if owned else (), storages if owned else (),
                                  materializations if owned else ()))
    return ResidentView(POLICY_VERSION, 'owner-local-point-in-time', id(provider), provider.epoch, tuple(facts))
