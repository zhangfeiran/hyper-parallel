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
"""Prepared selected-request metadata and an independent CPU descriptor oracle."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_indexer import validate_mixed_indexer_traces
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule, validate_device_phase_closure, validate_mixed_trace,
)

SELECTED_REQUESTS_MAGIC = 0x4850445341525131


@dataclass(frozen=True)
class SelectedRequestReference:
    """Deterministic active descriptors and complete membership occurrence counts."""

    requests: torch.Tensor
    membership: torch.Tensor
    selected_slots: int
    remote_keys: int
    remote_runs: int


def selected_request_rows(meta: DsaBatchMeta, device: torch.device | str) -> torch.Tensor:
    """Prepare canonical packed sequence starts and owner addresses outside the hot path."""
    rows = []
    for start, end in pairwise(meta.global_cu_seqlens):
        rows.extend((start, meta.token_owners[key], meta.token_local_offsets[key]) for key in range(start, end))
    return torch.tensor(rows, dtype=torch.int64, device=device)


def selected_request_reference(indices: torch.Tensor, meta: DsaBatchMeta) -> SelectedRequestReference:
    """Resolve sequence-relative slots and build a selected union with a CPU set oracle.

    This offline checker accepts duplicate/padded slots to exercise the device
    membership protocol. Future or cross-sequence slots do not participate.
    Every run requires consecutive packed destinations and owner-local offsets.
    No device tensor is transferred implicitly.
    """
    tokens = meta.global_valid_queries
    if (indices.device.type != "cpu" or indices.dtype != torch.int32 or indices.ndim != 3
            or indices.shape[:2] != (tokens, 1) or not 1 <= indices.shape[2] <= 2048):
        raise ValueError("selected request reference requires CPU int32 [global Q,1,K<=2048]")
    rows = selected_request_rows(meta, "cpu").tolist()
    keys, occurrences = set(), [0] * ((tokens + 7) // 8 * 8)
    for query, slots in enumerate(indices[:, 0].tolist()):
        start = rows[query][0]
        for index in slots:
            if 0 <= index <= query - start:
                key = start + index
                keys.add(key)
                occurrences[key] += 1
    descriptors = []
    remote_keys = 0
    for key in sorted(keys):
        _, owner, offset = rows[key]
        remote_keys += int(owner != meta.cp_rank)
        if (descriptors and descriptors[-1][0] == owner
                and descriptors[-1][1] + descriptors[-1][3] == offset
                and descriptors[-1][2] + descriptors[-1][3] == key):
            descriptors[-1][3] += 1
        else:
            descriptors.append([owner, offset, key, 1])
    return SelectedRequestReference(torch.tensor(descriptors, dtype=torch.int64).reshape(-1, 4),
                                    torch.tensor(occurrences, dtype=torch.int32), sum(occurrences), remote_keys,
                                    sum(row[0] != meta.cp_rank for row in descriptors))


def validate_selected_training_traces(traces: tuple[torch.Tensor, ...], schedule: MixedSfaSchedule,
                                       *, require_ld: bool) -> dict:
    """Require nine releases, including membership build and completed request/pull before SFA."""
    if len(traces) != 9:
        raise ValueError("selected training requires nine phase snapshots")
    snapshots = validate_device_phase_closure(traces, schedule)
    indexer = validate_mixed_indexer_traces(snapshots[:2], schedule, require_ld=require_ld)
    phases = [validate_mixed_trace(trace, schedule) for trace in snapshots[2:]]
    for trace in snapshots[2:]:
        for group in range(schedule.compute_groups):
            if any(int(trace[group, offset + 4]) != 0 for offset in (0, 16, 32)):
                raise ValueError("selected request/SFA/KL phase contains stale LI partition evidence")
    return {"indexer": indexer, "phases": phases, "device_phase_closure": True,
            "phase_order": ["li_main", "li_merge", "membership_init", "membership_build", "request_pack_pull",
                            "sfa", "kl_init", "kl_compute", "kl_post"]}


def validate_selected_requests(indices: torch.Tensor, meta: DsaBatchMeta, requests: torch.Tensor,
                                counts: torch.Tensor, membership: torch.Tensor, epoch: int) -> dict:
    """Compare completed CPU descriptor/count snapshots with the independent set oracle."""
    tokens = meta.global_valid_queries
    if (requests.device.type != "cpu" or requests.dtype != torch.int64 or requests.shape != (tokens, 4)
            or counts.device.type != "cpu" or counts.dtype != torch.int64 or counts.shape != (16,)
            or membership.device.type != "cpu" or membership.dtype != torch.int32
            or membership.shape != ((tokens + 7) // 8 * 8,)):
        raise ValueError("selected request validation requires explicit CPU snapshots with prepared shapes")
    oracle = selected_request_reference(indices, meta)
    run_count, unique = len(oracle.requests), int(oracle.membership.count_nonzero())
    expected = (SELECTED_REQUESTS_MAGIC, epoch, run_count, unique, oracle.selected_slots,
                oracle.remote_keys, oracle.remote_runs, 0)
    if tuple(counts[:8].tolist()) != expected or bool(counts[10:].any()):
        raise ValueError("selected device count/epoch/occurrence records differ from the independent oracle")
    if int(counts[8]) <= 0 or int(counts[9]) <= int(counts[8]):
        raise ValueError("selected device descriptor timestamps are incomplete")
    if not torch.equal(membership, oracle.membership) or not torch.equal(requests[:run_count], oracle.requests):
        raise ValueError("selected device membership/descriptors differ from the independent oracle")
    return {"request_count": run_count, "unique_keys": unique, "selected_slots": oracle.selected_slots,
            "remote_keys": oracle.remote_keys, "remote_runs": oracle.remote_runs,
            "duplicate_slots": oracle.selected_slots - unique, "request_capacity": tokens}
