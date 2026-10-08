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
"""Experimental two-phase LI with retained logical partials and no autograd."""

from __future__ import annotations

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule,
    validate_mixed_trace,
)
from hyper_parallel.core.multicore.torch.ops import _load_native

# Pinned arch22 H64/K2048: physical MM double buffers, logical Top-K lists and LD parameters.
MIXED_INDEXER_WORKSPACE_BYTES = 20 * (2 * 512 * 512 * 4 + 8 * 2 * 2 * 2048 * 4 + 8 * 2 * 16 * 8)


def validate_mixed_indexer_traces(traces: tuple[torch.Tensor, torch.Tensor],
                                 schedule: MixedSfaSchedule, *, require_ld: bool) -> dict:
    """Verify both completed CPU snapshots and all members' retained LD ownership.

    Args:
        traces: Explicit CPU main and merge snapshots. No device transfer occurs.
        schedule: Single-traversal mixed schedule used by both phases.
        require_ld: Require at least one cross-partition Top-K merge.

    Returns:
        Per-phase participation evidence and total logical LD partitions.

    Raises:
        ValueError: Missing or disagreeing member/phase records, impossible LD
            counts or an absent required cross-partition merge.
    """
    if len(traces) != 2 or schedule.rounds != 1:
        raise ValueError("mixed LI evidence requires two phases with rounds=1")
    phases = [validate_mixed_trace(trace, schedule) for trace in traces]
    records = [trace.tolist() for trace in traces]
    ld_total = 0
    for group in range(schedule.compute_groups):
        count = records[0][group][4]
        tasks = len(range(group, 20, schedule.compute_groups))
        if count < 0 or count > tasks:
            raise ValueError(f"physical group {group} has impossible LD count {count}")
        for phase in records:
            if any(phase[group][offset + 4] != count for offset in (0, 16, 32)):
                raise ValueError(f"physical group {group} members/phases disagree on LD ownership")
        ld_total += count
    if require_ld and ld_total == 0:
        raise ValueError("long-history mixed LI did not execute any cross-partition LD merge")
    return {"phases": phases, "ld_partitions": ld_total}


def mixed_indexer_forward_probe(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
                                cumulative_lengths: torch.Tensor, schedule: MixedSfaSchedule,
                                retained: torch.Tensor | None = None) -> tuple:
    """Run LI main and LD merge on one stream, preserving all logical partials.

    Args:
        query: Detached BF16 full ordered TND query [T,64,128].
        key: Detached BF16 full ordered key [T,1,128].
        weights: Already scaled BF16 or FP32 [T,64] merge weights.
        cumulative_lengths: Prepared CP1 nonempty packed int32 cumulative lengths.
            The caller must validate positive lengths and complete ordered Q/K.
        schedule: Reusable mixed-group schedule with rounds=1. Reuse is tested
            by repeating the complete main/merge pair, preserving their order.
        retained: Optional independent contiguous NPU uint8 scratch of at least
            MIXED_INDEXER_WORKSPACE_BYTES bytes. Reuse only on the same stream
            or after an explicit event dependency; concurrent calls are unsupported.

    Returns:
        Independent indices, values, two phase traces and the retained workspace.
        Traces require explicit CPU snapshots for validation. This probe has no
        backward and is not a CP or fused communication backend.

    Raises:
        ValueError: The schedule attempts to repeat a phase without the full pair.
    """
    if schedule.rounds != 1:
        raise ValueError("mixed LI requires rounds=1; repeat complete main/merge pairs instead")
    _load_native()
    if torch.ops.hyper_parallel.dsa_mixed_indexer_version() != 1:
        raise RuntimeError("mixed LI adapter ABI mismatch; rebuild this checkout's payload")
    device = query.device
    if retained is None:
        retained = torch.empty(MIXED_INDEXER_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    config = schedule.runtime_config(device)
    traces = (schedule.new_trace(device), schedule.new_trace(device))
    shape = (query.shape[0], 1, 2048)
    indices = torch.full(shape, -2, dtype=torch.int32, device=device)
    values = torch.full(shape, float("nan"), dtype=torch.bfloat16, device=device)
    for phase, trace in enumerate(traces):
        torch.ops.hyper_parallel.dsa_mixed_indexer_out(
            query, key, weights, cumulative_lengths, cumulative_lengths,
            config, trace, retained, phase, indices, values)
    return indices, values, traces, retained
