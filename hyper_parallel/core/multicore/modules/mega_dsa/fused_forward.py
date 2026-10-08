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
"""Experimental single-launch CP1 LI main/merge/SFA over reusable mixed workers."""

from __future__ import annotations

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.mixed_indexer import (
    MIXED_INDEXER_WORKSPACE_BYTES,
    validate_mixed_indexer_traces,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule,
    validate_device_phase_closure,
    validate_mixed_trace,
)
from hyper_parallel.core.multicore.torch.ops import _load_native


def fused_dsa_forward_probe(index_states: tuple[torch.Tensor, ...], main_states: tuple[torch.Tensor, ...],
                             cumulative_lengths: torch.Tensor, attention_scale: float,
                             schedule: MixedSfaSchedule, retained: torch.Tensor | None = None) -> tuple:
    """Run LI main, retained Top-K merge and SFA in one mixed AICore kernel.

    Args:
        index_states: Detached ordered BF16 query [T,64,128], key [T,1,128]
            and already scaled BF16/FP32 weights [T,64].
        main_states: Detached absorbed query [T,H32|64,512], shared compressed
            KV [T,512], query RoPE [T,H,64] and key RoPE [T,64], all BF16.
        cumulative_lengths: Prepared CP1 nonempty packed int32 cumulative lengths.
            The caller validates positive lengths and complete ordered Q/KV.
        attention_scale: Positive original attention scale applied exactly once.
        schedule: Single traversal with at least one reserved progress group.
        retained: Optional independently owned LI scratch. Reuse only on the same
            stream or with explicit dependencies, after the complete invocation.

    Returns:
        Native sequence-local indices, BF16 indexer values, compressed attention,
        FP32 maximum and denominator, three phase trace views and LI scratch.
        A reserved Vector releases each phase after all compute members arrive.
        No device-to-host transfer occurs. This raw forward probe has no autograd,
        selected-set KL, SHMEM transport or model installation.
    """
    if schedule.rounds != 1:
        raise ValueError("fused DSA requires rounds=1; repeat complete invocations instead")
    _load_native()
    if torch.ops.hyper_parallel.dsa_fused_forward_version() != 1:
        raise RuntimeError("fused DSA requires adapter ABI 1; rebuild this checkout's payload")
    query, compressed, query_rope, key_rope = main_states
    index_query, index_key, weights = index_states
    device = query.device
    if retained is None:
        retained = torch.empty(MIXED_INDEXER_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    config = schedule.runtime_config(device)
    trace = torch.zeros((3, 20, 64), dtype=torch.int64, device=device)
    shape = (query.shape[0], 1, 2048)
    indices = torch.full(shape, -2, dtype=torch.int32, device=device)
    values = torch.full(shape, float("nan"), dtype=torch.bfloat16, device=device)
    attention = torch.full_like(query, float("nan"))
    maximum = torch.full((1, query.shape[0], query.shape[1]), float("nan"), dtype=torch.float32, device=device)
    denominator = torch.full_like(maximum, float("nan"))
    torch.ops.hyper_parallel.dsa_fused_forward_out(
        index_query, index_key, query, compressed[:, None, :], query_rope, key_rope[:, None, :],
        weights, cumulative_lengths, config, trace, retained, attention_scale,
        indices, values, attention, maximum, denominator)
    return indices, values, attention, maximum, denominator, tuple(trace.unbind()), retained


def validate_fused_dsa_traces(traces: tuple[torch.Tensor, ...], schedule: MixedSfaSchedule,
                               *, require_ld: bool) -> dict:
    """Require every member's LI/merge/SFA tasks and three ordered progress releases."""
    if len(traces) != 3:
        raise ValueError("fused DSA requires LI-main/LI-merge/SFA phase snapshots")
    snapshots = validate_device_phase_closure(traces, schedule)
    indexer = validate_mixed_indexer_traces(snapshots[:2], schedule, require_ld=require_ld)
    attention = validate_mixed_trace(snapshots[2], schedule)
    for group in range(schedule.compute_groups):
        if any(int(snapshots[2][group, offset + 4]) != 0 for offset in (0, 16, 32)):
            raise ValueError("SFA phase contains stale LI partition evidence")
    return {"indexer": indexer, "attention": attention, "device_phase_closure": True,
            "progress_group": schedule.compute_groups, "arrivals_per_phase": 3 * schedule.compute_groups,
            "phase_order": ["li_main", "li_merge", "sfa"]}
