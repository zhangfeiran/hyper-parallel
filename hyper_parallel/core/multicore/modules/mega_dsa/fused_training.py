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
"""Single-launch LI/TopK/SFA and raw selected-KL over shared mixed worker groups."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.fused_forward import validate_fused_dsa_traces
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_indexer import MIXED_INDEXER_WORKSPACE_BYTES
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_kl import mixed_kl_workspace_bytes
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule, validate_device_phase_closure, validate_mixed_trace,
)
from hyper_parallel.core.multicore.torch.ops import _load_native


@dataclass(frozen=True)
class FusedTrainingResult:
    """Owned forward outputs and raw KL derivatives before any auxiliary scaling."""

    forward: tuple[torch.Tensor, ...]
    index_gradients: tuple[torch.Tensor, ...]
    loss: torch.Tensor
    traces: tuple[torch.Tensor, ...]
    retained: tuple[torch.Tensor, ...]


def fused_dsa_training_probe(index_states: tuple[torch.Tensor, ...], main_states: tuple[torch.Tensor, ...],
                              layout: CannDsaLayout, attention_scale: float,
                              schedule: MixedSfaSchedule) -> FusedTrainingResult:
    """Run six dependent math phases in one native launch with prepared packed metadata.

    Inputs are detached BF16 main/index states and BF16/FP32 signed merge weights.
    The native indexer creates the complete selection. Every invocation owns its
    outputs, LI/KL scratch and six phase traces; host cumulative lengths come from
    prepared metadata. This raw probe has no autograd or CP transport.
    """
    if schedule.rounds != 1 or torch.are_deterministic_algorithms_enabled():
        raise ValueError("fused training requires rounds=1 and non-deterministic KL")
    _load_native()
    if torch.ops.hyper_parallel.dsa_fused_training_version() != 1:
        raise RuntimeError("fused training ABI mismatch; rebuild this checkout's payload")
    query, compressed, query_rope, key_rope = main_states
    iq, ik, weight = index_states
    device = query.device
    retained = (torch.empty(MIXED_INDEXER_WORKSPACE_BYTES, dtype=torch.uint8, device=device),
                torch.empty(mixed_kl_workspace_bytes(query.shape[0], query.shape[1]),
                            dtype=torch.uint8, device=device))
    trace = torch.zeros((6, 20, 64), dtype=torch.int64, device=device)
    indices = torch.full((query.shape[0], 1, 2048), -2, dtype=torch.int32, device=device)
    values = torch.full_like(indices, float("nan"), dtype=torch.bfloat16)
    output = torch.full_like(query, float("nan"))
    maximum = torch.full((1, query.shape[0], query.shape[1]), float("nan"), dtype=torch.float32, device=device)
    denominator = torch.full_like(maximum, float("nan"))
    gradients = tuple(torch.full_like(tensor, float("nan")) for tensor in index_states)
    loss = torch.full((1,), float("nan"), dtype=torch.float32, device=device)
    torch.ops.hyper_parallel.dsa_fused_training_out(
        iq, ik[:, None], query, compressed[:, None], query_rope, key_rope[:, None], weight,
        layout.length_tensor, schedule.runtime_config(device), trace, *retained,
        layout.cumulative_lengths, attention_scale, indices, values, output, maximum, denominator,
        gradients[0], gradients[1][:, None], gradients[2], loss)
    return FusedTrainingResult((indices, values, output, maximum, denominator), gradients,
                               loss.reshape(()), tuple(trace.unbind()), retained)


def validate_fused_training_traces(traces: tuple[torch.Tensor, ...], schedule: MixedSfaSchedule,
                                    *, require_ld: bool) -> dict:
    """Require six ordered releases and complete tasks from every compute member."""
    if len(traces) != 6:
        raise ValueError("fused training requires six phase snapshots")
    snapshots = validate_device_phase_closure(traces, schedule)
    forward = validate_fused_dsa_traces(traces[:3], schedule, require_ld=require_ld)
    kl = [validate_mixed_trace(trace, schedule) for trace in snapshots[3:]]
    for trace in snapshots[3:]:
        for group in range(schedule.compute_groups):
            if any(int(trace[group, offset + 4]) != 0 for offset in (0, 16, 32)):
                raise ValueError("KL phase contains stale LI partition evidence")
    return {"forward": forward, "kl": kl, "device_phase_closure": True,
            "phase_order": ["li_main", "li_merge", "sfa", "kl_init", "kl_compute", "kl_post"]}
