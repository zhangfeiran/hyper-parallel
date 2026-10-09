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
"""Selected-KL callable phases with retained FP32 key/loss partials."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout, CannDsaReference, CannDsaSelection, CannDsaStats,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule, validate_mixed_trace
from hyper_parallel.core.multicore.torch.ops import _load_native


def mixed_kl_workspace_bytes(tokens: int, heads: int) -> int:
    """Return pinned K2048/Di128/Hindex64 non-deterministic retained HBM capacity."""
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    if not isinstance(heads, int) or isinstance(heads, bool) or heads not in (32, 64):
        raise ValueError("mixed KL supports H32/H64")
    per_core = 2 * (2048 * 576 * 2 + 2048 * 128 * 2 + heads * 2048 * 4
                    + 64 * 2048 * 4 * 2 + 2048 * 2 * 4 + 2048 * 128 * 4)
    return 20 * per_core + 512 + tokens * 128 * 4


@dataclass(frozen=True)
class MixedKlResult:
    """Owned raw index derivatives and unnormalized KL, plus explicit phase evidence."""

    gradients: tuple[torch.Tensor, ...]
    loss: torch.Tensor
    traces: tuple[torch.Tensor, ...]
    retained: torch.Tensor


def _submit_mixed_kl(main, index, indices, lengths, stats, config, scale):
    if torch.are_deterministic_algorithms_enabled():
        raise ValueError("mixed KL retains the non-deterministic native math")
    _load_native()
    if torch.ops.hyper_parallel.dsa_mixed_kl_version() != 1:
        raise RuntimeError("mixed KL ABI mismatch; rebuild this checkout's payload")
    query, compressed, query_rope, key_rope = main
    iq, ik, weight = index
    gradients = tuple(torch.full_like(tensor, float("nan")) for tensor in index)
    loss = torch.full((1,), float("nan"), dtype=torch.float32, device=query.device)
    # CANN inspects storage shape, so phase traces must each own their full storage.
    traces = tuple(torch.zeros((20, 64), dtype=torch.int64, device=query.device) for _ in range(3))
    retained = torch.empty(mixed_kl_workspace_bytes(query.shape[0], query.shape[1]),
                           dtype=torch.uint8, device=query.device)
    for phase in range(3):
        torch.ops.hyper_parallel.dsa_mixed_kl_out(
            query.detach(), compressed.detach()[:, None], iq.detach(), ik.detach()[:, None], weight.detach(),
            indices, stats.maximum.detach(), stats.denominator.detach(),
            query_rope.detach(), key_rope.detach()[:, None],
            lengths, lengths, config, traces[phase], retained, scale, phase,
            gradients[0], gradients[1][:, None], gradients[2], loss)
    return MixedKlResult(gradients, loss.reshape(()), traces, retained)


class MixedDsaKlProbe:
    """Explicit complete-selection KL phase probe, outside model/CP training dispatch."""

    def __init__(self, layout: CannDsaLayout, *, attention_scale: float, schedule: MixedSfaSchedule) -> None:
        """Reuse P0 selection ownership/admission and prepare fixed mixed-group metadata."""
        if schedule.rounds != 1:
            raise ValueError("mixed KL requires rounds=1; repeat whole invocations")
        self.reference = CannDsaReference(layout, attention_scale=attention_scale)
        self.schedule = schedule
        self.config = schedule.runtime_config(layout.device)
        self.lengths = layout.cumulative_lengths

    def gradients(self, main_inputs: tuple[torch.Tensor, ...], index_inputs: tuple[torch.Tensor, ...],
                  selection: CannDsaSelection, stats: CannDsaStats) -> MixedKlResult:
        """Evaluate raw selected KL and its three derivatives without auxiliary scaling."""
        self.reference._validate_main(main_inputs)
        self.reference._validate_index(index_inputs)
        indices = self.reference._native_indices(selection)
        return _submit_mixed_kl(main_inputs, index_inputs, indices, self.lengths, stats, self.config,
                                self.reference.attention_scale)

    def validate_trace(self, result: MixedKlResult, snapshots: tuple[torch.Tensor, ...]) -> dict:
        """Validate explicit completed CPU traces for initialization, compute and post."""
        if len(result.traces) != 3 or len(snapshots) != 3:
            raise ValueError("mixed KL requires three phase snapshots")
        return {"phase_order": ["initialize", "compute", "post"],
                "compute": [validate_mixed_trace(trace, self.schedule) for trace in snapshots]}
