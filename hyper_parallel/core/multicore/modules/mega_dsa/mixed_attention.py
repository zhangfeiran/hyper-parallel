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
"""Experimental CP1 absorbed SFA autograd over callable mixed forward/backward tiles."""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
    CannDsaSelection,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.torch.ops import _load_native


def mixed_sfa_grad_workspace_bytes(tokens: int, heads: int) -> int:
    """Bound pinned arch22 H32/H64 C512/Dr64/K2048 retained gradient workspace.

    Both scatter choices fit the three-slot upper bound. This is ordinary HBM,
    excluding the original kernel's internal 32 MiB system workspace. Host tiling
    independently checks the actual required capacity before every launch.
    """
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("tokens must be a positive integer")
    if not isinstance(heads, int) or isinstance(heads, bool) or heads not in (32, 64):
        raise ValueError("mixed SFA gradients support H32/H64")
    scratch_per_core = 128 * 576 * 2 * 4 + 128 * 512 * 2 * 2 + heads * 128 * 4 * 2 * 4
    gradients = sum((elements * 4 + 511) // 512 * 512
                    for elements in (tokens * heads * 576, tokens * 576, tokens * 512))
    scatter = 24 * 3 * 2048 * (576 + 512) * 4
    return min(tokens, 20) * scratch_per_core + gradients + scatter


def _ensure_native() -> None:
    _load_native()
    versions = (torch.ops.hyper_parallel.dsa_mixed_tile_version(), torch.ops.hyper_parallel.dsa_mixed_grad_version())
    if versions != (1, 1):
        raise RuntimeError("mixed SFA forward/backward ABI mismatch; rebuild this checkout's payload")


def mixed_sfa_backward_probe(states: tuple[torch.Tensor, ...], indices: torch.Tensor,
                             cumulative_lengths: torch.Tensor, config: torch.Tensor,
                             grad_output: torch.Tensor, forward: tuple[torch.Tensor, ...],
                             attention_scale: float, schedule: MixedSfaSchedule,
                             retained: torch.Tensor | None = None) -> tuple:
    """Run initialize/compute/post and return separate K/V gradients and phase traces.

    Args:
        states: Detached absorbed query, compressed K/V, query RoPE and key RoPE.
        indices: Complete owned sequence-local native selection [T,1,2048].
        cumulative_lengths: Prepared complete CP1 packed int32 lengths.
        config: Immutable prepared mixed-group runtime configuration.
        grad_output: BF16 compressed output gradient.
        forward: Saved independent output, maximum and denominator.
        attention_scale: Original model scale, used exactly once by native math.
        schedule: Single logical traversal; repeat whole three-phase invocations.
        retained: Optional independent scratch reused serially on one stream.

    Returns:
        Five native gradients (q, key, value, query RoPE, key RoPE), three phase
        traces and retained scratch. No device-to-host transfer occurs.

    Raises:
        ValueError: Unsupported deterministic mode or per-phase repeats.
    """
    if schedule.rounds != 1 or torch.are_deterministic_algorithms_enabled():
        raise ValueError("mixed SFA backward requires rounds=1 and non-deterministic mode")
    _ensure_native()
    query, compressed, query_rope, key_rope = (tensor.detach().contiguous() for tensor in states)
    key = compressed[:, None, :]
    rope = key_rope[:, None, :]
    output, maximum, denominator = (tensor.detach().contiguous() for tensor in forward)
    if retained is None:
        size = mixed_sfa_grad_workspace_bytes(query.shape[0], query.shape[1])
        retained = torch.empty(size, dtype=torch.uint8, device=query.device)
    gradients = tuple(torch.full_like(tensor, float("nan")) for tensor in (query, key, key, query_rope, rope))
    traces = tuple(schedule.new_trace(query.device) for _ in range(3))
    for phase, trace in enumerate(traces):
        torch.ops.hyper_parallel.dsa_mixed_grad_out(
            query, key, key, indices, grad_output.detach().contiguous(), output, maximum, denominator,
            cumulative_lengths, cumulative_lengths, query_rope, rope,
            config, trace, retained, attention_scale, phase, *gradients)
    return gradients, traces, retained


class _MixedAttention(torch.autograd.Function):
    """Keep forward saved state independent of backward accumulation scratch."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, compressed: torch.Tensor, query_rope: torch.Tensor,
                key_rope: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor,
                config: torch.Tensor, attention_scale: float, schedule: MixedSfaSchedule) -> torch.Tensor:
        """Save original versioned inputs and independent native output/statistics."""
        _ensure_native()
        native_query, native_c, native_qr, native_kr = (
            tensor.detach().contiguous() for tensor in (query, compressed, query_rope, key_rope))
        output = torch.full_like(native_query, float("nan"))
        maximum = torch.full((1, query.shape[0], query.shape[1]), float("nan"),
                             dtype=torch.float32, device=query.device)
        denominator = torch.full_like(maximum, float("nan"))
        trace = schedule.new_trace(query.device)
        key = native_c[:, None, :]
        torch.ops.hyper_parallel.dsa_mixed_tile_out(
            native_query, key, key, indices, lengths, lengths, native_qr, native_kr[:, None, :],
            config, trace, attention_scale, output, maximum, denominator)
        ctx.save_for_backward(query, compressed, query_rope, key_rope, indices, lengths,
                              config, output, maximum, denominator)
        ctx.attention_scale = attention_scale
        ctx.schedule = schedule
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Accumulate key/value contributions once and reject unsupported higher derivatives."""
        if torch.is_grad_enabled():
            raise ValueError("mixed SFA probe supports first-order backward only")
        query, compressed, query_rope, key_rope, indices, lengths, config, output, maximum, denominator = (
            ctx.saved_tensors)
        gradients, _, _ = mixed_sfa_backward_probe(
            (query, compressed, query_rope, key_rope), indices, lengths, config, grad_output,
            (output, maximum, denominator), ctx.attention_scale, ctx.schedule)
        grad_query, grad_key, grad_value, grad_qr, grad_kr = gradients
        return (grad_query, (grad_key + grad_value)[:, 0], grad_qr, grad_kr[:, 0],
                None, None, None, None, None)


class MixedDsaCoreProbe:
    """Explicit CP1 external-Top-K training probe with P0 selection admission.

    Projection parameters and KL remain outside this probe. Native atomic
    gradient accumulation uses the stock non-deterministic math. This object
    provides no automatic fallback, SHMEM communication or model integration.
    """

    def __init__(self, layout: CannDsaLayout, *, attention_scale: float, schedule: MixedSfaSchedule) -> None:
        """Prepare fixed runtime metadata while preserving the original attention scale."""
        if schedule.rounds != 1:
            raise ValueError("mixed SFA autograd requires rounds=1; repeat whole forward/backward invocations")
        self.reference = CannDsaReference(layout, attention_scale=attention_scale)
        self.schedule = schedule
        self.config = schedule.runtime_config(layout.device)

    def attention(self, query: torch.Tensor, compressed: torch.Tensor, query_rope: torch.Tensor,
                  key_rope: torch.Tensor, selection: CannDsaSelection) -> torch.Tensor:
        """Apply callable SFA to a complete opaque P0 selection and return compressed output."""
        # Internal coupling preserves P0 ownership/version guards without exporting raw selection storage.
        self.reference._validate_main((query, compressed, query_rope, key_rope))
        indices = self.reference._native_indices(selection)
        if query.shape[1] not in (32, 64):
            raise ValueError("mixed SFA autograd currently supports H32/H64")
        return _MixedAttention.apply(query, compressed, query_rope, key_rope, indices,
                                     self.reference.layout.length_tensor, self.config,
                                     self.reference.attention_scale, self.schedule)
