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
"""Unfused same-vendor backward control for the fused FP32 compressed VJP boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.mixed_attention import (
    mixed_sfa_backward_probe,
    mixed_sfa_gradient_partials,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


@dataclass(frozen=True)
class MixedFp32ControlResult:
    """Own gradients and FP32 snapshots independently of reusable native scratch.

    ``gradients`` contains query, compressed K/V, query RoPE and key RoPE VJPs.
    Key partials include RoPE in columns 512:576. Traces retain three phases
    per head chunk. This is diagnostic evidence, not an independent math oracle.
    """

    gradients: tuple[torch.Tensor, ...]
    key_partials: torch.Tensor
    value_partials: torch.Tensor
    traces: tuple[torch.Tensor, ...]
    head_ranges: tuple[tuple[int, int], ...]


def _validate_inputs(states: tuple[torch.Tensor, ...], grad_output: torch.Tensor,
                     forward: tuple[torch.Tensor, ...]) -> None:
    if len(states) != 4 or len(forward) != 3:
        raise ValueError("control requires four main states and three saved forward tensors")
    query, compressed, query_rope, key_rope = states
    if query.ndim != 3 or query.shape[0] <= 0 or query.shape[1] not in (32, 64, 128) or query.shape[2] != 512:
        raise ValueError("control requires complete nonempty TND query with H32/H64/H128 and C512")
    tokens, heads = query.shape[:2]
    shapes = ((tokens, 512), (tokens, heads, 64), (tokens, 64), query.shape, query.shape)
    tensors = (compressed, query_rope, key_rope, grad_output, forward[0])
    for tensor, shape in zip(tensors, shapes):
        if tensor.shape != shape or tensor.dtype != torch.bfloat16 or tensor.device != query.device:
            raise ValueError("control main states, output and cotangent must share BF16 complete-query geometry")
    if query.dtype != torch.bfloat16:
        raise ValueError("control requires BF16 query")
    for tensor in forward[1:]:
        if tensor.shape != (1, tokens, heads) or tensor.dtype != torch.float32 or tensor.device != query.device:
            raise ValueError("control requires saved FP32 maximum/sum with shape [1,T,H]")


@torch.no_grad()
def mixed_sfa_fp32_backward_control(
    states: tuple[torch.Tensor, ...], indices: torch.Tensor, cumulative_lengths: torch.Tensor,
    grad_output: torch.Tensor, forward: tuple[torch.Tensor, ...], *,
    attention_scale: float, schedule: MixedSfaSchedule,
) -> MixedFp32ControlResult:
    """Replay saved complete-query SFA with FP32 dK+dV and one activation cast.

    Args:
        states: BF16 complete packed query, compressed K/V, query/key RoPE.
        indices: Prepared sequence-local native selection [T,1,2048].
        cumulative_lengths: Prepared complete packed lengths on the device.
        grad_output: Actual BF16 compressed-output cotangent.
        forward: Actual saved TND output and FP32 [1,T,H] maximum/sum.
        attention_scale: Original SFA scale.
        schedule: Single-traversal mixed schedule, without CP/TP communication.

    Returns:
        Owned FP32 key/value planes, gradients and per-chunk device traces.
        H128 uses two H64 invocations with sliced original saved statistics;
        shared key/value and key-RoPE partials sum in FP32 before conversion.

    Note:
        Shares the locked vendor gradient tiles with the fused candidate.
        Never use this control to fit an independent acceptance profile.
        All work stays on the caller stream; scratch is reused only after its
        partials have been copied/added on that same stream. No host sync occurs.
        Group atomics remain nondeterministic. This API does not install autograd.
    """
    _validate_inputs(states, grad_output, forward)
    if schedule.rounds != 1 or torch.are_deterministic_algorithms_enabled():
        raise ValueError("control requires rounds=1 and non-deterministic mode")
    query, compressed, query_rope, key_rope = states
    config = schedule.runtime_config(query.device)
    ranges = tuple((start, min(start + 64, query.shape[1])) for start in range(0, query.shape[1], 64))
    query_gradients, rope_gradients, all_traces = [], [], []
    key_sum = torch.zeros((query.shape[0], 576), dtype=torch.float32, device=query.device)
    value_sum = torch.zeros_like(compressed, dtype=torch.float32)
    retained = None
    for start, end in ranges:
        chunk_states = (query[:, start:end], compressed, query_rope[:, start:end], key_rope)
        chunk_forward = (forward[0][:, start:end], forward[1][:, :, start:end], forward[2][:, :, start:end])
        gradients, traces, retained = mixed_sfa_backward_probe(
            chunk_states, indices, cumulative_lengths, config, grad_output[:, start:end],
            chunk_forward, attention_scale, schedule, retained)
        key, value = mixed_sfa_gradient_partials(
            retained, query_tokens=query.shape[0], key_tokens=query.shape[0], heads=end - start)
        key_sum.add_(key)
        value_sum.add_(value)
        query_gradients.append(gradients[0])
        rope_gradients.append(gradients[3])
        all_traces.extend(traces)
    result = (torch.cat(query_gradients, dim=1), (key_sum[:, :512] + value_sum).to(query.dtype),
              torch.cat(rope_gradients, dim=1), key_sum[:, 512:].to(query.dtype))
    return MixedFp32ControlResult(result, key_sum, value_sum, tuple(all_traces), ranges)
