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
"""Raw selected-KL derivatives for genuine local queries and complete packed keys."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedCpForwardResult, _layout_signature,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_kl import (
    mixed_kl_key_partials, mixed_kl_workspace_bytes,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule, validate_mixed_trace
from hyper_parallel.core.multicore.torch.ops import _load_native


@dataclass(frozen=True)
class LocalKlResult:
    """Invocation-owned raw derivatives, with FP32 key partials before owner reduction.

    Query and merge-weight derivatives use caller-local row order. Key outputs
    and partials use complete global packed order. Loss is the unnormalized sum
    over this tile only. Coefficients, normalization and upstream cotangents are
    applied by the training bridge, exactly once per contribution.
    """

    gradients: tuple[torch.Tensor, ...]
    key_partials: torch.Tensor
    loss: torch.Tensor
    traces: tuple[torch.Tensor, ...]
    retained: torch.Tensor


def local_kl_gradients(forward: FusedCpForwardResult) -> LocalKlResult:
    """Evaluate selected KL from immutable native local-forward state.

    Args:
        forward: This backend's invocation-owned local-query forward result.
            Main teacher activations are detached. Independent local gradients
            are restored to the original caller row order, with no Q collective.

    Returns:
        Raw first-order index derivatives and summed KL. Empty tiles produce
        owned zero key partials without launching a dummy query. Read results
        on the submission stream or establish an explicit event dependency.

    Raises:
        ValueError: Saved activations, geometry or index states have changed.
        RuntimeError: The loaded payload lacks the required native local KL ABI.
    """
    if not isinstance(forward, FusedCpForwardResult) or forward.saved is None:
        raise ValueError("local KL requires native forward-produced saved state")
    saved = forward.saved
    if saved.query_scope != "local" or saved.local_query_layout is None:
        raise ValueError("local KL requires local-query saved state")
    saved.validate_versions()
    backend = saved.backend
    if (_layout_signature(saved.batch_meta) != backend.signature
            or saved.geometry != (backend.heads, backend.scale, backend.schedule)):
        raise ValueError("local KL requires unchanged forward geometry and ownership")
    if (len(forward.index_states) != 3 or len(forward.index_versions) != 3
            or forward.index_versions != tuple(tensor._version for tensor in forward.index_states)):
        raise ValueError("local KL index states were modified or lack producer version metadata")
    if torch.are_deterministic_algorithms_enabled():
        raise ValueError("local KL retains the non-deterministic native math")
    _load_native()
    version = getattr(torch.ops.hyper_parallel, "dsa_local_mixed_kl_version", None)
    if version is None or version() != 1:
        raise RuntimeError("local KL requires adapter ABI 1; rebuild this checkout's payload")
    return _submit_local_kl(forward)


def _submit_local_kl(forward: FusedCpForwardResult) -> LocalKlResult:
    saved = forward.saved
    local = saved.local_query_layout
    query, key, query_rope, key_rope = saved.states
    iq, ik, weight = forward.index_states
    gradients = tuple(torch.full_like(tensor, float("nan")) for tensor in (iq, ik, weight))
    loss = torch.zeros((), dtype=torch.float32, device=query.device)
    if query.shape[0] == 0:
        for tensor in gradients:
            tensor.zero_()
        partials = torch.zeros(ik.shape, dtype=torch.float32, device=query.device)
        retained = torch.empty(0, dtype=torch.uint8, device=query.device)
        return LocalKlResult(gradients, partials, loss, (), retained)
    traces = tuple(torch.zeros((20, 64), dtype=torch.int64, device=query.device) for _ in range(3))
    retained = torch.empty(mixed_kl_workspace_bytes(key.shape[0], query.shape[1]),
                           dtype=torch.uint8, device=query.device)
    maximum, denominator = saved.forward[1:]
    for phase in range(3):
        torch.ops.hyper_parallel.dsa_local_mixed_kl_out(
            query.detach(), key.detach()[:, None], iq.detach(), ik.detach()[:, None], weight.detach(),
            saved.indices, maximum.detach(), denominator.detach(), query_rope.detach(), key_rope.detach()[:, None],
            local.cumulative_queries, saved.batch_meta.global_cu_seqlens[1:], saved.backend.config,
            traces[phase], retained, local.positions, saved.geometry[1], phase,
            gradients[0], gradients[1][:, None], gradients[2], loss.reshape(1))
    partials = mixed_kl_key_partials(retained, tokens=key.shape[0], heads=query.shape[1])
    restored = (local.restore(gradients[0]), gradients[1], local.restore(gradients[2]))
    return LocalKlResult(restored, partials, loss, traces, retained)


def validate_local_kl_trace(result: LocalKlResult, snapshots: tuple[torch.Tensor, ...],
                            schedule: MixedSfaSchedule) -> dict:
    """Validate explicit completed CPU phase snapshots, including a genuine empty tile."""
    if len(result.traces) != len(snapshots) or len(snapshots) not in (0, 3):
        raise ValueError("local KL requires exactly three phase snapshots or a genuine empty tile")
    return {"phase_order": ["initialize", "compute", "post"] if snapshots else [],
            "empty_query": not snapshots,
            "compute": [validate_mixed_trace(trace, schedule) for trace in snapshots]}
