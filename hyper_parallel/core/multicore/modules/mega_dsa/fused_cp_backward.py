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
"""Single-kernel CP attention backward with explicit FP32 owner gradient return."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedCpForwardResult,
    FusedCpSavedAttention,
    FusedDsaCpForwardProbe,
    _layout_signature,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_attention import (
    mixed_sfa_grad_workspace_bytes,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    validate_device_phase_closure,
    validate_mixed_trace,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
)
from hyper_parallel.core.multicore.torch.ops import _load_native

_GRAD_MAGIC = 0x4850445341434731


@dataclass(frozen=True)
class FusedCpBackwardResult:
    """Local gradients, FP32 owner sums and explicit compute/communication evidence."""

    gradients: tuple[torch.Tensor, ...]
    owner_fp32: tuple[torch.Tensor, ...]
    native_gradients: tuple[torch.Tensor, ...]
    partials: torch.Tensor
    phase_traces: tuple[torch.Tensor, ...]
    transport_trace: torch.Tensor
    retained: torch.Tensor
    epoch: int


class FusedDsaCpBackwardProbe:
    """Use local Q cotangents, original gradient math and peer-exclusive FP32 inbox stripes."""

    def __init__(self, forward: FusedDsaCpForwardProbe) -> None:
        """Reuse collectively prepared geometry and owner requests from the forward backend."""
        self.backend = forward
        self.workspace = forward.workspace
        meta = forward.layout.batch_meta
        offsets = tuple(meta.token_local_offsets[token] for token in meta.kv_global_ids)
        self.local_owner_order = torch.tensor(offsets, dtype=torch.long, device=forward.layout.device)

    def backward(self, forward: FusedCpForwardResult, grad_output: torch.Tensor) -> FusedCpBackwardResult:
        """Return Q, shared compressed-KV, Q-RoPE and K-RoPE gradients in original local orders.

        Only local Q rows contribute to this member's native cotangent. Every
        member calls collectively, including zero-row owners. FP32 owner sums
        include dK and dV once and cast only after ordered peer reduction.
        This explicit raw backward has no indexer/KL or autograd installation.
        """
        return self._backward_saved(forward.saved, tuple(forward.output.shape), grad_output)

    def _backward_saved(self, saved: FusedCpSavedAttention | None, shape: tuple,
                        grad_output: torch.Tensor) -> FusedCpBackwardResult:
        if (saved is None or saved.backend is not self.backend
                or _layout_signature(saved.batch_meta) != self.backend.signature):
            raise ValueError("fused CP backward requires this backend's saved attention state")
        saved.validate_versions()
        if saved.geometry != (self.backend.heads, self.backend.scale, self.backend.schedule):
            raise ValueError("fused CP backward requires unchanged forward geometry/scale/schedule")
        if torch.are_deterministic_algorithms_enabled():
            raise ValueError("fused CP backward retains the non-deterministic native math")
        if (grad_output.device != self.backend.layout.device or grad_output.dtype != torch.bfloat16
                or tuple(grad_output.shape) != shape or grad_output.requires_grad):
            raise ValueError("fused CP backward requires an explicitly detached local BF16 output cotangent")
        _load_native()
        if torch.ops.hyper_parallel.dsa_fused_grad_version() != 1:
            raise RuntimeError("fused CP backward requires adapter ABI 1; rebuild this checkout's payload")
        invocation = self.workspace.prepare(saved.batch_meta)
        with self.workspace.lease(invocation, direction="backward"):
            return self._submit(saved, grad_output, invocation)

    def _metadata(self, meta, epoch: int) -> torch.Tensor:
        offsets = {name: offset for name, offset, _, _ in self.workspace.layout}
        values = (_GRAD_MAGIC, 1, epoch, meta.cp_rank, len(meta.cp_ranks), meta.heap_generation,
                  meta.layer, meta.microbatch, meta.invocation, self.backend.layout_hash,
                  offsets["gradient_inbox"], self.workspace.spec.owner_gradient_tokens,
                  len(meta.kv_global_ids), offsets["events"], len(self.backend.runs),
                  meta.global_valid_queries, self.workspace.arena_bytes, self.workspace.spec.source_tokens)
        return torch.tensor(values, dtype=torch.int64).to(self.backend.layout.device, non_blocking=True)

    def _submit(self, saved, grad_output, invocation) -> FusedCpBackwardResult:
        query, compressed, query_rope, key_rope = saved.states
        local = self.backend.layout.local_query_ids
        full_grad = torch.zeros_like(query)
        full_grad.index_copy_(0, local, grad_output)
        gradients = tuple(torch.full_like(tensor, float("nan")) for tensor in
                          (query, compressed[:, None], compressed[:, None], query_rope, key_rope[:, None]))
        traces = torch.zeros((3, 20, 64), dtype=torch.int64, device=query.device)
        transport_trace = torch.zeros(32, dtype=torch.int64, device=query.device)
        retained = torch.empty(mixed_sfa_grad_workspace_bytes(query.shape[0], query.shape[1]),
                               dtype=torch.uint8, device=query.device)
        owner = torch.full((self.workspace.spec.owner_gradient_tokens, 576), float("nan"),
                           dtype=torch.float32, device=query.device)
        partials = torch.full((query.shape[0], 1088), float("nan"), dtype=torch.float32, device=query.device)
        epoch = self.workspace.next_transport_epoch(invocation)
        metadata = self._metadata(saved.batch_meta, epoch)
        lengths = self.backend.native_layout.length_tensor
        torch.ops.hyper_parallel.dsa_fused_cp_grad_out(
            query, compressed[:, None], compressed[:, None], saved.indices, full_grad,
            *saved.forward, lengths, lengths, query_rope, key_rope[:, None], self.backend.config,
            traces, retained, self.backend.scale, *gradients, self.workspace.arena,
            metadata, self.backend.requests, transport_trace, owner, partials)
        restored = owner.index_select(0, self.local_owner_order)
        owner_fp32 = (restored[:, :512].contiguous(), restored[:, 512:].contiguous())
        local_gradients = (gradients[0].index_select(0, local), owner_fp32[0].bfloat16(),
                           gradients[3].index_select(0, local), owner_fp32[1].bfloat16())
        return FusedCpBackwardResult(local_gradients, owner_fp32, gradients, partials, tuple(traces.unbind()),
                                      transport_trace, retained, epoch)

    def validate_trace(self, result: FusedCpBackwardResult, snapshots: tuple[torch.Tensor, ...],
                       transport: torch.Tensor) -> dict:
        """Certify all gradient phases, completed peer writes and ordered owner reduction."""
        if transport.device.type != "cpu" or transport.dtype != torch.int64 or transport.shape != (32,):
            raise ValueError("gradient transport evidence requires an explicit CPU int64 [32] snapshot")
        if len(snapshots) != 3:
            raise ValueError("fused CP gradients require initialize/compute/post phase snapshots")
        phases = validate_device_phase_closure(snapshots, self.backend.schedule)
        compute = tuple(validate_mixed_trace(trace, self.backend.schedule) for trace in phases)
        meta = self.backend.layout.batch_meta
        size, local, total = len(meta.cp_ranks), len(meta.kv_global_ids), meta.global_valid_queries
        expected = (result.epoch, result.epoch, size, size, total * 2816, (total - local) * 2816,
                    size * local * 2816, (size - 1) * local * 2816)
        if tuple(transport[:8].tolist()) != expected:
            raise ValueError("fused CP gradients lack complete ready/write/ACK or exact transport bytes")
        times = tuple(int(transport[index]) for index in (8, 9, 10, 11))
        if times[0] <= 0 or not times[0] < times[1] <= times[2] < times[3]:
            raise ValueError("fused CP gradients lack ordered transfer and owner-reduction intervals")
        offsets = tuple(int(transport[index]) for index in (20, 21))
        sizes = (total * 576 * 4, total * 512 * 4)
        if offsets[0] < 0 or offsets[1] < offsets[0] + sizes[0] or offsets[1] + sizes[1] > result.retained.numel():
            raise ValueError("fused CP gradient accumulator offsets exceed the retained workspace")
        reserved = torch.cat((transport[12:20], transport[22:]))
        if bool(reserved.any()):
            raise ValueError("fused CP gradient trace contains unsupported control records")
        return {"compute": compute, "phase_order": ["initialize", "compute", "post_and_owner_return"],
                "ready_write_ack": True, "epoch": result.epoch, "remote_sent_bytes": expected[5],
                "remote_received_bytes": expected[7], "fp32_owner_order": list(range(size)),
                "dk_dv_workspace_offsets": offsets}


class _FusedCpAttention(torch.autograd.Function):
    """Versioned first-order main-attention bridge; indexer/KL gradients remain separate."""

    @staticmethod
    def forward(ctx: Any, backend: FusedDsaCpBackwardProbe, invocation: DsaWorkspaceInvocation,
                query: torch.Tensor, compressed: torch.Tensor, query_rope: torch.Tensor,
                key_rope: torch.Tensor, index_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Save versioned originals and independent native buffers for the main attention VJP."""
        states = (query, compressed, query_rope, key_rope)
        result = backend.backend.forward(invocation, tuple(tensor.detach() for tensor in states),
                                         tuple(tensor.detach() for tensor in index_states))
        saved = result.saved
        ctx.save_for_backward(*states, *index_states, *saved.states, saved.indices, *saved.forward)
        ctx.backend = backend
        ctx.batch_meta = saved.batch_meta
        ctx.versions = saved.versions
        ctx.geometry = saved.geometry
        ctx.token = saved._token
        ctx.output_shape = tuple(result.output.shape)
        return result.output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return owner-local main gradients without installing a gradient path for hard selection."""
        if torch.is_grad_enabled():
            raise ValueError("fused CP attention supports first-order backward only")
        if ctx.geometry != (ctx.backend.backend.heads, ctx.backend.backend.scale, ctx.backend.backend.schedule):
            raise ValueError("fused CP backward requires unchanged forward geometry/scale/schedule")
        tensors = ctx.saved_tensors
        if len(tensors) != 15:
            raise RuntimeError("fused CP attention saved-state contract is incomplete")
        saved = FusedCpSavedAttention(ctx.batch_meta, tuple(tensors[7:11]), tensors[11], tuple(tensors[12:]),
                                      ctx.versions, ctx.backend.backend, ctx.geometry, ctx.token)
        result = ctx.backend._backward_saved(saved, ctx.output_shape, grad_output.detach().contiguous())
        return (None, None, *result.gradients, None)


class FusedDsaCpAttentionProbe:
    """Explicit CP main-attention autograd over fused forward/backward and native owner return.

    This prototype computes native LI selection but detaches its inputs from
    the main attention objective. Selected-set KL, projection ownership and
    declared downstream loss normalization remain external requirements.
    """

    def __init__(self, forward: FusedDsaCpForwardProbe) -> None:
        """Prepare the backward owner permutation once for an existing forward backend."""
        self.backward_backend = FusedDsaCpBackwardProbe(forward)

    def attention(self, invocation: DsaWorkspaceInvocation, main_states: tuple[torch.Tensor, ...],
                  index_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute local-Q compressed output with one gradient contribution per local owner input."""
        if len(main_states) != 4 or len(index_states) != 3:
            raise ValueError("fused CP attention requires four main and three index states")
        return _FusedCpAttention.apply(self.backward_backend, invocation, *main_states, index_states)
