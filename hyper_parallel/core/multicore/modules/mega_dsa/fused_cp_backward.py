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

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedCpForwardResult,
    FusedCpSavedAttention,
    FusedDsaCpForwardProbe,
    _layout_signature,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_query import (
    DsaLocalQueryLayout,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_attention import (
    local_sfa_grad_workspace_bytes,
    mixed_sfa_grad_workspace_bytes,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule,
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
    _runtime_buffers: tuple[torch.Tensor, ...] = field(default=(), repr=False, compare=False)


class FusedDsaCpBackwardProbe:
    """Use local Q cotangents, original gradient math and peer-exclusive FP32 inbox stripes."""

    def __init__(self, forward: FusedDsaCpForwardProbe, *, sparse_return: bool = False,
                 schedule: MixedSfaSchedule | None = None, group_private_partials: bool = False) -> None:
        """Reuse collectively prepared geometry and declare a common gradient-return protocol.

        Args:
            forward: Prepared CP forward backend with a bound shared workspace.
            sparse_return: Clear sender-exclusive owner inboxes before readiness
                and omit numerically zero merged main-gradient rows. FP32 sums
                retain sender order. This optional protocol supports CP up to
                eight and requires the sparse-return native capability.
            schedule: Optional independently declared backward compute schedule.
                Defaults to the forward schedule. Forward state and LI/KL
                transport geometry remain unchanged; backward requires rounds=1.
            group_private_partials: Use private FP32 KV accumulators per compute
                group and a fixed-order reduction before post/owner return.
        """
        self.backend = forward
        self.workspace = forward.workspace
        error = None
        chosen = forward.schedule if schedule is None else schedule
        if not isinstance(sparse_return, bool):
            error = "sparse gradient return must be a boolean"
        elif not isinstance(group_private_partials, bool):
            error = "group-private partials must be a boolean"
        elif sparse_return and forward.layout.cp_size > 8:
            error = "sparse gradient return supports CP sizes up to eight"
        elif not isinstance(chosen, MixedSfaSchedule) or chosen.rounds != 1:
            error = "backward requires a MixedSfaSchedule with rounds=1"
        schedule_words = ((chosen.compute_groups, chosen.rounds) if isinstance(chosen, MixedSfaSchedule) else None)
        declaration = (sparse_return, error, schedule_words, group_private_partials)
        declarations = [declaration]
        if forward.layout.cp_size > 1:
            declarations = [None] * forward.layout.cp_size
            dist.all_gather_object(declarations, declaration, group=forward.layout.group)
        errors = [value[1] for value in declarations if value[1] is not None]
        if errors:
            raise ValueError("; ".join(errors))
        if any(value[0] != sparse_return for value in declarations):
            raise ValueError("CP members disagree on sparse gradient return")
        if any(len(value) != 4 or value[2] != schedule_words for value in declarations):
            raise ValueError("CP members disagree on backward schedule")
        if any(value[3] != group_private_partials for value in declarations):
            raise ValueError("CP members disagree on group-private partials")
        self.group_private_partials = group_private_partials
        self._group_private_declaration = group_private_partials
        self.sparse_return = sparse_return
        self._sparse_declaration = sparse_return
        self.schedule = chosen
        self._schedule_declaration = chosen
        self.config = (forward.config if chosen == forward.schedule else chosen.runtime_config(forward.layout.device))
        self._config_identity = self.config
        self._config_version = self.config._version
        meta = forward.layout.batch_meta
        offsets = tuple(meta.token_local_offsets[token] for token in meta.kv_global_ids)
        self.local_owner_order = torch.tensor(offsets, dtype=torch.long, device=forward.layout.device)

    @staticmethod
    def check_native_support() -> None:
        """Require the compact valid-prefix gradient adapter before entering a native lease."""
        _load_native()
        if torch.ops.hyper_parallel.dsa_fused_grad_version() != 2:
            raise RuntimeError("fused CP backward requires adapter ABI 2; rebuild this checkout's payload")

    @staticmethod
    def check_group_partials_native_support() -> None:
        """Require the explicit private-group capability before publication or acquiring a lease."""
        _load_native()
        version = getattr(torch.ops.hyper_parallel, "dsa_group_partials_version", None)
        if version is None or version() != 1:
            raise RuntimeError("group-private gradients require adapter ABI 1; rebuild this checkout's payload")

    def backward(self, forward: FusedCpForwardResult, grad_output: torch.Tensor) -> FusedCpBackwardResult:
        """Return Q, shared compressed-KV, Q-RoPE and K-RoPE gradients in original local orders.

        Only local Q rows contribute to this member's native cotangent. Every
        member calls collectively, including zero-row owners. FP32 owner sums
        include dK and dV once and cast only after ordered peer reduction.
        This explicit raw backward has no indexer/KL or autograd installation.
        """
        return self._backward_saved(forward.saved, tuple(forward.output.shape), grad_output)

    @staticmethod
    def check_local_native_support() -> None:
        """Require the explicit-position local gradient ABI before acquiring a workspace lease."""
        _load_native()
        version = getattr(torch.ops.hyper_parallel, "dsa_local_cp_grad_version", None)
        if version is None or version() != 1:
            raise RuntimeError("local CP backward requires adapter ABI 1; rebuild this checkout's payload")

    def backward_local(self, forward: FusedCpForwardResult, grad_output: torch.Tensor, *,
                       defer_owner_cast: bool = False) -> FusedCpBackwardResult:
        """Return main-attention gradients for actual local Q, including empty tiles.

        Query cotangents use caller order and stay local. Complete-key FP32
        partials are returned to owners before casting; this raw baseline still
        uses complete owner return and has no KL or autograd installation.
        With defer_owner_cast=True, shared C/K-RoPE gradients remain FP32 for
        a subsequent TP SUM; query-head gradients retain their native dtype.
        """
        return self._backward_saved(forward.saved, tuple(forward.output.shape), grad_output, local_queries=True,
                                    defer_owner_cast=defer_owner_cast)

    def _backward_saved(self, saved: FusedCpSavedAttention | None, shape: tuple,
                        grad_output: torch.Tensor, *, local_queries: bool = False,
                        defer_owner_cast: bool = False) -> FusedCpBackwardResult:
        if (not isinstance(self.group_private_partials, bool)
                or self.group_private_partials != self._group_private_declaration):
            raise ValueError("group-private partials were modified; prepare a new probe")
        if self.group_private_partials:
            self.check_group_partials_native_support()
        if not isinstance(defer_owner_cast, bool):
            raise TypeError("owner cast deferral must be a boolean")
        if not isinstance(self.sparse_return, bool) or self.sparse_return != self._sparse_declaration:
            raise ValueError("gradient return mode was modified; prepare a new probe")
        if (self.schedule != self._schedule_declaration or self.config is not self._config_identity
                or self.config._version != self._config_version):
            raise ValueError("backward schedule or runtime configuration was modified; prepare a new probe")
        if self.sparse_return:
            _load_native()
            version = getattr(torch.ops.hyper_parallel, "dsa_sparse_cp_grad_version", None)
            if version is None or version() != 1:
                raise RuntimeError("sparse gradient return requires adapter ABI 1; rebuild this checkout's payload")
        if (saved is None or saved.backend is not self.backend
                or _layout_signature(saved.batch_meta) != self.backend.signature):
            raise ValueError("fused CP backward requires this backend's saved attention state")
        if local_queries and saved.query_scope != "local":
            raise ValueError("local CP backward requires local-query saved state")
        if not local_queries and saved.query_scope != "global":
            raise ValueError("local-query saved state requires the local backward adapter")
        saved.validate_versions()
        if saved.geometry != (self.backend.heads, self.backend.scale, self.backend.schedule):
            raise ValueError("fused CP backward requires unchanged forward geometry/scale/schedule")
        if torch.are_deterministic_algorithms_enabled():
            raise ValueError("fused CP backward retains the non-deterministic native math")
        if (grad_output.device != self.backend.layout.device or grad_output.dtype != torch.bfloat16
                or tuple(grad_output.shape) != shape or grad_output.requires_grad):
            raise ValueError("fused CP backward requires an explicitly detached local BF16 output cotangent")
        if local_queries:
            self.check_local_native_support()
        else:
            self.check_native_support()
        invocation = self.workspace.prepare(saved.batch_meta)
        self.backend.check_mapping_support()
        with self.workspace.lease(invocation, direction="backward"):
            return self._submit(saved, grad_output, invocation, local_queries=local_queries,
                                defer_owner_cast=defer_owner_cast)

    def _metadata(self, meta, epoch: int) -> torch.Tensor:
        offsets = {name: offset for name, offset, _, _ in self.workspace.layout}
        protocol = (2 if self.sparse_return else 1) + (2 if self.workspace.mapped_transport else 0)
        values = (_GRAD_MAGIC, protocol, epoch, meta.cp_rank, len(meta.cp_ranks),
                  meta.heap_generation,
                  meta.layer, meta.microbatch, meta.invocation, self.backend.layout_hash,
                  offsets["gradient_inbox"], self.workspace.spec.owner_gradient_tokens,
                  len(meta.kv_global_ids), offsets["events"], len(self.backend.runs),
                  meta.global_valid_queries, self.workspace.arena_bytes, self.workspace.spec.source_tokens)
        return torch.tensor(self.workspace.transport_metadata(values, meta), dtype=torch.int64).to(
            self.backend.layout.device, non_blocking=True)

    def _submit(self, saved, grad_output, invocation, *, local_queries=False,
                defer_owner_cast=False) -> FusedCpBackwardResult:
        query, compressed, query_rope, key_rope = saved.states
        local = self.backend.layout.local_query_ids
        layout = saved.local_query_layout if local_queries else None
        if local_queries:
            full_grad = layout.pack(grad_output)
        else:
            full_grad = torch.zeros_like(query)
            full_grad.index_copy_(0, local, grad_output)
        gradients = tuple(torch.full_like(tensor, float("nan")) for tensor in
                          (query, compressed[:, None], compressed[:, None], query_rope, key_rope[:, None]))
        traces = torch.zeros((3, 20, 64), dtype=torch.int64, device=query.device)
        transport_trace = torch.zeros(32, dtype=torch.int64, device=query.device)
        required = (local_sfa_grad_workspace_bytes(query.shape[0], compressed.shape[0], query.shape[1])
                    if local_queries else mixed_sfa_grad_workspace_bytes(query.shape[0], query.shape[1]))
        retained = torch.empty(required, dtype=torch.uint8, device=query.device)
        owner = torch.full((self.workspace.spec.owner_gradient_tokens, 576), float("nan"),
                           dtype=torch.float32, device=query.device)
        partial_shape = ((self.schedule.compute_groups+1, compressed.shape[0], 1088)
                         if self.group_private_partials else (compressed.shape[0], 1088))
        partial_storage = torch.full(partial_shape, float("nan"), dtype=torch.float32, device=query.device)
        partials = partial_storage[0] if self.group_private_partials else partial_storage
        epoch = self.workspace.next_transport_epoch(invocation)
        metadata = self._metadata(saved.batch_meta, epoch)
        lengths = self.backend.native_layout.length_tensor
        native = (torch.ops.hyper_parallel.dsa_local_cp_grad_out if local_queries
                  else torch.ops.hyper_parallel.dsa_fused_cp_grad_out)
        positions = (layout.positions,) if local_queries else ()
        native(
            query, compressed[:, None], compressed[:, None], saved.indices, full_grad,
            *saved.forward, layout.query_lengths if local_queries else lengths, lengths,
            query_rope, key_rope[:, None], self.config,
            traces, retained, self.backend.scale, *gradients, self.workspace.arena,
            metadata, self.backend.requests, transport_trace, owner, partial_storage, *positions)
        restored = owner.index_select(0, self.local_owner_order)
        owner_fp32 = (restored[:, :512].contiguous(), restored[:, 512:].contiguous())
        query_gradients = (tuple(layout.restore(gradients[index]) for index in (0, 3)) if local_queries
                           else tuple(gradients[index].index_select(0, local) for index in (0, 3)))
        owner_gradients = owner_fp32 if defer_owner_cast else tuple(tensor.bfloat16() for tensor in owner_fp32)
        local_gradients = (query_gradients[0], owner_gradients[0], query_gradients[1], owner_gradients[1])
        return FusedCpBackwardResult(local_gradients, owner_fp32, gradients, partials, tuple(traces.unbind()),
                                      transport_trace, retained, epoch, (metadata, self.config, partial_storage))

    def validate_trace(self, result: FusedCpBackwardResult, snapshots: tuple[torch.Tensor, ...],
                       transport: torch.Tensor) -> dict:
        """Certify all gradient phases, completed peer writes and ordered owner reduction."""
        if transport.device.type != "cpu" or transport.dtype != torch.int64 or transport.shape != (32,):
            raise ValueError("gradient transport evidence requires an explicit CPU int64 [32] snapshot")
        if len(snapshots) != 3:
            raise ValueError("fused CP gradients require initialize/compute/post phase snapshots")
        if self.group_private_partials:
            checked = []
            for phase, snapshot in enumerate(snapshots):
                if snapshot.device.type != "cpu" or snapshot.dtype != torch.int64 or snapshot.shape != (20, 64):
                    raise ValueError("group-partial readiness requires explicit CPU int64 [20,64] snapshots")
                if int(snapshot[self.schedule.compute_groups, 26]) != (3 if phase == 2 else 0):
                    raise ValueError("group-private partials lack ordered reduction readiness")
                value = snapshot.clone()
                value[self.schedule.compute_groups, 26] = 0
                checked.append(value)
            snapshots = tuple(checked)
        phases = validate_device_phase_closure(snapshots, self.schedule)
        compute = tuple(validate_mixed_trace(trace, self.schedule) for trace in phases)
        meta = self.backend.layout.batch_meta
        size, local, total = len(meta.cp_ranks), len(meta.kv_global_ids), meta.global_valid_queries
        sparse = None
        if self.sparse_return:
            sent, skipped, received, remote_received = (int(transport[index]) for index in (12, 13, 14, 15))
            remote_sent = int(transport[5]) // 2816
            if (not 0 <= sent <= total or skipped != total - sent
                    or not 0 <= remote_sent <= min(sent, total - local)
                    or not 0 <= received <= size * local
                    or not 0 <= remote_received <= min(received, (size - 1) * local)
                    or tuple(transport[16:19].tolist()) != ((size - 1) * 8, (size - 1) * 8, size * local * 576 * 4)):
                raise ValueError("sparse gradient counters or zero-cleared owner extent are inconsistent")
            sparse = {"sent_rows": sent, "skipped_zero_rows": skipped, "received_rows": received,
                      "remote_received_rows": remote_received, "remote_counter_control_bytes": (size - 1) * 8,
                      "owner_inbox_clear_bytes": size * local * 576 * 4,
                      "owner_inbox_read_bytes": size * local * 576 * 4}
            expected = (result.epoch, result.epoch, size, size, sent * 2816, remote_sent * 2816,
                        received * 2816, remote_received * 2816)
        else:
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
        reserved = torch.cat((transport[19:20] if self.sparse_return else transport[12:20], transport[22:]))
        if bool(reserved.any()):
            raise ValueError("fused CP gradient trace contains unsupported control records")
        return {"compute": compute, "phase_order": ["initialize", "compute", "post_and_owner_return"],
                "ready_write_ack": True, "epoch": result.epoch, "remote_sent_bytes": expected[5],
                "remote_received_bytes": expected[7], "fp32_owner_order": list(range(size)),
                "dk_dv_workspace_offsets": offsets, "sparse_return": sparse,
                "group_private_partials": self.group_private_partials,
                "group_partial_ready": self.group_private_partials}


def _save_attention(ctx, backend, states, index_states, result):
    saved = result.saved
    ctx.save_for_backward(*states, *index_states, *saved.states, saved.indices, *saved.forward)
    ctx.backend = backend
    ctx.batch_meta = saved.batch_meta
    ctx.versions = saved.versions
    ctx.geometry = saved.geometry
    ctx.token = saved._token
    ctx.output_shape = tuple(result.output.shape)
    ctx.query_scope = saved.query_scope
    ctx.local_query_layout = saved.local_query_layout


def _attention_backward(ctx, grad_output):
    if torch.is_grad_enabled():
        raise ValueError("fused CP attention supports first-order backward only")
    if ctx.geometry != (ctx.backend.backend.heads, ctx.backend.backend.scale, ctx.backend.backend.schedule):
        raise ValueError("fused CP backward requires unchanged forward geometry/scale/schedule")
    tensors = ctx.saved_tensors
    if len(tensors) != 15:
        raise RuntimeError("fused CP attention saved-state contract is incomplete")
    saved = FusedCpSavedAttention(ctx.batch_meta, tuple(tensors[7:11]), tensors[11], tuple(tensors[12:]),
                                  ctx.versions, ctx.backend.backend, ctx.geometry, ctx.token,
                                  ctx.query_scope, ctx.local_query_layout)
    cotangent = grad_output.detach().contiguous()
    if ctx.query_scope == "local":
        return ctx.backend._backward_saved(saved, ctx.output_shape, cotangent, local_queries=True)
    return ctx.backend._backward_saved(saved, ctx.output_shape, cotangent)


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
        _save_attention(ctx, backend, states, index_states, result)
        return result.output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return owner-local main gradients without installing a gradient path for hard selection."""
        result = _attention_backward(ctx, grad_output)
        return (None, None, *result.gradients, None)


class _FusedLocalCpAttention(torch.autograd.Function):
    """First-order local attention with a caller-owned cold layout for checkpoint replay."""

    @staticmethod
    def forward(ctx: Any, backend: FusedDsaCpBackwardProbe, invocation: DsaWorkspaceInvocation,
                query: torch.Tensor, compressed: torch.Tensor, query_rope: torch.Tensor,
                key_rope: torch.Tensor, index_states: tuple[torch.Tensor, ...],
                query_layout: DsaLocalQueryLayout) -> torch.Tensor:
        """Save native activations through tensor hooks and retain immutable cold query metadata."""
        states = (query, compressed, query_rope, key_rope)
        result = backend.backend.forward_local(invocation, tuple(tensor.detach() for tensor in states),
                                                tuple(tensor.detach() for tensor in index_states),
                                                query_layout=query_layout)
        _save_attention(ctx, backend, states, index_states, result)
        # Empty tiles expose the native output itself; keep its autograd metadata detached.
        return result.output.detach()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return four owner-local main gradients while keeping hard selection detached."""
        result = _attention_backward(ctx, grad_output)
        return (None, None, *result.gradients, None, None)


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

    def attention_local(self, invocation: DsaWorkspaceInvocation, main_states: tuple[torch.Tensor, ...],
                        index_states: tuple[torch.Tensor, ...], *, query_layout: DsaLocalQueryLayout) -> torch.Tensor:
        """Compute local-query main attention with explicit immutable tile identity.

        Args:
            invocation: Prepared workspace invocation with unchanged owner geometry.
            main_states: Caller-ordered local Q/Q-RoPE and complete owner-local C/K-RoPE.
            index_states: Local index Q/weights and complete owner-local index K; no main-loss gradients.
            query_layout: Layout returned by this backend's cold preparation. Capture it
                explicitly in checkpoint callbacks so replay survives preparation of later tiles.

        Returns:
            Compressed BF16 output in caller query order, with first-order main gradients.
            Selected KL and auxiliary loss normalization remain separate.
        """
        if len(main_states) != 4 or len(index_states) != 3:
            raise ValueError("local CP attention requires four main and three index states")
        if not isinstance(query_layout, DsaLocalQueryLayout):
            raise TypeError("local CP attention requires its cold prepared query layout")
        return _FusedLocalCpAttention.apply(self.backward_backend, invocation, *main_states, index_states, query_layout)
