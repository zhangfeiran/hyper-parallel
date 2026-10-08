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
"""External Top-K CP sparse attention with fused native forward and owner backward."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaStats
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    _SAVED_ATTENTION_TOKEN,
    FusedCpSavedAttention,
    FusedDsaCpForwardProbe,
    _layout_signature,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_backward import (
    FusedDsaCpBackwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule,
    validate_device_phase_closure,
    validate_mixed_trace,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.torch.ops import _load_native

_CORE_SELECTION_TOKEN = object()


class MegaDsaCoreSelection:
    """Owned, versioned external selection prepared collectively outside the execution hot path.

    CPU preparation checks global IDs and uniqueness before device admission.
    Future device indexer producers require a separate admission path. Exports
    never share the saved native selection's storage.
    """

    def __init__(self, backend: FusedDsaCpForwardProbe, indices: torch.Tensor, *, backward_ready: bool = False,
                 _token: object = None) -> None:
        """Bind an admitted immutable selection to one collectively prepared CP layout."""
        if _token is not _CORE_SELECTION_TOKEN:
            raise ValueError("obtain a core selection from MegaDsaCore.prepare_selection")
        self._backend = backend
        self._backward_ready = backward_ready
        with torch.inference_mode(False):
            self._indices = indices.detach().clone()
        self._version = self._indices._version

    @property
    def backward_ready(self) -> bool:
        """Whether every owner selection matches the existing native causal-count contract."""
        return self._backward_ready

    def validate(self, backend: FusedDsaCpForwardProbe) -> None:
        """Reject foreign layout admission and mutations before acquiring a workspace lease."""
        if backend is not self._backend:
            raise ValueError("core selection belongs to another prepared CP backend")
        if self._indices._version != self._version:
            raise ValueError("core selection storage was modified; prepare a new selection")

    def to_global_indices(self) -> torch.Tensor:
        """Export independent local-Q global packed int32 indices on the prepared device."""
        self.validate(self._backend)
        return self._backend.native_layout.sequence_to_global_indices(self._indices).index_select(
            0, self._backend.layout.local_query_ids)

    def native_indices(self, backend: FusedDsaCpForwardProbe) -> torch.Tensor:
        """Return an independent native selection after validating the admitted producer."""
        self.validate(backend)
        return self._indices.clone()


@dataclass(frozen=True)
class MegaDsaCoreForwardResult:
    """Owned main output, native statistics and explicit forward transport evidence."""

    output: torch.Tensor
    maximum: torch.Tensor
    denominator: torch.Tensor
    phase_traces: tuple[torch.Tensor, ...]
    transport_trace: torch.Tensor
    epoch: int
    saved: FusedCpSavedAttention


class _CoreAttention(torch.autograd.Function):
    """Keep all saved tensors under autograd hooks and return only four main input gradients."""

    @staticmethod
    def forward(ctx: Any, core: MegaDsaCore, invocation: DsaWorkspaceInvocation,
                selection: MegaDsaCoreSelection, query: torch.Tensor, compressed: torch.Tensor,
                query_rope: torch.Tensor, key_rope: torch.Tensor) -> tuple:
        """Save original versions plus independent native activations for delayed backward."""
        states = (query, compressed, query_rope, key_rope)
        result = core.raw_forward(invocation, tuple(tensor.detach() for tensor in states), selection)
        saved = result.saved
        ctx.save_for_backward(*states, *saved.states, saved.indices, *saved.forward)
        ctx.backend = core.backward_backend
        ctx.batch_meta, ctx.versions = saved.batch_meta, saved.versions
        ctx.geometry, ctx.token = saved.geometry, saved._token
        ctx.output_shape = tuple(result.output.shape)
        ctx.mark_non_differentiable(result.maximum, result.denominator)
        ctx.set_materialize_grads(False)
        return result.output, result.maximum, result.denominator

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor | None, _maximum: Any, _denominator: Any) -> tuple:
        """Run native CP backward once for each local attention objective contribution."""
        if torch.is_grad_enabled():
            raise ValueError("MegaDsaCore supports first-order backward only")
        if ctx.geometry != (ctx.backend.backend.heads, ctx.backend.backend.scale, ctx.backend.backend.schedule):
            raise ValueError("core backward requires unchanged forward geometry/scale/schedule")
        tensors = ctx.saved_tensors
        saved = FusedCpSavedAttention(ctx.batch_meta, tuple(tensors[4:8]), tensors[8], tuple(tensors[9:]),
                                      ctx.versions, ctx.backend.backend, ctx.geometry, ctx.token)
        if grad_output is None:
            grad_output = tensors[0].new_zeros(ctx.output_shape)
        result = ctx.backend._backward_saved(saved, ctx.output_shape, grad_output.detach().contiguous())
        return (None, None, None, *result.gradients)


class MegaDsaCore(torch.nn.Module):
    """Parameter-free external Top-K CP attention with a declared BF16 C512/RoPE64 backend.

    Call every member in matching collective order, including empty owners.
    Preparation is collective; forward/backward replicate packed Q, pull main
    KV inside the native attention launch, and return merged FP32 owner sums.
    This initial path admits CPU-prepared selections, not arbitrary device IDs.
    Projection, selected KL and model loss normalization belong to the caller.
    """

    def __init__(self, workspace: MegaDsaWorkspace, invocation: DsaWorkspaceInvocation, *, heads: int,
                 attention_scale: float, schedule: MixedSfaSchedule) -> None:
        """Prepare layout, transport geometry and backward owner permutations collectively."""
        super().__init__()
        self.backend = FusedDsaCpForwardProbe(workspace, invocation, heads=heads,
                                             attention_scale=attention_scale, schedule=schedule)
        self.backward_backend = FusedDsaCpBackwardProbe(self.backend)
        self.prepared_invocation = invocation

    def prepare_selection(self, global_indices: torch.Tensor) -> MegaDsaCoreSelection:
        """Admit CPU int32 [local Q,2048], then collectively gather and convert once.

        Reject unknown IDs, duplicates and foreign shapes. Cross-sequence and
        future IDs are masked; -1 may occur anywhere and compacts stably to the
        tail. Every peer observes a preparation error before the device gather.
        """
        error = None
        backward_ready = False
        try:
            backward_ready = self._validate_selection(global_indices)
        except ValueError as exc:
            error = str(exc)
        layout = self.backend.layout
        declarations = [(error, backward_ready)]
        if layout.cp_size > 1:
            declarations = [None] * layout.cp_size
            dist.all_gather_object(declarations, (error, backward_ready), group=layout.group)
        errors = [item[0] for item in declarations if item[0] is not None]
        if errors:
            raise ValueError("; ".join(errors))
        backward_ready = all(item[1] for item in declarations)
        local = global_indices.to(layout.device).index_select(0, layout.query_order)
        padded = torch.full((layout.padded_tokens, 2048), -1, dtype=torch.int32, device=layout.device)
        padded[:layout.local_tokens] = local
        gathered = torch.empty((layout.cp_size * layout.padded_tokens, 2048), dtype=torch.int32,
                               device=layout.device)
        if layout.cp_size == 1:
            gathered.copy_(padded)
        else:
            dist.all_gather_into_tensor(gathered, padded, group=layout.group)
        full = gathered.index_select(0, layout.global_order)
        native = self.backend.native_layout.global_to_sequence_indices(full)
        return MegaDsaCoreSelection(self.backend, native, backward_ready=backward_ready, _token=_CORE_SELECTION_TOKEN)

    def _validate_selection(self, indices: torch.Tensor) -> bool:
        meta = self.backend.layout.batch_meta
        if (not isinstance(indices, torch.Tensor) or indices.device.type != "cpu" or indices.dtype != torch.int32
                or indices.shape != (len(meta.q_global_ids), 2048)):
            raise ValueError("core preparation requires CPU int32 [local Q,2048] global packed IDs")
        backward_ready = True
        for query, row in zip(meta.q_global_ids, indices.tolist()):
            sequence, position = meta.sequence_position(query)
            legal = 0
            seen = set()
            for token in row:
                if token == -1:
                    continue
                key_sequence, key_position = meta.sequence_position(token)
                legal += int(key_sequence == sequence and key_position <= position)
                if token in seen:
                    raise ValueError("external Top-K must not contain duplicate valid token IDs")
                seen.add(token)
            backward_ready = backward_ready and legal == min(2048, position + 1)
        return backward_ready

    def _validate_invocation(self, invocation: DsaWorkspaceInvocation) -> None:
        meta, prepared = invocation.batch_meta, self.backend.layout.batch_meta
        if (invocation.workspace is not self.backend.workspace or _layout_signature(meta) != self.backend.signature
                or (meta.q_global_ids, meta.kv_global_ids, meta.cp_rank) != (
                    prepared.q_global_ids, prepared.kv_global_ids, prepared.cp_rank)):
            raise ValueError("core invocation must preserve prepared ownership, local orders and heap generation")

    def raw_forward(self, invocation: DsaWorkspaceInvocation, states: tuple[torch.Tensor, ...],
                    selection: MegaDsaCoreSelection) -> MegaDsaCoreForwardResult:
        """Submit detached main states under one lease and return owned native evidence."""
        self._validate_invocation(invocation)
        if not isinstance(selection, MegaDsaCoreSelection):
            raise TypeError("core requires an admitted MegaDsaCoreSelection")
        selection.validate(self.backend)
        if len(states) != 4 or any(tensor.requires_grad for tensor in states):
            raise ValueError("core raw forward requires four detached main tensors")
        _load_native()
        if torch.ops.hyper_parallel.dsa_cp_attention_version() != 1:
            raise RuntimeError("core CP attention requires adapter ABI 1; rebuild this checkout's payload")
        with torch.inference_mode(False), self.backend.workspace.lease(invocation, direction="forward"):
            return self._submit(invocation, states, selection)

    def _submit(self, invocation, states, selection) -> MegaDsaCoreForwardResult:
        backend = self.backend
        query, compressed, query_rope, key_rope = states
        full_query, full_rope = backend.layout.gather_query_fields(
            (query, query_rope), ((backend.heads, 512), (backend.heads, 64)))
        full_query, full_rope = full_query.contiguous(), full_rope.contiguous()
        tokens, device = full_query.shape[0], full_query.device
        keys = tuple(torch.full((tokens, 1, width), float("nan"), dtype=torch.bfloat16, device=device)
                     for width in (512, 64))
        output = torch.full_like(full_query, float("nan"))
        maximum = torch.full((1, tokens, backend.heads), float("nan"), dtype=torch.float32, device=device)
        denominator = torch.full_like(maximum, float("nan"))
        trace = torch.zeros((20, 64), dtype=torch.int64, device=device)
        transport = torch.zeros(32, dtype=torch.int64, device=device)
        # Own the selection per invocation; retained backward cannot depend on the preparation object.
        indices = selection.native_indices(backend)
        backend.workspace.publish_main_states(invocation, (compressed, key_rope))
        epoch = backend.workspace.next_transport_epoch(invocation)
        torch.ops.hyper_parallel.dsa_cp_attention_out(
            full_query, keys[0], full_rope, keys[1], indices, backend.native_layout.length_tensor,
            backend.config, trace, backend.workspace.arena, backend._metadata(invocation, epoch),
            backend.requests, transport, backend.scale, output, maximum, denominator)
        saved_states = (full_query, keys[0][:, 0], full_rope, keys[1][:, 0])
        tensors = (*saved_states, indices, output, maximum, denominator)
        saved = FusedCpSavedAttention(invocation.batch_meta, saved_states, indices, (output, maximum, denominator),
                                      tuple(tensor._version for tensor in tensors), backend,
                                      (backend.heads, backend.scale, backend.schedule), _SAVED_ATTENTION_TOKEN)
        local = backend.layout.local_query_ids
        return MegaDsaCoreForwardResult(output.index_select(0, local), maximum.index_select(1, local),
                                         denominator.index_select(1, local), (trace,), transport, epoch, saved)

    def forward(self, query: torch.Tensor, compressed_kv: torch.Tensor, query_rope: torch.Tensor,
                key_rope: torch.Tensor, topk_indices: MegaDsaCoreSelection,
                batch_meta: DsaBatchMeta) -> tuple[torch.Tensor, CannDsaStats]:
        """Return local-Q compressed attention and nondifferentiable native max/sum statistics.

        Inputs use their prepared owner-local storage order. The caller retains
        parameter ownership; hard selected IDs have no main-objective gradient.
        Layer/microbatch/invocation may change without reconstructing layout.
        """
        if not isinstance(topk_indices, MegaDsaCoreSelection):
            raise TypeError("core requires an admitted MegaDsaCoreSelection")
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (
                query, compressed_kv, query_rope, key_rope)) and not topk_indices.backward_ready:
            raise NotImplementedError("external Top-K training with extra padding awaits the native selected-count fix")
        invocation = replace(self.prepared_invocation, batch_meta=batch_meta)
        output, maximum, denominator = _CoreAttention.apply(
            self, invocation, topk_indices, query, compressed_kv, query_rope, key_rope)
        return output, CannDsaStats(maximum, denominator)

    def validate_trace(self, result: MegaDsaCoreForwardResult, snapshots: tuple[torch.Tensor, ...],
                       transport: torch.Tensor) -> dict:
        """Certify attention team closure, actual main-KV bytes and complete ready/read ACK."""
        if len(snapshots) != 1:
            raise ValueError("core forward requires one attention phase snapshot")
        phase = validate_device_phase_closure(snapshots, self.backend.schedule)[0]
        compute = validate_mixed_trace(phase, self.backend.schedule)
        if transport.device.type != "cpu" or transport.dtype != torch.int64 or transport.shape != (32,):
            raise ValueError("core transport evidence requires CPU int64 [32]")
        layout = self.backend.layout
        total = layout.batch_meta.global_valid_queries
        remote = total - layout.local_tokens
        expected = (0, result.epoch, layout.cp_size, layout.cp_size, 0, total * 1152, 0, remote * 1152)
        if tuple(transport[:8].tolist()) != expected or bool(torch.cat((transport[8:10], transport[12:])).any()):
            raise ValueError("core transport lacks complete ready/ACK or exact main-only byte counts")
        if not 0 < int(transport[10]) < int(transport[11]):
            raise ValueError("core transport lacks ordered main-KV copy timestamps")
        return {"compute": compute, "ready_ack": True, "epoch": result.epoch,
                "remote_main_bytes": remote * 1152, "remote_index_bytes": 0}
