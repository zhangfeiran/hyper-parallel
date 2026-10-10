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
"""Experimental CP query replication with device KV pull during fused LI/SFA forward."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from itertools import pairwise

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.cp_reference import DsaCpLayout
from hyper_parallel.core.multicore.modules.mega_dsa.fused_forward import (
    validate_fused_dsa_traces,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_indexer import (
    MIXED_INDEXER_WORKSPACE_BYTES,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_kl import mixed_kl_workspace_bytes
from hyper_parallel.core.multicore.modules.mega_dsa.fused_training import validate_fused_training_traces
from hyper_parallel.core.multicore.modules.mega_dsa.selected_requests import (
    selected_request_rows, validate_selected_requests, validate_selected_training_traces,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.torch.ops import _load_native

_TRANSPORT_MAGIC = 0x4850445341435031
_SAVED_ATTENTION_TOKEN = object()


def cp_owner_runs(meta: DsaBatchMeta) -> tuple[tuple[int, ...], ...]:
    """Cover every global key once with contiguous owner-offset and destination runs."""
    runs = []
    for token, (owner, offset) in enumerate(zip(meta.token_owners, meta.token_local_offsets)):
        if runs and runs[-1][0] == owner and runs[-1][1] + runs[-1][3] == offset:
            peer, source, destination, count = runs[-1]
            runs[-1] = (peer, source, destination, count + 1)
        else:
            runs.append((owner, offset, token, 1))
    return tuple(runs)


def _layout_signature(meta: DsaBatchMeta) -> tuple:
    return (meta.global_cu_seqlens, meta.token_owners, meta.token_local_offsets,
            meta.cp_ranks, meta.root_pes, meta.layout_id, meta.heap_generation)


@dataclass(frozen=True)
class FusedCpSavedAttention:
    """Invocation-owned attention activations, independent of reusable symmetric scratch."""

    batch_meta: DsaBatchMeta
    states: tuple[torch.Tensor, ...]
    indices: torch.Tensor
    forward: tuple[torch.Tensor, ...]
    versions: tuple[int, ...]
    backend: FusedDsaCpForwardProbe
    geometry: tuple[int, float, MixedSfaSchedule]
    _token: object = field(default=None, repr=False, compare=False)

    def validate_versions(self) -> None:
        """Reject mutation of any saved activation before a delayed or retained backward."""
        if self._token is not _SAVED_ATTENTION_TOKEN:
            raise ValueError("fused CP backward requires native forward-produced saved state")
        tensors = (*self.states, self.indices, *self.forward)
        if tuple(tensor._version for tensor in tensors) != self.versions:
            raise ValueError("fused CP saved attention state was modified after forward")


@dataclass(frozen=True)
class FusedCpForwardResult:
    """Owned local outputs and explicit native transport/compute evidence.

    Global key buffers are ordinary invocation-owned HBM, independent of the
    symmetric publication arena and suitable for a future saved backward state.
    Optional KL outputs are raw saved derivatives and loss; the training module
    owns normalization and autograd integration.
    """

    output: torch.Tensor
    global_indices: torch.Tensor
    values: torch.Tensor
    maximum: torch.Tensor
    denominator: torch.Tensor
    global_keys: tuple[torch.Tensor, ...]
    phase_traces: tuple[torch.Tensor, ...]
    transport_trace: torch.Tensor
    epoch: int
    saved: FusedCpSavedAttention | None = None
    index_states: tuple[torch.Tensor, ...] = ()
    kl_gradients: tuple[torch.Tensor, ...] = ()
    kl_loss: torch.Tensor | None = None
    selected_requests: tuple[torch.Tensor, ...] = ()


class FusedDsaCpForwardProbe:
    """Replicate Q through the CP gather and pull actual remote KV in one math kernel.

    The reserved progress Vector copies index-K before LI starts, then copies
    compressed KV/RoPE while LI executes. It waits for every peer's completed
    read receipt before kernel completion permits publication storage reuse.
    Support is one node, MTE-reachable CP/root members, H32/H64 and K2048.
    selected_kl extends the same launch with three KL phases. Native CP
    backward and model installation use their separate entry points.
    selected_pull inserts three membership and request/pull stages after LI merge;
    it preserves full query scope and destination addresses.
    """

    def __init__(self, workspace: MegaDsaWorkspace, invocation: DsaWorkspaceInvocation,
                 *, heads: int, attention_scale: float, schedule: MixedSfaSchedule) -> None:
        """Collectively prepare ownership, full packed layout and immutable device requests."""
        if heads not in (32, 64) or schedule.rounds != 1 or not math.isfinite(attention_scale) or attention_scale <= 0:
            raise ValueError("fused CP requires H32/H64 and rounds=1")
        if workspace.spec.dtype != torch.bfloat16 or (workspace.spec.compressed_dim, workspace.spec.rope_dim,
                                                     workspace.spec.index_dim) != (512, 64, 128):
            raise ValueError("fused CP requires BF16 C512/RoPE64/index128 publication")
        if workspace.buffers["events"].numel() * 4 < 256:
            raise ValueError("fused CP requires independent 128-byte ready and ACK cache lines")
        if invocation.workspace is not workspace:
            raise ValueError("fused CP preparation requires the invocation's owning workspace")
        declaration = (heads, float(attention_scale))
        if len(workspace.root.members) > 1:
            declarations = [None] * len(workspace.root.members)
            dist.all_gather_object(declarations, declaration, group=workspace.root.group)
            if any(item != declaration for item in declarations):
                raise ValueError("fused CP members disagree on head count or attention scale")
        self.workspace = workspace
        self.layout = DsaCpLayout(invocation.batch_meta, workspace.arena.device, group=workspace.root.group)
        self.signature = _layout_signature(invocation.batch_meta)
        self.heads = heads
        self.scale = attention_scale
        self.schedule = schedule
        self.config = schedule.runtime_config(self.layout.device)
        self.runs = cp_owner_runs(invocation.batch_meta)
        self.requests = torch.tensor(self.runs, dtype=torch.int64, device=self.layout.device)
        lengths = tuple(end - start for start, end in zip(invocation.batch_meta.global_cu_seqlens,
                                                        invocation.batch_meta.global_cu_seqlens[1:]))
        self.native_layout = CannDsaLayout(DsaBatchMeta.packed(lengths), self.layout.device)
        self.layout_hash = int.from_bytes(hashlib.sha256(repr(self.signature).encode()).digest()[:8], "little") % 2**63
        self.selected_rows: torch.Tensor | None = None
        self.selected_rows_version: int | None = None

    def prepare_selected_requests(self) -> None:
        """Prepare immutable row addresses collectively before selected-transfer submissions."""
        if self.layout.batch_meta.global_valid_queries > (2**31 - 1) // 2048:
            raise ValueError("selected membership occurrences exceed int32 capacity")
        self.selected_rows = selected_request_rows(self.layout.batch_meta, self.layout.device)
        self.selected_rows_version = self.selected_rows._version

    def _metadata(self, invocation: DsaWorkspaceInvocation, epoch: int) -> torch.Tensor:
        meta = invocation.batch_meta
        offsets = {name: offset for name, offset, _, _ in self.workspace.layout}
        values = (_TRANSPORT_MAGIC, 1, epoch, meta.cp_rank, len(meta.cp_ranks), meta.heap_generation,
                  meta.layer, meta.microbatch, meta.invocation, self.layout_hash,
                  offsets["source_index"], offsets["source_compressed"], offsets["source_rope"],
                  offsets["events"], len(self.runs), meta.global_valid_queries,
                  self.workspace.arena_bytes, self.workspace.spec.source_tokens)
        return torch.tensor(values, dtype=torch.int64, device=self.layout.device)

    def forward(self, invocation: DsaWorkspaceInvocation, main_states: tuple[torch.Tensor, ...],
                index_states: tuple[torch.Tensor, ...], *, selected_kl: bool = False,
                selected_pull: bool = False) -> FusedCpForwardResult:
        """Submit a complete forward under one lease without host barriers or device snapshots.

        Main states are owner-local Q, shared compressed KV, Q-RoPE, K-RoPE;
        index states are Q-index [Tq,64,128], K-index [Tkv,128] and scaled
        BF16/FP32 weights [Tq,64]. All inputs must be explicitly detached and
        follow the prepared Q/KV storage orders. Every rank calls collectively.
        selected_kl additionally emits raw KL derivatives/loss in the same kernel;
        all members must agree. Normalization and auxiliary scaling belong to the caller.
        selected_pull requires prepared row addresses and generates
        its main-KV request count on device after TopK merge.
        """
        if invocation.workspace is not self.workspace or _layout_signature(invocation.batch_meta) != self.signature:
            raise ValueError("fused CP invocation must preserve the prepared ownership/layout/generation")
        meta = invocation.batch_meta
        prepared = self.layout.batch_meta
        if (meta.q_global_ids, meta.kv_global_ids, meta.cp_rank) != (
                prepared.q_global_ids, prepared.kv_global_ids, prepared.cp_rank):
            raise ValueError("fused CP invocation must preserve the prepared local Q/KV storage orders")
        if len(main_states) != 4 or len(index_states) != 3 or any(tensor.requires_grad
                                                                for tensor in (*main_states, *index_states)):
            raise ValueError("fused CP raw forward requires four detached main and three detached index states")
        if selected_pull and self.selected_rows is None:
            raise ValueError("selected pull requires prepared row addresses")
        if selected_pull and self.selected_rows._version != self.selected_rows_version:
            raise ValueError("selected row addresses were modified; prepare a new backend")
        _load_native()
        if selected_pull and selected_kl and torch.ops.hyper_parallel.dsa_selected_cp_training_version() != 1:
            raise RuntimeError("selected CP training requires adapter ABI 1; rebuild this checkout's payload")
        if selected_kl and torch.ops.hyper_parallel.dsa_fused_training_version() != 1:
            raise RuntimeError("fused CP training requires adapter ABI 1; rebuild this checkout's payload")
        if selected_pull and not selected_kl and torch.ops.hyper_parallel.dsa_selected_cp_forward_version() != 1:
            raise RuntimeError("selected CP forward requires adapter ABI 1; rebuild this checkout's payload")
        if not selected_kl and not selected_pull and torch.ops.hyper_parallel.dsa_fused_forward_version() != 2:
            raise RuntimeError("fused CP requires adapter ABI 2; rebuild this checkout's payload")
        with self.workspace.lease(invocation, direction="forward"):
            return self._submit(invocation, main_states, index_states,
                                selected_kl=selected_kl, selected_pull=selected_pull)

    def _submit(self, invocation: DsaWorkspaceInvocation, main_states: tuple,
                index_states: tuple, *, selected_kl: bool = False, selected_pull: bool = False) -> FusedCpForwardResult:
        query, compressed, query_rope, key_rope = main_states
        index_query, index_key, weights = index_states
        full_iq, full_q, full_qr = self.layout.gather_query_fields(
            (index_query, query, query_rope), ((64, 128), (self.heads, 512), (self.heads, 64)))
        full_iq, full_q, full_qr = (tensor.contiguous() for tensor in (full_iq, full_q, full_qr))
        full_weights, = self.layout.gather_query_fields((weights,), ((64,),))
        full_weights = full_weights.contiguous()
        tokens = invocation.batch_meta.global_valid_queries
        device = self.layout.device
        keys = tuple(torch.full((tokens, 1, width), float("nan"), dtype=torch.bfloat16, device=device)
                     for width in (128, 512, 64))
        indices = torch.full((tokens, 1, 2048), -2, dtype=torch.int32, device=device)
        values = torch.full_like(indices, float("nan"), dtype=torch.bfloat16)
        output = torch.full_like(full_q, float("nan"))
        maximum = torch.full((1, tokens, self.heads), float("nan"), dtype=torch.float32, device=device)
        denominator = torch.full_like(maximum, float("nan"))
        phases = 6 if selected_kl else 3
        if selected_pull:
            phases += 3
        trace = torch.zeros((phases, 20, 64), dtype=torch.int64, device=device)
        transport_trace = torch.zeros(32, dtype=torch.int64, device=device)
        retained = torch.empty(MIXED_INDEXER_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self.workspace.publish_owner_states(invocation, (compressed, key_rope, index_key))
        epoch = self.workspace.next_transport_epoch(invocation)
        metadata = self._metadata(invocation, epoch)
        kl_gradients, kl_loss = (), None
        selected_buffers = ()
        extra = ()
        if selected_pull:
            membership = torch.empty((tokens + 7) // 8 * 8, dtype=torch.int32, device=device)
            selected_table = torch.empty((tokens, 4), dtype=torch.int64, device=device)
            selected_counts = torch.empty(16, dtype=torch.int64, device=device)
            selected_buffers = (selected_table, selected_counts, membership)
            extra = (self.selected_rows, membership, selected_table, selected_counts)
        if selected_kl:
            kl_retained = torch.empty(mixed_kl_workspace_bytes(tokens, self.heads), dtype=torch.uint8, device=device)
            kl_gradients = tuple(torch.full_like(tensor, float("nan")) for tensor in
                                 (full_iq, keys[0][:, 0], full_weights))
            kl_loss = torch.full((1,), float("nan"), dtype=torch.float32, device=device)
            submit = torch.ops.hyper_parallel.dsa_fused_cp_training_out
            if selected_pull:
                submit = torch.ops.hyper_parallel.dsa_selected_cp_training_out
            submit(
                full_iq, keys[0], full_q, keys[1], full_qr, keys[2], full_weights,
                self.native_layout.length_tensor, self.config, trace, retained, kl_retained,
                self.native_layout.cumulative_lengths, self.scale, indices, values, output, maximum, denominator,
                kl_gradients[0], kl_gradients[1][:, None], kl_gradients[2], kl_loss,
                self.workspace.arena, metadata, self.requests, transport_trace, *extra)
            kl_loss = kl_loss.reshape(())
        elif selected_pull:
            torch.ops.hyper_parallel.dsa_selected_cp_forward_out(
                full_iq, keys[0], full_q, keys[1], full_qr, keys[2], full_weights,
                self.native_layout.length_tensor, self.config, trace, retained,
                self.native_layout.cumulative_lengths, self.scale, indices, values, output, maximum, denominator,
                self.workspace.arena, metadata, self.requests, transport_trace, *extra)
        else:
            torch.ops.hyper_parallel.dsa_fused_cp_forward_out(
                full_iq, keys[0], full_q, keys[1], full_qr, keys[2], full_weights,
                self.native_layout.length_tensor, self.config, trace, retained, self.scale,
                indices, values, output, maximum, denominator,
                self.workspace.arena, metadata, self.requests, transport_trace)
        local = self.layout.local_query_ids
        global_indices = self.native_layout.sequence_to_global_indices(indices)
        states = (full_q, keys[1][:, 0], full_qr, keys[2][:, 0])
        tensors = (*states, indices, output, maximum, denominator)
        saved = FusedCpSavedAttention(invocation.batch_meta, states, indices, (output, maximum, denominator),
                                      tuple(tensor._version for tensor in tensors), self,
                                      (self.heads, self.scale, self.schedule), _SAVED_ATTENTION_TOKEN)
        return FusedCpForwardResult(output.index_select(0, local), global_indices.index_select(0, local),
                                    values.index_select(0, local), maximum.index_select(1, local),
                                    denominator.index_select(1, local), keys, tuple(trace.unbind()),
                                    transport_trace, epoch, saved, (full_iq, keys[0][:, 0], full_weights),
                                    kl_gradients, kl_loss, selected_buffers)

    @staticmethod
    def _transport_interval(transport: torch.Tensor) -> tuple[int, int]:
        if transport.device.type != "cpu" or transport.dtype != torch.int64 or transport.shape != (32,):
            raise ValueError("transport evidence requires an explicit CPU int64 [32] snapshot")
        index_start, index_end, start, end = (int(transport[word]) for word in (8, 9, 10, 11))
        if index_start <= 0 or index_end <= index_start or start < index_end:
            raise ValueError("transport timestamps do not certify index-before-main field ordering")
        if bool(transport[14:].any()) or int(transport[12]) != 0:
            raise ValueError("transport trace contains unsupported control records")
        if start <= 0 or end <= start:
            raise ValueError("fused CP does not prove an ordered main-KV transfer interval")
        return start, end

    def _compute_overlap(self, snapshots: tuple[torch.Tensor, ...], start: int, end: int) -> list:
        overlap = []
        for group in range(self.schedule.compute_groups):
            for member in range(3):
                row = snapshots[0][group]
                begin, finish = (int(row[member * 16 + word]) for word in (8, 9))
                if int(row[member * 16 + 7]) != 1 or begin <= 0 or finish < begin:
                    raise ValueError("compute interval lacks completed ordered device timestamps")
                if max(begin, start) < min(finish, end):
                    overlap.append((group, member))
        return overlap

    def validate_trace(self, result: FusedCpForwardResult, snapshots: tuple[torch.Tensor, ...],
                       transport: torch.Tensor, *, require_overlap: bool,
                       selected_snapshots: tuple[torch.Tensor, ...] | None = None) -> dict:
        """Decode explicitly completed CPU snapshots and exact local/remote byte counts."""
        start, end = self._transport_interval(transport)
        lengths = self.native_layout.batch_meta.global_cu_seqlens
        require_ld = any(end - begin > 2048 for begin, end in pairwise(lengths))
        selected = None
        if result.selected_requests:
            if selected_snapshots is None or len(selected_snapshots) != 4:
                raise ValueError("selected transfer requires explicit CPU request/count/membership/index snapshots")
            compute = validate_selected_training_traces(
                snapshots, self.schedule, require_ld=require_ld, with_kl=result.kl_loss is not None)
            selected = validate_selected_requests(selected_snapshots[3], result.saved.batch_meta,
                                                 *selected_snapshots[:3], result.epoch)
            if int(selected_snapshots[1][9]) > start:
                raise ValueError("selected main-KV transfer started before descriptor publication completed")
        elif result.kl_loss is not None:
            compute = validate_fused_training_traces(snapshots, self.schedule, require_ld=require_ld)
        else:
            compute = validate_fused_dsa_traces(snapshots, self.schedule, require_ld=require_ld)
        total = self.layout.batch_meta.global_valid_queries
        remote_tokens = total - self.layout.local_tokens
        main_tokens = selected["unique_keys"] if selected else total
        remote_main_tokens = selected["remote_keys"] if selected else remote_tokens
        expected = (result.epoch, result.epoch, self.layout.cp_size, self.layout.cp_size,
                    total * 256, main_tokens * 1152, remote_tokens * 256, remote_main_tokens * 1152)
        startup = 0 if selected else 3 * self.schedule.compute_groups
        if tuple(transport[:8].tolist()) != expected or int(transport[13]) != startup:
            raise ValueError("fused CP did not certify complete ready/ACK, startup or field byte counts")
        overlap = self._compute_overlap(snapshots, start, end)
        if require_overlap and not overlap:
            raise ValueError("fused CP does not prove the required LI/main-KV transfer overlap")
        return {"compute": compute, "ready_ack": True, "epoch": result.epoch,
                "remote_index_bytes": expected[6], "remote_main_bytes": expected[7],
                "compute_members_overlapping_transfer": overlap, "field_runs": len(self.runs),
                "main_bytes": expected[5], "selected_requests": selected}
