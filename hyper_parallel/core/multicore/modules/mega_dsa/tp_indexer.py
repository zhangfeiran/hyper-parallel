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
"""Exact native LI selection with TP score SUM before every candidate tile's Top-K."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_indexer import (
    MIXED_INDEXER_WORKSPACE_BYTES,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumInvocation,
    TpTileSumProbe,
)
from hyper_parallel.core.multicore.torch.ops import _load_native


@dataclass(frozen=True)
class TpIndexerInvocation:
    """Prepared compact query order, genuine packed lengths and TP protocol identity."""

    backend: TpLocalIndexerProbe
    batch_meta: DsaBatchMeta
    transport: TpTileSumInvocation
    query_order: torch.Tensor
    restore_order: torch.Tensor
    query_lengths: torch.Tensor
    key_lengths: torch.Tensor
    positions: torch.Tensor
    sequence_starts: torch.Tensor
    config: torch.Tensor
    versions: tuple[int, ...]


@dataclass(frozen=True)
class TpIndexerResult:
    """Independent global-packed selection rows and BF16 values in caller query order."""

    global_indices: torch.Tensor
    values: torch.Tensor
    phase_trace: torch.Tensor
    _runtime_buffers: tuple = field(default=(), repr=False, compare=False)


class TpLocalIndexerProbe:
    """Native score exchange and exact Top-K for local Q and complete replicated index K.

    This forward-only selection boundary consumes original Hi64 split across TP.
    It pads each local head shard with zero queries/weights to the locked native
    H64 geometry, preserving the caller's model weight scale. No heads or scores
    are gathered to the host. CP fetch/replication must finish before invocation.
    It uses one launch for LI main, per-tile TP SUM, local Top-K and LD merge.
    Native KL teacher TP exchange and full CP×TP training remain separate work.
    """

    def __init__(self, transport: TpTileSumProbe, *, schedule: MixedSfaSchedule) -> None:
        """Require one bounded width-512 source per physical compute vector slot."""
        if schedule.rounds != 1 or transport.spec.width != 512 or transport.spec.slots != 2 * schedule.compute_groups:
            raise ValueError("TP LI requires one width-512 slot per compute vector and rounds=1")
        if transport.participates and 64 % len(transport.members):
            raise ValueError("original Hi64 must divide the TP group size")
        self.transport = transport
        self.schedule = schedule
        self.local_heads = 64 // len(transport.members) if transport.participates else 0
        self.retained = None

    def prepare(self, batch_meta: DsaBatchMeta, *, epoch: int) -> TpIndexerInvocation:
        """Collectively prepare explicit packed Q positions, storage order and row identity.

        KV must cover all global packed IDs in increasing order. Query rows can
        be reordered or empty. TP peers declare the same Q/K metadata, while
        their CP group memberships can differ. Setup uses host declarations;
        no projected device tensor is copied to the host.
        """
        if not self.transport.participates:
            raise ValueError("inactive root members cannot prepare TP LI work")
        if batch_meta.kv_global_ids != tuple(range(batch_meta.global_valid_queries)):
            raise ValueError("TP LI requires complete globally ordered replicated index K")
        signature = (batch_meta.global_cu_seqlens, batch_meta.q_global_ids, batch_meta.kv_global_ids,
                     batch_meta.cache_identity, self.schedule.compute_groups)
        declarations = [None] * len(self.transport.members)
        dist.all_gather_object(declarations, signature, group=self.transport.group)
        if any(item != signature for item in declarations):
            raise ValueError("TP LI peers disagree on query/key rows or invocation")
        row_id = int.from_bytes(hashlib.sha256(json.dumps(signature).encode()).digest()[:8], 'little') % 2**63
        total = batch_meta.global_valid_queries
        rows = len(batch_meta.q_global_ids)
        upper_rounds = max(1, rows * ((total + 511) // 512 + 1))
        identity = (batch_meta.invocation, batch_meta.layer, batch_meta.microbatch, batch_meta.heap_generation, row_id)
        meta = self.transport.prepare(rounds=upper_rounds, epoch=epoch, identity=identity)
        device = self.transport.arena.device
        order = tuple(sorted(range(rows), key=lambda row: batch_meta.q_global_ids[row]))
        query_order = torch.tensor(order, dtype=torch.long, device=device)
        restore = query_order.argsort()
        ids = tuple(batch_meta.q_global_ids[row] for row in order)
        positions = torch.tensor(tuple(batch_meta.sequence_position(token)[1] for token in ids),
                                 dtype=torch.int64, device=device)
        starts = torch.tensor(tuple(token - batch_meta.sequence_position(token)[1] for token in ids),
                              dtype=torch.int32, device=device).reshape(rows, 1)
        counts = tuple(sum(start <= token < end for token in ids)
                       for start, end in zip(batch_meta.global_cu_seqlens, batch_meta.global_cu_seqlens[1:]))
        cumulative = tuple(sum(counts[:end]) for end in range(1, len(counts) + 1))
        query_lengths = torch.tensor(cumulative, dtype=torch.int32, device=device)
        key_lengths = torch.tensor(batch_meta.global_cu_seqlens[1:], dtype=torch.int32, device=device)
        config = self.schedule.runtime_config(device)
        buffers = (query_order, restore, query_lengths, key_lengths, positions, starts, config)
        return TpIndexerInvocation(self, batch_meta, meta, *buffers, tuple(tensor._version for tensor in buffers))

    def forward(self, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
                invocation: TpIndexerInvocation) -> TpIndexerResult:
        """Queue complete native LI with bounded score SUM, then restore caller query order.

        Query is detached BF16 [local Tq,64/TP,128], key detached BF16 [global Tk,128],
        and weights detached BF16/FP32 [local Tq,64/TP], already model-scaled.
        This Top-K operation has no autograd path. The caller retains selections
        for backward independently of the reusable native scratch and arena.
        """
        if invocation.backend is not self:
            raise ValueError("TP LI invocation belongs to another backend")
        self.transport.check_invocation(invocation.transport)
        buffers = (invocation.query_order, invocation.restore_order, invocation.query_lengths,
                   invocation.key_lengths, invocation.positions, invocation.sequence_starts, invocation.config)
        if tuple(tensor._version for tensor in buffers) != invocation.versions:
            raise ValueError("TP LI prepared query metadata was modified")
        rows = len(invocation.batch_meta.q_global_ids)
        total = invocation.batch_meta.global_valid_queries
        device = self.transport.arena.device
        if (query.shape != (rows, self.local_heads, 128) or key.shape != (total, 128)
                or weights.shape != (rows, self.local_heads) or query.dtype != torch.bfloat16
                or key.dtype != torch.bfloat16 or weights.dtype not in (torch.bfloat16, torch.float32)
                or any(tensor.device != device or tensor.requires_grad for tensor in (query, key, weights))):
            raise ValueError("TP LI fields must be detached and match the prepared local-head/global-key geometry")
        _load_native()
        version = getattr(torch.ops.hyper_parallel, 'dsa_tp_local_indexer_version', None)
        if version is None or version() != 1:
            raise RuntimeError("native payload lacks TP local indexer version 1")
        padded_query = torch.zeros((rows, 64, 128), dtype=torch.bfloat16, device=device)
        padded_weights = torch.zeros((rows, 64), dtype=weights.dtype, device=device)
        padded_query[:, :self.local_heads].copy_(query.index_select(0, invocation.query_order))
        padded_weights[:, :self.local_heads].copy_(weights.index_select(0, invocation.query_order))
        dense_key = key.contiguous().view(total, 1, 128)
        required = MIXED_INDEXER_WORKSPACE_BYTES if rows else 0
        if self.retained is None or self.retained.numel() < required:
            self.retained = torch.empty(required, dtype=torch.uint8, device=device)
        indices = torch.full((rows, 1, 2048), -1, dtype=torch.int32, device=device)
        values = torch.full(indices.shape, -torch.inf, dtype=torch.bfloat16, device=device)
        trace = torch.zeros((2, 20, 64), dtype=torch.int64, device=device)
        with self.transport.lease(invocation.transport):
            torch.ops.hyper_parallel.dsa_tp_local_indexer_out(
                padded_query, dense_key, padded_weights, invocation.query_lengths, invocation.key_lengths,
                invocation.config, trace, self.retained, invocation.positions, self.transport.arena,
                invocation.transport.metadata, indices, values)
        relative = indices[:, 0]
        global_indices = torch.where(relative >= 0, relative + invocation.sequence_starts, -1)
        ordered_indices = global_indices.index_select(0, invocation.restore_order)
        ordered_values = values[:, 0].index_select(0, invocation.restore_order)
        return TpIndexerResult(ordered_indices, ordered_values, trace,
                               (padded_query, padded_weights, dense_key, indices, values, invocation, self.retained))
