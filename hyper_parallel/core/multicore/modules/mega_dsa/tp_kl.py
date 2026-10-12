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
"""Native TP selected-KL teacher/predictor reduction and local-head derivatives."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.local_query import (
    DsaLocalQueryLayout,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_kl import (
    mixed_kl_key_partials,
    mixed_kl_workspace_bytes,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumInvocation,
    TpTileSumProbe,
)
from hyper_parallel.core.multicore.torch.ops import _load_native


@dataclass(frozen=True)
class TpKlInvocation:
    """Cold TP row declaration, selected-set version and independent transport epoch."""

    backend: TpLocalKlProbe
    layout: DsaLocalQueryLayout
    transport: TpTileSumInvocation
    global_indices: torch.Tensor
    indices_version: int
    config: torch.Tensor
    config_version: int


@dataclass(frozen=True)
class TpKlResult:
    """Raw local index VJP and FP32 global-key partials, before CP/TP owner reduction.

    Loss is replicated across TP and summed over the local query tile. Backward
    consumers use an identity VJP through the replicated predictor SUM; they
    must not SUM the cotangent or divide by TP again. Apply the model coefficient,
    CP normalization and upstream cotangent once outside this primitive.
    """

    gradients: tuple[torch.Tensor, ...]
    key_partials: torch.Tensor
    loss: torch.Tensor
    traces: tuple[torch.Tensor, ...]
    retained: torch.Tensor
    _runtime_buffers: tuple = field(default=(), repr=False, compare=False)


class TpLocalKlProbe:
    """Selected KL with native bounded TP exchange and complete replicated shared K.

    Original Hi64 is head-sharded across TP and zero-padded to native Hi64.
    Main attention heads are evenly sharded with local H32/H64. Teacher heads
    remain detached; their per-head softmax statistics must come from the same
    selected set. All TP peers must supply identical ordered selected IDs from
    the TP LI producer. This primitive does not perform CP or TP key return.
    """

    def __init__(self, transport: TpTileSumProbe, *, global_heads: int,
                 attention_scale: float, schedule: MixedSfaSchedule) -> None:
        """Require a width-2048 source per vector and evenly sharded main/index heads."""
        size = len(transport.members)
        if (not transport.participates or size not in (1, 2, 4, 8) or global_heads % size
                or global_heads // size not in (32, 64) or schedule.rounds != 1
                or transport.spec.width != 2048 or transport.spec.slots != 2 * schedule.compute_groups
                or not math.isfinite(attention_scale) or attention_scale <= 0):
            raise ValueError("TP KL requires local H32/H64, original Hi64, width-2048 vector slots and rounds=1")
        self.transport = transport
        self.global_heads = global_heads
        self.local_heads = global_heads // size
        self.index_heads = 64 // size
        self.scale = attention_scale
        self.schedule = schedule

    def prepare(self, batch_meta: DsaBatchMeta, global_indices: torch.Tensor, *, epoch: int) -> TpKlInvocation:
        """Collectively declare identical logical query rows and freeze the TP LI selection.

        Selection is int32 [caller Tq,2048] in the global_packed namespace. It
        must contain the complete causal prefix up to K valid IDs, as required
        by the native selected-KL contract. No selection or activation is copied
        to the host. Same selected IDs across TP are a producer precondition.
        """
        if batch_meta.kv_global_ids != tuple(range(batch_meta.global_valid_queries)):
            raise ValueError("TP KL requires complete globally ordered replicated keys")
        self.transport.bind()
        device = self.transport.arena.device
        rows = len(batch_meta.q_global_ids)
        if (global_indices.shape != (rows, 2048) or global_indices.dtype != torch.int32
                or global_indices.device != device or global_indices.requires_grad):
            raise ValueError("TP KL selection must be detached int32 [local queries,2048] on the root device")
        signature = (batch_meta.global_cu_seqlens, batch_meta.q_global_ids, batch_meta.kv_global_ids,
                     batch_meta.cache_identity, self.global_heads, self.scale, self.schedule.compute_groups)
        declarations = [None] * len(self.transport.members)
        dist.all_gather_object(declarations, signature, group=self.transport.group)
        if any(item != signature for item in declarations):
            raise ValueError("TP KL peers disagree on rows, head geometry or invocation")
        row_id = int.from_bytes(hashlib.sha256(json.dumps(signature).encode()).digest()[:8], 'little') % 2**63
        identity = (batch_meta.invocation, batch_meta.layer, batch_meta.microbatch, batch_meta.heap_generation, row_id)
        invocation = self.transport.prepare(rounds=max(1, rows), epoch=epoch, identity=identity)
        layout = DsaLocalQueryLayout(batch_meta, device)
        config = self.schedule.runtime_config(device)
        return TpKlInvocation(self, layout, invocation, global_indices, global_indices._version,
                              config, config._version)

    def gradients(self, main: tuple[torch.Tensor, ...], index: tuple[torch.Tensor, ...],
                  statistics: tuple[torch.Tensor, ...], invocation: TpKlInvocation) -> TpKlResult:
        """Queue initialize, TP-reduced compute and post phases, returning raw local VJP.

        Main contains BF16 query [Tq,Hlocal,512], global compressed [Tk,512],
        query RoPE [Tq,Hlocal,64] and global key RoPE [Tk,64]. Index contains
        BF16 query [Tq,64/TP,128], global key [Tk,128], and already scaled signed
        BF16/FP32 merge weights [Tq,64/TP]. Statistics are detached FP32 maximum
        and denominator [1,Tq,Hlocal] in caller row order.
        """
        self._validate(main, index, statistics, invocation)
        _load_native()
        version = getattr(torch.ops.hyper_parallel, 'dsa_tp_local_mixed_kl_version', None)
        if version is None or version() != 1:
            raise RuntimeError("native payload lacks TP local KL version 1")
        return self._submit(main, index, statistics, invocation)

    def _validate(self, main, index, statistics, invocation):
        if not isinstance(invocation, TpKlInvocation) or invocation.backend is not self:
            raise ValueError("TP KL invocation belongs to another backend")
        self.transport.check_invocation(invocation.transport)
        invocation.layout.validate_versions()
        if (invocation.global_indices._version != invocation.indices_version
                or invocation.config._version != invocation.config_version):
            raise ValueError("TP KL selection/config was modified after preparation")
        if torch.are_deterministic_algorithms_enabled():
            raise ValueError("TP KL retains non-deterministic native math")
        rows = len(invocation.layout.global_ids)
        total = invocation.layout.batch_meta.global_valid_queries
        shapes = ((rows, self.local_heads, 512), (total, 512), (rows, self.local_heads, 64), (total, 64),
                  (rows, self.index_heads, 128), (total, 128), (rows, self.index_heads),
                  (1, rows, self.local_heads), (1, rows, self.local_heads))
        if len(main) != 4 or len(index) != 3 or len(statistics) != 2:
            raise ValueError("TP KL requires four main states, three index states and two statistics")
        for position, (tensor, shape) in enumerate(zip((*main, *index, *statistics), shapes)):
            dtypes = ((torch.bfloat16, torch.float32) if position == 6 else
                      (torch.float32,) if position >= 7 else (torch.bfloat16,))
            if (tensor.shape != shape or tensor.dtype not in dtypes or tensor.requires_grad
                    or tensor.device != invocation.layout.device):
                raise ValueError("TP KL fields must be detached and match prepared shapes/device/dtypes")

    def _submit(self, main, index, statistics, invocation):
        layout = invocation.layout
        query, key, query_rope, key_rope = main
        iq, ik, weight = index
        rows, total = query.shape[0], key.shape[0]
        device = query.device
        padded_query = torch.zeros((rows, 64, 128), dtype=iq.dtype, device=device)
        padded_weight = torch.zeros((rows, 64), dtype=weight.dtype, device=device)
        padded_query[:, :self.index_heads].copy_(layout.pack(iq))
        padded_weight[:, :self.index_heads].copy_(layout.pack(weight))
        gradients = (torch.empty_like(padded_query), torch.empty_like(ik), torch.empty_like(padded_weight))
        loss = torch.zeros(1, dtype=torch.float32, device=device)
        traces = tuple(torch.zeros((20, 64), dtype=torch.int64, device=device) for _ in range(3)) if rows else ()
        retained = torch.empty(mixed_kl_workspace_bytes(total, self.local_heads) if rows else 0,
                               dtype=torch.uint8, device=device)
        if not rows:
            for tensor in gradients:
                tensor.zero_()
            partials = torch.zeros(ik.shape, dtype=torch.float32, device=device)
            with self.transport.lease(invocation.transport):
                return TpKlResult((iq.clone(), gradients[1], weight.clone()), partials, loss.reshape(()), (), retained)
        starts = torch.tensor(tuple(token - layout.batch_meta.sequence_position(token)[1]
                                    for token in layout.global_ids), dtype=torch.int32, device=device).reshape(rows, 1)
        selected = layout.pack(invocation.global_indices)
        native_indices = torch.where(selected >= 0, selected - starts, -1).reshape(rows, 1, 2048).contiguous()
        packed = (layout.pack(query), key.contiguous()[:, None], padded_query, ik.contiguous()[:, None],
                  padded_weight, native_indices, statistics[0].index_select(1, layout.pack_order).contiguous(),
                  statistics[1].index_select(1, layout.pack_order).contiguous(), layout.pack(query_rope),
                  key_rope.contiguous()[:, None])
        for phase in range(3):
            args = (*packed, layout.cumulative_queries, layout.batch_meta.global_cu_seqlens[1:],
                    invocation.config, traces[phase], retained, layout.positions)
            outputs = (self.scale, phase, gradients[0], gradients[1][:, None], gradients[2], loss)
            if phase == 1:
                with self.transport.lease(invocation.transport):
                    torch.ops.hyper_parallel.dsa_tp_local_mixed_kl_out(
                        *args, self.transport.arena, invocation.transport.metadata, *outputs)
            else:
                torch.ops.hyper_parallel.dsa_local_mixed_kl_out(*args, *outputs)
        partials = mixed_kl_key_partials(retained, tokens=total, heads=self.local_heads)
        restored = (layout.restore(gradients[0][:, :self.index_heads]), gradients[1],
                    layout.restore(gradients[2][:, :self.index_heads]))
        return TpKlResult(restored, partials, loss.reshape(()), traces, retained,
                          (*packed, *gradients, invocation))
