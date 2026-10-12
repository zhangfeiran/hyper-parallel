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
"""Common TP declarations and one-root-lease fused local CP forward submission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedCpForwardResult,
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_query import (
    DsaLocalQueryLayout,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumInvocation,
    TpTileSumProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
)


@dataclass(frozen=True)
class FusedCpTpInvocation:
    """Caller-owned CP query layout and matching LI/KL bounded TP declarations."""

    backend: FusedDsaCpTpForwardProbe
    cp: DsaWorkspaceInvocation
    local: DsaLocalQueryLayout
    transports: tuple[TpTileSumInvocation, ...]
    selected_kl: bool


class FusedDsaCpTpForwardProbe:
    """Compose local CP forward with native TP score and selected-KL reductions.

    Reserve CP, TP LI and TP KL consumers before root activation. Bind them in
    the same root-wide order. Main heads are local H32/H64; original Hi64 is
    sharded over TP and internally padded to native Hi64. The native call owns
    the CP consumer lease and records one completion event for all arenas.
    This detached primitive does not install model TP projections or backward.
    """

    def __init__(self, cp: FusedDsaCpForwardProbe, li: TpTileSumProbe,
                 kl: TpTileSumProbe | None = None) -> None:
        """Require disjoint, pre-reserved transports within the CP workspace root."""
        size = len(li.members)
        if (not li.participates or size not in (1, 2, 4, 8)
                or li.root is not cp.workspace.root or li.spec.width != 512
                or li.spec.slots != 2*cp.schedule.compute_groups):
            raise ValueError("fused CP TP requires local H32/H64, original Hi64 and matching LI root/slots")
        if kl is not None and (kl is li or kl.root is not li.root or kl.members != li.members
                               or kl.spec.width != 2048 or kl.spec.slots != li.spec.slots):
            raise ValueError("fused KL requires a distinct width-2048 transport with matching root and TP members")
        self.cp = cp
        self.li = li
        self.kl = kl
        self.index_heads = 64//size

    def prepare(self, invocation: DsaWorkspaceInvocation, *, epoch: int,
                query_ids: tuple[int, ...] | None = None, selected_kl: bool = False) -> FusedCpTpInvocation:
        """Collectively freeze identical TP rows and ownership outside execution.

        Different CP groups use different global rank names. Their TP peers must
        nevertheless agree on CP ordinal, logical query rows, owner offsets and
        owner-local key order. The common signature covers both LI and KL; only
        their candidate widths and maximum round bounds differ.
        """
        if invocation.workspace is not self.cp.workspace or (selected_kl and self.kl is None):
            raise ValueError("fused preparation requires its CP workspace and a KL transport when enabled")
        local = self.cp.prepare_local_queries(query_ids)
        meta = invocation.batch_meta
        signature = (meta.global_cu_seqlens, local.global_ids, meta.kv_global_ids, meta.token_owners,
                     meta.token_local_offsets, meta.cp_rank, meta.layout_id, meta.layer, meta.microbatch,
                     meta.invocation, meta.heap_generation, self.cp.heads, self.cp.scale,
                     self.cp.schedule.compute_groups, selected_kl)
        declarations = [None]*len(self.li.members)
        dist.all_gather_object(declarations, signature, group=self.li.group)
        if any(value != signature for value in declarations):
            raise ValueError("fused TP peers disagree on logical rows, ownership, geometry or KL branch")
        row_id = int.from_bytes(hashlib.sha256(json.dumps(signature).encode()).digest()[:8], 'little') % 2**63
        identity = (meta.invocation, meta.layer, meta.microbatch, meta.heap_generation, row_id)
        rows = len(local.global_ids)
        li = self.li.prepare(rounds=max(1, rows*((meta.global_valid_queries+511)//512+1)),
                             epoch=epoch, identity=identity)
        transports = (li,)
        if selected_kl:
            transports += (self.kl.prepare(rounds=max(1, rows), epoch=epoch, identity=identity),)
        return FusedCpTpInvocation(self, invocation, local, transports, selected_kl)

    def forward(self, main: tuple[torch.Tensor, ...], index: tuple[torch.Tensor, ...],
                invocation: FusedCpTpInvocation) -> FusedCpForwardResult:
        """Submit one detached native CP/TP forward, restoring caller-local index heads.

        Main states follow the local CP probe contract. Index contains BF16
        query [Tq,64/TP,128], complete owner-local BF16 key [Tk,128], and signed
        already scaled BF16/FP32 weights [Tq,64/TP]. No extra TP normalization
        is applied to the replicated raw KL loss or saved local-head VJP.
        """
        if not isinstance(invocation, FusedCpTpInvocation) or invocation.backend is not self:
            raise ValueError("fused CP TP invocation belongs to another backend")
        if len(index) != 3:
            raise ValueError("fused CP TP requires three local-head index states")
        query, key, weights = index
        rows = len(invocation.local.global_ids)
        device = self.cp.layout.device
        for tensor, shape, dtypes in (
                (query, (rows, self.index_heads, 128), (torch.bfloat16,)),
                (key, (len(invocation.cp.batch_meta.kv_global_ids), 128), (torch.bfloat16,)),
                (weights, (rows, self.index_heads), (torch.bfloat16, torch.float32))):
            if (tensor.shape != shape or tensor.device != device or tensor.dtype not in dtypes
                    or tensor.requires_grad or not tensor.is_contiguous()):
                raise ValueError("fused TP index states must be detached contiguous local-head tensors")
        padded_query = torch.zeros((rows, 64, 128), dtype=query.dtype, device=device)
        padded_weights = torch.zeros((rows, 64), dtype=weights.dtype, device=device)
        padded_query[:, :self.index_heads].copy_(query)
        padded_weights[:, :self.index_heads].copy_(weights)
        result = self.cp.forward_local(invocation.cp, main, (padded_query, key, padded_weights),
                                       query_layout=invocation.local, selected_kl=invocation.selected_kl,
                                       tp_invocations=invocation.transports)
        states = (result.index_states[0][:, :self.index_heads], result.index_states[1],
                  result.index_states[2][:, :self.index_heads])
        gradients = result.kl_gradients
        if gradients:
            gradients = (gradients[0][:, :self.index_heads], gradients[1], gradients[2][:, :self.index_heads])
        return replace(result, index_states=states, kl_gradients=gradients,
                       index_versions=tuple(tensor._version for tensor in states))
