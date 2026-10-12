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
"""Compose native CP index-K owner return with FP32 shared-gradient TP SUM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import _layout_signature
from hyper_parallel.core.multicore.modules.mega_dsa.index_return import (
    IndexGradReturnResult,
    SparseIndexGradReturnProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_owner_return import (
    TpOwnerGradientReturnProbe,
    TpOwnerReturnInvocation,
    TpOwnerReturnResult,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
)


@dataclass(frozen=True)
class CpTpSharedReturnInvocation:
    """Matched CP ownership and TP ordered-owner identity for one first-order backward."""

    backend: CpTpSharedGradientReturnProbe
    cp: DsaWorkspaceInvocation
    tp: TpOwnerReturnInvocation


@dataclass(frozen=True)
class CpTpSharedReturnResult:
    """Native CP receipt plus TP FP32 sums and the final projection-boundary gradients."""

    index_cp: IndexGradReturnResult | None
    shared_tp: TpOwnerReturnResult


class CpTpSharedGradientReturnProbe:
    """Return global index-K contributions through CP, then SUM all shared fields through TP.

    Main C/K-RoPE inputs must already be the FP32 owner sums from this CP
    backward. C contains dK+dV once. Index-K starts in complete global packed
    order. Coefficients, CP normalization and upstream cotangents must already
    be applied; this boundary changes neither scale nor loss replication.
    """

    def __init__(self, index_return: SparseIndexGradReturnProbe, tp_return: TpOwnerGradientReturnProbe) -> None:
        """Require the CP and TP transports to share one encompassing SHMEM root."""
        if index_return.workspace.root is not tp_return.transport.root:
            raise ValueError("CP and TP shared-gradient return must use the same SHMEM root")
        self.index_return = index_return
        self.tp_return = tp_return

    def prepare(self, invocation: DsaWorkspaceInvocation, *, epoch: int,
                presence: tuple[bool, ...] = (True, True, True)) -> CpTpSharedReturnInvocation:
        """Collectively freeze branch presence across both CP and TP before backward execution."""
        meta = invocation.batch_meta
        if (invocation.workspace is not self.index_return.workspace
                or _layout_signature(meta) != self.index_return.backend.signature
                or meta.kv_global_ids != self.index_return.backend.layout.batch_meta.kv_global_ids):
            raise ValueError("CP/TP return requires the prepared index-return workspace")
        if not isinstance(presence, tuple) or len(presence) != 3 or any(not isinstance(x, bool) for x in presence):
            raise ValueError("CP/TP return requires three immutable branch flags")
        group = self.index_return.backend.layout.group
        declaration = (presence, meta.cache_identity)
        declarations = [None] * dist.get_world_size(group)
        dist.all_gather_object(declarations, declaration, group=group)
        if any(value != declaration for value in declarations):
            raise ValueError("CP members disagree on shared-gradient branches or invocation")
        tp = self.tp_return.prepare(invocation.batch_meta, epoch=epoch, presence=presence)
        return CpTpSharedReturnInvocation(self, invocation, tp)

    def return_owner(self, index_partial: torch.Tensor | None,
                     main_owner_fp32: tuple[torch.Tensor | None, ...], invocation: CpTpSharedReturnInvocation, *,
                     input_dtypes: tuple[torch.dtype, ...]) -> CpTpSharedReturnResult:
        """Queue native CP index return followed by one shared TP SUM and one final dtype conversion.

        Args:
            index_partial: Detached FP32 [global N,128] local-query/head
                contributions, or None for an inactive index branch.
            main_owner_fp32: Detached caller-ordered CP owner C and K-RoPE sums,
                including already combined dK+dV, or None for inactive fields.
            invocation: Cold collective branch and owner declaration.
            input_dtypes: Original projection output dtypes for C/K-RoPE/index-K.

        Returns:
            CP index-return receipt and TP owner FP32 sums/final gradients.
            Same-stream submission preserves the dependency between the two
            native transports; other streams require an explicit event wait.
        """
        self._validate(index_partial, main_owner_fp32, invocation, input_dtypes)
        self.tp_return.check_native_support()
        cp_result = None
        if index_partial is not None:
            self.index_return.check_native_support()
            cp_result = self.index_return.return_owner(invocation.cp, index_partial.contiguous())
        owner_key = None if cp_result is None else cp_result.owner_fp32
        shared = self.tp_return.return_owner((*main_owner_fp32, owner_key), invocation.tp, input_dtypes=input_dtypes)
        return CpTpSharedReturnResult(cp_result, shared)

    def _validate(self, index_partial, main, invocation, dtypes):
        if (not isinstance(invocation, CpTpSharedReturnInvocation) or invocation.backend is not self
                or invocation.cp.workspace is not self.index_return.workspace):
            raise ValueError("CP/TP return invocation belongs to another backend")
        if len(main) != 2 or len(dtypes) != 3 or any(dtype not in (torch.bfloat16, torch.float32) for dtype in dtypes):
            raise ValueError("CP/TP return requires two main owner fields and three BF16/FP32 dtypes")
        rows = len(invocation.tp.owner_ids)
        presence = invocation.tp.presence
        device = self.tp_return.transport.arena.device
        self.tp_return.transport.check_invocation(invocation.tp.transport)
        for field, width, active in zip(main, (512, 64), presence[:2]):
            if active != (field is not None):
                raise ValueError("CP/TP main branch presence changed after collective preparation")
            if field is not None and (field.shape != (rows, width) or field.dtype != torch.float32
                                      or field.device != device or field.requires_grad):
                raise ValueError("CP/TP main owner fields must be detached FP32 in caller owner order")
        if presence[2] != (index_partial is not None):
            raise ValueError("CP/TP index branch presence changed after collective preparation")
        if index_partial is not None and (index_partial.shape != (invocation.cp.batch_meta.global_valid_queries, 128)
                                          or index_partial.dtype != torch.float32 or index_partial.device != device
                                          or index_partial.requires_grad):
            raise ValueError("CP/TP index contributions must be detached FP32 in global packed order")
