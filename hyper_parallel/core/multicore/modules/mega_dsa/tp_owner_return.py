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
"""Native TP SUM of CP owner-local shared gradients, retaining FP32 until one final cast."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumInvocation,
    TpTileSumProbe,
)
from hyper_parallel.core.multicore.torch.ops import _load_native

_SHARED_WIDTHS = (512, 64, 128)


@dataclass(frozen=True)
class TpOwnerReturnInvocation:
    """Matching caller owner-row order, branch presence and monotonic transport identity."""

    backend: TpOwnerGradientReturnProbe
    owner_ids: tuple[int, ...]
    presence: tuple[bool, ...]
    transport: TpTileSumInvocation


@dataclass(frozen=True)
class TpOwnerReturnResult:
    """Independent FP32 owner sums and gradients converted once to projection input dtypes."""

    owner_fp32: tuple[torch.Tensor | None, ...]
    gradients: tuple[torch.Tensor | None, ...]


class TpOwnerGradientReturnProbe:
    """SUM shared C/K-RoPE/index-K after native CP owner return, without another CP/TP divisor.

    Each field must already include its upstream cotangent, model coefficient
    and CP normalization. C is the combined dK+dV from main attention, included
    exactly once. The bounded symmetric arena is reused across regular FP32
    tiles; it does not grow with owner row count. This boundary is first-order
    only and must execute in matching backward order on every TP member.
    """

    def __init__(self, transport: TpTileSumProbe) -> None:
        """Require a live explicit TP transport with width-512 vector slots."""
        if not transport.participates or transport.spec.width != 512:
            raise ValueError("TP owner return requires active width-512 transport")
        self.transport = transport

    def prepare(self, batch_meta: DsaBatchMeta, *, epoch: int,
                presence: tuple[bool, ...] = (True, True, True)) -> TpOwnerReturnInvocation:
        """Collectively freeze owner IDs, branch presence and the CP/TP invocation identity.

        Caller KV rows are owner-local storage rows, not a replicated global K
        view. TP peers may belong to different CP groups but must own exactly
        the same ordered global IDs. Root-wide arena binding must finish before
        subgroup preparation when the root contains inactive members.
        """
        if (not isinstance(presence, tuple) or len(presence) != 3
                or any(not isinstance(value, bool) for value in presence)
                or any(batch_meta.token_owners[token] != batch_meta.cp_rank for token in batch_meta.kv_global_ids)):
            raise ValueError("TP owner return requires three branch flags and exclusively owner-local KV rows")
        signature = (batch_meta.global_cu_seqlens, batch_meta.kv_global_ids, batch_meta.cache_identity, presence)
        declarations = [None] * len(self.transport.members)
        dist.all_gather_object(declarations, signature, group=self.transport.group)
        if any(value != signature for value in declarations):
            raise ValueError("TP owner return peers disagree on owner rows, branches or invocation")
        words = len(batch_meta.kv_global_ids)*sum(width for width, active in zip(_SHARED_WIDTHS, presence) if active)
        rounds = max(1, (words+self.transport.spec.slots*512-1)//(self.transport.spec.slots*512))
        row_id = int.from_bytes(hashlib.sha256(json.dumps(signature).encode()).digest()[:8], 'little') % 2**63
        identity = (batch_meta.invocation, batch_meta.layer, batch_meta.microbatch, batch_meta.heap_generation, row_id)
        invocation = self.transport.prepare(rounds=rounds, epoch=epoch, identity=identity)
        return TpOwnerReturnInvocation(self, batch_meta.kv_global_ids, presence, invocation)

    def return_owner(self, partials: tuple[torch.Tensor | None, ...], invocation: TpOwnerReturnInvocation, *,
                     input_dtypes: tuple[torch.dtype, ...]) -> TpOwnerReturnResult:
        """Queue one bounded native SUM, then perform the sole projection-boundary cast.

        Args:
            partials: Detached FP32 owner-local C [N,512], K-RoPE [N,64] and
                index-K [N,128] after CP owner return. Inactive branches are None.
            invocation: Cold prepared ordered owner rows, branch mask and epoch.
            input_dtypes: The three original shared projection output dtypes,
                each BF16 or FP32. No input tensor or CP result is mutated.

        Returns:
            FP32 TP sums in caller owner order and their final converted gradients.
            Read on the submission stream or establish an explicit event dependency.
        """
        self.validate_partials(partials, invocation, input_dtypes=input_dtypes)
        self.check_native_support()
        words = sum(tensor.numel() for tensor in partials if tensor is not None)
        shape = (invocation.transport.rounds, self.transport.spec.slots, 512)
        packed = torch.zeros(shape, dtype=torch.float32, device=self.transport.arena.device)
        if words:
            packed.reshape(-1)[:words].copy_(torch.cat(tuple(tensor.reshape(-1) for tensor in partials
                                                           if tensor is not None)))
        summed = self.transport.run(packed, invocation.transport).reshape(-1)
        offset = 0
        result = []
        for tensor in partials:
            if tensor is None:
                result.append(None)
            else:
                result.append(summed[offset:offset+tensor.numel()].reshape(tensor.shape))
                offset += tensor.numel()
        fp32 = tuple(result)
        gradients = tuple(None if tensor is None else tensor.to(dtype) for tensor, dtype in zip(fp32, input_dtypes))
        return TpOwnerReturnResult(fp32, gradients)

    @staticmethod
    def check_native_support() -> None:
        """Reject an incompatible payload before CP or TP workspace publication."""
        _load_native()
        version = getattr(torch.ops.hyper_parallel, 'dsa_tp_tile_sum_version', None)
        if version is None or version() != 1:
            raise RuntimeError("native payload lacks TP tile SUM version 1")

    def validate_partials(self, partials: tuple[torch.Tensor | None, ...],
                          invocation: TpOwnerReturnInvocation, *, input_dtypes: tuple[torch.dtype, ...]) -> None:
        """Validate all prepared owner fields without claiming an epoch or launching native work."""
        dtypes = input_dtypes
        if not isinstance(invocation, TpOwnerReturnInvocation) or invocation.backend is not self:
            raise ValueError("TP owner invocation belongs to another backend")
        self.transport.check_invocation(invocation.transport)
        if (len(partials) != 3 or len(dtypes) != 3 or
                any(dtype not in (torch.bfloat16, torch.float32) for dtype in dtypes)):
            raise ValueError("TP owner return requires three partials and BF16/FP32 input dtypes")
        for tensor, width, active in zip(partials, _SHARED_WIDTHS, invocation.presence):
            if active != (tensor is not None):
                raise ValueError("TP owner branch presence changed after collective preparation")
            if tensor is not None and (tensor.shape != (len(invocation.owner_ids), width)
                                       or tensor.dtype != torch.float32 or tensor.requires_grad
                                       or tensor.device != self.transport.arena.device):
                raise ValueError("TP owner partials must be detached FP32 in declared owner-row order")
