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
"""Device sparse index-K owner return with ordered FP32 accumulation and explicit reuse receipts."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
    _layout_signature,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
)
from hyper_parallel.core.multicore.torch.ops import _load_native

_INDEX_MAGIC = 0x4850445341494731


@dataclass(frozen=True)
class IndexGradReturnResult:
    """Caller-ordered owner FP32 sums and explicitly completed native transport evidence."""

    owner_fp32: torch.Tensor
    transport_trace: torch.Tensor
    epoch: int
    _runtime_buffers: tuple[torch.Tensor, ...] = field(default=(), repr=False, compare=False)


class SparseIndexGradReturnProbe:
    """Return only nonzero index-K partial rows through sender-exclusive symmetric inboxes.

    All CP members call in matching collective order, including empty owners and
    query tiles. Input contributions already include coefficient, normalization
    and upstream. This operation applies none of them and returns FP32 before
    the caller performs the sole final input-dtype conversion.
    """

    def __init__(self, forward: FusedDsaCpForwardProbe) -> None:
        """Bind the forward backend's immutable geometry, owner requests and workspace."""
        if forward.layout.cp_size > 8:
            raise ValueError("device index-K return supports CP sizes up to eight")
        self.backend = forward
        self.workspace = forward.workspace
        meta = forward.layout.batch_meta
        self.owner_order = torch.tensor(tuple(meta.token_local_offsets[token] for token in meta.kv_global_ids),
                                         dtype=torch.long, device=forward.layout.device)
        self._owner_order_version = self.owner_order._version
        self._owner_order_reference = self.owner_order

    @staticmethod
    def check_native_support() -> None:
        """Reject an old payload before entering any native workspace lease."""
        _load_native()
        version = getattr(torch.ops.hyper_parallel, "dsa_sparse_index_grad_version", None)
        if version is None or version() != 1:
            raise RuntimeError("device index-K return requires adapter ABI 1; rebuild this checkout's payload")

    def return_owner(self, invocation: DsaWorkspaceInvocation, partial: torch.Tensor) -> IndexGradReturnResult:
        """Submit one native return for detached global-packed FP32 [N,128] contributions.

        Args:
            invocation: Prepared backward invocation with unchanged CP owner geometry.
            partial: Already scaled input contributions in global packed key order.

        Returns:
            Owner-local FP32 gradients in the prepared caller KV order, an owned
            trace and the communication epoch. Numerical zero rows, including
            signed zeros, may be omitted and normalize to zero.
        """
        meta = invocation.batch_meta
        if (invocation.workspace is not self.workspace or _layout_signature(meta) != self.backend.signature
                or meta.kv_global_ids != self.backend.layout.batch_meta.kv_global_ids):
            raise ValueError("index-K return requires unchanged prepared key ownership")
        if (self.owner_order is not self._owner_order_reference
                or self.owner_order._version != self._owner_order_version):
            raise ValueError("index-K owner permutation was modified; prepare a new probe")
        if (partial.shape != (meta.global_valid_queries, 128) or partial.dtype != torch.float32
                or partial.device != self.backend.layout.device or not partial.is_contiguous()
                or partial.requires_grad):
            raise ValueError("index-K partial must be detached contiguous global-packed FP32 [N,128]")
        self.check_native_support()
        with self.workspace.lease(invocation, direction="backward"):
            epoch = self.workspace.next_transport_epoch(invocation)
            offsets = {name: offset for name, offset, _, _ in self.workspace.layout}
            values = (_INDEX_MAGIC, 1, epoch, meta.cp_rank, len(meta.cp_ranks), meta.heap_generation,
                      meta.layer, meta.microbatch, meta.invocation, self.backend.layout_hash,
                      offsets["gradient_inbox"], self.workspace.spec.owner_gradient_tokens,
                      len(meta.kv_global_ids), offsets["events"], len(self.backend.runs),
                      meta.global_valid_queries, self.workspace.arena_bytes, self.workspace.spec.source_tokens)
            metadata = torch.tensor(values, dtype=torch.int64).to(partial.device, non_blocking=True)
            trace = torch.zeros(32, dtype=torch.int64, device=partial.device)
            owner = torch.full((self.workspace.spec.owner_gradient_tokens, 128), float("nan"),
                               dtype=torch.float32, device=partial.device)
            torch.ops.hyper_parallel.dsa_sparse_index_grad_out(partial, self.workspace.arena, metadata,
                                                               self.backend.requests, trace, owner)
            restored = owner.index_select(0, self.owner_order)
            return IndexGradReturnResult(restored, trace, epoch, (metadata, owner, partial))

    def validate_trace(self, result: IndexGradReturnResult, trace: torch.Tensor) -> dict:
        """Validate caller-supplied completed CPU receipts without implicit device synchronization."""
        if trace.device.type != "cpu" or trace.dtype != torch.int64 or trace.shape != (32,):
            raise ValueError("index-K transport evidence requires a completed CPU int64 [32] snapshot")
        meta = self.backend.layout.batch_meta
        size, local, total = len(meta.cp_ranks), len(meta.kv_global_ids), meta.global_valid_queries
        sent, skipped, received, remote_received = (int(trace[word]) for word in (12, 13, 14, 15))
        remote_sent = int(trace[5])//512
        if (not 0 <= sent <= total or skipped != total-sent or not 0 <= remote_sent <= min(sent, total-local)
                or not 0 <= received <= size*local
                or not 0 <= remote_received <= min(received, (size-1)*local)):
            raise ValueError("index-K transport row counters are inconsistent")
        expected = (result.epoch, result.epoch, size, size, sent*512, remote_sent*512,
                    received*512, remote_received*512)
        if tuple(trace[:8].tolist()) != expected:
            raise ValueError("index-K transport lacks complete ready/write/ACK or exact row payload bytes")
        times = tuple(int(trace[word]) for word in (8, 9, 10, 11))
        if times[0] <= 0 or not times[0] < times[1] <= times[2] < times[3]:
            raise ValueError("index-K transport lacks ordered publication and owner reduction intervals")
        if tuple(trace[16:19].tolist()) != ((size-1)*8, (size-1)*8, size*local*512) or bool(trace[19:].any()):
            raise ValueError("index-K control or owner-clear extent is invalid")
        return {"ready_write_ack": True, "epoch": result.epoch, "sent_rows": sent, "skipped_zero_rows": skipped,
                "received_rows": received, "remote_sent_bytes": expected[5], "remote_received_bytes": expected[7],
                "remote_counter_control_bytes": (size-1)*8, "owner_inbox_clear_bytes": size*local*512,
                "fp32_owner_order": list(range(size))}
