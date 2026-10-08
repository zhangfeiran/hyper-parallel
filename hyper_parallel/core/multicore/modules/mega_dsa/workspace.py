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
"""DSA owner publication, staging and gradient scratch in a frozen shared root."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.shmem.consumer import (
    SharedShmemRoot,
    ShmemConsumerSpec,
    allocation_budget,
)

from .metadata import DsaBatchMeta

_ALIGNMENT = 512


@dataclass(frozen=True)
class DsaWorkspaceSpec:
    """Static capacities, independent of expert counts or MoE budget formulas."""

    source_tokens: int
    staging_tokens: int
    owner_gradient_tokens: int
    cp_size: int
    event_slots: int = 4
    compressed_dim: int = 512
    rope_dim: int = 64
    index_dim: int = 128
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        for name in ("source_tokens", "staging_tokens", "owner_gradient_tokens", "cp_size", "event_slots",
                     "compressed_dim", "rope_dim", "index_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("DSA source/staging dtype must be BF16 or FP16")

    def buffers(self) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
        """Declare one arena's fields; peer-exclusive gradient stripes accumulate in FP32."""
        width = self.compressed_dim + self.rope_dim + self.index_dim
        return (("source_compressed", (self.source_tokens, self.compressed_dim), self.dtype),
                ("source_rope", (self.source_tokens, self.rope_dim), self.dtype),
                ("source_index", (self.source_tokens, self.index_dim), self.dtype),
                ("staging_compressed", (self.staging_tokens, self.compressed_dim), self.dtype),
                ("staging_rope", (self.staging_tokens, self.rope_dim), self.dtype),
                ("staging_index", (self.staging_tokens, self.index_dim), self.dtype),
                ("gradient_inbox", (self.cp_size, self.owner_gradient_tokens, width), torch.float32),
                ("events", (2, self.cp_size, self.event_slots, 16), torch.int32))

    def arena_layout(self) -> tuple[int, tuple[tuple[str, int, tuple[int, ...], torch.dtype], ...]]:
        """Return arena bytes and aligned offsets; every independent signal occupies 64 bytes."""
        offset = 0
        fields = []
        for name, shape, dtype in self.buffers():
            offset = (offset + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
            fields.append((name, offset, shape, dtype))
            offset += math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        return (offset + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT, tuple(fields)


@dataclass(frozen=True)
class DsaWorkspaceInvocation:
    """Prepared logical identity and local-storage permutation, independent of scratch contents."""

    batch_meta: DsaBatchMeta
    owner_order: torch.Tensor
    workspace: MegaDsaWorkspace


class MegaDsaWorkspace:
    """Own bounded DSA scratch; saved backward activations remain caller-owned.

    Reserve before any root consumer binds. Forward/backward claim fresh leases
    and republish owner-local states at the current generation. This supplies
    storage for native tiles; it does not implement attention, a gradient bridge
    or a remote ready/ACK protocol. Publication only enqueues local copies.
    """

    def __init__(self, root: SharedShmemRoot, consumer_name: str, specification: DsaWorkspaceSpec) -> None:
        """Declare DSA bytes in the shared registry without allocating native memory."""
        if len(root.members) != specification.cp_size:
            raise ValueError("DSA CP must have the same ordered membership as the SHMEM root")
        self.root = root
        self.spec = specification
        self.arena_bytes, self.layout = specification.arena_layout()
        declaration = tuple((name, offset, shape, str(dtype)) for name, offset, shape, dtype in self.layout)
        self.consumer = root.reserve(ShmemConsumerSpec(consumer_name, "mega_dsa",
                                                       allocation_budget((self.arena_bytes,), torch.uint8, _ALIGNMENT),
                                                       declaration))
        self.arena: torch.Tensor | None = None
        self.buffers: dict[str, torch.Tensor] = {}
        self.active_invocation: tuple | None = None
        self.closed = False

    def bind(self) -> None:
        """Allocate one symmetric arena after all consumers have declared their budgets."""
        if self.closed:
            raise RuntimeError("DSA workspace is closed")
        if self.arena is not None:
            return
        self.consumer.bind()
        try:
            with self.consumer.access():
                self.arena = shmem.empty((self.arena_bytes,), dtype=torch.uint8, alignment=_ALIGNMENT)
                for name, offset, shape, dtype in self.layout:
                    size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                    self.buffers[name] = self.arena[offset:offset + size].view(dtype).reshape(shape)
                self.arena.zero_()
        except Exception:
            with self.consumer.access():
                if self.arena is not None:
                    shmem.free(self.arena)
                    self.arena = None
                self.buffers.clear()
            self.consumer.unbind()
            raise

    def prepare(self, batch_meta: DsaBatchMeta) -> DsaWorkspaceInvocation:
        """Prepare owner-local publication order outside the execution hot path."""
        self.bind()
        if (batch_meta.cp_ranks != self.root.members or batch_meta.cp_rank != self.root.rank
                or batch_meta.root_pes != tuple(range(self.spec.cp_size))):
            raise ValueError("DSA metadata must use the same ordered CP/root membership and direct root PE mapping")
        if batch_meta.heap_generation != self.root.generation:
            raise ValueError("DSA metadata heap_generation must match the bound root")
        owned = {token for token, owner in enumerate(batch_meta.token_owners) if owner == batch_meta.cp_rank}
        if set(batch_meta.kv_global_ids) != owned:
            raise ValueError("DSA source publication requires the complete owner-local KV shard")
        if len(owned) > min(self.spec.source_tokens, self.spec.owner_gradient_tokens):
            raise ValueError("DSA owner shard exceeds the declared source/gradient capacity")
        order = tuple(sorted(range(len(owned)), key=lambda row: batch_meta.token_local_offsets[
            batch_meta.kv_global_ids[row]]))
        return DsaWorkspaceInvocation(batch_meta, torch.tensor(order, dtype=torch.long, device=self.arena.device), self)

    @contextmanager
    def lease(self, invocation: DsaWorkspaceInvocation, *, direction: str) -> Iterator[MegaDsaWorkspace]:
        """Bind a forward/backward lease to invocation, layout and physical heap generation."""
        if invocation.workspace is not self or invocation.batch_meta.heap_generation != self.root.generation:
            raise ValueError("DSA invocation belongs to another workspace or heap generation")
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be forward or backward")
        self.consumer.claim()
        meta = invocation.batch_meta
        self.active_invocation = (meta.layer, meta.microbatch, meta.invocation, meta.layout_id,
                                  meta.heap_generation, direction)
        try:
            yield self
        finally:
            self.consumer.release()
            self.active_invocation = None

    def publish_owner_states(self, invocation: DsaWorkspaceInvocation, states: tuple[torch.Tensor, ...]) -> None:
        """Enqueue detached c/K-RoPE/index-K copies in canonical owner-offset order.

        The caller/native worker owns remote readiness and ACK ordering. No
        barrier, host synchronization or autograd path is implied by publication.
        Saved activations must remain independent of the reusable arena.
        """
        meta = invocation.batch_meta
        if (invocation.workspace is not self or self.active_invocation is None
                or self.root.lease_owner is not self.consumer):
            raise RuntimeError("publish_owner_states requires this workspace's active invocation lease")
        expected = (meta.layer, meta.microbatch, meta.invocation, meta.layout_id, meta.heap_generation)
        if self.active_invocation[:5] != expected:
            raise ValueError("DSA publication belongs to another invocation/layout/generation")
        names = ("source_compressed", "source_rope", "source_index")
        if len(states) != len(names):
            raise ValueError("publish_owner_states requires compressed KV, key RoPE and index key")
        for name, tensor in zip(names, states):
            destination = self.buffers[name]
            if (tensor.device != destination.device or tensor.dtype != destination.dtype
                    or tensor.shape != (len(meta.kv_global_ids), destination.shape[1])):
                raise ValueError("DSA publication tensors must match owner rows and prepared dtype/device/dimensions")
        with torch.no_grad():
            for name, tensor in zip(names, states):
                reordered = tensor.detach().index_select(0, invocation.owner_order)
                self.buffers[name][:len(meta.kv_global_ids)].copy_(reordered)

    def close(self) -> None:
        """Quiesce all streams and collectively free this consumer's arena once."""
        if self.closed:
            return
        if self.active_invocation is not None:
            raise RuntimeError("cannot close DSA workspace during its active lease")
        if self.arena is not None:
            with self.consumer.access():
                torch.npu.synchronize(self.root.device)
                shmem.host_barrier()
                shmem.free(self.arena)
                self.arena = None
                self.buffers.clear()
                shmem.host_barrier()
        self.consumer.close()
        self.closed = True
