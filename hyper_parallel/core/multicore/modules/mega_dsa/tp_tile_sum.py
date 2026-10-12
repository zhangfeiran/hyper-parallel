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
"""Experimental bounded FP32 SHMEM tile SUM for mapped TP head shards."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.shmem.consumer import (
    SharedShmemRoot,
    ShmemConsumer,
    ShmemConsumerSpec,
    allocation_budget,
)
from hyper_parallel.core.multicore.torch.ops import _load_native

_TP_MAGIC = 0x4850445341545031
_SLOT_BYTES = 80 * 8


@dataclass(frozen=True)
class TpTileSumSpec:
    """Bound simultaneous vector slots and candidate width, independently of scan length."""

    slots: int = 40
    width: int = 512

    def __post_init__(self) -> None:
        if (isinstance(self.slots, bool) or not isinstance(self.slots, int) or not 1 <= self.slots <= 40
                or isinstance(self.width, bool) or not isinstance(self.width, int)
                or not 8 <= self.width <= 2048 or self.width % 8):
            raise ValueError("TP tile SUM requires slots in [1,40] and width in [8,2048], aligned to eight")

    @property
    def control_offset(self) -> int:
        """Return the first control cache line after all bounded source tiles."""
        return (self.slots * self.width * 4 + 63) // 64 * 64

    @property
    def arena_bytes(self) -> int:
        """Return fixed source/control storage, rounded to allocator alignment."""
        return (self.control_offset + self.slots * _SLOT_BYTES + 511) // 512 * 512

    def metadata(self, *, rounds: int, epoch: int, rank: int, root_pes: tuple[int, ...], root_size: int,
                 identity: tuple[int, int, int, int, int]) -> tuple[int, ...]:
        """Encode bounded transport geometry and invocation/ordered-row identity.

        Identity is (invocation, layer, microbatch, heap generation, row signature).
        The caller prepares one collision-free row-signature ID for matching TP
        candidates; native peers compare it before consuming the first tile.
        """
        if not isinstance(root_pes, tuple) or not isinstance(identity, tuple) or len(identity) != 5:
            raise ValueError("TP root mapping and five-word identity must be host tuples")
        values = (rounds, epoch, rank, root_size, *root_pes, *identity)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("TP transport metadata must contain host integers")
        if (not 0 < rounds < 2**32 or not 0 < epoch < 2**31 or not 1 <= len(root_pes) <= 8
                or not 0 <= rank < len(root_pes) or root_size < len(root_pes)
                or len(set(root_pes)) != len(root_pes) or any(not 0 <= pe < root_size for pe in root_pes)
                or any(not 0 <= value < 2**63 for value in identity)):
            raise ValueError("invalid TP transport epoch, membership or invocation identity")
        return (_TP_MAGIC, 1, epoch, rank, len(root_pes), root_size, self.slots, self.width, 0,
                self.control_offset, rounds, *identity, *root_pes, *(-1 for _ in range(8 - len(root_pes))))


@dataclass(frozen=True)
class TpTileSumInvocation:
    """Immutable device metadata, retained until this queued native invocation completes."""

    probe: TpTileSumProbe
    rounds: int
    epoch: int
    generation: int
    metadata: torch.Tensor
    metadata_version: int
    identity: tuple[int, ...] = ()


class TpTileSumProbe:
    """Reserve root-wide storage and execute only members of an explicit mapped TP group.

    All root members reserve/bind/close in matching collective order, including
    NON_GROUP_MEMBER ranks. Participating TP groups can have independent inputs
    and epochs. This is a transport probe, separate from native LI/SFA/KL math.
    Each invocation uses one ready/ACK-protected source tile per runnable slot;
    scan rounds do not grow the symmetric arena. Execution is serial per root.
    """

    def __init__(self, root: SharedShmemRoot, consumer_name: str, specification: TpTileSumSpec,
                 *, tp_group: dist.ProcessGroup) -> None:
        """Reserve identical source/control capacities before any root consumer binds."""
        if tp_group is None:
            raise ValueError("native TP transport requires an explicit TP group")
        self.root = root
        self.group = tp_group
        self.spec = specification
        self.participates = tp_group != dist.GroupMember.NON_GROUP_MEMBER
        self.members = tuple(dist.get_process_group_ranks(tp_group)) if self.participates else ()
        self.rank = dist.get_rank(tp_group) if self.participates else -1
        if self.participates and (not set(self.members).issubset(root.members) or not 1 <= len(self.members) <= 8):
            raise ValueError("TP group must contain up to eight members within the SHMEM root")
        self.root_pes = tuple(root.members.index(rank) for rank in self.members)
        if self.participates and (not 0 <= self.rank < len(self.members) or self.root_pes[self.rank] != root.rank):
            raise ValueError("TP ordinal must map to the caller SHMEM root PE")
        self.consumer = root.reserve(ShmemConsumerSpec(consumer_name, "mega_dsa_tp_tile_sum",
            allocation_budget((specification.arena_bytes,), torch.uint8, 512),
            (specification.slots, specification.width, specification.control_offset, _SLOT_BYTES)))
        self.arena = None
        self.last_epoch = 0
        self.closed = False

    def bind(self) -> None:
        """Collectively allocate and zero one symmetric arena, including inactive root members."""
        if self.closed:
            raise RuntimeError("TP transport probe is closed")
        if self.arena is not None:
            return
        self.consumer.bind()
        try:
            with self.consumer.access():
                self.arena = shmem.empty((self.spec.arena_bytes,), dtype=torch.uint8, alignment=512)
                self.arena.zero_()
        except Exception:
            with self.consumer.access():
                if self.arena is not None:
                    shmem.free(self.arena)
                    self.arena = None
            self.consumer.unbind()
            raise

    def prepare(self, *, rounds: int, epoch: int, identity: tuple[int, int, int, int, int]) -> TpTileSumInvocation:
        """Prepare matching TP geometry and identity outside execution, without device reads."""
        if not self.participates:
            raise ValueError("inactive root members cannot prepare TP work")
        self.bind()
        words = self.spec.metadata(rounds=rounds, epoch=epoch, rank=self.rank, root_pes=self.root_pes,
                                   root_size=len(self.root.members), identity=identity)
        if identity[3] != self.root.generation:
            raise ValueError("TP metadata heap generation must match the bound root")
        metadata = torch.tensor(words, dtype=torch.int64, device=self.arena.device)
        return TpTileSumInvocation(self, rounds, epoch, self.root.generation, metadata, metadata._version, identity)

    def run(self, partial: torch.Tensor, invocation: TpTileSumInvocation) -> torch.Tensor:
        """Queue detached FP32 tile reductions, returning an ordinary caller-owned output.

        Partial is [rounds,slots,width]. Peers execute the same ordered rounds.
        Native sums root-mapped peers in fixed TP order, acknowledges reads and
        waits for all final readers before releasing its source storage.
        """
        self.check_invocation(invocation)
        shape = (invocation.rounds, self.spec.slots, self.spec.width)
        if (partial.device != self.arena.device or partial.dtype != torch.float32 or partial.requires_grad
                or not partial.is_contiguous() or partial.shape != shape):
            raise ValueError("native TP partials must be detached contiguous FP32 [rounds,slots,width] on the root NPU")
        _load_native()
        version = getattr(torch.ops.hyper_parallel, "dsa_tp_tile_sum_version", None)
        if version is None or version() != 1:
            raise RuntimeError("native payload lacks DSA TP tile SUM version 1")
        output = torch.empty_like(partial)
        with self.lease(invocation):
            torch.ops.hyper_parallel.dsa_tp_tile_sum_out(partial, self.arena, invocation.metadata, output)
        return output

    def check_invocation(self, invocation: TpTileSumInvocation) -> None:
        """Reject stale identity/metadata before any native publication or root claim."""
        if (self.closed or self.arena is None or not self.participates or invocation.probe is not self
                or invocation.generation != self.root.generation or invocation.epoch <= self.last_epoch
                or invocation.metadata._version != invocation.metadata_version):
            raise ValueError("TP invocation must be live, matching and newer than prior work")

    def consume_in_root_lease(self, invocation: TpTileSumInvocation, owner: ShmemConsumer) -> None:
        """Consume an epoch under a fused caller's existing root lease.

        The caller queues one kernel using both consumers' arenas. Its completion
        event protects every arena; this method neither claims nor releases the
        root. Validate all arithmetic and transport arguments before consuming.
        """
        if owner.root is not self.root or self.root.lease_owner is not owner:
            raise ValueError("fused TP submission requires an active owner lease on the same root")
        with owner.access():
            self.check_invocation(invocation)
            self.last_epoch = invocation.epoch

    @contextmanager
    def lease(self, invocation: TpTileSumInvocation) -> Iterator[None]:
        """Claim one monotonic epoch for a native tile consumer, recording completion on release.

        Native capability and tensor validation must finish before entering this
        lease. A queued LI consumer can use fewer than its declared round bound;
        it must acknowledge all actually published tiles before returning.
        """
        self.check_invocation(invocation)
        self.consumer.claim()
        try:
            self.last_epoch = invocation.epoch
            yield
        finally:
            self.consumer.release()

    def close(self) -> None:
        """Quiesce and free this arena collectively after matching TP/root completion."""
        if self.closed:
            return
        if self.arena is not None:
            with self.consumer.access():
                torch.npu.synchronize(self.root.device)
                shmem.host_barrier()
                shmem.free(self.arena)
                self.arena = None
                shmem.host_barrier()
        self.consumer.close()
        self.closed = True
