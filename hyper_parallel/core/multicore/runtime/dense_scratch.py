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
"""Bounded per-stream scratch reuse, separate from retained autograd values."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from weakref import WeakSet, WeakValueDictionary

import torch

_POOLS: WeakValueDictionary[tuple[torch.device, int], DenseScratchPool] = WeakValueDictionary()
_POOLS_LOCK = Lock()


@dataclass
class DenseScratch:
    """Kernel-only storage; no inputs, outputs or saved activations are retained."""

    pointers: torch.Tensor
    events: torch.Tensor
    workspace: torch.Tensor
    overflow: torch.Tensor

    def view(self, pointers: int, event_elements: int) -> DenseScratch:
        """Expose exact descriptor lengths from grow-only backing storage.

        Args:
            pointers: Address count for this invocation.
            event_elements: Padded event words used by this invocation.
        """
        if pointers == self.pointers.numel() and event_elements == self.events.numel():
            return self
        return DenseScratch(self.pointers[:pointers], self.events[:event_elements], self.workspace, self.overflow)

    def prepare(self, addresses: tuple[int, ...]) -> None:
        """Refresh invocation addresses and reset joins in the current stream.

        Args:
            addresses: Current inputs and freshly allocated outputs in descriptor order.
        """
        if len(addresses) != self.pointers.numel():
            raise ValueError("Dense scratch pointer count differs from the bound descriptor")
        # Fresh CPU storage and a blocking copy preserve source lifetime across queued launches.
        self.pointers.copy_(torch.tensor(addresses, dtype=torch.int64, device="cpu"), non_blocking=False)
        self.events.zero_()
        self.overflow.zero_()


@dataclass
class _ScratchSlot:
    buffers: DenseScratch
    active: bool = False


class DenseScratchPool:
    """Keep bounded stream-owned slots and private storage for overlapping host calls.

    A cached slot stays on its original stream. Reset, pointer copies and launches
    use that stream's ordering; different streams never share a slot. Scratch is
    released after enqueue, while forward values remain owned by autograd. The
    native launcher records every device allocation on its captured stream.
    """

    def __init__(self, device: torch.device, pointers: int, event_elements: int, workspace_bytes: int,
                 max_cached_streams: int = 2) -> None:
        """Configure lazy storage without allocating any device memory.

        Args:
            device: Device owning the bound executable; CPU is used by protocol tests.
            pointers: Number of compact tensor-address slots.
            event_elements: Int32 elements, including native cache-line padding.
            workspace_bytes: Exact library workspace size reported by the selected SDK.
            max_cached_streams: Bound on persistent stream slots; zero keeps calls private.
        """
        sizes = (pointers, event_elements, workspace_bytes, max_cached_streams)
        if any(type(size) not in (int,) or size < 0 for size in sizes) or pointers == 0:
            raise ValueError("Dense scratch sizes must be nonnegative integers with positive pointer count")
        self.device = device
        self.pointers = pointers
        self.event_elements = event_elements
        self.workspace_bytes = workspace_bytes
        self.max_cached_streams = max_cached_streams
        self._slots: dict[object, _ScratchSlot] = {}
        self._owners: WeakSet[object] = WeakSet()
        self._lock = Lock()
        self._closed = False
        self._allocations = 0
        self._reuses = 0

    def _allocate(self, pointers, event_elements) -> _ScratchSlot:
        buffers = DenseScratch(
            torch.empty(pointers, dtype=torch.int64, device=self.device),
            torch.empty(event_elements, dtype=torch.int32, device=self.device),
            torch.empty(self.workspace_bytes, dtype=torch.uint8, device=self.device),
            torch.empty(8, dtype=torch.uint8, device=self.device),
        )
        self._allocations += 1
        return _ScratchSlot(buffers)

    def _grow(self, slot, pointers, event_elements):
        buffers = slot.buffers
        addresses, events = buffers.pointers, buffers.events
        if pointers > addresses.numel():
            addresses = torch.empty(pointers, dtype=torch.int64, device=self.device)
        if event_elements > events.numel():
            events = torch.empty(event_elements, dtype=torch.int32, device=self.device)
        if addresses is not buffers.pointers or events is not buffers.events:
            # Queued handlers retain the old Tensor objects when a capacity grows.
            slot.buffers = DenseScratch(addresses, events, buffers.workspace, buffers.overflow)

    def register_owner(self, owner: object) -> bool:
        """Attach a weak executable owner while admitting no owners after close.

        Args:
            owner: Bound executable retaining this pool.

        Returns:
            Whether this still-open pool admitted the owner.
        """
        with self._lock:
            if self._closed:
                return False
            self._owners.add(owner)
            return True

    def release_owner(self, owner: object) -> None:
        """Detach a closed executable, releasing the cache after its last live peer closes.

        Args:
            owner: Bound executable whose future forwards have been rejected.
        """
        with self._lock:
            self._owners.discard(owner)
            if not self._owners:
                self._closed = True
                self._slots.clear()

    @contextmanager
    def lease(self, stream: object, *, pointers: int | None = None,
              event_elements: int | None = None) -> Iterator[DenseScratch]:
        """Reserve a stream's slot until all its launch commands have been enqueued.

        Args:
            stream: Hashable current-stream object, equal for wrappers of the same native stream.
            pointers: Descriptor address count; defaults to the constructor's count.
            event_elements: Padded event words; defaults to the constructor's count.

        Yields:
            Resettable scratch exclusive to this host invocation.
        """
        pointers = self.pointers if pointers is None else pointers
        event_elements = self.event_elements if event_elements is None else event_elements
        if (type(pointers) not in (int,) or pointers <= 0
                or type(event_elements) not in (int,) or event_elements < 0):
            raise ValueError("Dense scratch descriptor counts must be valid nonnegative integers")
        with self._lock:
            if self._closed:
                raise RuntimeError("Dense scratch pool is closed")
            slot = self._slots.get(stream)
            if slot is not None and not slot.active:
                self._grow(slot, pointers, event_elements)
                self._reuses += 1
            else:
                slot = self._allocate(pointers, event_elements)
                if stream not in self._slots and len(self._slots) < self.max_cached_streams:
                    self._slots[stream] = slot
            slot.active = True
        try:
            yield slot.buffers.view(pointers, event_elements)
        finally:
            with self._lock:
                slot.active = False

    def statistics(self) -> dict[str, int]:
        """Report persistent bytes and slot allocation/reuse counts without device synchronization."""
        with self._lock:
            sizes = [sum(value.numel() * value.element_size() for value in
                         (slot.buffers.pointers, slot.buffers.events, slot.buffers.workspace, slot.buffers.overflow))
                     for slot in self._slots.values()]
            size = max(sizes, default=self.pointers * 8 + self.event_elements * 4 + self.workspace_bytes + 8)
            return {"cached_streams": len(self._slots), "max_cached_streams": self.max_cached_streams,
                    "cached_bytes": sum(sizes), "bytes_per_slot": size,
                    "allocations": self._allocations, "reuses": self._reuses}

    def close(self) -> None:
        """Drop cached storage; active leases and queued native calls retain their own references."""
        with self._lock:
            self._closed = True
            self._owners.clear()
            self._slots.clear()


def shared_dense_scratch(device: torch.device, pointers: int, event_elements: int, workspace_bytes: int,
                         owner: object) -> DenseScratchPool:
    """Bind a weakly registered pool shared across layers and token shapes on one device.

    Args:
        device: Normalized device owning the executable.
        pointers: Initial descriptor address count.
        event_elements: Initial padded event count; leases may grow its capacity.
        workspace_bytes: Exact SDK workspace size, isolating different workspace contracts.
        owner: Executable retaining the pool until close or garbage collection.
    """
    key = (device, workspace_bytes)
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None or not pool.register_owner(owner):
            pool = DenseScratchPool(device, pointers, event_elements, workspace_bytes)
            pool.register_owner(owner)
            _POOLS[key] = pool
        return pool
