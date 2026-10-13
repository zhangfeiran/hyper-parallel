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
"""Coordinate replacement of a root heap through registered allocation owners."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from . import _lifecycle
from ._debug import debug_state

# Heap replacement is internal to SHMEM and requires its private lifecycle protocol.
# pylint: disable=protected-access

# Physical payload pages and virtual-address reservation have separate granularities.
_HEAP_GRANULARITY_BYTES = 2 * 1024 * 1024
_MANAGER = None


def _round_heap_bytes(value: int) -> int:
    """Round required payload bytes to physical pages."""
    return (value + _HEAP_GRANULARITY_BYTES - 1) // _HEAP_GRANULARITY_BYTES * _HEAP_GRANULARITY_BYTES


@dataclass
class _Entry:
    key: Any
    layout: tuple[int, tuple[Any, ...], Any]
    resource: Any = None


class ShmemHeapManager:
    """Manage one root's byte reservations and quiescent collective heap replacement.

    Layouts contain byte demand, a deterministic signature and an opaque owner payload.
    Owners expose heap_state(), release_heap() and restore_heap(payload, allocated).
    """

    def __init__(self, group: Any, tensor: torch.Tensor, reservations: tuple[Any, ...]) -> None:
        """Reserve every known layout without taking a native owner reference."""
        self.group = group
        self.members = _lifecycle.root_members(group)
        self.device = tensor.device
        self.element_size = tensor.element_size()
        self.entries: list[_Entry] = []
        self.epoch = 0
        self.state = "ready"
        self.growth_records: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._rebuild_thread = None
        configured = os.getenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE")
        self.heap_limit = None
        self._reserve(reservations)
        _lifecycle._reconfiguration_stage(lambda: self._configure(configured), group=group,
                                         value=(configured, str(tensor.dtype), tuple(item[1] for item in reservations)))

    def _configure(self, configured: str | None) -> None:
        """Validate the explicit heap limit and select the initial physical heap budget."""
        required = self.required_bytes([entry.layout for entry in self.entries])
        if configured is not None:
            try:
                self.heap_limit = int(configured)
            except ValueError as error:
                raise ValueError("HYPER_PARALLEL_SHMEM_HEAP_SIZE must be a positive integer number of bytes") from error
            if self.heap_limit <= 0:
                raise ValueError("HYPER_PARALLEL_SHMEM_HEAP_SIZE must be positive")
            if self.heap_limit < required:
                raise RuntimeError(f"HYPER_PARALLEL_SHMEM_HEAP_SIZE is too small: requires at least {required} bytes")
            if self.heap_limit % _HEAP_GRANULARITY_BYTES:
                raise ValueError("HYPER_PARALLEL_SHMEM_HEAP_SIZE must be a multiple of the 2 MiB physical page size")
        self.heap_bytes = self.budget(required)

    @property
    def resources(self) -> list[Any]:
        """Return bound owners in deterministic allocation order."""
        return [entry.resource for entry in self.entries if entry.resource is not None]

    @contextmanager
    def access(self) -> Iterator[None]:
        """Exclude concurrent submissions while permitting the rebuild thread to restore owners."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("SHMEM root does not support concurrent heap operations")
        try:
            if self.state == "failed" or (self.state != "ready" and self._rebuild_thread != threading.get_ident()):
                raise RuntimeError(f"SHMEM heap is {self.state}; no further execution is allowed")
            yield
        finally:
            self._lock.release()

    def _reserve(self, reservations: tuple[Any, ...]) -> None:
        for key, layout in reservations:
            if not any(entry.key is key for entry in self.entries):
                self.entries.append(_Entry(key, layout))

    def reserve(self, reservations: tuple[Any, ...], *, allow_rebuild: bool) -> None:
        """Include lazy owners while preserving already grown reservations.

        Args:
            reservations: Ordered pairs of identity keys and byte/signature/payload layouts.
            allow_rebuild: Whether bound owners support adding a larger layout.
        """
        with self.access():
            previous = len(self.entries)
            self._reserve(reservations)
            layouts = [entry.layout for entry in self.entries]
            try:
                _lifecycle._reconfiguration_stage(lambda: None, group=self.group,
                                                 value=tuple(layouts))
                required = self.required_bytes(layouts)
                if required <= self.heap_bytes:
                    return
                if self.resources and not allow_rebuild:
                    raise RuntimeError("Create all resources before initializing a fixed SHMEM layout")
                self._rebuild(layouts, self.budget(required))
            except Exception:
                if self.state == "ready":
                    del self.entries[previous:]
                raise

    def bind(self, resource: Any, key: Any) -> Any:
        """Bind an acquired owner and return its current opaque layout payload.

        Args:
            resource: Acquired owner providing heap state and release/restore operations.
            key: Identity key previously reserved for this owner.
        """
        for entry in self.entries:
            if entry.key is key:
                if entry.resource is not None:
                    raise RuntimeError("SHMEM heap owner is already bound")
                self.heap_bytes = debug_state()["config"]["heap_size_bytes"]
                entry.resource = resource
                return entry.layout[2]
        raise RuntimeError("SHMEM heap owner was not reserved")

    def remove(self, resource: Any) -> None:
        """Forget a closed owner after its allocations and native reference are released.

        Args:
            resource: Successfully closed owner.
        """
        self.entries[:] = [entry for entry in self.entries if entry.resource is not resource]

    @staticmethod
    def required_bytes(layouts: list[Any]) -> int:
        """Return the page-aligned total of byte reservations.

        Args:
            layouts: Byte/signature/payload triples for every reservation.
        """
        return _round_heap_bytes(sum(layout[0] for layout in layouts))

    def budget(self, required: int) -> int:
        """Apply an explicit limit or preserve the existing grow-only heap size.

        Args:
            required: Page-aligned total demand in bytes.
        """
        if self.heap_limit is not None and required > self.heap_limit:
            raise RuntimeError(
                f"SHMEM growth exceeds explicit heap budget: required={required}, limit={self.heap_limit}")
        return self.heap_limit if self.heap_limit is not None else max(getattr(self, "heap_bytes", 0), required)

    def replace(self, resource: Any, layout: tuple[Any, ...]) -> None:
        """Replace one owner's reservation and collectively rebuild the physical heap.

        Args:
            resource: Bound owner whose demand changes.
            layout: Target byte demand, collective signature and opaque owner payload.
        """
        with self.access():
            if not any(entry.resource is resource for entry in self.entries):
                raise RuntimeError("SHMEM heap owner was not bound")
            layouts = [layout if entry.resource is resource else entry.layout for entry in self.entries]
            self._rebuild(layouts, self.budget(self.required_bytes(layouts)))

    def _preflight(self, states: list[Any]) -> None:
        """Check ownership and layout before destroying any buffers."""
        state = debug_state()
        # SHMEM owns both the native reference count and the registered automatic-pool pins.
        if state["reference_count"] != len(self.resources) + _lifecycle._retained_users:
            raise RuntimeError("unmanaged SHMEM owner prevents heap growth")
        pointers = set()
        for entry in self.entries:
            snapshot = entry.resource.heap_state() if entry.resource is not None else None
            states.append(snapshot)
            if snapshot is None:
                continue
            allocations, in_use, _ = snapshot
            if in_use:
                raise RuntimeError("active workspace lease prevents heap growth")
            pointers.update(allocations)
        if pointers != {record["allocation_base"] for record in state["active_allocations"]}:
            raise RuntimeError("unmanaged SHMEM allocation prevents heap growth")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("SHMEM heap growth is unavailable during graph capture")

    def _rebuild(self, layouts: list[Any], heap_bytes: int) -> None:
        """Rebuild managed buffers through SHMEM's private coordinated lifecycle."""
        with _lifecycle._reconfiguration():
            start = time.perf_counter()
            states = []
            _lifecycle._reconfiguration_stage(
                lambda: self._preflight(states), group=self.group,
                value=lambda: (self.epoch, heap_bytes, tuple(
                    (layout, state is not None, state is not None and state[2])
                    for layout, state in zip(layouts, states))))
            timings = {"quiesce": _lifecycle._reconfiguration_stage(lambda: torch.npu.synchronize(self.device),
                                                                 group=self.group)}
            self.state = "rebuilding"
            self._rebuild_thread = threading.get_ident()
            old_bytes = self.heap_bytes
            try:
                for index, (entry, state) in enumerate(zip(self.entries, states)):
                    if state is not None and state[2]:
                        timings[f"free_{index}"] = _lifecycle._reconfiguration_stage(entry.resource.release_heap,
                                                                                   group=self.group)
                timings.update(_lifecycle._reinitialize(heap_bytes))
                for index, (entry, layout, state) in enumerate(zip(self.entries, layouts, states)):
                    if state is not None:
                        timings[f"allocate_{index}"] = _lifecycle._reconfiguration_stage(
                            lambda e=entry, p=layout[2], a=state[2]: e.resource.restore_heap(p, a), group=self.group)
                _lifecycle._reconfiguration_stage(lambda: None, group=self.group, value=("commit", self.epoch + 1))
                for entry, layout in zip(self.entries, layouts):
                    entry.layout = layout
                self.heap_bytes = heap_bytes
                self.epoch += 1
                self.state = "ready"
                self.growth_records.append({"epoch": self.epoch, "old_heap_bytes": old_bytes,
                                            "heap_bytes": heap_bytes, "stages_ms": timings,
                                            "total_ms": (time.perf_counter() - start) * 1000})
            except Exception:
                self.state = "failed"
                _lifecycle._invalidate_runtime()
                raise
            finally:
                self._rebuild_thread = None


def get_heap_manager(group: Any, tensor: torch.Tensor, reservations: tuple[Any, ...], *,
                     allow_rebuild: bool = True) -> ShmemHeapManager:
    """Select the active root coordinator and reserve all known owner layouts.

    Args:
        group: Ordered SHMEM root process group.
        tensor: Reference defining device and element size.
        reservations: All known ordered byte reservations, including lazy owners.
        allow_rebuild: Whether bound owners support reserving a larger layout.
    """
    global _MANAGER  # pylint: disable=global-statement
    if _MANAGER is None or (not _MANAGER.resources and not _lifecycle._retained_users):
        _MANAGER = ShmemHeapManager(group, tensor, reservations)
        return _MANAGER
    if _MANAGER.members != _lifecycle.root_members(group) or _MANAGER.device != tensor.device:
        raise RuntimeError("SHMEM heap requires one device and ordered root per process")
    if _MANAGER.element_size != tensor.element_size():
        raise ValueError("SHMEM heap owners must use the same element size")
    _MANAGER.reserve(reservations, allow_rebuild=allow_rebuild)
    return _MANAGER
