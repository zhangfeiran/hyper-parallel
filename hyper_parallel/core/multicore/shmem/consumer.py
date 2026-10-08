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
"""Frozen byte budgets and serial execution ownership for a shared SHMEM root."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from . import _lifecycle

_PAGE_BYTES = 2 * 1024 * 1024


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def allocation_budget(shape: tuple[int, ...], dtype: torch.dtype, alignment: int = 512) -> int:
    """Return conservative native bytes including alignment and vendor rounding."""
    if not shape or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape):
        raise ValueError("shape must contain nonnegative integer dimensions")
    _positive_integer("alignment", alignment)
    size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    return ((size + alignment - 1 + 15) // 16) * 16


def _host_layout(value: Any) -> bool:
    if isinstance(value, tuple):
        return all(_host_layout(item) for item in value)
    return value is None or isinstance(value, (str, int, float, bool))


@dataclass(frozen=True)
class ShmemConsumerSpec:
    """Consumer-specific byte reservation; layout remains owned by the consumer."""

    name: str
    kind: str
    budget_bytes: int
    layout: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.kind, str) or not self.name or not self.kind:
            raise ValueError("consumer name and kind must be nonempty strings")
        _positive_integer("budget_bytes", self.budget_bytes)
        if not isinstance(self.layout, tuple) or not _host_layout(self.layout):
            raise TypeError("consumer layout must be a host declaration tuple without tensors or mutable containers")


class ShmemConsumer:
    """One registered consumer with scoped allocations and a root-wide execution lease."""

    def __init__(self, root: SharedShmemRoot, specification: ShmemConsumerSpec) -> None:
        """Create an unbound consumer owned by the registry."""
        self.root = root
        self.specification = specification
        self.bound = False
        self.closed = False
        self.allocated_bytes = 0
        self.allocations: dict[int, int] = {}

    def bind(self) -> None:
        """Freeze and initialize the complete registry before this consumer can allocate."""
        with _lifecycle._lock:
            if self.bound or self.closed:
                raise RuntimeError("consumer is already bound or closed")
            if self.root.lease_owner is not None:
                raise RuntimeError("consumer binding requires a quiescent shared root")
            self.root.activate()
            self.bound = True

    @contextmanager
    def access(self) -> Iterator[None]:
        """Scope setup/teardown calls and exclude other consumers during execution."""
        with self.root.access(self):
            yield

    def claim(self) -> int:
        """Acquire the exclusive root lease after the previous stream's completion event."""
        return self.root.claim(self)

    def release(self) -> None:
        """Record completion of this consumer before another stream or consumer runs."""
        self.root.release_lease(self)

    def wait_for_reuse(self) -> None:
        """Order setup communication after prior root work without a host synchronization."""
        with self.access():
            self.root.wait_for_reuse()

    def unbind(self) -> None:
        """Roll back a failed binding after all partial allocations are freed."""
        with self.access():
            if self.allocations or self.root.lease_owner is self:
                raise RuntimeError("cannot unbind a consumer with allocations or an active lease")
            self.bound = False

    def close(self) -> None:
        """Retire a consumer after its queued work and allocations have been released."""
        if self.closed:
            return
        if not self.bound:
            self.closed = True
            return
        with self.access():
            if self.allocations or self.root.lease_owner is self:
                raise RuntimeError("consumer close requires freed allocations and no active lease")
            self.closed = True
            self.bound = False


class SharedShmemRoot:
    """Declare every consumer before binding; keep one native reference and frozen heap.

    The first version requires identical ordered CP/EP/root membership and
    serial execution. Consumer layouts are independent. No consumer can grow
    or rebuild this root. Every member declares/binds/allocates/frees/closes in
    the same collective order. Close consumers first, then close this root.
    Legacy standalone SHMEM lifecycles remain independent of this opt-in API.
    """

    def __init__(
        self, device: torch.device | str, *, root_group: dist.ProcessGroup | None = None,
        heap_size_bytes: int | None = None,
    ) -> None:
        """Prepare root identity and an optional explicit whole-root byte limit."""
        self.device = torch.device(device)
        self.group = root_group
        self.rank = dist.get_rank(root_group) if dist.is_initialized() else 0
        self.members = (tuple(dist.get_process_group_ranks(dist.group.WORLD if root_group is None else root_group))
                        if dist.is_initialized() else (0,))
        if not dist.is_initialized() and root_group is not None:
            raise ValueError("root_group requires initialized torch.distributed")
        if heap_size_bytes is None and os.environ.get("HYPER_PARALLEL_SHMEM_HEAP_SIZE") is not None:
            heap_size_bytes = int(os.environ["HYPER_PARALLEL_SHMEM_HEAP_SIZE"])
        if heap_size_bytes is not None:
            _positive_integer("heap_size_bytes", heap_size_bytes)
            if heap_size_bytes % _PAGE_BYTES:
                raise ValueError("heap_size_bytes must be a multiple of 2 MiB")
        self.heap_size_bytes = heap_size_bytes
        self.consumers: dict[str, ShmemConsumer] = {}
        self.state = "planning"
        self.generation = 0
        self.completion_event = None
        self.used = False
        self.lease_owner: ShmemConsumer | None = None
        self.lease_thread: int | None = None
        self._scope_owner: ShmemConsumer | None = None

    def reserve(self, specification: ShmemConsumerSpec) -> ShmemConsumer:
        """Reserve one named byte budget without initializing native SHMEM."""
        with _lifecycle._lock:
            if self.state != "planning":
                raise RuntimeError("shared root consumer registry is frozen after binding")
            if specification.name in self.consumers:
                raise ValueError("consumer name is already reserved")
            consumer = ShmemConsumer(self, specification)
            self.consumers[specification.name] = consumer
            return consumer

    def _consensus(self, value: Any, error: str | None) -> None:
        declarations = [(value, error)]
        if len(self.members) > 1:
            declarations = [None] * len(self.members)
            dist.all_gather_object(declarations, (value, error), group=self.group)
        failures = [item[1] for item in declarations if item[1] is not None]
        if failures:
            raise RuntimeError(f"shared root preparation failed: {failures}")
        if any(item[0] != value for item in declarations):
            raise RuntimeError("shared root membership, generation, budget or consumer order differs across ranks")

    def activate(self) -> None:
        """Collectively freeze byte declarations before the first native initialization."""
        with _lifecycle._lock:
            if self.state == "ready":
                return
            if self.state != "planning":
                raise RuntimeError(f"shared root is {self.state}")
            required = sum(item.specification.budget_bytes for item in self.consumers.values())
            budget = self.heap_size_bytes or ((required + _PAGE_BYTES - 1) // _PAGE_BYTES) * _PAGE_BYTES
            error = self._activation_error(required, budget)
            declaration = (self.members, budget, _lifecycle._heap_generation + 1,
                           tuple(item.specification for item in self.consumers.values()))
            self._consensus(declaration, error)
            event = torch.npu.Event()
            _lifecycle.acquire(self.group, heap_size_bytes=budget, _registry=self)
            self.heap_size_bytes = budget
            self.generation = _lifecycle._heap_generation
            self.completion_event = event
            self.state = "ready"

    def _activation_error(self, required: int, budget: int) -> str | None:
        if not self.consumers:
            return "shared root must declare at least one consumer"
        if budget < required:
            return f"shared root byte budget is too small: required={required}, configured={budget}"
        if _lifecycle._reference_count():
            return "shared root must be planned before any unmanaged SHMEM owner initializes"
        if (self.device.type != "npu"
                or torch.device("npu", torch.npu.current_device()) != self.device):
            return "shared root requires its prepared NPU as the current device"
        return None

    def _check_consumer(self, consumer: ShmemConsumer) -> None:
        if (self.state != "ready" or not consumer.bound or consumer.closed
                or _lifecycle._shutdown_failed or _lifecycle._registry_token is not self):
            raise RuntimeError("shared root requires a bound live consumer")
        if self.consumers.get(consumer.specification.name) is not consumer:
            raise RuntimeError("consumer belongs to another shared root")
        if torch.npu.current_device() != self.device.index:
            raise RuntimeError("SHMEM consumer requires its prepared NPU as the current device")
        if self.generation != _lifecycle._heap_generation:
            raise RuntimeError("shared root heap generation changed; stale addresses cannot execute")
        if self.lease_owner is not None and (self.lease_owner is not consumer
                                            or self.lease_thread != threading.get_ident()):
            raise RuntimeError("shared root does not support concurrent consumer execution")

    @contextmanager
    def access(self, consumer: ShmemConsumer) -> Iterator[None]:
        """Guard a consumer operation against root teardown and concurrent host submissions."""
        if not _lifecycle._lock.acquire(blocking=False):
            raise RuntimeError("shared root does not support concurrent host submissions")
        previous = self._scope_owner
        try:
            self._check_consumer(consumer)
            if previous is not None and previous is not consumer:
                raise RuntimeError("nested scopes must belong to the same SHMEM consumer")
            self._scope_owner = consumer
            yield
        finally:
            self._scope_owner = previous
            _lifecycle._lock.release()

    def wait_for_reuse(self) -> None:
        """Insert device stream ordering after the last consumer without a host readback."""
        if self.used:
            torch.npu.current_stream(self.device).wait_event(self.completion_event)

    def claim(self, consumer: ShmemConsumer) -> int:
        """Take a serial execution lease bound to this physical heap generation."""
        with self.access(consumer):
            if self.lease_owner is not None:
                raise RuntimeError("shared root lease is already active")
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("shared SHMEM consumer leases do not support graph capture")
            self.wait_for_reuse()
            self.lease_owner = consumer
            self.lease_thread = threading.get_ident()
            return self.generation

    def release_lease(self, consumer: ShmemConsumer) -> None:
        """Commit a completion event before publishing the root as reusable."""
        with self.access(consumer):
            if self.lease_owner is not consumer:
                raise RuntimeError("consumer has no active shared root lease")
            self.completion_event.record(torch.npu.current_stream(self.device))
            self.lease_owner = None
            self.lease_thread = None
            self.used = True

    def validate_submission(self, operation: str) -> None:
        """Reject unregistered API submissions while this registry owns the native reference."""
        consumer = self._scope_owner or self.lease_owner
        if consumer is None:
            raise RuntimeError("managed SHMEM submissions require a consumer scope or lease")
        self._check_consumer(consumer)
        if operation not in ("empty", "free", "host_barrier") and self.lease_owner is not consumer:
            raise RuntimeError("one-sided SHMEM submissions require the consumer execution lease")

    def allocation_requested(self, shape: tuple, dtype: torch.dtype, alignment: int | None) -> int:
        """Check a conservative allocation bill before entering native allocation."""
        if self.lease_owner is not None:
            raise RuntimeError("symmetric allocation must complete before claiming the execution lease")
        consumer = self._scope_owner or self.lease_owner
        bill = allocation_budget(shape, dtype, alignment or 16)
        if consumer.allocated_bytes + bill > consumer.specification.budget_bytes:
            raise RuntimeError(f"consumer {consumer.specification.name} exceeds its declared byte budget")
        return bill

    def allocation_created(self, tensor: torch.Tensor, bill: int) -> None:
        """Attribute the successful native allocation to the current consumer."""
        consumer = self._scope_owner or self.lease_owner
        consumer.allocations[tensor.data_ptr()] = bill
        consumer.allocated_bytes += bill

    def allocation_freeing(self, tensor: torch.Tensor) -> tuple[ShmemConsumer, int, int]:
        """Validate allocation ownership before the native call invalidates its storage."""
        if self.lease_owner is not None:
            raise RuntimeError("cannot free symmetric storage during an active execution lease")
        consumer = self._scope_owner or self.lease_owner
        pointer = tensor.data_ptr()
        if pointer not in consumer.allocations:
            raise RuntimeError("allocation is not owned by the current SHMEM consumer")
        return consumer, pointer, consumer.allocations[pointer]

    @staticmethod
    def allocation_freed(record: tuple[ShmemConsumer, int, int]) -> None:
        """Remove allocation accounting only after native free succeeded."""
        consumer, pointer, bill = record
        del consumer.allocations[pointer]
        consumer.allocated_bytes -= bill

    def close(self) -> None:
        """Collectively release the frozen heap after every bound consumer has closed."""
        with _lifecycle._lock:
            if self.state == "closed":
                return
            if self.lease_owner is not None or any(item.bound or item.allocations for item in self.consumers.values()):
                raise RuntimeError("shared root close requires all consumers closed and no active leases")
            if self.state == "ready":
                _lifecycle.release(_registry=self)
            for consumer in self.consumers.values():
                consumer.closed = True
            self.completion_event = None
            self.state = "closed"
