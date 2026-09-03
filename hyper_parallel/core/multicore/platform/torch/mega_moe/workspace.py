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
"""Owned directional workspace for the managed Torch MegaMoe backend."""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from hyper_parallel.core.multicore.scheduler.config import (
    EVENT_COUNTER_BYTES as _EVENT_COUNTER_BYTES,
    READY_CACHE_LINE_BYTES as _READY_CACHE_LINE_BYTES,
    ROUTE_ARRIVAL_STRIDE_BYTES as _ROUTE_ARRIVAL_STRIDE_BYTES,
    event_workspace_bytes,
)

from ._framework import get_torch
from .config import (
    _COMMUNICATION_SPLIT,
    MegaMoeConfig,
    _resolve_fusion_modes,
    _resolve_receive_capacity,
)

_GMM_WORKSPACE_BYTES = 32 * 1024 * 1024
_SWIGLU_GRAD_WORKSPACE_BYTES = 16 * 1024 * 1024
_WORKSPACE_ALIGNMENT = 512
_HEAP_GRANULARITY_BYTES = 64 * 1024 * 1024


def _round_up(value: int, granularity: int) -> int:
    """Round ``value`` up to a positive ``granularity``."""
    return (value + granularity - 1) // granularity * granularity


def _event_workspace_bytes(
    *,
    ep_size: int,
    local_num_tokens: int,
    num_experts: int,
    dispatch_split: int,
    fusion_mode: str,
) -> int:
    """Return event storage including isolated route-dispatch arrivals."""
    standard_bytes = event_workspace_bytes(ep_size)
    if fusion_mode not in {"permute", "both"}:
        return standard_bytes
    tasks_per_expert = (
        local_num_tokens + dispatch_split - 1
    ) // dispatch_split
    return (
        standard_bytes
        + num_experts * tasks_per_expert * _ROUTE_ARRIVAL_STRIDE_BYTES
    )


def _spec_workspace_bytes(specification: Mapping[str, Any], element_size: int) -> int:
    """Return planned symmetric bytes for one resource group."""
    local_tokens = specification["local_num_tokens"]
    routed_slots = local_tokens * specification["top_k"]
    capacity = _resolve_receive_capacity(
        specification["expert_capacity_factor"], routed_slots
    )
    tensor_bytes = (
        (capacity + routed_slots) * specification["hidden_size"] * element_size
    )
    forward_mode, backward_mode = _resolve_fusion_modes(specification["top_k"])
    forward_event_bytes = _event_workspace_bytes(
        ep_size=specification["ep_size"],
        local_num_tokens=local_tokens,
        num_experts=specification["num_experts"],
        dispatch_split=_COMMUNICATION_SPLIT,
        fusion_mode=forward_mode,
    )
    backward_event_bytes = _event_workspace_bytes(
        ep_size=specification["ep_size"],
        local_num_tokens=local_tokens,
        num_experts=specification["num_experts"],
        dispatch_split=_COMMUNICATION_SPLIT,
        fusion_mode=backward_mode,
    )
    return (
        tensor_bytes
        + forward_event_bytes
        + backward_event_bytes
        + 4 * (_WORKSPACE_ALIGNMENT - 1)
    )


def configure_symmetric_heap(
    active_specifications: tuple[Any, ...],
    tensor: Any,
) -> int:
    """Size the fixed SHMEM heap once for all known resource groups.

    Args:
        active_specifications: Specifications for every live compatible resource group.
        tensor: First input tensor used to determine element size.

    Returns:
        Configured symmetric heap size in bytes.
    """
    required_bytes = sum(
        _spec_workspace_bytes(specification, tensor.element_size())
        for specification in active_specifications
    )
    required_bytes = _round_up(required_bytes, _HEAP_GRANULARITY_BYTES)
    configured = os.getenv("SYMMETRIC_MEMORY_HEAP_SIZE")
    if configured is None:
        os.environ["SYMMETRIC_MEMORY_HEAP_SIZE"] = str(required_bytes)
        return required_bytes
    try:
        configured_bytes = int(configured)
    except ValueError as error:
        raise ValueError(
            "SYMMETRIC_MEMORY_HEAP_SIZE must be a positive integer number of bytes, "
            f"got {configured!r}."
        ) from error
    if configured_bytes <= 0:
        raise ValueError("SYMMETRIC_MEMORY_HEAP_SIZE must be positive.")
    if configured_bytes < required_bytes:
        raise RuntimeError(
            "SYMMETRIC_MEMORY_HEAP_SIZE is too small for active MegaMoe resources: "
            f"configured {configured_bytes}, requires at least {required_bytes}."
        )
    return configured_bytes


@dataclass
class MegaMoeWorkspace:
    """Buffers owned by one standalone or explicitly shared resource group."""

    shared: bool
    dtype: Any | None = None
    device: Any | None = None
    expert_capacity: int = 0
    routed_slots: int = 0
    expert_buffer: Any | None = None
    routed_buffer: Any | None = None
    forward_event_counters: Any | None = None
    backward_event_counters: Any | None = None
    gmm_workspace: Any | None = None
    swiglu_grad_workspace: Any | None = None
    in_use: bool = False
    reuse_event: Any | None = None
    reuse_event_pending: bool = False
    forward_ready_initialized: bool = False
    backward_ready_initialized: bool = False
    forward_route_arrivals_enabled: bool = False
    backward_route_arrivals_enabled: bool = False
    route_arrival_offset_bytes: int = 0
    lock: Any = field(default_factory=threading.Lock, repr=False)

    def ensure(
        self, config: MegaMoeConfig, dtype: Any, device: Any, capacity: int
    ) -> None:
        """Allocate or collectively grow buffers for an exact route capacity.

        Args:
            config: Validated local-token and expert topology configuration.
            dtype: Element dtype for symmetric data buffers.
            device: Device that owns local and symmetric buffers.
            capacity: Exact receive capacity required by the current route.
        """
        torch = get_torch()
        symm = importlib.import_module("hyper_parallel.core.symmetric_memory")

        requested_capacity = config.receive_capacity if self.shared else capacity
        if self.expert_buffer is not None:
            compatible = (
                self.dtype == dtype
                and self.device == device
                and self.routed_slots == config.routed_slots
            )
            if not compatible:
                raise ValueError(
                    "MegaMoe workspace cannot change device, dtype or routed shape."
                )
            if self.expert_capacity >= requested_capacity:
                return
            torch.npu.synchronize()
            symm.barrier()
            self._free_symmetric_tensors()
            symm.barrier()

        self.dtype = dtype
        self.device = device
        self.expert_capacity = requested_capacity
        self.routed_slots = config.routed_slots
        self.expert_buffer = symm.aligned_empty(
            (requested_capacity, config.hidden_size), dtype, _WORKSPACE_ALIGNMENT
        )
        self.routed_buffer = symm.aligned_empty(
            (config.routed_slots, config.hidden_size), dtype, _WORKSPACE_ALIGNMENT
        )
        self.forward_route_arrivals_enabled = config.forward_permute_fused
        self.backward_route_arrivals_enabled = config.backward_permute_fused
        self.route_arrival_offset_bytes = (
            event_workspace_bytes(config.ep_size)
            if (
                self.forward_route_arrivals_enabled
                or self.backward_route_arrivals_enabled
            )
            else 0
        )
        forward_event_bytes = _event_workspace_bytes(
            ep_size=config.ep_size,
            local_num_tokens=config.local_num_tokens,
            num_experts=config.num_experts,
            dispatch_split=config.dispatch_split,
            fusion_mode=config.forward_fusion_mode,
        )
        backward_event_bytes = _event_workspace_bytes(
            ep_size=config.ep_size,
            local_num_tokens=config.local_num_tokens,
            num_experts=config.num_experts,
            dispatch_split=config.dispatch_split,
            fusion_mode=config.backward_fusion_mode,
        )
        self.forward_event_counters = symm.aligned_empty(
            (forward_event_bytes,), torch.uint8, _WORKSPACE_ALIGNMENT
        )
        self.backward_event_counters = symm.aligned_empty(
            (backward_event_bytes,), torch.uint8, _WORKSPACE_ALIGNMENT
        )
        self.gmm_workspace = torch.empty(
            (_GMM_WORKSPACE_BYTES,), dtype=torch.uint8, device=device
        )
        self.swiglu_grad_workspace = torch.empty(
            (_SWIGLU_GRAD_WORKSPACE_BYTES,), dtype=torch.uint8, device=device
        )

    def prepare_event_counters(self, *, forward: bool) -> Any:
        """Reset per-call counters while preserving the ready generation."""
        if forward:
            field_name = "forward_event_counters"
            events = self.forward_event_counters
            ready_initialized = self.forward_ready_initialized
            route_arrivals_enabled = self.forward_route_arrivals_enabled
        else:
            field_name = "backward_event_counters"
            events = self.backward_event_counters
            ready_initialized = self.backward_ready_initialized
            route_arrivals_enabled = self.backward_route_arrivals_enabled
        if events is None:
            raise RuntimeError(f"MegaMoe workspace {field_name} is not initialized.")
        events[:_EVENT_COUNTER_BYTES].zero_()
        if route_arrivals_enabled:
            events[self.route_arrival_offset_bytes:].zero_()
        if events.numel() == _EVENT_COUNTER_BYTES or ready_initialized:
            return events

        # Initialize the persistent generation before any peer can signal it.
        symm = importlib.import_module("hyper_parallel.core.symmetric_memory")

        events[_EVENT_COUNTER_BYTES:].zero_()
        symm.barrier()
        if forward:
            self.forward_ready_initialized = True
        else:
            self.backward_ready_initialized = True
        return events

    def claim(self) -> None:
        """Claim the fixed serial workspace lease."""
        with self.lock:
            if self.in_use:
                raise RuntimeError(
                    "MegaMoe execution resources do not support concurrent calls."
                )
            self._wait_for_reuse()
            self.reuse_event_pending = False
            self.in_use = True

    def wait_for_reuse(self) -> None:
        """Order route exchange after the previous lease without a host wait."""
        with self.lock:
            self._wait_for_reuse()

    def _wait_for_reuse(self) -> None:
        """Enqueue the pending event wait while the lease lock is held."""
        torch = get_torch()

        if not self.reuse_event_pending:
            return
        if self.reuse_event is None or self.device is None:
            raise RuntimeError("MegaMoe workspace has an incomplete reuse event.")
        torch.npu.current_stream(self.device).wait_event(self.reuse_event)

    def release(self) -> None:
        """Release the workspace lease after an enqueued kernel call."""
        torch = get_torch()

        with self.lock:
            if not self.in_use:
                raise RuntimeError("MegaMoe execution resources are not currently claimed.")
            if self.reuse_event is None:
                self.reuse_event = torch.npu.Event()
            if self.device is None:
                raise RuntimeError("MegaMoe workspace device is not initialized.")
            self.reuse_event.record(torch.npu.current_stream(self.device))
            self.reuse_event_pending = True
            self.in_use = False

    def _free_symmetric_tensors(self) -> None:
        """Free each unique SHMEM allocation and invalidate its tensor view."""
        symm = importlib.import_module("hyper_parallel.core.symmetric_memory")

        for field_name in (
            "expert_buffer",
            "routed_buffer",
            "forward_event_counters",
            "backward_event_counters",
        ):
            tensor = getattr(self, field_name)
            if tensor is None:
                continue
            symm.free(tensor)
            tensor.resize_(0)
            setattr(self, field_name, None)
        self.expert_capacity = 0
        self.routed_slots = 0
        self.forward_ready_initialized = False
        self.backward_ready_initialized = False
        self.forward_route_arrivals_enabled = False
        self.backward_route_arrivals_enabled = False
        self.route_arrival_offset_bytes = 0
