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

from ._framework import get_torch
from .config import MegaMoeConfig, _resolve_receive_capacity

_GMM_WORKSPACE_BYTES = 32 * 1024 * 1024
_SWIGLU_GRAD_WORKSPACE_BYTES = 16 * 1024 * 1024
_EVENT_COUNTER_BYTES = 4096
_WORKSPACE_ALIGNMENT = 512
_HEAP_GRANULARITY_BYTES = 64 * 1024 * 1024


def _round_up(value: int, granularity: int) -> int:
    """Round ``value`` up to a positive ``granularity``."""
    return (value + granularity - 1) // granularity * granularity


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
    return tensor_bytes + 2 * _EVENT_COUNTER_BYTES + 4 * (_WORKSPACE_ALIGNMENT - 1)


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
    used: bool = False
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
        self.forward_event_counters = symm.aligned_empty(
            (_EVENT_COUNTER_BYTES,), torch.uint8, _WORKSPACE_ALIGNMENT
        )
        self.backward_event_counters = symm.aligned_empty(
            (_EVENT_COUNTER_BYTES,), torch.uint8, _WORKSPACE_ALIGNMENT
        )
        self.gmm_workspace = torch.empty(
            (_GMM_WORKSPACE_BYTES,), dtype=torch.uint8, device=device
        )
        self.swiglu_grad_workspace = torch.empty(
            (_SWIGLU_GRAD_WORKSPACE_BYTES,), dtype=torch.uint8, device=device
        )

    def claim(self) -> None:
        """Claim the fixed serial workspace lease."""
        torch = get_torch()

        with self.lock:
            if self.in_use:
                raise RuntimeError(
                    "MegaMoe execution resources do not support concurrent calls."
                )
            if self.used and not self.shared:
                torch.npu.synchronize()
            self.in_use = True

    def release(self) -> None:
        """Release the workspace lease after an enqueued kernel call."""
        torch = get_torch()

        if self.shared:
            torch.npu.synchronize()
        with self.lock:
            self.in_use = False
            self.used = True

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
