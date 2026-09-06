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
"""Owned directional workspace for managed Torch MegaMoe execution."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from hyper_parallel.core.multicore.scheduler.config import (
    EVENT_COUNTER_BYTES as _EVENT_COUNTER_BYTES,
    NUM_WORKERS_CUBE,
    READY_CACHE_LINE_BYTES,
    event_workspace_bytes,
)
from hyper_parallel.core.multicore.scheduler.config import (
    ROUTE_ARRIVAL_STRIDE_BYTES as _ROUTE_ARRIVAL_STRIDE_BYTES,
)

from .spec import (
    _COMMUNICATION_SPLIT,
    MegaMoeSpec,
    _resolve_fusion_modes,
    _resolve_receive_capacity,
)

torch = import_module("torch")

_GMM_WORKSPACE_BYTES = 32 * 1024 * 1024
_SWIGLU_GRAD_WORKSPACE_BYTES = 16 * 1024 * 1024
_CYCLE_PROFILE_STORAGE_BYTES = 4 * 1024
_CYCLE_PROFILE_CATEGORIES = (
    "descriptor_decode",
    "dependency_wait",
    "gmm",
    "dispatch_dy",
    "dispatch_hidden",
    "combine",
    "local_compute",
    "synchronization",
)
_CYCLE_PROFILE_VALUES_PER_WORKER = len(_CYCLE_PROFILE_CATEGORIES)
_CYCLE_PROFILE_WORKER_SLOTS = 2 * NUM_WORKERS_CUBE
_CYCLE_PROFILE_BYTES = (
    _CYCLE_PROFILE_WORKER_SLOTS
    * _CYCLE_PROFILE_VALUES_PER_WORKER
    * 8
)
_ROUTE_SIGNAL_ARENAS = 2
_WORKSPACE_ALIGNMENT = 512
_HEAP_GRANULARITY_BYTES = 64 * 1024 * 1024
# Retain the private alias used by the workspace sizing tests.
_READY_CACHE_LINE_BYTES = READY_CACHE_LINE_BYTES


@dataclass(frozen=True)
class _WeightGradScratchPlan:
    """Whole-matrix placement shared conceptually with the device helper."""

    retained_w1_experts: int
    retained_w2_experts: int
    tail_bytes: int


def _round_up(value: int, granularity: int) -> int:
    """Round ``value`` up to a positive ``granularity``."""
    return (value + granularity - 1) // granularity * granularity


def _plan_weight_grad_scratch(
    *,
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    element_size: int,
    receive_capacity: int,
) -> _WeightGradScratchPlan:
    """Pack whole dW matrices into one dead retained up-projection window.

    W1 is exactly twice W2 for the gated-MLP layout. Packing W1 first and W2
    second therefore maximizes occupied prefix bytes without splitting a
    matrix. Remaining W2 and W1 matrices retain deterministic tail order.
    """
    dimensions = (
        local_experts,
        hidden_size,
        intermediate_size,
        element_size,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError("weight-gradient scratch dimensions must be positive.")
    if receive_capacity < 0:
        raise ValueError("receive_capacity cannot be negative.")

    w1_matrix_bytes = (
        hidden_size * 2 * intermediate_size * element_size
    )
    w2_matrix_bytes = hidden_size * intermediate_size * element_size
    prefix_bytes = receive_capacity * 2 * intermediate_size * element_size
    retained_w1_experts = min(local_experts, prefix_bytes // w1_matrix_bytes)
    prefix_bytes -= retained_w1_experts * w1_matrix_bytes
    retained_w2_experts = min(local_experts, prefix_bytes // w2_matrix_bytes)
    tail_bytes = (
        (local_experts - retained_w2_experts) * w2_matrix_bytes
        + (local_experts - retained_w1_experts) * w1_matrix_bytes
    )
    return _WeightGradScratchPlan(
        retained_w1_experts=retained_w1_experts,
        retained_w2_experts=retained_w2_experts,
        tail_bytes=tail_bytes,
    )


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
    route_arena_bytes = (
        num_experts * tasks_per_expert * _ROUTE_ARRIVAL_STRIDE_BYTES
    )
    # The first arena carries dispatch arrivals. The second carries the
    # receiver's copy-complete acknowledgement before dY reuses that tile.
    return standard_bytes + _ROUTE_SIGNAL_ARENAS * route_arena_bytes


def _spec_workspace_bytes(specification: Mapping[str, Any], element_size: int) -> int:
    """Return planned symmetric bytes for one execution-resource group."""
    local_tokens = specification["local_num_tokens"]
    routed_slots = local_tokens * specification["top_k"]
    capacity = _resolve_receive_capacity(
        specification["expert_capacity_factor"],
        routed_slots,
        specification["ep_size"],
    )
    tensor_bytes = (
        (capacity + routed_slots) * specification["hidden_size"] * element_size
    )
    forward_mode, backward_mode = _resolve_fusion_modes(
        specification["top_k"],
        routed_slots,
    )
    physical_num_experts = (
        specification["num_experts"]
        + specification["ep_size"]
        * specification.get("replica_slots_per_rank", 0)
    )
    forward_event_bytes = _event_workspace_bytes(
        ep_size=specification["ep_size"],
        local_num_tokens=local_tokens,
        num_experts=physical_num_experts,
        dispatch_split=_COMMUNICATION_SPLIT,
        fusion_mode=forward_mode,
    )
    backward_route_arrivals = any(
        mode in {"permute", "both"}
        for mode in (forward_mode, backward_mode)
    )
    backward_event_bytes = _event_workspace_bytes(
        ep_size=specification["ep_size"],
        local_num_tokens=local_tokens,
        num_experts=physical_num_experts,
        dispatch_split=_COMMUNICATION_SPLIT,
        # Multi-wave W1 replay route-gathers a token-major forward input even
        # when backward dY dispatch itself uses the legacy endpoint.
        fusion_mode="permute" if backward_route_arrivals else backward_mode,
    )
    return (
        tensor_bytes
        + forward_event_bytes
        + backward_event_bytes
        + 4 * (_WORKSPACE_ALIGNMENT - 1)
    )


def configure_symmetric_heap(
    active_specifications: tuple[Any, ...],
    tensor: torch.Tensor,
) -> int:
    """Size the fixed SHMEM heap once for all active resource groups.

    Args:
        active_specifications: Specifications for every live compatible resource group.
        tensor: First input tensor used to determine element size.

    Returns:
        Configured symmetric heap size in bytes.

    Raises:
        ValueError: If the configured heap size is invalid.
        RuntimeError: If an explicit heap is smaller than the required workspace.
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
        raise ValueError(
            f"SYMMETRIC_MEMORY_HEAP_SIZE must be positive, got {configured_bytes}."
        )
    if configured_bytes < required_bytes:
        raise RuntimeError(
            "SYMMETRIC_MEMORY_HEAP_SIZE is too small for active MegaMoe resources: "
            f"configured {configured_bytes} bytes, requires at least {required_bytes} bytes."
        )
    return configured_bytes


@dataclass
class MegaMoeWorkspace:
    """Buffers owned by one standalone or explicitly shared resource group."""

    symmetric_memory: Any
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
    completion_event: Any | None = None
    in_use: bool = False
    used: bool = False
    routed_buffer_generation: int = 0
    forward_ready_initialized: bool = False
    backward_ready_initialized: bool = False
    forward_route_arrivals_enabled: bool = False
    backward_route_arrivals_enabled: bool = False
    route_arrival_offset_bytes: int = 0
    lock: Any = field(default_factory=threading.Lock, repr=False)

    def ensure(self, spec: MegaMoeSpec, dtype: Any, device: Any) -> None:
        """Allocate the fixed configured route capacity once.

        Args:
            spec: First-forward-bound shape and capacity specification.
            dtype: Element dtype for symmetric data buffers.
            device: Device that owns local and symmetric buffers.
        """
        requested_capacity = spec.receive_capacity
        if self.expert_buffer is not None:
            compatible = (
                self.dtype == dtype
                and self.device == device
                and self.routed_slots == spec.routed_slots
                and self.expert_capacity == requested_capacity
            )
            if not compatible:
                raise ValueError(
                    "MegaMoe workspace cannot change device, dtype, routed shape, or capacity."
                )
            return

        self.dtype = dtype
        self.device = device
        self.expert_capacity = requested_capacity
        self.routed_slots = spec.routed_slots
        self.forward_route_arrivals_enabled = spec.forward_permute_fused
        self.backward_route_arrivals_enabled = (
            spec.forward_permute_fused or spec.backward_permute_fused
        )
        self.route_arrival_offset_bytes = (
            event_workspace_bytes(spec.ep_size)
            if (
                self.forward_route_arrivals_enabled
                or self.backward_route_arrivals_enabled
            )
            else 0
        )
        forward_event_bytes = _event_workspace_bytes(
            ep_size=spec.ep_size,
            local_num_tokens=spec.local_num_tokens,
            num_experts=spec.physical_num_experts,
            dispatch_split=spec.dispatch_split,
            fusion_mode=spec.forward_fusion_mode,
        )
        backward_event_bytes = _event_workspace_bytes(
            ep_size=spec.ep_size,
            local_num_tokens=spec.local_num_tokens,
            num_experts=spec.physical_num_experts,
            dispatch_split=spec.dispatch_split,
            fusion_mode=(
                "permute"
                if self.backward_route_arrivals_enabled
                else spec.backward_fusion_mode
            ),
        )
        try:
            self.expert_buffer = self.symmetric_memory.aligned_empty(
                (requested_capacity, spec.hidden_size),
                dtype,
                _WORKSPACE_ALIGNMENT,
            )
            self.routed_buffer = self.symmetric_memory.aligned_empty(
                (spec.routed_slots, spec.hidden_size),
                dtype,
                _WORKSPACE_ALIGNMENT,
            )
            self.forward_event_counters = self.symmetric_memory.aligned_empty(
                (forward_event_bytes,),
                torch.uint8,
                _WORKSPACE_ALIGNMENT,
            )
            self.backward_event_counters = self.symmetric_memory.aligned_empty(
                (backward_event_bytes,),
                torch.uint8,
                _WORKSPACE_ALIGNMENT,
            )
            self.gmm_workspace = torch.empty(
                (_GMM_WORKSPACE_BYTES,),
                dtype=torch.uint8,
                device=device,
            )
            self.swiglu_grad_workspace = torch.empty(
                (_SWIGLU_GRAD_WORKSPACE_BYTES + _CYCLE_PROFILE_STORAGE_BYTES,),
                dtype=torch.uint8,
                device=device,
            )
            self.completion_event = torch.npu.Event()
        except Exception:
            self._free_symmetric_tensors()
            self._free_local_tensors()
            raise

    def prepare_cycle_profile(self, *, enabled: bool) -> None:
        """Clear the fixed diagnostic tail only for an instrumented launch."""
        if not enabled:
            return
        workspace = self.swiglu_grad_workspace
        if workspace is None:
            raise RuntimeError("MegaMoe SwiGLU-grad workspace is not initialized.")
        workspace.narrow(
            0,
            _SWIGLU_GRAD_WORKSPACE_BYTES,
            _CYCLE_PROFILE_STORAGE_BYTES,
        ).zero_()

    def collect_cycle_profile(self, num_cube_cores: int) -> dict[str, Any]:
        """Return counters from the last profiled backward.

        Args:
            num_cube_cores: Number of populated Cube and Vector worker slots.

        Returns:
            Per-core counters and aggregate sums and maxima.
        """
        if not 0 < num_cube_cores <= NUM_WORKERS_CUBE:
            raise ValueError(
                f"num_cube_cores must be in [1, {NUM_WORKERS_CUBE}], got "
                f"{num_cube_cores}."
            )
        workspace = self.swiglu_grad_workspace
        if workspace is None:
            raise RuntimeError("MegaMoe SwiGLU-grad workspace is not initialized.")
        values = (
            workspace.narrow(
                0,
                _SWIGLU_GRAD_WORKSPACE_BYTES,
                _CYCLE_PROFILE_BYTES,
            )
            .view(torch.int64)
            .reshape(
                _CYCLE_PROFILE_WORKER_SLOTS,
                _CYCLE_PROFILE_VALUES_PER_WORKER,
            )
            .cpu()
            .tolist()
        )

        def _summarize(rows: list[list[int]]) -> dict[str, Any]:
            return {
                "per_core": rows,
                "sum": {
                    category: sum(row[index] for row in rows)
                    for index, category in enumerate(_CYCLE_PROFILE_CATEGORIES)
                },
                "max": {
                    category: max(row[index] for row in rows)
                    for index, category in enumerate(_CYCLE_PROFILE_CATEGORIES)
                },
            }

        cube_rows = values[:num_cube_cores]
        vector_rows = values[
            NUM_WORKERS_CUBE : NUM_WORKERS_CUBE + num_cube_cores
        ]
        return {
            "unit": "system_cycle",
            "categories": list(_CYCLE_PROFILE_CATEGORIES),
            "cube": _summarize(cube_rows),
            "vector": _summarize(vector_rows),
            "accounting": (
                "worker categories are sequential within one core, but worker "
                "maxima are diagnostic and are not additive wall time"
            ),
        }

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
        events[_EVENT_COUNTER_BYTES:].zero_()
        self.symmetric_memory.barrier()
        if forward:
            self.forward_ready_initialized = True
        else:
            self.backward_ready_initialized = True
        return events

    def wait_for_reuse(self) -> None:
        """Order count exchange after the previous workspace lease."""
        with self.lock:
            if self.used:
                torch.npu.current_stream(self.device).wait_event(self.completion_event)

    def prepare_gmm_workspace(
        self,
        weight1: Any,
        weight2: Any,
        wave_count: int,
        *,
        receive_capacity: int,
        reuse_retained_scratch: bool,
        atomic_dw_accumulation: bool,
    ) -> None:
        """Grow dW scratch while preserving capacity for outstanding backwards.

        A workspace starts with only the 32 MiB GMM workspace. Non-atomic
        multi-wave calls grow a model-shape-bounded tail for temporary dW.
        Smaller calls retain that high-watermark until the workspace is freed.

        Args:
            weight1: First expert weight tensor used to size dW1 scratch.
            weight2: Second expert weight tensor used to size dW2 scratch.
            wave_count: Number of kernel-internal communication waves.
            receive_capacity: Fixed number of receive rows in each wave.
            reuse_retained_scratch: Whether whole matrices may occupy the dead
                first retained up-projection window.
            atomic_dw_accumulation: Whether later waves atomically add into the
                persistent dW outputs and therefore need no matrix scratch.
        """

        required_bytes = _GMM_WORKSPACE_BYTES
        if wave_count > 1 and not atomic_dw_accumulation:
            if reuse_retained_scratch:
                if (
                    weight1.dim() != 3
                    or weight2.dim() != 3
                    or weight1.shape[0] != weight2.shape[0]
                    or weight1.shape[1] != weight2.shape[2]
                    or weight1.shape[2] != 2 * weight2.shape[1]
                    or weight1.element_size() != weight2.element_size()
                ):
                    raise ValueError(
                        "retained dW scratch requires matching gated-MLP weights."
                    )
                plan = _plan_weight_grad_scratch(
                    local_experts=weight1.shape[0],
                    hidden_size=weight1.shape[1],
                    intermediate_size=weight2.shape[1],
                    element_size=weight1.element_size(),
                    receive_capacity=receive_capacity,
                )
                required_bytes += plan.tail_bytes
            else:
                required_bytes += (
                    weight1.numel() * weight1.element_size()
                    + weight2.numel() * weight2.element_size()
                )
        if self.gmm_workspace is None or self.device is None:
            raise RuntimeError("MegaMoe GMM workspace is not initialized.")
        # A W1 forward may run before an earlier W2 backward consumes this
        # workspace. Its smaller requirement must not discard the live tail.
        if self.gmm_workspace.numel() >= required_bytes:
            return
        torch.npu.synchronize(self.device)
        self.gmm_workspace.resize_(0)
        self.gmm_workspace = torch.empty(
            (required_bytes,),
            dtype=torch.uint8,
            device=self.device,
        )

    def claim(self) -> None:
        """Claim the serial workspace and order it after the previous stream."""
        with self.lock:
            if self.in_use:
                raise RuntimeError(
                    "MegaMoe execution resources do not support concurrent calls."
                )
            if self.used:
                torch.npu.current_stream().wait_event(self.completion_event)
            self.in_use = True

    def mark_routed_buffer_write(self) -> int:
        """Record one enqueued launch that changes the routed SHMEM buffer."""
        with self.lock:
            if not self.in_use:
                raise RuntimeError(
                    "MegaMoe routed-buffer writes require an active workspace lease."
                )
            self.routed_buffer_generation += 1
            return self.routed_buffer_generation

    def release(self) -> None:
        """Record completion ordering and release the serial workspace lease."""
        with self.lock:
            if not self.in_use:
                raise RuntimeError(
                    "MegaMoe workspace was released without an active lease."
                )
            self.completion_event.record(torch.npu.current_stream())
            self.in_use = False
            self.used = True

    def _free_symmetric_tensors(self) -> None:
        """Free each unique SHMEM allocation and invalidate its tensor view."""
        for field_name in (
            "expert_buffer",
            "routed_buffer",
            "forward_event_counters",
            "backward_event_counters",
        ):
            tensor = getattr(self, field_name)
            if tensor is None:
                continue
            self.symmetric_memory.free(tensor)
            setattr(self, field_name, None)
        self.expert_capacity = 0
        self.routed_slots = 0
        self.forward_ready_initialized = False
        self.backward_ready_initialized = False
        self.forward_route_arrivals_enabled = False
        self.backward_route_arrivals_enabled = False
        self.route_arrival_offset_bytes = 0

    def _free_local_tensors(self) -> None:
        """Release local-only workspaces after all queued kernels complete."""
        for field_name in ("gmm_workspace", "swiglu_grad_workspace"):
            tensor = getattr(self, field_name)
            if tensor is None:
                continue
            tensor.untyped_storage().resize_(0)
            setattr(self, field_name, None)

    def close(self) -> None:
        """Synchronize, release all buffers, and reset the workspace."""
        with self.lock:
            if self.in_use:
                raise RuntimeError(
                    "cannot close MegaMoe workspace during an active call."
                )
            if self.expert_buffer is None and self.gmm_workspace is None:
                return
        torch.npu.synchronize(self.device)
        self.symmetric_memory.barrier()
        self._free_symmetric_tensors()
        self.symmetric_memory.barrier()
        self._free_local_tensors()
        self.dtype = None
        self.device = None
        self.completion_event = None
        self.used = False
