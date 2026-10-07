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
"""Opt-in diagnostics for replica benchmarks; never enabled in timed steps."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from collections.abc import Iterator
from typing import Any
import sys
from functools import wraps
import importlib
import time
from unittest.mock import patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import routing
from hyper_parallel.core.expert_parallel.hot_replica.device import DeviceExpertExecutionPlan
from hyper_parallel.core.expert_parallel.hot_replica.pool import replica_pool
from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.modules.mega_moe import function


class HostMeasurements:
    """Measure inclusive host spans and stream intervals in a diagnostic invocation."""

    def __init__(self, *, device_intervals: bool = True) -> None:
        """Optionally omit device events for a lightweight host-only observation."""
        self.device_intervals = device_intervals
        self.records = []
        self.stack = []
        self.patches = ExitStack()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Record queue intervals without synchronizing each observed operation.

        Args:
            name: Stage name for the inclusive interval.
        """
        start_event, end_event = None, None
        if self.device_intervals:
            start_event, end_event = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start_event.record()
        parent = self.stack[-1] if self.stack else None
        self.stack.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            if end_event is not None:
                end_event.record()
            self.stack.pop()
            self.records.append({"stage": name, "parent": parent, "host_ms": elapsed,
                                 "events": (start_event, end_event)})

    def _wrap(self, original, name):
        @wraps(original)
        def _measured(*args, **kwargs):
            cache = kwargs.get("cache") if name == "invocation_metadata" else None
            previous = None if cache is None else cache.get(id(args[0]))
            with self.span(name):
                result = original(*args, **kwargs)
            if name == "invocation_metadata":
                self.records[-1].update(base_bytes=args[0].numel(), image_bytes=result.numel(),
                                       cache_hit=previous is not None and previous[2] is result)
            return result
        return _measured

    def __enter__(self) -> HostMeasurements:
        """Observe existing call boundaries without replacing their behavior."""
        device = importlib.import_module("hyper_parallel.core.expert_parallel.hot_replica.device")
        for module, name, stage in (
                (routing, "build_expert_replica_plan", "cpu_solver"),
                (routing, "build_device_expert_replica_plan", "device_plan"),
                (device, "launch_device_planner", "aiv_launch"),
                (DeviceExpertExecutionPlan, "host_summary", "control_summary"),
                (routing, "_remap_replica_ids", "remap"),
                (routing, "_upload_route_metadata", "upload_metadata"),
                (function, "_split_runtime", "invocation_metadata"),
                (function, "return_gradients", "gradient_return_enqueue")):
            self.patches.enter_context(patch.object(module, name, self._wrap(getattr(module, name), stage)))
        self.patches.enter_context(patch.object(torch.Tensor, "cpu", self._wrap(torch.Tensor.cpu, "device_to_host")))
        original_prefetch = function.prefetch_weights
        measurements = self

        class _Work:
            def __init__(self, work: Any) -> None:
                """Retain the actual asynchronous collective work."""
                self.work = work

            def wait(self) -> Any:
                """Time the original wait without adding a device synchronization."""
                with measurements.span("count_gather_wait_enqueue"):
                    return self.work.wait()

        def _gather_wrapper(original):
            @wraps(original)
            def _gather(*args, **kwargs):
                with self.span("count_gather_enqueue"):
                    work = original(*args, **kwargs)
                return _Work(work) if work is not None else None
            return _gather

        @contextmanager
        def _prefetch(*args, **kwargs):
            context = original_prefetch(*args, **kwargs)
            with self.span("weight_prefetch_enqueue"):
                pool = context.__enter__()
            try:
                yield pool
            finally:
                with self.span("weight_lease_release"):
                    context.__exit__(*sys.exc_info())

        for name in ("all_gather", "all_gather_into_tensor"):
            self.patches.enter_context(patch.object(routing.dist, name, _gather_wrapper(getattr(routing.dist, name))))
        self.patches.enter_context(patch.object(function, "prefetch_weights", _prefetch))
        return self

    def __exit__(self, *args: Any) -> None:
        """Restore all functions before draining the diagnostic events."""
        self.patches.close()
        if self.device_intervals:
            torch.npu.synchronize()
        for record in self.records:
            start, end = record.pop("events")
            if start is not None:
                record["stream_interval_ms"] = start.elapsed_time(end)


def _payload_bytes(tensors):
    """Count distinct pool views, excluding shared backing allocation and padding."""
    views = {(tensor.data_ptr(), tensor.numel(), tensor.element_size())
             for tensor in tensors if isinstance(tensor, torch.Tensor)}
    return sum(elements * width for _, elements, width in views)


def memory_snapshot(module: object, values: torch.Tensor, output: torch.Tensor | None = None) -> dict:
    """Separate external SHMEM reservations from overlapping allocator totals.

    Args:
        module: MegaMoe executor owning the execution resources.
        values: Input used to resolve the active workspace.
        output: Optional graph root for saved-storage accounting.

    Returns:
        Memory totals, overlapping component bytes and capacity information.
    """
    resources = module._get_execution_resources(values)
    state = shmem.debug_state()
    weights = (module.gate_up_weight, module.down_weight)
    budget = resources.spec.replica_slots_per_rank
    provider = resources.workspace.replica_provider
    pool = None if not budget else (provider.pool if provider is not None else
                                   replica_pool(weights, budget, resources.spec.ep_group))

    def _bytes(tensors):
        unique = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                  for tensor in tensors if isinstance(tensor, torch.Tensor)}
        return sum(unique.values())

    saved = []
    pending, visited = [None if output is None else output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        saved.extend(getattr(node, "saved_tensors", ()))
        pending.extend(parent for parent, _ in node.next_functions)
    parameters = {weight.untyped_storage().data_ptr() for weight in weights}
    saved = [value for value in saved if value.untyped_storage().data_ptr() not in parameters]
    heap = resources.heap_manager.heap_bytes
    return {"torch_allocated": torch.npu.memory_allocated(), "torch_reserved": torch.npu.memory_reserved(),
            "torch_peak_allocated": torch.npu.max_memory_allocated(),
            "torch_peak_reserved": torch.npu.max_memory_reserved(), "shmem_heap_reserved": heap,
            "shmem_allocated": state.get("allocated_bytes"),
            "shmem_allocations": state.get("active_allocations"),
            "guest_weights": 0 if pool is None else _payload_bytes(pool.weights),
            "guest_gradients": 0 if pool is None else _payload_bytes(pool.gradients or ()),
            "guest_pool_backing_storage": 0 if pool is None else _bytes(pool.weights + (pool.gradients or ())),
            "gradient_scratch": getattr(provider, "gradient_scratch_bytes", 0),
            "replica_runtime_image_bytes": _bytes(tuple(entry[2] for entry in
                getattr(resources.workspace, "replica_runtime_images", {}).values())),
            "saved_nonparameter_storages": _bytes(saved),
            "home_dw_fp32_expected": sum(weight.numel() for weight in weights) * 4 if budget else 0,
            "capacity": resources.workspace.capacity_floor, "maximum_capacity": resources.spec.maximum_receive_capacity,
            "heap_epoch": resources.heap_manager.epoch,
            "accounting": "Components overlap Torch/SHMEM totals. Saved storage is an observed lower bound. "
                          "Add the constant SHMEM heap to Torch peak only within a no-growth steady window."}
