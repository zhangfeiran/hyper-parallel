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
"""Opt-in, untimed traces and all-rank kernel cycles for the layer benchmark."""

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from typing import Any

from examples.torch.multicore.mega_moe_p2p_profile import (
    _P2PObserver,
    _validate_payload_balance,
)
from examples.torch.multicore.mega_moe_profile_context import ProfileContext

_PACKAGE = "hyper_parallel.core.multicore.modules.mega_moe"
_CYCLE_PROFILE_ENV = "HP_MEGA_MOE_KERNEL_CYCLE_PROFILE"
_SPANS = {
    "route": (
        "_start_count_gather",
        "_finish_static_count_gather",
        "_build_replica_plan",
        "_remap_topk_ids_to_physical",
        "_physical_copy_runs",
        "_materialize_replica_counts",
        "_compute_route_metadata",
        "_prepare_route_endpoints",
    ),
    "function": (
        "_prepare_forward_buffers",
        "_launch_forward",
        "_prepare_backward_route",
        "_allocate_backward_intermediates",
        "_launch_backward",
        "_finish_backward",
    ),
    "replica": ("_launch_transfers", "_wait_handles"),
    "replica.ExpertReplicaManager": (
        "start_shared_prefetch",
        "_all_ranks_can_reuse",
        "_prefetch_transfers",
        "_gradient_transfers",
        "_complete_gradient_transfers",
        "_reduce_gradient_group",
        "reduce_replica_gradients",
    ),
    "replica.SharedExpertWeightLease": ("wait", "prepare_backward"),
}


def _annotated(function: Any, record_function: Any, name: str, context: ProfileContext | None = None) -> Any:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Preserve the wrapped call while recording its inclusive CPU scope."""
        with record_function(f"megamoe::{name}"), \
                context.span("scope", scope=name) if context is not None else nullcontext():
            return function(*args, **kwargs)

    return wrapped


@contextmanager
def _profile_spans(record_function: Any, context: ProfileContext | None = None) -> Iterator[None]:
    """Temporarily annotate Python launch scopes without device synchronization."""
    # Patch only the single-threaded example worker, not library hot paths. An
    # ExitStack also restores earlier patches if a later target cannot resolve.
    with ExitStack() as stack:
        for path, names in _SPANS.items():
            module_name, _, class_name = path.partition(".")
            target = importlib.import_module(f"{_PACKAGE}.{module_name}")
            if class_name:
                target = getattr(target, class_name)
            for name in names:
                original = getattr(target, name)
                stack.callback(setattr, target, name, original)
                setattr(target, name, _annotated(original, record_function, f"{path}.{name}", context))
        yield


def profile_trace_paths(output: Path, world_size: int, *, all_ranks: bool = False) -> tuple[Path, ...]:
    """Resolve trace destinations without creating files or importing a profiler.

    Args:
        output: Base JSON path; retained verbatim for default rank-zero capture.
        world_size: Number of ranks in the benchmark group.
        all_ranks: Give every rank a distinct ``.rankN.json`` filename.

    Returns:
        Paths ordered by rank, or only rank zero's path for default capture.
    """
    if world_size <= 0:
        raise ValueError("trace world size must be positive")
    if output.suffix != ".json":
        raise ValueError("trace output must have a .json suffix")
    if not all_ranks:
        return (output,)
    return tuple(output.with_name(f"{output.stem}.rank{rank}.json") for rank in range(world_size))


def _memory_records(raw_path: Path) -> list[dict[str, Any]]:
    """Decode memory already collected by Torch-NPU using its version-matched reader."""
    package = "torch_npu.profiler.analysis"
    config = importlib.import_module(f"{package}._profiler_config").ProfilerConfig()
    config.load_info(str(raw_path))
    parser = importlib.import_module(f"{package}.prof_parse._fwk_file_parser").FwkFileParser(str(raw_path))
    tag = importlib.import_module(f"{package}.prof_common_func._file_tag").FileTag.MEMORY
    return [{"timestamp_ns": item.time_ns, "pid": item.pid, "tid": item.tid,
             "args": {"Bytes": item.alloc_size_for_db, "Total Allocated": item.total_allocated_for_db,
                      "Total Reserved": item.total_reserved_for_db, "Address": item.ptr,
                      "Device Type": item.device_type, "Device Id": item.device_index,
                      "Allocator Type": item.allocator_type, "Data Type": item.data_type}}
            for item in parser.get_file_data_by_tag(tag)]


def _export_trace(capture: Any, output: Path) -> None:
    """Reject incomplete files when the vendor exporter logs an error without raising."""
    capture.export_chrome_trace(str(output))
    if not output.is_file():
        raise RuntimeError(f"NPU profiler did not export {output}")
    try:
        document = json.loads(output.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"Incomplete NPU trace export: {output}; inspect CANN exporter logs") from exc
    if not isinstance(document, (list, dict)):
        raise TypeError(f"Invalid NPU trace export: {output}")


def profile_layer_step(
    sweep: Any, route: tuple, output: Path, *, all_ranks: bool = False, context: ProfileContext | None = None,
) -> dict[str, Any]:
    """Capture one extra AdamW step after all benchmark timing has finished.

    Args:
        sweep: Initialized layer benchmark; all EP ranks must call this function.
        route: Already sampled route for the diagnostic step.
        output: New Chrome trace path, or filename base for all-rank capture.
        all_ranks: Capture every rank with distinct files; default records rank zero.
        context: Correlate payloads and CPU scopes in this exact execution when provided.

    Note:
        CPU annotations are inclusive launch scopes, not device durations. The
        NPU timeline must be used to attribute asynchronous compute and HCCL.
        Profiling perturbs execution and is excluded from performance results.
    """
    worker = sweep.worker
    enabled = all_ranks or worker._RANK == 0  # pylint: disable=protected-access
    paths = profile_trace_paths(output, worker.dist.get_world_size() if all_ranks else 1, all_ranks=all_ranks)
    # All workers share the example's output filesystem. Preflight every rank
    # before any collective so one existing artifact cannot strand its peers.
    for path in (output, *paths):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    output = paths[worker._RANK] if all_ranks else paths[0]  # pylint: disable=protected-access
    if enabled:
        output.parent.mkdir(parents=True, exist_ok=True)
    # Return to the selected route and finish a real update before capture;
    # post-measurement finite validation otherwise leaves a warm version hit.
    sweep._train_step(route)  # pylint: disable=protected-access
    sweep.optimizer.step()
    worker.dist.barrier()
    worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
    record = worker.torch.autograd.profiler.record_function
    capture = nullcontext()
    if enabled:
        # Selected ranks own separate profilers and raw-artifact prefixes;
        # inactive ranks must not create artifacts through profiler destructors.
        profiler = importlib.import_module("torch_npu.profiler")
        capture = profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=profiler.tensorboard_trace_handler(
                str(output.parent),
                worker_name=output.stem,
                analyse_flag=False,
            ),
        )
    observer = None
    if context is not None:
        context.record_function = record if enabled else None
        observer = _P2PObserver(worker.torch.npu, worker._DEVICE, context=context, device_timing=False)
        replica = importlib.import_module(f"{_PACKAGE}.replica")
    with capture, ExitStack() as stack:
        if context is not None:
            stack.enter_context(context.observe(worker.torch, sweep.layers, checkpoint="reentrant"))
            stack.enter_context(observer.observe(replica))
        if enabled:
            stack.enter_context(_profile_spans(record, context))
        worker.dist.barrier()
        with record("megamoe::fwd_bwd") if enabled else nullcontext():
            sweep._train_step(route)  # pylint: disable=protected-access
            worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
        with record("megamoe::adamw") if enabled else nullcontext():
            sweep.optimizer.step()
            worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
    memory, error = [], None
    if enabled:
        try:
            _export_trace(capture, output)
            if context is not None:
                memory = _memory_records(Path(capture.prof_if.prof_path))
        except (RuntimeError, TypeError, ValueError, OSError) as exc:
            if context is None:
                raise
            error = str(exc)
    if context is not None:
        errors = [None] * worker.dist.get_world_size()
        worker.dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(f"Diagnostic trace export failed: {errors}")
    worker.dist.barrier()
    if observer is None:
        return {}
    local = {"rank": worker._RANK, "context": context.manifest(), **observer.collect(),
             "memory_events": memory, "memory_clock_domain": "host_aligned_epoch_ns",
             "memory_source": "existing Torch-NPU framework memory records"}
    ranks = [None] * worker.dist.get_world_size()
    worker.dist.all_gather_object(ranks, local)
    return {"schema_version": 2, "capture_id": context.base["capture_id"], "per_rank": ranks,
            "payload_balance": _validate_payload_balance(ranks)}


def profile_layer_cycles(
    sweep: Any, route: tuple, *, context: ProfileContext | None = None, tasks: bool = False,
) -> dict[str, Any]:
    """Collect all ranks' last forward/backward counters after an update.

    Args:
        sweep: Initialized MegaMoe benchmark; every EP rank must participate.
        route: Already sampled route for the diagnostic step.

    Returns:
        Counters for layer zero's final checkpoint recomputation and backward,
        not the entire training step. Each direction has independent storage,
        overwritten on every instrumented launch in that direction.

    Note:
        Call only after performance and memory measurements. Instrumentation
        changes kernel execution; neither counters nor worker maxima are wall
        timings.
    """
    # Reuse the ST collector so rank aggregation and native counter decoding
    # have one contract; this coupling is confined to diagnostic example code.
    worker = sweep.worker
    previous = os.environ.get(_CYCLE_PROFILE_ENV)
    previous_tasks = os.environ.get("HP_MEGA_MOE_TASK_PROFILE")
    sweep._train_step(route)  # pylint: disable=protected-access
    sweep.optimizer.step()
    worker.dist.barrier()
    worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
    os.environ[_CYCLE_PROFILE_ENV] = "1"
    if tasks:
        os.environ["HP_MEGA_MOE_TASK_PROFILE"] = "1"
    correlated = []
    try:
        with context.observe(worker.torch, sweep.layers, checkpoint="reentrant") if context else nullcontext():
            sweep._train_step(route)  # pylint: disable=protected-access
        worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
        counters = worker._collect_kernel_cycle_profile(sweep.layers[0])  # pylint: disable=protected-access
        forward_counters = worker._collect_kernel_cycle_profile(  # pylint: disable=protected-access
            sweep.layers[0],
            forward=True,
        )
        if counters is None or forward_counters is None:
            raise RuntimeError("MegaMoe forward/backward cycle counters were not collected")
        detailed = []
        if os.getenv("HP_MEGA_MOE_TASK_PROFILE", "0") == "1":
            resources = sweep.layers[0]._resource_group.resources  # pylint: disable=protected-access
            local = {
                "rank": worker._RANK,  # pylint: disable=protected-access
                "forward": resources.workspace.collect_task_profile(resources.spec.num_cube_cores, forward=True),
                "backward": resources.workspace.collect_task_profile(resources.spec.num_cube_cores),
            }
            detailed = [None] * worker.dist.get_world_size()
            worker.dist.all_gather_object(detailed, local)
        if context is not None:
            groups = []
            seen = set()
            for entry, layer in zip(context.layer_registry, sweep.layers):
                group = entry["resource_group"]
                if group in seen:
                    continue
                seen.add(group)
                resources = layer._resource_group.resources  # pylint: disable=protected-access
                item = {"resource_group": group, "cycles": {}, "tasks": {}}
                for direction in ("forward", "backward"):
                    metadata = context.last_launch[(group, direction)]
                    item["cycles"][direction] = {
                        "context": metadata, "coverage": "last launch only",
                        "profile": resources.workspace.collect_cycle_profile(
                            resources.spec.num_cube_cores, forward=direction == "forward",
                        ),
                    }
                    if os.getenv("HP_MEGA_MOE_TASK_PROFILE", "0") == "1":
                        item["tasks"][direction] = {
                            "context": metadata, "coverage": "last launch per resource group and direction",
                            "profile": resources.workspace.collect_task_profile(
                                resources.spec.num_cube_cores, forward=direction == "forward",
                            ),
                        }
                groups.append(item)
            local = {"rank": worker._RANK, "context": context.manifest(), "resource_groups": groups}
            correlated = [None] * worker.dist.get_world_size()
            worker.dist.all_gather_object(correlated, local)
        sweep.optimizer.step()
        worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
    finally:
        if previous is None:
            os.environ.pop(_CYCLE_PROFILE_ENV, None)
        else:
            os.environ[_CYCLE_PROFILE_ENV] = previous
        if previous_tasks is None:
            os.environ.pop("HP_MEGA_MOE_TASK_PROFILE", None)
        else:
            os.environ["HP_MEGA_MOE_TASK_PROFILE"] = previous_tasks
    return {
        "layer_index": 0,
        "scope": "last forward recomputation and backward launch only; not all layers",
        "timing": "extra post-update step after all performance and memory measurements",
        "counters": counters,
        "forward_counters": forward_counters,
        "task_profiles": detailed,
        "captures": correlated,
    }


def profile_scoped_events(
    torch_module: Any, call: Any, *, context: ProfileContext | None = None,
) -> list[dict[str, Any]]:
    """Capture CPU launch scopes and current-stream intervals in an extra step.

    Scopes are inclusive and may nest or overlap across streams. HCCL launch
    wrappers are CPU-only; the enclosing gradient reduction's current-stream
    interval includes its real Work.wait dependencies. This diagnostic is never
    used as the profile-off throughput measurement.
    """
    records = []
    scope_groups = {
        "route": (
            "_build_replica_plan",
            "_physical_copy_runs",
            "_remap_topk_ids_to_physical",
            "_compute_route_metadata",
            "_prepare_route_endpoints",
        ),
        "function": (
            "_prepare_forward_buffers",
            "_launch_forward",
            "_prepare_backward_route",
            "_allocate_backward_intermediates",
            "_launch_backward",
            "_finish_backward",
        ),
        "replica.ExpertReplicaManager": ("reduce_replica_gradients",),
    }

    def _wrap(original: Any, name: str, device: bool) -> Any:
        @wraps(original)
        def _measured(*args: Any, **kwargs: Any) -> Any:
            events = (
                (torch_module.npu.Event(enable_timing=True), torch_module.npu.Event(enable_timing=True))
                if device
                else ()
            )
            stream = torch_module.npu.current_stream() if device else None
            if events:
                events[0].record(stream)
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                cpu_us = (time.perf_counter_ns() - start) / 1000
                if events:
                    events[1].record(stream)
                records.append({"scope": name, "cpu_us": cpu_us, "events": events,
                                "context": context.metadata() if context is not None else None})

        return _measured

    with ExitStack() as stack:
        for path, names in scope_groups.items():
            module_name, _, class_name = path.partition(".")
            target = importlib.import_module(f"{_PACKAGE}.{module_name}")
            if class_name:
                target = getattr(target, class_name)
            for name in names:
                original = getattr(target, name)
                stack.callback(setattr, target, name, original)
                setattr(
                    target,
                    name,
                    _wrap(original, f"{path}.{name}", name not in {"_build_replica_plan", "_physical_copy_runs"}),
                )
        call()
    torch_module.npu.synchronize()
    return [
        {
            "scope": item["scope"],
            "cpu_us": item["cpu_us"],
            "stream_us": item["events"][0].elapsed_time(item["events"][1]) * 1000 if item["events"] else None,
            "context": item["context"],
            "clock_domain": "stream_elapsed_us_and_cpu_perf_counter_elapsed_us",
        }
        for item in records
    ]
