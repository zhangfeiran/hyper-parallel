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
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from typing import Any

_PACKAGE = "hyper_parallel.core.multicore.modules.mega_moe"
_CYCLE_PROFILE_ENV = "HP_MEGA_MOE_KERNEL_CYCLE_PROFILE"
_SPANS = {
    "route": ("_start_count_gather", "_finish_static_count_gather", "_build_replica_plan",
              "_remap_topk_ids_to_physical", "_materialize_replica_counts", "_prepare_route_endpoints"),
    "function": ("_prepare_forward_buffers", "_launch_forward", "_prepare_backward_route",
                  "_allocate_backward_intermediates", "_launch_backward", "_finish_backward"),
    "replica": ("_launch_transfers", "_wait_handles"),
    "replica.ExpertReplicaManager": ("start_shared_prefetch", "_all_ranks_can_reuse", "_prefetch_transfers",
                                     "_gradient_transfers", "reduce_replica_gradients"),
    "replica.SharedExpertWeightLease": ("wait", "prepare_backward"),
}


def _annotated(function: Any, record_function: Any, name: str) -> Any:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Preserve the wrapped call while recording its inclusive CPU scope."""
        with record_function(f"megamoe::{name}"):
            return function(*args, **kwargs)

    return wrapped


@contextmanager
def _profile_spans(record_function: Any) -> Iterator[None]:
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
                setattr(target, name, _annotated(original, record_function, f"{path}.{name}"))
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


def profile_layer_step(sweep: Any, route: tuple, output: Path, *, all_ranks: bool = False) -> None:
    """Capture one extra AdamW step after all benchmark timing has finished.

    Args:
        sweep: Initialized layer benchmark; all EP ranks must call this function.
        route: Already sampled route for the diagnostic step.
        output: New Chrome trace path, or filename base for all-rank capture.
        all_ranks: Capture every rank with distinct files; default records rank zero.

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
            on_trace_ready=profiler.tensorboard_trace_handler(
                str(output.parent), worker_name=output.stem, analyse_flag=False,
            ),
        )
    with capture:
        with _profile_spans(record) if enabled else nullcontext():
            worker.dist.barrier()
            with record("megamoe::fwd_bwd") if enabled else nullcontext():
                sweep._train_step(route)  # pylint: disable=protected-access
                worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
            with record("megamoe::adamw") if enabled else nullcontext():
                sweep.optimizer.step()
                worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
    if enabled:
        capture.export_chrome_trace(str(output))
        if not output.is_file():
            raise RuntimeError(f"NPU profiler did not export {output}")
    worker.dist.barrier()


def profile_layer_cycles(sweep: Any, route: tuple) -> dict[str, Any]:
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
    sweep._train_step(route)  # pylint: disable=protected-access
    sweep.optimizer.step()
    worker.dist.barrier()
    worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
    os.environ[_CYCLE_PROFILE_ENV] = "1"
    try:
        sweep._train_step(route)  # pylint: disable=protected-access
        worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
        counters = worker._collect_kernel_cycle_profile(sweep.layers[0])  # pylint: disable=protected-access
        forward_counters = worker._collect_kernel_cycle_profile(  # pylint: disable=protected-access
            sweep.layers[0], forward=True,
        )
        if counters is None or forward_counters is None:
            raise RuntimeError("MegaMoe forward/backward cycle counters were not collected")
        sweep.optimizer.step()
        worker.torch.npu.synchronize(worker._DEVICE)  # pylint: disable=protected-access
    finally:
        if previous is None:
            os.environ.pop(_CYCLE_PROFILE_ENV, None)
        else:
            os.environ[_CYCLE_PROFILE_ENV] = previous
    return {"layer_index": 0, "scope": "last forward recomputation and backward launch only; not all layers",
            "timing": "extra post-update step after all performance and memory measurements",
            "counters": counters, "forward_counters": forward_counters}
