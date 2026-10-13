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
"""把既有 MoE/MHC/Gate 调度转换为共同的静态事件 IR，并验证兼容性 lowering。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from hyper_parallel.core.multicore.backends.dynamic_events import bind_moe_dynamic_events, moe_count_source
from hyper_parallel.core.multicore.backends.events import decode_event_wire, lower_event_fields
from hyper_parallel.core.multicore.ir.events import (
    EventBoundaryIR,
    EventGraphIR,
    EventIndexMap,
    EventTensorIR,
    TaskGridIR,
    WorkerQueueIR,
)


def _grid(name, tasks, waits, notifies, notification="local"):
    shape = (len(tasks),)
    return TaskGridIR(name, shape, tuple(tasks), EventIndexMap.from_values(shape, tuple(waits)),
                      EventIndexMap.from_values(shape, tuple(notifies)), notification)


def _pipeline(plan, direction, image, view):
    partitions = plan.schedule.partitions
    if direction == "backward":
        partitions = tuple(part for part in partitions if part.row_count)
    grids = tuple(_grid(stage.logical_name, (index,) * len(partitions),
                        (None,) * len(partitions), (None,) * len(partitions))
                  for index, stage in enumerate(image.stages))
    queues = tuple(WorkerQueueIR("vector", part.worker_id,
                                 tuple((index, worker) for index in range(len(grids))))
                   for worker, part in enumerate(partitions))
    boundaries = ((EventBoundaryIR("external_backward", calls=plan.external_backward_calls),)
                  if direction == "backward" else ())
    if any(event is not None for event in (*view.waits, *view.notifies)):
        raise ValueError("Gate worker pipelines cannot contain global native event dependencies")
    return EventGraphIR("gate", direction, "worker_pipeline", None, grids, queues, boundaries,
                        (("vector", len(partitions)),))


def _dag_grids(image, view, family):
    grids = []
    transport = _transport_type(view) if family == "moe" else None
    for stage in image.stages:
        name, first, count = stage[:3]
        tasks = tuple(range(first, first + count))
        kinds = {view.task_types[task] for task in tasks}
        scope = "native_transport" if transport in kinds else "local"
        if scope == "native_transport" and kinds != {transport}:
            raise ValueError("Native transport and local producers require separate task grids")
        grids.append(_grid(name, tasks, view.waits[first:first + count],
                           view.notifies[first:first + count], scope))
    return tuple(grids) + _termination_grids(grids, view)


def _termination_grids(grids, view):
    covered = {task for grid in grids for task in grid.native_tasks}
    terminal = tuple(task for task in range(view.task_count) if task not in covered)
    if any(view.task_types[task] != 0 for task in terminal):
        raise ValueError("Native event stages omit a non-termination task")
    if terminal:
        return (_grid("native_termination", terminal, tuple(view.waits[task] for task in terminal),
                      tuple(view.notifies[task] for task in terminal)),)
    return ()


def _dag_queues(image, grids, view):
    references = {native: (number, index) for number, grid in enumerate(grids)
                  for index, native in enumerate(grid.native_tasks)}
    workers = getattr(image, "worker_counts", None) or image.workers
    result = []
    for resource, queue, count in zip(("cube", "vector", "mix"), view.queues, workers):
        if type(count) not in (int,) or count <= 0:
            raise ValueError("Static event queue requires a positive worker count")
        result.extend(WorkerQueueIR(resource, worker, tuple(references[task] for task in queue[worker::count]))
                      for worker in range(count) if queue[worker::count])
    return tuple(result)


def _transport_type(view):
    return next(binding.numeric_id for binding in view.abi.task_types
                if binding.native_name == "TASK_SHMEM_PUT_MEM_SIGNAL")


def _moe_boundaries(plan, view, image):
    transport = _transport_type(view)
    external = {event for kind, event in zip(view.task_types, view.notifies)
                if kind == transport and event is not None}
    boundaries = [EventBoundaryIR(f"native_{plan.spec.dispatch_mode}_transport", tuple(sorted(external)))]
    header = view.abi.structure("RuntimeConfigC")
    # 入场/退场握手由原生 worker 发布，不能伪装成图内任务的通知。
    for name in ("ready_event", "completion_event"):
        offset = header.offset(name)
        value = int.from_bytes(image.normal[offset:offset + 4], "little")
        if value:
            boundaries.append(EventBoundaryIR(name, (value,)))
    if plan.spec.replica_slots_per_rank:
        boundaries.append(EventBoundaryIR(f"replica_protocol_{plan.spec.replica_transport}"))
    return tuple(boundaries)


def build_event_graphs(plan: Any) -> tuple[EventGraphIR, ...]:
    """生成三类计划的前向、反向和可选副本回退事件图。

    Args:
        plan: 已有 MoE、MHC 或 Gate 兼容性计划；dense 属于独立实现范围。
    """
    family = plan.export_manifest()["family"]
    if family not in ("gate", "mhc", "moe"):
        raise ValueError("Static event adapters support MoE, MHC and Gate")
    graphs = []
    for direction in ("forward", "backward", "backward_no_replica"):
        image = getattr(plan, direction, None)
        if image is None:
            continue
        view = decode_event_wire(family, image.normal)
        if family == "gate":
            graph = _pipeline(plan, direction, image, view)
        else:
            grids = _dag_grids(image, view, family)
            tensor = EventTensorIR("completion", (view.event_capacity,), view.thresholds)
            boundaries = _moe_boundaries(plan, view, image) if family == "moe" else ()
            workers = getattr(image, "worker_counts", None) or image.workers
            graph = EventGraphIR(family, direction, "fixed_stride", tensor, grids,
                                 _dag_queues(image, grids, view), boundaries,
                                 tuple(zip(("cube", "vector", "mix"), workers)))
            if family == "moe":
                graph = bind_moe_dynamic_events(graph, image.normal, view, moe_count_source(plan.spec, direction))
        for wire in (image.normal, image.profiled):
            if lower_event_fields(graph, wire) != wire:
                raise ValueError(f"{family} {direction} event lowering changed the original native contract")
        if family != "moe":
            graph.simulate()
        graphs.append(graph)
    return tuple(graphs)


def attach_event_graphs(plan: Any) -> Any:
    """在编译阶段保存事件 IR，执行调用无需再次解析事件图。

    Args:
        plan: MoE、MHC 或 Gate 的完整主机计划。
    """
    return replace(plan, event_graphs=build_event_graphs(plan))
