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
"""将事件 IR 绑定到固定算子族 ABI；保留数值描述符、通信协议和性能记录字段。"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from hyper_parallel.core.multicore.ir.events import EventGraphIR
from hyper_parallel.core.multicore.runtime.abi import FamilyABI, family_abi

INVALID_EVENT = 0xFFFFFFFF


def _field(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


@dataclass(frozen=True)
class EventWireView:
    """按族 schema 解码的静态事件字段与队列，数值任务内容仍归原生实现。"""

    abi: FamilyABI
    task_count: int
    task_capacity: int
    event_capacity: int
    task_offset: int
    queue_offset: int
    thresholds: tuple[int, ...]
    task_types: tuple[int, ...]
    waits: tuple[int | None, ...]
    notifies: tuple[int | None, ...]
    queues: tuple[tuple[int, ...], ...]


def _queues(data, offset, capacity, count, task_offset, task_abi):
    sizes = struct.unpack_from("<4i", data, offset)
    result = []
    for number, size in enumerate(sizes[:3]):
        if not 0 <= size <= capacity:
            raise ValueError("Native event queue length exceeds task capacity")
        queue = struct.unpack_from(f"<{size}i", data, offset + 16 + number * capacity * 4)
        for task in queue:
            if not 0 <= task < capacity:
                raise ValueError("Native event queue references an invalid descriptor")
            kind = _field(data, task_offset + task * task_abi.size + task_abi.offset("task_type"))
            if task >= count and kind != 0:
                raise ValueError("Only a termination descriptor may be outside the active task domain")
        result.append(tuple(task for task in queue if task < count))
    return tuple(result)


def _layout(abi, data):
    header, task_abi, event_abi = (abi.structure(name) for name in ("RuntimeConfigC", "TaskDescC", "EventDescC"))
    if not isinstance(data, bytes) or len(data) < header.size:
        raise ValueError("Native event image is truncated")
    count = _field(data, header.offset("task_num"))
    capacity = _field(data, header.offset("task_capacity"))
    events = _field(data, header.offset("event_capacity"))
    if count > capacity or capacity % 16 or events < 1024 or events % 16:
        raise ValueError("Invalid native event/task capacities")
    task_offset = header.size + events * 4
    queue_offset = task_offset + capacity * task_abi.size + events * event_abi.size
    if len(data) < queue_offset + 16 + capacity * 12:
        raise ValueError("Native event image arrays are truncated")
    return header, task_abi, count, capacity, events, task_offset, queue_offset


def _task_fields(data, count, task_offset, task_abi):
    return tuple(tuple(_field(data, task_offset + index * task_abi.size + task_abi.offset(name))
                       for name in ("task_type", "dependent_event", "trigger_event")) for index in range(count))


def decode_event_wire(family: str, data: bytes) -> EventWireView:
    """验证事件与队列范围，并依据固定族 schema 读取依赖字段。

    Args:
        family: gate、moe 或 mhc。
        data: 普通或性能记录运行时镜像。
    """
    abi = family_abi(family)
    header, task_abi, count, capacity, events, task_offset, queue_offset = _layout(abi, data)
    thresholds = struct.unpack_from(f"<{events}i", data, header.size)
    if any(value < 0 for value in thresholds):
        raise ValueError("Native event wait counts cannot be negative")
    fields = _task_fields(data, count, task_offset, task_abi)
    waits = tuple(None if wait == INVALID_EVENT else wait for _, wait, _ in fields)
    notifies = tuple(None if notify == INVALID_EVENT else notify for _, _, notify in fields)
    if any(event is not None and event >= events for event in (*waits, *notifies)):
        raise ValueError("Native event identity exceeds allocated capacity")
    queues = _queues(data, queue_offset, capacity, count, task_offset, task_abi)
    return EventWireView(abi, count, capacity, events, task_offset, queue_offset, thresholds,
                         tuple(kind for kind, _, _ in fields), waits, notifies, queues)


def _bindings(graph):
    result = {}
    for grid in graph.grids:
        for native, wait, notify in zip(grid.native_tasks, grid.waits.values(), grid.notifies.values()):
            pair = (wait, notify)
            if native in result and result[native] != pair:
                raise ValueError("Broadcast descriptors cannot encode different per-worker event mappings")
            result[native] = pair
    return result


def _native_queues(graph):
    return {(queue.resource, queue.worker): tuple(graph.grids[number].native_tasks[index]
                                                 for number, index in queue.tasks) for queue in graph.queues}


def _check_placement(graph, view, template):
    actual = _native_queues(graph)
    counts = dict(graph.worker_counts)
    if graph.mode == "worker_pipeline":
        expected = {("vector", worker): view.queues[1] for worker in range(counts.get("vector", 0))}
    else:
        workers = _field(template, view.abi.structure("RuntimeConfigC").offset("num_workers"))
        if counts != {"cube": workers // 2, "vector": workers, "mix": workers // 2}:
            raise ValueError("Event IR topology differs from native fixed-stride workers")
        expected = {(resource, worker): queue[worker::counts[resource]]
                    for resource, queue in zip(("cube", "vector", "mix"), view.queues)
                    for worker in range(counts[resource]) if queue[worker::counts[resource]]}
    if actual != expected:
        raise ValueError("Event lowering requires the original native task placement and FIFO order")


def lower_event_fields(graph: EventGraphIR, template: bytes) -> bytes:
    """由 IR 写入依赖、通知和阈值，拒绝当前 ABI 无法表示的事件映射。

    Args:
        graph: 已校验的静态事件图；本阶段固定原始任务放置和协议。
        template: 保存数值任务、队列、通信与性能记录内容的原生镜像。
    """
    view = decode_event_wire(graph.family, template)
    _check_placement(graph, view, template)
    bindings = _bindings(graph)
    if graph.mode == "worker_pipeline" and graph.event_tensor is not None:
        raise ValueError("Gate worker pipeline cannot lower global event tensors")
    if set(bindings) != set(range(view.task_count)):
        raise ValueError("Event IR must bind every native task descriptor")
    if graph.event_tensor is None:
        if any(view.thresholds):
            raise ValueError("An event-free pipeline cannot lower nonzero native counters")
        thresholds = view.thresholds
    else:
        thresholds = graph.event_tensor.wait_counts
        if len(thresholds) != view.event_capacity:
            raise ValueError("Event IR tensor differs from native event capacity")
    data = bytearray(template)
    struct.pack_into(f"<{len(thresholds)}i", data, view.abi.structure("RuntimeConfigC").size, *thresholds)
    task_abi = view.abi.structure("TaskDescC")
    for native, pair in bindings.items():
        for name, event in zip(("dependent_event", "trigger_event"), pair):
            offset = view.task_offset + native * task_abi.size + task_abi.offset(name)
            struct.pack_into("<I", data, offset, INVALID_EVENT if event is None else event)
    return bytes(data)


def _check_mhc_ownership(graph, view, template):
    counts = dict(graph.worker_counts)
    workers = _field(template, view.abi.structure("RuntimeConfigC").offset("num_workers"))
    if counts != {"cube": workers // 2, "vector": workers, "mix": workers // 2}:
        raise ValueError("Worker queue transformation cannot change native topology")
    actual = _native_queues(graph)
    for resource, queue in zip(("cube", "vector", "mix"), view.queues):
        for worker in range(counts[resource]):
            expected = queue[worker::counts[resource]]
            selected = actual.get((resource, worker), ())
            if resource == "vector":
                if len(selected) != len(expected) or set(selected) != set(expected):
                    raise ValueError("Worker queue transformation must preserve per-worker task ownership")
            elif selected != expected:
                raise ValueError("Worker queue transformation cannot reorder Cube or Mix tasks")
    return actual, workers


def lower_mhc_worker_queues(graph: EventGraphIR, template: bytes) -> bytes:
    """降级 MHC 反向的同 worker 队列变换；事件和数值描述符必须保持不变。

    Args:
        graph: 已通过固定 FIFO 进展检查的 MHC 反向事件图。
        template: 变换前的普通或 profiling 镜像。
    """
    if graph.family != "mhc" or graph.direction != "backward" or graph.boundaries:
        raise ValueError("Worker queue transformations require a local MHC backward graph")
    if any(grid.notification != "local" for grid in graph.grids):
        raise ValueError("MHC worker queue transformations require local event publication")
    view = decode_event_wire("mhc", template)
    actual, workers = _check_mhc_ownership(graph, view, template)
    if _bindings(graph) != dict(enumerate(zip(view.waits, view.notifies))):
        raise ValueError("Worker queue transformation cannot change native event mappings")
    if graph.event_tensor is None or graph.event_tensor.wait_counts != view.thresholds:
        raise ValueError("Worker queue transformation cannot change native event thresholds")
    graph.simulate()
    data = bytearray(template)
    offset = view.queue_offset + 16 + view.task_capacity * 4
    if _field(template, view.queue_offset + 4) != len(view.queues[1]):
        raise ValueError("Worker queue transformation cannot move inactive termination descriptors")
    for worker in range(workers):
        for index, task in enumerate(actual.get(("vector", worker), ())):
            struct.pack_into("<i", data, offset + (worker + index * workers) * 4, task)
    return lower_event_fields(graph, bytes(data))
