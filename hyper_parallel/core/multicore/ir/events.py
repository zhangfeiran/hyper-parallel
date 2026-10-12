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
"""任务域、事件张量与分段仿射映射；不包含设备地址或任意可执行索引函数。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import prod


def domain_size(shape: tuple[int, ...]) -> int:
    """校验静态任务域并返回元素数；零维长度表示空域。

    Args:
        shape: 不可变、非负整数维度；本阶段尚不接受运行时符号。
    """
    if not isinstance(shape, tuple) or not shape or any(type(size) not in (int,) or size < 0 for size in shape):
        raise ValueError("Event/task domains require a nonempty tuple of nonnegative static integers")
    return prod(shape)


@dataclass(frozen=True)
class IndexRun:
    """将线性任务区间映射为 base + floor((index-begin)/repeat)*step。"""

    begin: int
    end: int
    base: int | None
    step: int = 0
    repeat: int = 1

    def __post_init__(self) -> None:
        sizes = (self.begin, self.end, self.step, self.repeat)
        if any(type(value) not in (int,) for value in sizes) or not 0 <= self.begin < self.end or self.repeat <= 0:
            raise ValueError("Invalid affine event run interval, step or repeat")
        if self.base is not None and (type(self.base) not in (int,) or self.base < 0):
            raise ValueError("Event run base must be a nonnegative integer or None")
        if self.base is None and self.step:
            raise ValueError("An absent event cannot have an affine step")

    def at(self, index: int) -> int | None:
        """读取该区间内的映射值。

        Args:
            index: 任务域的线性坐标。
        """
        if type(index) not in (int,) or not self.begin <= index < self.end:
            raise ValueError("Task index is outside its affine event run")
        return None if self.base is None else self.base + (index - self.begin) // self.repeat * self.step


@dataclass(frozen=True)
class EventIndexMap:
    """覆盖整个任务域的受限分段仿射事件映射。"""

    shape: tuple[int, ...]
    runs: tuple[IndexRun, ...]

    def __post_init__(self) -> None:
        count, end = domain_size(self.shape), 0
        if not isinstance(self.runs, tuple):
            raise ValueError("Event map runs must be immutable")
        for run in self.runs:
            if not isinstance(run, IndexRun) or run.begin != end:
                raise ValueError("Event map runs must cover the domain without gaps or overlap")
            end = run.end
        if end != count:
            raise ValueError("Event map runs must cover the complete task domain")

    @classmethod
    def from_values(cls, shape: tuple[int, ...], values: tuple[int | None, ...]) -> EventIndexMap:
        """将已有事件坐标压缩为常量或线性连续区间。

        Args:
            shape: 静态任务域。
            values: 每个任务通知或等待的事件；None 表示无事件。
        """
        if not isinstance(values, tuple) or len(values) != domain_size(shape):
            raise ValueError("Event map values differ from the task domain")
        runs, begin = [], 0
        while begin < len(values):
            base, end = values[begin], begin + 1
            step = (values[end] - base if end < len(values) and base is not None
                    and values[end] is not None else 0)
            while end < len(values) and values[end] == (None if base is None else base + (end - begin) * step):
                end += 1
            runs.append(IndexRun(begin, end, base, step))
            begin = end
        return cls(shape, tuple(runs))

    def values(self) -> tuple[int | None, ...]:
        """实例化受限映射，供编译校验和兼容性绑定使用。"""
        return tuple(run.at(index) for run in self.runs for index in range(run.begin, run.end))

    def at(self, coordinate: tuple[int, ...]) -> int | None:
        """将任务域的多维坐标映射到事件张量的线性坐标。

        Args:
            coordinate: 与任务域维度对应的静态整数坐标。
        """
        if not isinstance(coordinate, tuple) or len(coordinate) != len(self.shape):
            raise ValueError("Task coordinate rank differs from its domain")
        index = 0
        for position, size in zip(coordinate, self.shape):
            if type(position) not in (int,) or not 0 <= position < size:
                raise ValueError("Task coordinate exceeds its domain")
            index = index * size + position
        return next(run.at(index) for run in self.runs if run.begin <= index < run.end)


@dataclass(frozen=True)
class EventTensorIR:
    """从零递增至阈值的事件张量；保留当前 Ascend 发布契约。"""

    name: str
    shape: tuple[int, ...]
    wait_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Event tensor requires a name")
        if not isinstance(self.wait_counts, tuple) or len(self.wait_counts) != domain_size(self.shape):
            raise ValueError("Event wait counts differ from the tensor domain")
        if any(type(count) not in (int,) or not 0 <= count <= 0x7FFFFFFF for count in self.wait_counts):
            raise ValueError("Event wait counts must be nonnegative int32 values")


@dataclass(frozen=True)
class TaskGridIR:
    """一个分块任务域及其生产者、消费者映射；原生描述符允许广播复用。"""

    name: str
    shape: tuple[int, ...]
    native_tasks: tuple[int, ...]
    waits: EventIndexMap
    notifies: EventIndexMap
    notification: str = "local"

    def __post_init__(self) -> None:
        count = domain_size(self.shape)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Task grid requires a name")
        if not isinstance(self.native_tasks, tuple) or len(self.native_tasks) != count:
            raise ValueError("Native task mapping differs from the task domain")
        if any(type(task) not in (int,) or task < 0 for task in self.native_tasks):
            raise ValueError("Native task identities must be nonnegative integers")
        if self.waits.shape != self.shape or self.notifies.shape != self.shape:
            raise ValueError("Event maps must share their task grid domain")
        if self.notification not in ("local", "native_transport"):
            raise ValueError("Unknown task event publication scope")


@dataclass(frozen=True)
class WorkerQueueIR:
    """固定执行资源的任务 FIFO；引用由 grid 编号和域内线性坐标组成。"""

    resource: str
    worker: int
    tasks: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.resource not in ("cube", "vector", "mix") or type(self.worker) not in (int,) or self.worker < 0:
            raise ValueError("Invalid worker queue resource or identity")
        if not isinstance(self.tasks, tuple):
            raise ValueError("Worker task references must be immutable")


@dataclass(frozen=True)
class EventBoundaryIR:
    """兼容执行域之外的发布或调用；仿真不能自行假定它们已经完成。"""

    name: str
    events: tuple[int, ...] = ()
    calls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("External event boundary requires a name")
        if not isinstance(self.events, tuple) or any(type(event) not in (int,) or event < 0 for event in self.events):
            raise ValueError("External event identities must be immutable nonnegative integers")
        if not isinstance(self.calls, tuple) or any(not isinstance(call, str) or not call for call in self.calls):
            raise ValueError("External calls must be immutable names")


@dataclass(frozen=True)
class EventGraphIR:
    """统一静态依赖 IR；不声明设备指令、通信或动态队列已经通过验证。"""

    family: str
    direction: str
    mode: str
    event_tensor: EventTensorIR | None
    grids: tuple[TaskGridIR, ...]
    queues: tuple[WorkerQueueIR, ...]
    boundaries: tuple[EventBoundaryIR, ...] = ()
    worker_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.family not in ("gate", "mhc", "moe") or self.direction not in (
                "forward", "backward", "backward_no_replica"):
            raise ValueError("Unknown event graph family or direction")
        if self.mode not in ("worker_pipeline", "fixed_stride"):
            raise ValueError("Only static event graph scheduling is admitted")
        if (self.family == "gate") != (self.mode == "worker_pipeline"):
            raise ValueError("Event graph mode differs from its admitted family execution")
        if any(not isinstance(items, tuple) for items in (self.grids, self.queues, self.boundaries)):
            raise ValueError("Event graph collections must be immutable")
        if len({grid.name for grid in self.grids}) != len(self.grids):
            raise ValueError("Task grid names must be unique")
        self._check_maps()
        self._check_queues()
        self._check_counts()
        self._check_workers()

    def _check_workers(self) -> None:
        if not isinstance(self.worker_counts, tuple):
            raise ValueError("Execution topology must be immutable")
        counts = dict(self.worker_counts)
        if len(counts) != len(self.worker_counts) or any(resource not in ("cube", "vector", "mix")
                                                      or type(count) not in (int,) or count <= 0
                                                      for resource, count in self.worker_counts):
            raise ValueError("Invalid event graph execution topology")
        if counts and any(queue.worker >= counts.get(queue.resource, 0) for queue in self.queues):
            raise ValueError("Worker queue owner exceeds the execution topology")

    def _task_events(self) -> dict[tuple[int, int], tuple[int | None, int | None, str]]:
        return {(number, index): (wait, notify, grid.notification)
                for number, grid in enumerate(self.grids)
                for index, (wait, notify) in enumerate(zip(grid.waits.values(), grid.notifies.values()))}

    def _external_events(self) -> set[int]:
        return {event for boundary in self.boundaries for event in boundary.events}

    def _check_maps(self) -> None:
        capacity = len(self.event_tensor.wait_counts) if self.event_tensor is not None else 0
        referenced = [event for grid in self.grids for mapping in (grid.waits, grid.notifies)
                      for event in mapping.values() if event is not None]
        referenced.extend(self._external_events())
        if any(event < 0 or event >= capacity for event in referenced):
            raise ValueError("Task/boundary event mapping exceeds the event tensor domain")

    def _check_queues(self) -> None:
        refs = [ref for queue in self.queues for ref in queue.tasks]
        if any(not isinstance(ref, tuple) or len(ref) != 2
               or any(type(index) not in (int,) for index in ref) for ref in refs):
            raise ValueError("Worker references require immutable grid/index pairs")
        if Counter(refs) != Counter(self._task_events().keys()):
            raise ValueError("Worker queues must execute each logical task exactly once")
        owners = [(queue.resource, queue.worker) for queue in self.queues]
        if len(set(owners)) != len(owners):
            raise ValueError("Worker queues must have unique execution owners")

    def _check_counts(self) -> None:
        if self.event_tensor is None:
            return
        entries = self._task_events().values()
        supplied = Counter(notify for _, notify, scope in entries if notify is not None and scope == "local")
        external = self._external_events()
        for wait, _, _ in self._task_events().values():
            if wait is not None and wait not in external and supplied[wait] < self.event_tensor.wait_counts[wait]:
                raise ValueError(f"Event {wait} has fewer local producers than its wait count")

    def simulate(self, *, external_counts: Mapping[int, int] | None = None) -> dict[str, object]:
        """验证固定 FIFO 进展；外部计数只能由调用方显式提供。

        Args:
            external_counts: 对原生通信或握手计数的明确假设，不构成通信完成证据。
        """
        thresholds = self.event_tensor.wait_counts if self.event_tensor is not None else ()
        counters = [0] * len(thresholds)
        supplied = dict(external_counts or {})
        if set(supplied) - self._external_events():
            raise ValueError("Simulation cannot inject a local event counter")
        if any(type(event) not in (int,) for event in supplied):
            raise ValueError("External event identities must be integers")
        if any(type(value) not in (int,) or not 0 <= value <= 0x7FFFFFFF for value in supplied.values()):
            raise ValueError("External event counts must be nonnegative integers")
        if any(supplied.get(event, 0) < thresholds[event] for event in self._external_events()):
            raise ValueError("Simulation requires explicit external event counts for native boundaries")
        for event, value in supplied.items():
            counters[event] = value
        tasks, cursors, visited = self._task_events(), [0] * len(self.queues), []
        while len(visited) < len(tasks):
            progress = self._advance(tasks, thresholds, counters, cursors, visited)
            if not progress:
                raise ValueError("Static FIFO has an unsatisfied dependency or external event")
        return {"visited": tuple(visited), "counters": tuple(counters),
                "assumed_external_events": tuple(sorted(supplied)),
                "external_boundaries": tuple(boundary.name for boundary in self.boundaries),
                "instruction_simulation": False}

    def _advance(self, tasks, thresholds, counters, cursors, visited) -> bool:
        progress = False
        for worker, queue in enumerate(self.queues):
            if cursors[worker] == len(queue.tasks):
                continue
            ref = queue.tasks[cursors[worker]]
            wait, notify, scope = tasks[ref]
            if wait is not None and counters[wait] < thresholds[wait]:
                continue
            if notify is not None and scope == "local":
                counters[notify] += 1
            visited.append(ref)
            cursors[worker] += 1
            progress = True
        return progress

    def export_manifest(self) -> dict[str, object]:
        """导出可审查的任务域、事件映射与边界，不声称设备放行。"""
        return {"format_version": 1, "status": "static_compatibility_ir", "counter_semantics": "zero_to_threshold",
                **asdict(self)}
