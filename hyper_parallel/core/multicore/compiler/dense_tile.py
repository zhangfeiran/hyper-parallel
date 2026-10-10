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
"""Row-separable dense DAG tiling with explicit cube/vector ownership and joins."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from hyper_parallel.core.multicore.ir.program import Value
from hyper_parallel.core.multicore.language.types import DType
from hyper_parallel.core.multicore.runtime.dense import DenseKernelPlan


@dataclass(frozen=True)
class DenseTilePolicy:
    """按行和输出列静态分区，不依赖专家拓扑或本次调用的地址。"""

    row_input: str = "x"
    rows_per_tile: int = 64
    cube_workers: int = 24
    prefetch_tiles: int = 2
    column_workers: int = 1

    def __post_init__(self) -> None:
        sizes = (self.rows_per_tile, self.cube_workers, self.prefetch_tiles, self.column_workers)
        if any(type(size) not in (int,) or size <= 0 for size in sizes):
            raise ValueError("Dense tile sizes and worker counts must be positive integers")
        if self.cube_workers > 24 or self.prefetch_tiles > 8 or self.column_workers > 24:
            raise ValueError("Dense tile policy exceeds Ascend910B worker/prefetch bounds")


@dataclass(frozen=True)
class DenseTileTask:
    """One owned primitive slice and its producer-count dependencies."""

    operation: int
    tile: int
    worker_kind: str
    worker: int
    row_begin: int
    row_end: int
    column_begin: int
    column_end: int
    column_tile: int
    dependencies: tuple[tuple[int, int], ...]
    trigger: int


@dataclass(frozen=True)
class DenseTilePlan:
    """Concrete forward worker queues; device execution has separate admission."""

    dense: DenseKernelPlan
    policy: DenseTilePolicy
    rows: int
    tiled_values: tuple[int, ...]
    tasks: tuple[DenseTileTask, ...]
    event_count: int

    def queues(self) -> dict[tuple[str, int], tuple[DenseTileTask, ...]]:
        """Preserve actual per-worker head order with bounded producer prefetch."""
        queues = defaultdict(list)
        for task in self.tasks:
            queues[(task.worker_kind, task.worker)].append(task)
        result = {}
        for worker, tasks in queues.items():
            result[worker] = tuple(sorted(tasks, key=lambda task: (
                task.tile // (self.policy.cube_workers * self.policy.prefetch_tiles),
                task.operation, task.tile, task.column_tile)))
        return result

    def export_manifest(self) -> dict[str, object]:
        """Export exact joins and buffer retention without claiming device acceptance."""
        return {"format_version": 1, "execution_mode": "resident_dense_tile_candidate",
                "device_status": "unvalidated", "rows": self.rows, "policy": asdict(self.policy),
                "events": self.event_count, "event_stride_bytes": 128, "tiled_values": self.tiled_values,
                "tasks": [asdict(task) for task in self.tasks],
                "buffers": [{"value_id": buffer.value_id, "bytes": buffer.size_bytes,
                             "ownership": "invocation", "retain_until": "backward" if buffer.saved_for_backward
                             else "last_consumer_or_caller"} for buffer in self.dense.buffers]}


def _check_row_operation(operation, tiled):
    arguments = dict(operation.arguments)
    if operation.logical_name == "dense.matmul":
        left, right = arguments["left"], arguments["right"]
        if left.id not in tiled or right.id in tiled or arguments["transpose_left"]:
            raise ValueError("Dense forward tiles require row-local left data and invariant right weights")
    elif operation.logical_name == "dense.swiglu":
        if arguments["packed"].id not in tiled:
            raise ValueError("Dense activation must consume row-local data")
    else:
        raise ValueError("No resident tile implementation for this dense primitive")


def _tiled_values(plan, policy):
    matches = [value for value in plan.ir.inputs if value.name == policy.row_input]
    if len(matches) != 1:
        raise ValueError("Dense row input must identify exactly one semantic tensor parameter")
    types = dict(plan.value_types)
    if any(tensor_type.dtype != DType.BF16 for tensor_type in types.values()):
        raise ValueError("Resident dense tile candidates currently require BF16 storage")
    row_input, = matches
    rows = types[row_input.id].shape[0]
    tiled = {row_input.id}
    for operation in plan.ir.operations:
        _check_row_operation(operation, tiled)
        if any(types[value.id].shape[0] != rows for value in operation.outputs):
            raise ValueError("Dense tiling must preserve its row axis")
        tiled.update(value.id for value in operation.outputs)
    if any(value.id not in tiled for value in plan.ir.outputs):
        raise ValueError("Dense tile outputs must retain row ownership")
    return rows, tuple(sorted(tiled))


def compile_dense_tiles(plan: DenseKernelPlan, policy: DenseTilePolicy = DenseTilePolicy()) -> DenseTilePlan:
    """将可按行分离的 SSA 展开为二维工作队列及整行生产者汇合。

    Args:
        plan: Canonical dense primitive DAG.
        policy: 行输入、tile 高度、Cube 数、预取窗口及目标列分区数。
    """
    rows, values = _tiled_values(plan, policy)
    tasks = []
    count = len(plan.tasks)
    if count == 0 or count > 16 or len(plan.value_types) > 32:
        raise ValueError("Dense tile descriptor requires 1..16 primitives and at most 32 tensor values")
    if any(dimension <= 0 for _, tensor_type in plan.value_types for dimension in tensor_type.shape[1:]):
        raise ValueError("Resident dense tiles require nonzero channel dimensions")
    tiles = (rows + policy.rows_per_tile - 1) // policy.rows_per_tile
    if tiles * count > (2**32 - 1) // 32:
        raise ValueError("Dense tile event offsets exceed their native uint32 boundary")
    partitions = tuple(column_partition(plan, task.index, policy) for task in plan.tasks)
    producer_counts = tuple(parts * (2 if task.provider.worker == "vector" else 1)
                            for task, (_, parts) in zip(plan.tasks, partitions))
    for tile in range(tiles):
        first, end = tile * policy.rows_per_tile, min((tile + 1) * policy.rows_per_tile, rows)
        for task in plan.tasks:
            kind = task.provider.worker
            if kind not in ("cube", "vector"):
                raise ValueError("Dense task requires a known cube/vector worker implementation")
            joins = tuple((tile * count + producer, producer_counts[producer]) for producer in task.dependencies)
            trigger = tile * count + task.index
            width, parts = partitions[task.index]
            output, = plan.ir.operations[task.index].outputs
            columns = dict(plan.value_types)[output.id].shape[1]
            for part in range(parts):
                column_begin, column_end = part * width, min((part + 1) * width, columns)
                cube = (tile * parts + part) % policy.cube_workers
                slices = ((cube, first, end),) if kind == "cube" else (
                    (cube * 2, first, first + (end - first + 1) // 2),
                    (cube * 2 + 1, first + (end - first + 1) // 2, end))
                for worker, begin, stop in slices:
                    tasks.append(DenseTileTask(task.index, tile, kind, worker, begin, stop,
                                               column_begin, column_end, part, joins, trigger))
    return DenseTilePlan(plan, policy, rows, values, tuple(tasks), tiles * count)


def column_partition(plan: DenseKernelPlan, operation: int, policy: DenseTilePolicy) -> tuple[int, int]:
    """返回输出列片段宽度和实际片段数，保留非对齐尾部。

    Args:
        plan: 已绑定具体形状的稠密 SSA。
        operation: 原语索引。
        policy: 目标列分区数。
    """
    output, = plan.ir.operations[operation].outputs
    columns = dict(plan.value_types)[output.id].shape[1]
    if policy.column_workers == 1:
        return columns, 1
    width = ((columns + policy.column_workers * 16 - 1) // (policy.column_workers * 16)) * 16
    return width, (columns + width - 1) // width


def simulate_dense_tiles(plan: DenseTilePlan) -> tuple[DenseTileTask, ...]:
    """Check real queue-head scheduling and all joins, including empty vector lanes.

    Args:
        plan: Expanded dense forward worker queues.
    """
    queues = {worker: list(tasks) for worker, tasks in plan.queues().items()}
    counters, completed = [0] * plan.event_count, []
    while any(queues.values()):
        progress = False
        for tasks in queues.values():
            if not tasks or any(counters[event] < required for event, required in tasks[0].dependencies):
                continue
            task = tasks.pop(0)
            counters[task.trigger] += 1
            completed.append(task)
            progress = True
        if not progress:
            raise ValueError("Dense worker queues deadlock at their actual blocked heads")
    return tuple(completed)


def tensor_arguments(plan: DenseTilePlan, operation: int) -> tuple[Value, ...]:
    """Expose semantic tensor argument order for native binding generation.

    Args:
        plan: Typed resident candidate.
        operation: Primitive index in the semantic program.
    """
    return tuple(value for _, value in plan.dense.ir.operations[operation].arguments if isinstance(value, Value))
