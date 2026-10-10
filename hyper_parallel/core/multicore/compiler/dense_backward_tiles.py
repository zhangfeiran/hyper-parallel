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
"""反向行流水和完整收缩工作队列，使用跨轴汇合避免预取窗口死锁。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from hyper_parallel.core.multicore.compiler.dense_backward import DenseBackwardPlan, compile_dense_backward
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy, DenseTileTask, compile_dense_tiles


@dataclass(frozen=True)
class DenseBackwardRegion:
    """物理矩形域独立于原始 token 行，SwiGLUGrad 按成对通道写回。"""

    node: int
    phase: int
    rows: int
    columns: int
    kind: str
    width: int
    parts: int
    events: tuple[int, ...]
    signals: int
    matrix: tuple[int, int, int, bool, bool] | None
    transpose_output: bool


@dataclass(frozen=True)
class DenseBackwardTilePlan:
    """显式反向的真实 worker 队列；原生序列化和设备准入仍未绑定。"""

    backward: DenseBackwardPlan
    policy: DenseTilePolicy
    output_mask: tuple[bool, ...]
    regions: tuple[DenseBackwardRegion, ...]
    aliases: tuple[tuple[int, int], ...]
    tasks: tuple[DenseTileTask, ...]
    event_count: int

    def queues(self) -> dict[tuple[str, int], tuple[DenseTileTask, ...]]:
        """先完成全部行窗口，再让 worker 进入完整收缩阶段。"""
        regions = {region.node: region for region in self.regions}
        queues = defaultdict(list)
        for task in self.tasks:
            queues[(task.worker_kind, task.worker)].append(task)
        window = self.policy.cube_workers * self.policy.prefetch_tiles
        return {worker: tuple(sorted(tasks, key=lambda task: (
            regions[task.operation].phase, task.tile // window, task.operation, task.tile, task.column_tile)))
                for worker, tasks in queues.items()}

    def export_manifest(self) -> dict[str, object]:
        """展示矩形域、全 token 汇合和转置输出，保留未绑定状态。"""
        return {"format_version": 1, "native_status": "unbound", "execution_mode": "resident_backward_candidate",
                "numeric_policy": self.backward.forward.ir.numeric_policy, "policy": asdict(self.policy),
                "output_mask": self.output_mask, "regions": [asdict(region) for region in self.regions],
                "aliases": self.aliases, "tasks": [asdict(task) for task in self.tasks],
                "event_count": self.event_count, "event_stride_bytes": 128,
                "reduction": "complete_contracted_axis_per_output_fragment"}


def _geometry(node, row_values):
    phase = 0 if node.target in row_values else 1
    if node.matrix is not None:
        rows, columns, contracted = node.matrix[:3]
        kind = "zero" if contracted == 0 else "matmul"
    else:
        rows, columns = node.output.type.shape
        kind = "swiglu_grad" if node.kind == "swiglu_input_grad" else "gradient_add"
        if kind == "swiglu_grad":
            columns //= 2
    if any(dimension > 2**31 - 1 for dimension in (rows, columns, *(node.matrix[:3] if node.matrix else ()))):
        raise ValueError("Dense backward matrix dimensions exceed the signed SDK boundary")
    return phase, rows, columns, kind


def _regions(backward, policy, row_values, defined):
    regions, producers, aliases = [], {}, []
    event_count = 0
    for node in backward.nodes:
        if node.output.id not in defined:
            continue
        if node.kind == "gradient_add":
            active = [value for value in node.inputs if value in defined]
            if len(active) == 1:
                producers[node.output.id] = producers.get(active[0])
                aliases.append((node.output.id, active[0]))
                continue
        phase, rows, columns, kind = _geometry(node, row_values)
        width = columns if policy.column_workers == 1 else (
            (columns + policy.column_workers * 16 - 1) // (policy.column_workers * 16) * 16)
        parts = (columns + width - 1) // width if width else 0
        count = (rows + policy.rows_per_tile - 1) // policy.rows_per_tile
        if event_count + count > (2**32 - 1) // 32:
            raise ValueError("Dense backward event offsets exceed the native uint32 boundary")
        events = tuple(range(event_count, event_count + count))
        signals = parts * (1 if kind == "matmul" else 2)
        region = DenseBackwardRegion(node.index, phase, rows, columns, kind, width, parts, events, signals,
                                      node.matrix, node.transpose_output)
        regions.append(region)
        producers[node.output.id] = node.index
        event_count += count
    return tuple(regions), producers, tuple(aliases), event_count


def _dependencies(node, region, tile, regions, producers, defined):
    result = set()
    for value in node.inputs:
        if value not in defined or producers.get(value) is None:
            continue
        parent = regions[producers[value]]
        events = (parent.events[tile],) if region.phase == parent.phase == 0 else parent.events
        result.update((event, parent.signals) for event in events)
    return tuple(sorted(result))


def _tasks(backward, policy, regions, producers, defined):
    result = []
    lookup = {region.node: region for region in regions}
    for region in regions:
        node = backward.nodes[region.node]
        worker_kind = "cube" if region.kind == "matmul" else "vector"
        for tile, trigger in enumerate(region.events):
            first, end = tile * policy.rows_per_tile, min((tile + 1) * policy.rows_per_tile, region.rows)
            dependencies = _dependencies(node, region, tile, lookup, producers, defined)
            for part in range(region.parts):
                column, stop = part * region.width, min((part + 1) * region.width, region.columns)
                cube = (tile * region.parts + part) % policy.cube_workers
                slices = ((cube, first, end),) if worker_kind == "cube" else (
                    (cube * 2, first, first + (end - first + 1) // 2),
                    (cube * 2 + 1, first + (end - first + 1) // 2, end))
                for worker, begin, finish in slices:
                    result.append(DenseTileTask(node.index, tile, worker_kind, worker, begin, finish,
                                                column, stop, part, dependencies, trigger))
    return tuple(result)


def compile_dense_backward_tiles(backward: DenseBackwardPlan, policy: DenseTilePolicy = DenseTilePolicy(),
                                 output_mask: tuple[bool, ...] | None = None) -> DenseBackwardTilePlan:
    """按行数据流和完整 token 收缩分阶段，不拆分 BF16 权重梯度的 K 轴。

    Args:
        backward: 来自已准入显式 recipe 的完整梯度 SSA。
        policy: 每个矩形域的行高、输出列分区及 worker 数量。
        output_mask: 本次调用各余切的主机存在性；缺省为全部存在。
    """
    if backward != compile_dense_backward(backward.forward):
        raise ValueError("Dense backward tile plan differs from its canonical explicit recipes")
    forward = compile_dense_tiles(backward.forward, policy)
    mask = (True,) * len(backward.seeds) if output_mask is None else output_mask
    defined = backward.defined_values(mask)
    regions, producers, aliases, count = _regions(backward, policy, set(forward.tiled_values), defined)
    return DenseBackwardTilePlan(backward, policy, mask, regions, aliases,
                                 _tasks(backward, policy, regions, producers, defined), count)


def simulate_dense_backward_tiles(plan: DenseBackwardTilePlan) -> tuple[DenseTileTask, ...]:
    """回放真实队头，验证全轴汇合、完整发布和无阻塞环。

    Args:
        plan: 反向 worker 队列及逐矩形域事件计划。
    """
    queues = {worker: list(tasks) for worker, tasks in plan.queues().items()}
    counters, completed = [0] * plan.event_count, []
    while any(queues.values()):
        progress = False
        for tasks in queues.values():
            if not tasks:
                continue
            task = tasks[0]
            if (task.trigger < 0 or task.trigger >= plan.event_count
                    or any(event < 0 or event >= plan.event_count or required <= 0
                           for event, required in task.dependencies)):
                raise ValueError("Dense backward task contains invalid event dependencies")
            if any(counters[event] < required for event, required in task.dependencies):
                continue
            completed.append(tasks.pop(0))
            counters[task.trigger] += 1
            progress = True
        if not progress:
            raise ValueError("Dense backward worker queues contain a dependency deadlock")
    if any(counters[event] != region.signals for region in plan.regions for event in region.events):
        raise ValueError("Dense backward worker queues did not publish every producer fragment")
    return tuple(completed)
