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
"""绑定 MoE 受限动态域与原生 DYNAMIC_DSV3_MOE 描述符。"""

from dataclasses import replace
from math import prod
from typing import Any

from hyper_parallel.core.multicore.ir.dynamic_events import MoeCountSourceIR, MoeTaskDomainIR
from hyper_parallel.core.multicore.ir.events import EventGraphIR, EventIndexMap, IndexRun
from hyper_parallel.core.multicore.runtime.generated.moe_abi import DynamicDataC, DynamicType, TaskDescC, TaskType


def moe_count_source(spec: Any, direction: str) -> MoeCountSourceIR:
    """由绑定规格生成计数别名和无损行数上界。

    Args:
        spec: 已有 MegaMoeSpec，包括副本执行槽位。
        direction: forward、backward 或 backward_no_replica。
    """
    positions = (6, 10) if direction == "forward" else (19,)
    return MoeCountSourceIR(positions, spec.local_experts, spec.local_num_tokens * spec.ep_size,
                            spec.maximum_receive_capacity)


def _expert_map(shape, values):
    if any(value is None or value == 0xFFFFFFFF for value in values):
        raise ValueError("MoE dynamic mappings require native expert events")
    entries = values[::shape[1]]
    if any(len(set(values[index:index + shape[1]])) != 1 for index in range(0, len(values), shape[1])):
        raise ValueError("MoE dynamic mappings require one event per expert task group")
    step = 0 if len(entries) == 1 else entries[1] - entries[0]
    if any(value != entries[0] + index * step for index, value in enumerate(entries)):
        raise ValueError("MoE dynamic mappings require bounded affine expert event identities")
    return EventIndexMap(shape, (IndexRun(0, prod(shape), entries[0], step, shape[1]),))


def _check_native_partition(tasks, source):
    kinds = {task.task_type for task in tasks}
    supported = {TaskType.TASK_GROUPED_MATMUL, TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD}
    if not kinds & supported:
        return None
    if len(kinds) != 1 or not all(task.inputs[0].dynamic_shape == 1 for task in tasks):
        raise ValueError("MoE dynamic stages require homogeneous native dynamic descriptors")
    if len(tasks) % source.expert_count or any(task.task_index != index or task.task_split_num != len(tasks)
                                              for index, task in enumerate(tasks)):
        raise ValueError("MoE dynamic task indices must cover the reserved expert domain")
    return kinds.pop()


def _grouped_domain(tasks, source, shape, cubes):
    if shape[1] != cubes or any(task.inputs[2].input_position not in source.input_positions
                              or task.inputs[2].data_type != 8 for task in tasks):
        raise ValueError("MoE GMM lanes or count source differ from native bindings")
    return MoeTaskDomainIR("grouped_matmul", shape)


def _vector_domain(tasks, shape):
    rows = tasks[0].task_split_value
    if any(task.task_split_value != rows for task in tasks):
        raise ValueError("MoE dynamic Vector row tiles must match within a stage")
    return MoeTaskDomainIR("swiglu", shape, rows)


def _dynamic_grid(grid, tasks, source, cubes):
    kind = _check_native_partition(tasks, source)
    if kind is None:
        return grid
    shape = (source.expert_count, len(tasks) // source.expert_count)
    domain = (_grouped_domain(tasks, source, shape, cubes) if kind == TaskType.TASK_GROUPED_MATMUL
              else _vector_domain(tasks, shape))
    domain.row_intervals(source, (0,) * source.expert_count)
    return replace(grid, shape=shape, dynamic_domain=domain,
                   waits=_expert_map(shape, tuple(task.dependent_event for task in tasks)),
                   notifies=_expert_map(shape, tuple(task.trigger_event for task in tasks)))


def bind_moe_dynamic_events(graph: EventGraphIR, template: bytes, view: Any,
                           source: MoeCountSourceIR) -> EventGraphIR:
    """校验原生计数头和任务算术，绑定受限动态工作域与仿射专家事件映射。

    Args:
        graph: 原生固定队列事件图。
        template: 普通或 profiling 运行时镜像。
        view: 同一镜像经过边界检查的事件视图。
        source: 从绑定规格取得的容量与 ABI 输入位置。
    """
    offset = view.queue_offset + 16 + view.task_capacity * 12
    if len(template) < offset + view.abi.structure("DynamicDataC").size:
        raise ValueError("MoE native dynamic count header is truncated")
    dynamic = DynamicDataC.from_buffer_copy(template, offset)
    if (dynamic.dynamic_type != DynamicType.DYNAMIC_DSV3_MOE
            or dynamic.dynamic_input_position != source.input_positions[0]
            or dynamic.dynamic_group_size != source.expert_count or dynamic.dynamic_max_seq_len != -1):
        raise ValueError("MoE dynamic count header differs from the bounded native contract")
    tasks = tuple(TaskDescC.from_buffer_copy(template, view.task_offset + index * view.abi.structure("TaskDescC").size)
                  for index in range(view.task_count))
    cubes = dict(graph.worker_counts)["cube"]
    grids = tuple(_dynamic_grid(grid, tuple(tasks[index] for index in grid.native_tasks), source, cubes)
                  for grid in graph.grids)
    return replace(graph, grids=grids, count_source=source)
