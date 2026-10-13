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
"""通过事件 IR 实施受限静态队列变换，保持原生任务归属和 scratch 契约。"""

from dataclasses import replace
from itertools import zip_longest
from typing import Any

from hyper_parallel.core.multicore.backends.events import (
    decode_event_wire, lower_mhc_final_polling, lower_mhc_worker_queues,
)
from hyper_parallel.core.multicore.compiler.events import attach_event_graphs
from hyper_parallel.core.multicore.ir.events import EventGraphIR
from hyper_parallel.core.multicore.runtime.mhc_abi import FAST_DEPENDENCY_POLL_INTERVAL_US


def _check_mhc_backward(graph):
    names = tuple(grid.name for grid in graph.grids)
    expected = ("grad_output_init", "rms_norm_grad", "mhc_grad_prev_a_and_mapping",
                "mhc_grad_phi_rms", "mhc_grad_prev_x_and_post")
    if graph.family != "mhc" or graph.direction != "backward" or names != expected:
        raise ValueError("MHC overlap requires the original five-stage backward graph")


def overlap_mhc_backward(graph: EventGraphIR) -> EventGraphIR:
    """保留 RMS 阶段，交错各 worker 的 Prepare/Final，按事件等待宏块。

    Args:
        graph: 原始 MHC 反向固定 FIFO 图。
    """
    _check_mhc_backward(graph)
    queues = []
    for queue in graph.queues:
        if queue.resource != "vector":
            queues.append(queue)
            continue
        stages = tuple(tuple(ref for ref in queue.tasks if ref[0] == number) for number in range(5))
        middle = tuple(ref for pair in zip_longest(stages[2], stages[4]) for ref in pair if ref is not None)
        queues.append(replace(queue, tasks=stages[0] + stages[1] + middle))
    result = replace(graph, queues=tuple(queues))
    result.simulate()
    return result


def wavefront_mhc_backward(graph: EventGraphIR) -> EventGraphIR:
    """先推进一整波 Prepare，再放行已经覆盖完整宏块生产者的 Final。

    Args:
        graph: 原始 MHC 反向固定 FIFO 图；原生事件和任务归属保持不变。
    """
    _check_mhc_backward(graph)
    prepare, cube, final = graph.grids[2:]
    producers = {}
    for index, event in enumerate(prepare.notifies.values()):
        producers[event] = index + 1
    ready = {notify: producers[wait] for wait, notify in zip(cube.waits.values(), cube.notifies.values())}
    prefixes = tuple(ready[event] for event in final.waits.values())
    workers = dict(graph.worker_counts)["vector"]
    queues = []
    for queue in graph.queues:
        if queue.resource != "vector":
            queues.append(queue)
            continue
        head = tuple(ref for ref in queue.tasks if ref[0] in (0, 1))
        middle = []
        for begin in range(0, len(prepare.native_tasks), workers):
            end = min(begin + workers, len(prepare.native_tasks))
            middle.extend(ref for ref in queue.tasks if ref[0] == 2 and begin <= ref[1] < end)
            middle.extend(ref for ref in queue.tasks if ref[0] == 4 and begin < prefixes[ref[1]] <= end)
        queues.append(replace(queue, tasks=head + tuple(middle)))
    result = replace(graph, queues=tuple(queues))
    result.simulate()
    return result


def _fast_final_image(image, graph):
    normal, profiled = (lower_mhc_final_polling(graph, wire) for wire in (image.normal, image.profiled))
    finals = {task for grid in graph.grids if grid.name == "mhc_grad_prev_x_and_post" for task in grid.native_tasks}
    tasks = tuple((*task[:-1], FAST_DEPENDENCY_POLL_INTERVAL_US) if index in finals else task
                  for index, task in enumerate(image.tasks))
    return replace(image, normal=normal, profiled=profiled, tasks=tasks)


def apply_event_schedule(plan: Any, policy: str = "compatibility") -> Any:
    """选择已实现的事件调度，并同时更新设备镜像和可审查 IR。

    Args:
        plan: 带事件图的兼容性主机计划。
        policy: compatibility 或显式 MHC 反向实验策略。
    """
    if policy == "compatibility":
        return plan
    transforms = {"mhc_backward_overlap": overlap_mhc_backward, "mhc_backward_wavefront": wavefront_mhc_backward,
                  "mhc_backward_wavefront_fast": wavefront_mhc_backward}
    if policy not in transforms:
        raise ValueError(f"Unknown event schedule policy: {policy}")
    if plan.export_manifest()["family"] != "mhc" or plan.backward is None:
        raise ValueError(f"{policy} requires an MHC plan with backward")
    original = next(graph for graph in plan.event_graphs if graph.direction == "backward")
    graph = transforms[policy](original)
    normal = lower_mhc_worker_queues(graph, plan.backward.normal)
    profiled = lower_mhc_worker_queues(graph, plan.backward.profiled)
    queues = decode_event_wire("mhc", normal).queues
    image = replace(plan.backward, normal=normal, profiled=profiled, queues=queues)
    if policy == "mhc_backward_wavefront_fast":
        image = _fast_final_image(image, graph)
    result = attach_event_graphs(replace(plan, backward=image, event_schedule=policy))
    if next(item for item in result.event_graphs if item.direction == "backward") != graph:
        raise ValueError("MHC event schedule differs from its lowered native queues")
    return result
