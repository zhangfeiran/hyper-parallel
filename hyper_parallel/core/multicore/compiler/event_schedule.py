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

from hyper_parallel.core.multicore.backends.events import decode_event_wire, lower_mhc_worker_queues
from hyper_parallel.core.multicore.compiler.events import attach_event_graphs
from hyper_parallel.core.multicore.ir.events import EventGraphIR


def overlap_mhc_backward(graph: EventGraphIR) -> EventGraphIR:
    """保留 RMS 阶段，交错各 worker 的 Prepare/Final，按事件等待宏块。

    Args:
        graph: 原始 MHC 反向固定 FIFO 图。
    """
    names = tuple(grid.name for grid in graph.grids)
    expected = ("grad_output_init", "rms_norm_grad", "mhc_grad_prev_a_and_mapping",
                "mhc_grad_phi_rms", "mhc_grad_prev_x_and_post")
    if graph.family != "mhc" or graph.direction != "backward" or names != expected:
        raise ValueError("MHC overlap requires the original five-stage backward graph")
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


def apply_event_schedule(plan: Any, policy: str = "compatibility") -> Any:
    """选择已实现的事件调度，并同时更新设备镜像和可审查 IR。

    Args:
        plan: 带事件图的兼容性主机计划。
        policy: compatibility 或显式实验策略 mhc_backward_overlap。
    """
    if policy == "compatibility":
        return plan
    if policy != "mhc_backward_overlap":
        raise ValueError(f"Unknown event schedule policy: {policy}")
    if plan.export_manifest()["family"] != "mhc" or plan.backward is None:
        raise ValueError("mhc_backward_overlap requires an MHC plan with backward")
    original = next(graph for graph in plan.event_graphs if graph.direction == "backward")
    graph = overlap_mhc_backward(original)
    normal = lower_mhc_worker_queues(graph, plan.backward.normal)
    profiled = lower_mhc_worker_queues(graph, plan.backward.profiled)
    queues = decode_event_wire("mhc", normal).queues
    image = replace(plan.backward, normal=normal, profiled=profiled, queues=queues)
    result = attach_event_graphs(replace(plan, backward=image, event_schedule=policy))
    if next(item for item in result.event_graphs if item.direction == "backward") != graph:
        raise ValueError("MHC event schedule differs from its lowered native queues")
    return result
