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
"""Route acknowledgements that release a reused symmetric dY tile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyper_parallel.core.multicore.scheduler.config import (
    ROUTE_COMPUTE_ACK_FLAG,
    RuntimeConfigC,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)
from hyper_parallel.core.multicore.scheduler.graph import OperatorNode
from hyper_parallel.core.multicore.tasks.task_base import FillConfig
from hyper_parallel.core.multicore.tasks.utils import advance_tsv_vector_only

_INVALID_DESCRIPTOR_VALUE = 0xFFFFFFFF


@dataclass
class RouteComputeAckFillConfig(FillConfig):
    """Publish per-source expert reuse after both dY GMM consumers finish."""

    activation_producer: Any
    weight_producer: Any
    route_tiles_per_expert: int

    def fill(
        self,
        cfg: RuntimeConfigC,
        op: OperatorNode,
        tsv: TaskSplitValue,
    ) -> None:
        """Append one acknowledgement publisher per local expert and source.

        Args:
            cfg: Runtime configuration receiving acknowledgement tasks.
            op: Route acknowledgement graph node being lowered.
            tsv: Mutable scheduler counters and dimensions.
        """
        activation_base = self.activation_producer.completion_event_base
        weight_base = self.weight_producer.completion_event_base
        if activation_base is None or weight_base is None:
            raise RuntimeError(
                f"{op.name} must be serialized after both dY GMM consumers."
            )
        expected_tasks = tsv.single_rank_expert_num * tsv.ep
        if op.task_num != expected_tasks:
            raise ValueError(
                f"{op.name} requires {expected_tasks} tasks, got {op.task_num}."
            )
        if self.route_tiles_per_expert <= 0:
            raise ValueError("route acknowledgement requires at least one tile.")

        for index in range(op.task_num):
            expert = index // tsv.ep
            task = TaskDescC()
            task.task_type = TaskType.TASK_LOCAL_COPY
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
            task.dependent_event = activation_base + expert + 1
            task.trigger_event = _INVALID_DESCRIPTOR_VALUE
            task.task_index = index
            task.task_split_num = op.task_num
            task.task_split_value = 1
            task.tiling_data_position = _INVALID_DESCRIPTOR_VALUE
            task.extra_value_0 = ROUTE_COMPUTE_ACK_FLAG
            task.extra_value_1 = weight_base + expert + 1
            task.extra_value_2 = self.route_tiles_per_expert

            task_id = tsv.pre_task_num + index
            cfg.all_tasks[task_id] = task
            cfg.vector_task_indices[tsv.pre_vector_task_num + index] = task_id

        cfg.task_index_num[1] += op.task_num
        advance_tsv_vector_only(tsv, op.task_num)


__all__ = ["RouteComputeAckFillConfig"]
