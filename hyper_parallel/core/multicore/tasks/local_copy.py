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
"""Local device-copy descriptors used by kernel-internal MegaMoe waves."""

from __future__ import annotations

from dataclasses import dataclass

from hyper_parallel.core.multicore.scheduler.config import (
    ROUTE_ARRIVAL_WAIT_FLAG,
    ROUTE_COPY_ACK_FLAG,
    RuntimeConfigC,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)
from hyper_parallel.core.multicore.scheduler.graph import OperatorNode
from hyper_parallel.core.multicore.tasks.task_base import FillConfig
from hyper_parallel.core.multicore.tasks.utils import (
    advance_tsv_vector,
    make_tensor_desc,
)


@dataclass
class LocalCopyFillConfig(FillConfig):
    """Copy one wave's expert-major hidden rows into ordinary HBM scratch."""

    global_trigger_event: int | None = None
    rank_in_event: bool = False
    route_tile_ack: bool = False
    route_arrival_only: bool = False
    route_generation_phase: int = 0
    route_generation_count: int = 1

    def fill(
        self,
        cfg: RuntimeConfigC,
        op: OperatorNode,
        tsv: TaskSplitValue,
    ) -> None:
        """Append local-copy tasks and advance scheduler counters.

        Args:
            cfg: Runtime descriptor receiving the generated tasks.
            op: Local-copy graph node whose tensor layout is serialized.
            tsv: Mutable scheduler counters and topology values.
        """
        task_num = op.task_num
        tasks_per_expert = task_num // tsv.single_rank_expert_num
        if self.route_tile_ack and self.route_arrival_only:
            raise ValueError(
                "LocalCopy cannot copy-and-ack and wait-only in one task."
            )
        route_tile_wait = self.route_tile_ack or self.route_arrival_only
        if self.route_tile_ack and len(op.inputs) != 3:
            raise ValueError(
                "route-tile LocalCopy requires data, source-offset and size inputs."
            )
        if route_tile_wait:
            if self.route_generation_count <= 1:
                raise ValueError(
                    "route-tile wait requires multiple route generations."
                )
            if not 0 <= self.route_generation_phase < self.route_generation_count:
                raise ValueError(
                    "route-tile LocalCopy generation phase is out of range."
                )
        for index in range(task_num):
            task = TaskDescC()
            task.task_type = TaskType.TASK_LOCAL_COPY
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
            task.num_inputs = len(op.inputs)
            task.num_outputs = 1

            for tensor_index, spec in enumerate(op.inputs):
                task.inputs[tensor_index] = make_tensor_desc(
                    tensor_type=spec.tensor_type,
                    data_type=spec.dtype_size,
                    input_position=op.param_positions[tensor_index],
                    dynamic_shape=int(spec.is_dynamic),
                    shape=spec.shape,
                )

            output_spec = op.outputs[0]
            task.outputs[0] = make_tensor_desc(
                tensor_type=output_spec.tensor_type,
                data_type=output_spec.dtype_size,
                input_position=op.param_positions[len(op.inputs)],
                dynamic_shape=int(output_spec.is_dynamic),
                shape=output_spec.shape,
            )

            expert = index // tasks_per_expert
            if route_tile_wait:
                task.dependent_event = 0xFFFFFFFF
                task.extra_value_0 = (
                    ROUTE_COPY_ACK_FLAG
                    if self.route_tile_ack
                    else ROUTE_ARRIVAL_WAIT_FLAG
                )
                task.extra_value_1 = self.route_generation_phase
                task.extra_value_2 = self.route_generation_count
            else:
                rank_event_offset = (
                    tsv.single_rank_expert_num * tsv.rank_id
                    if self.rank_in_event
                    else 0
                )
                task.dependent_event = (
                    tsv.pre_pre_event_num + rank_event_offset + expert + 1
                )
            task.trigger_event = (
                self.global_trigger_event
                if self.global_trigger_event is not None
                else tsv.pre_event_num + expert + 1
            )
            cfg.all_event_num_triggers[task.trigger_event] = (
                task_num
                if self.global_trigger_event is not None
                else tasks_per_expert
            )
            task.task_index = index
            task.task_split_num = task_num
            task.task_split_value = op.split_value
            task.tiling_data_position = 0xFFFFFFFF

            task_id = tsv.pre_task_num + index
            cfg.all_tasks[task_id] = task
            cfg.vector_task_indices[tsv.pre_vector_task_num + index] = task_id

        cfg.task_index_num[1] += task_num
        advance_tsv_vector(
            tsv,
            task_num,
            event_group_size=tsv.single_rank_expert_num,
        )


__all__ = ["LocalCopyFillConfig"]
