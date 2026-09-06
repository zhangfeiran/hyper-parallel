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
"""Cross-worker phase barrier descriptors for multi-wave kernels."""

from dataclasses import dataclass

from hyper_parallel.core.multicore.scheduler.config import (
    NUM_WORKERS_CUBE,
    NUM_WORKERS_VECTOR,
    RuntimeConfigC,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)
from hyper_parallel.core.multicore.scheduler.graph import OperatorNode
from hyper_parallel.core.multicore.tasks.task_base import FillConfig

_INVALID_DESCRIPTOR_VALUE = 0xFFFFFFFF


@dataclass
class PhaseBarrierFillConfig(FillConfig):
    """Make every Cube and active Vector worker execute one barrier descriptor."""

    def fill(
        self,
        cfg: RuntimeConfigC,
        op: OperatorNode,
        tsv: TaskSplitValue,
    ) -> None:
        """Append a shared phase barrier without advancing logical events.

        Args:
            cfg: Runtime configuration receiving the barrier descriptor.
            op: Tensor-free phase-barrier graph node.
            tsv: Mutable scheduler counters and dimensions.
        """
        if op.task_num != 1 or op.inputs or op.outputs:
            raise ValueError("phase barrier must contain one tensor-free task.")

        task = TaskDescC()
        task.task_type = TaskType.TASK_PHASE_BARRIER
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_MIX
        task.dependent_event = _INVALID_DESCRIPTOR_VALUE
        task.trigger_event = _INVALID_DESCRIPTOR_VALUE
        task.task_split_num = 1
        task.task_split_value = 1
        task.tiling_data_position = _INVALID_DESCRIPTOR_VALUE

        task_id = tsv.pre_task_num
        cfg.all_tasks[task_id] = task
        active_vector_workers = NUM_WORKERS_VECTOR // 2
        if active_vector_workers != NUM_WORKERS_CUBE:
            raise ValueError(
                "phase barrier requires the same number of Cube and active Vector workers."
            )
        for offset in range(NUM_WORKERS_CUBE):
            cfg.cube_task_indices[tsv.pre_cube_task_num + offset] = task_id
            cfg.vector_task_indices[tsv.pre_vector_task_num + offset] = task_id
        cfg.task_index_num[0] += NUM_WORKERS_CUBE
        cfg.task_index_num[1] += active_vector_workers

        tsv.pre_task_num += 1
        tsv.pre_cube_task_num += NUM_WORKERS_CUBE
        tsv.pre_vector_task_num += active_vector_workers


__all__ = ["PhaseBarrierFillConfig"]
