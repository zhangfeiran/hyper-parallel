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
"""Wave-local weight-gradient accumulation descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyper_parallel.core.multicore.scheduler.config import (
    ATOMIC_DW_ACCUMULATION_FLAG,
    RETAINED_DW_SCRATCH_FLAG,
    RuntimeConfigC,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)
from hyper_parallel.core.multicore.scheduler.graph import OperatorNode
from hyper_parallel.core.multicore.tasks.task_base import FillConfig
from hyper_parallel.core.multicore.tasks.utils import (
    advance_tsv_vector_only,
    make_tensor_desc,
)

_INVALID_DESCRIPTOR_VALUE = 0xFFFFFFFF


@dataclass
class LocalAccumulateFillConfig(FillConfig):
    """Accumulate one ordinary GMM scratch matrix into persistent HBM."""

    producer: Any
    num_vector_cores: int
    reuse_retained_scratch: bool = False
    atomic_accumulation: bool = False

    def fill(
        self,
        cfg: RuntimeConfigC,
        op: OperatorNode,
        tsv: TaskSplitValue,
    ) -> None:
        """Append one matrix partition per vector core and local expert.

        Args:
            cfg: Runtime descriptor receiving the generated tasks.
            op: Local-accumulate graph node whose tensor layout is serialized.
            tsv: Mutable scheduler counters and topology values.
        """
        dependency_base = self.producer.completion_event_base
        if dependency_base is None:
            raise RuntimeError(
                f"{op.name} must be serialized after its GMM producer."
            )
        task_num = op.task_num
        tasks_per_expert = task_num // tsv.single_rank_expert_num
        for index in range(task_num):
            task = TaskDescC()
            task.task_type = TaskType.TASK_LOCAL_ACCUMULATE
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
            task.num_inputs = len(op.inputs)
            task.num_outputs = len(op.outputs)

            for tensor_index, spec in enumerate(op.inputs):
                task.inputs[tensor_index] = make_tensor_desc(
                    tensor_type=spec.tensor_type,
                    data_type=spec.dtype_size,
                    input_position=op.param_positions[tensor_index],
                    shape=spec.shape,
                )

            output_spec = op.outputs[0]
            task.outputs[0] = make_tensor_desc(
                tensor_type=output_spec.tensor_type,
                data_type=output_spec.dtype_size,
                input_position=op.param_positions[len(op.inputs)],
                shape=output_spec.shape,
            )

            expert = index // tasks_per_expert
            task.dependent_event = dependency_base + expert + 1
            task.trigger_event = _INVALID_DESCRIPTOR_VALUE
            task.task_index = index
            task.task_split_num = task_num
            task.task_split_value = op.split_value
            task.tiling_data_position = _INVALID_DESCRIPTOR_VALUE
            if self.reuse_retained_scratch:
                task.extra_value_0 |= RETAINED_DW_SCRATCH_FLAG
            if self.atomic_accumulation:
                task.extra_value_0 |= ATOMIC_DW_ACCUMULATION_FLAG

            task_id = tsv.pre_task_num + index
            cfg.all_tasks[task_id] = task
            cfg.vector_task_indices[tsv.pre_vector_task_num + index] = task_id

        cfg.task_index_num[1] += task_num
        advance_tsv_vector_only(tsv, task_num)


__all__ = ["LocalAccumulateFillConfig"]
