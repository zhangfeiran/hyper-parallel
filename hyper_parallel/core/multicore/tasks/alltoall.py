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
"""AllToAll fill config: dispatch and combine (forward + backward)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hyper_parallel.core.multicore.scheduler.config import (
    ROUTE_COPY_ACK_FLAG,
    ROUTE_EXPERT_ACK_FLAG,
    RuntimeConfigC,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
    TensorDescC,
)
from hyper_parallel.core.multicore.scheduler.graph import OperatorNode
from hyper_parallel.core.multicore.tasks.task_base import FillConfig
from hyper_parallel.core.multicore.tasks.utils import (
    advance_tsv_vector,
    advance_tsv_vector_only,
)


class AllToAllType(Enum):
    """
    MoE AllToAll semantic type — determines event wiring.

    DISPATCH — scatter tokens from model-parallel ranks to expert-holding ranks.
        dependent_event = pre_pre_event_num + 0
        trigger_event   = pre_event_num + (i // per_g_e_num) + 1
        trigger_count   = task_num * ep // all_expert_num

    COMBINE — gather expert results back to the originating rank.
        dependent_event = pre_pre_event_num + (i // per_g_e_num) % sre + 1
        trigger_event   = all_event_num
        trigger_count   = task_num

    OTHER — reserved for AllToAll patterns outside the MoE dispatch/combine semantic.
    """
    DISPATCH = 1
    COMBINE  = 2
    OTHER    = 3


@dataclass
class AllToAllFillConfig(FillConfig):
    """
    AllToAll fill config covering dispatch and combine for fwd and bwd.

    moe_type : AllToAllType
        Determines event wiring; see AllToAllType for details.

    advance : "vector" | "vector_only"
        "vector"      — advance_tsv_vector(tsv, task_num, event_group_size=event_group)
                        Used by: dispatch (fwd/bwd), combine forward.
        "vector_only" — advance_tsv_vector_only(tsv, task_num)
                        Only advances pre_task_num/pre_vector_task_num.
                        Used by: combine backward.

    event_group : int
        Only effective when advance="vector".
        dispatch (fwd/bwd): all_expert_num
        combine forward:    1

    route_top_k : int | None
        Encodes route-slot indices back to source-token rows inside a fused
        dispatch task.

    """
    moe_type:    AllToAllType = AllToAllType.DISPATCH
    advance:     str          = "vector"   # "vector" | "vector_only"
    event_group: int          = 1          # only used when advance="vector"
    dependent_event: int | None = None
    trigger_event: int | None = None
    route_index_position: int | None = None
    route_weight_position: int | None = None
    route_top_k: int | None = None
    route_completion_event: int | None = None
    additional_dependent_event: int | None = None
    isolated_route_signals: bool = False
    route_generation_phase: int = 0
    route_generation_count: int = 1
    source_base_ptr_offset: int = 0
    offsets_in_tokens: bool = False
    wait_for_copy_ack: bool = False
    expert_copy_ack: bool = False
    align_isolated_event_group: bool = False
    destination_wave_fence: bool = False

    def fill(
        self,
        cfg: RuntimeConfigC,
        op: OperatorNode,
        tsv: TaskSplitValue,
    ) -> None:
        """Append dispatch or combine tasks and advance scheduler counters.

        Args:
            cfg: Runtime configuration receiving task descriptors.
            op: Dispatch or combine graph node being lowered.
            tsv: Mutable scheduler counters and dimensions.
        """
        task_num    = op.task_num
        per_g_e_num = task_num // tsv.all_expert_num
        param       = op.param_positions
        trigger_count = task_num * tsv.ep // tsv.all_expert_num
        if self.wait_for_copy_ack and not self.isolated_route_signals:
            raise ValueError(
                "copy acknowledgement requires isolated route signals."
            )
        if self.expert_copy_ack and not self.wait_for_copy_ack:
            raise ValueError(
                "expert acknowledgement requires copy-ack waiting."
            )
        if self.isolated_route_signals:
            if not 1 <= self.route_generation_count <= 0x7F:
                raise ValueError(
                    "route_generation_count must fit in the positive packed "
                    f"arrival threshold, got {self.route_generation_count}."
                )
            if not 0 <= self.route_generation_phase < self.route_generation_count:
                raise ValueError(
                    "route_generation_phase must be in "
                    f"[0, {self.route_generation_count}), got "
                    f"{self.route_generation_phase}."
                )
            if not 1 <= trigger_count <= 0xFFFF:
                raise ValueError(
                    "route arrival trigger count must fit in 16 bits, got "
                    f"{trigger_count}."
                )
            if self.align_isolated_event_group:
                # Route-arrival waiters recover the local expert from the
                # absolute event id. A late dispatch therefore has to start
                # on a local-expert boundary, just like the first event group.
                event_padding = (
                    -tsv.pre_event_num
                ) % tsv.single_rank_expert_num
                tsv.pre_event_num += event_padding

        for i in range(task_num):
            task = TaskDescC()
            task.task_type        = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
            task.num_inputs       = len(op.inputs)
            task.num_outputs      = len(op.outputs)

            for j, spec in enumerate(op.inputs):
                td = TensorDescC()
                td.data_type       = spec.dtype_size
                td.input_position  = param[j]
                td.base_ptr_offset = 0

                if j == 1:   # src data: tensor_type and is_dynamic from TensorSpec
                    td.tensor_type   = spec.tensor_type
                    td.dynamic_shape = int(spec.is_dynamic)
                    td.base_ptr_offset = self.source_base_ptr_offset
                else:        # metadata (target_offset / src_offset / size): fixed type=0
                    td.tensor_type     = 0
                    td.base_ptr_offset = i // per_g_e_num   # expert-group index

                task.inputs[j] = td

            out_spec = op.outputs[0]
            out = TensorDescC()
            out.tensor_type     = out_spec.tensor_type
            out.data_type       = out_spec.dtype_size
            out.input_position  = param[task.num_inputs]
            out.base_ptr_offset = 0
            out.dynamic_shape   = int(out_spec.is_dynamic)
            task.outputs[0] = out

            if self.moe_type in {AllToAllType.DISPATCH, AllToAllType.OTHER}:
                res = i // per_g_e_num
                task.dependent_event = (
                    self.dependent_event
                    if self.dependent_event is not None
                    else tsv.pre_pre_event_num
                )
                task.trigger_event   = tsv.pre_event_num + res + 1
                # Route-fused dispatch publishes one cache-line-isolated SET
                # per source/expert/tile. A negative threshold tells the
                # device waiter to consume those slots instead of one shared
                # remote atomic ADD counter.
                if self.isolated_route_signals:
                    if self.route_generation_count == 1:
                        route_threshold = trigger_count
                    else:
                        route_threshold = (
                            trigger_count
                            | (self.route_generation_phase << 16)
                            | (self.route_generation_count << 24)
                        )
                    cfg.all_event_num_triggers[task.trigger_event] = -route_threshold
                else:
                    cfg.all_event_num_triggers[task.trigger_event] = trigger_count
            else:   # COMBINE
                current_dep = (i // per_g_e_num) % tsv.single_rank_expert_num
                task.dependent_event = tsv.pre_pre_event_num + current_dep + 1
                task.trigger_event   = (
                    self.trigger_event
                    if self.trigger_event is not None
                    else tsv.all_event_num
                )
                cfg.all_event_num_triggers[task.trigger_event] = task_num

            task.task_index           = i
            task.task_split_num       = task_num
            task.task_split_value     = op.split_value
            task.tiling_data_position = 0xFFFFFFFF
            task.extra_value_0 = (
                self.route_index_position
                if self.route_index_position is not None
                else 0xFFFFFFFF
            )
            task.extra_value_1 = (
                self.route_weight_position
                if self.route_weight_position is not None
                else 0xFFFFFFFF
            )
            if (
                self.moe_type is AllToAllType.COMBINE
                and self.additional_dependent_event is not None
            ):
                task.extra_value_2 = self.additional_dependent_event
            else:
                task.extra_value_2 = (
                    self.route_completion_event
                    if self.route_completion_event is not None
                    else 0xFFFFFFFF
                )
            task.extra_value_3 = self.moe_type.value
            if self.route_generation_phase or self.route_generation_count != 1:
                task.extra_value_3 |= (
                    self.route_generation_phase << 8
                ) | (self.route_generation_count << 16)
            if self.wait_for_copy_ack:
                task.extra_value_3 |= ROUTE_COPY_ACK_FLAG
            if self.expert_copy_ack:
                task.extra_value_3 |= ROUTE_EXPERT_ACK_FLAG
            task.extra_value_4 = int(self.offsets_in_tokens) | (
                (self.route_top_k or 0) << 8
            )
            cfg.all_tasks[tsv.pre_task_num + i] = task
            cfg.vector_task_indices[tsv.pre_vector_task_num + i] = tsv.pre_task_num + i

        cfg.task_index_num[1] += task_num
        if self.route_completion_event is not None:
            cfg.all_event_num_triggers[self.route_completion_event] = task_num
        if self.advance == "vector":
            advance_tsv_vector(tsv, task_num, event_group_size=self.event_group)
        else:   # "vector_only"
            advance_tsv_vector_only(tsv, task_num)
