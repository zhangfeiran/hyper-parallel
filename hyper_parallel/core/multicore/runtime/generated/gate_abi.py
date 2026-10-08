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

"""Generated wire definitions; regenerate with backends.schema."""

import ctypes
from enum import IntEnum
from typing import Any


class DynamicType(IntEnum):
    """Family wire enum from the shared schema."""

    DYNAMIC_DSV3_MOE = 101
    DYNAMIC_EMPTY = 0

class EventType(IntEnum):
    """Family wire enum from the shared schema."""

    EVENT_EMPTY = 900
    EVENT_END_OF_TASK_GRAPH = 910
    EVENT_INVALID = 999
    EVENT_LAUNCH_DEPENDENT_TASKS = 903
    EVENT_LAUNCH_MASSIVE_TASKS = 902
    EVENT_LAUNCH_TASKS = 901
    EVENT_TERMINATION = 911

class TaskAiCoreType(IntEnum):
    """Family wire enum from the shared schema."""

    TASK_AICORE_CUBE = 1
    TASK_AICORE_INVALID = 0
    TASK_AICORE_MIX = 3
    TASK_AICORE_VECTOR = 2

class TaskType(IntEnum):
    """Family wire enum from the shared schema."""

    TASK_TERMINATE = 0
    TASK_BEGIN_TASK_GRAPH = 10
    TASK_ADD_CUSTOM = 101
    TASK_SWI_GLU = 102
    TASK_MATMUL = 103
    TASK_GROUPED_MATMUL = 104
    TASK_SHMEM_PUT_MEM_SIGNAL = 105
    TASK_SWI_GLU_GRAD = 106
    TASK_MEGA_GATE_ROUTE = 107
    TASK_GATE_SOFTPLUS = 108
    TASK_GATE_SQRT = 109
    TASK_GATE_ADD_BIAS = 110
    TASK_GATE_TOPK = 111
    TASK_GATE_GATHER = 112
    TASK_GATE_REDUCE_SUM = 113
    TASK_GATE_ADD_EPSILON = 114
    TASK_GATE_DIV = 115
    TASK_GATE_MUL_SCALE = 116
    TASK_GATE_CAST_INDEX = 117
    TASK_GATE_GRAD_MULS_SCALE = 118
    TASK_GATE_GRAD_NEG = 119
    TASK_GATE_GRAD_DIV_SELECTED = 120
    TASK_GATE_GRAD_DIV_SELECTED_RATIO = 121
    TASK_GATE_GRAD_MUL_CROSS = 122
    TASK_GATE_GRAD_DIV_DIRECT = 123
    TASK_GATE_GRAD_REDUCE_SUM = 124
    TASK_GATE_GRAD_BROADCAST_ROW_SUM = 125
    TASK_GATE_GRAD_ADD_SELECTED = 126
    TASK_GATE_GRAD_ZEROS = 127
    TASK_GATE_GRAD_LINEAR_INDEX = 128
    TASK_GATE_GRAD_SCATTER_SELECTED = 129
    TASK_GATE_GRAD_MULS_DOUBLE_SCORE = 130
    TASK_GATE_GRAD_DIV_SQRT = 131
    TASK_GATE_GRAD_SOFTPLUS = 132
    TASK_GATE_GRAD_ADD_DIRECT = 133
    TASK_GATE_GRAD_BROADCAST_DENOMINATOR = 134


class TensorDescC(ctypes.Structure):
    """Family wire structure from the shared schema."""

    _fields_ = [
        ("tensor_type", ctypes.c_uint32),
        ("num_dims", ctypes.c_uint32),
        ("dim", ctypes.c_uint32 * 4),
        ("stride", ctypes.c_uint32 * 4),
        ("data_type", ctypes.c_uint32),
        ("input_position", ctypes.c_uint32),
        ("base_ptr_offset", ctypes.c_uint32),
        ("transpose_flag", ctypes.c_uint32),
        ("dynamic_shape", ctypes.c_uint32),
        ("dynamic_dim", ctypes.c_uint32),
    ]


class TaskDescC(ctypes.Structure):
    """Family wire structure from the shared schema."""

    _fields_ = [
        ("task_type", ctypes.c_uint32),
        ("task_aicore_type", ctypes.c_uint32),
        ("num_inputs", ctypes.c_uint32),
        ("num_outputs", ctypes.c_uint32),
        ("trigger_event", ctypes.c_uint32),
        ("dependent_event", ctypes.c_uint32),
        ("inputs", TensorDescC * 4),
        ("outputs", TensorDescC * 4),
        ("tiling_data_position", ctypes.c_uint32),
        ("tiling_data_offset", ctypes.c_uint32),
        ("task_index", ctypes.c_uint32),
        ("task_split_num", ctypes.c_uint32),
        ("task_split_value", ctypes.c_uint32),
        ("extra_value_0", ctypes.c_uint32),
        ("extra_value_1", ctypes.c_uint32),
        ("extra_value_2", ctypes.c_uint32),
        ("profile_desc_id", ctypes.c_uint32),
        ("profile_owner_id", ctypes.c_uint32),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Retain invalid profiling sentinels for descriptors without explicit ownership.

        Args:
            *args: Positional ctypes field values.
            **kwargs: Named ctypes field values.
        """
        has_desc = len(args) >= len(self._fields_) - 1 or "profile_desc_id" in kwargs
        has_owner = len(args) >= len(self._fields_) or "profile_owner_id" in kwargs
        super().__init__(*args, **kwargs)
        if not has_desc:
            self.profile_desc_id = 0xFFFFFFFF
        if not has_owner:
            self.profile_owner_id = 0xFFFFFFFF


class EventDescC(ctypes.Structure):
    """Family wire structure from the shared schema."""

    _fields_ = [
        ("event_type", ctypes.c_uint32),
        ("num_triggers", ctypes.c_uint32),
        ("first_task_id", ctypes.c_uint32),
        ("last_task_id", ctypes.c_uint32),
    ]


class DynamicDataC(ctypes.Structure):
    """Family wire structure from the shared schema."""

    _fields_ = [
        ("dynamic_type", ctypes.c_uint32),
        ("dynamic_input_position", ctypes.c_uint32),
        ("dynamic_group_size", ctypes.c_uint32),
        ("dynamic_max_seq_len", ctypes.c_int32),
    ]


class RuntimeConfigC(ctypes.Structure):
    """Family wire structure from the shared schema."""

    _fields_ = [
        ("task_num", ctypes.c_uint32),
        ("num_workers", ctypes.c_uint32),
        ("task_capacity", ctypes.c_uint32),
        ("event_capacity", ctypes.c_uint32),
        ("ready_event", ctypes.c_uint32),
        ("cycle_profiling_enabled", ctypes.c_uint32),
        ("aic_profile_record_capacity", ctypes.c_uint32),
        ("aiv_profile_record_capacity", ctypes.c_uint32),
        ("_padding", ctypes.c_uint32 * 8),
    ]
