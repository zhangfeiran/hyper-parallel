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
"""An optional, rank-local W2 phase join for large received-token workloads."""

import ctypes

from hyper_parallel.core.multicore.scheduler.config import (
    ATOMIC_ADD_VALUE_LEN,
    EVENT_INVALID_ID,
    INVALID_PROFILE_DESC_ID,
    INVALID_PROFILE_OWNER_ID,
    RuntimeConfigC,
    TaskSplitValue,
    TaskType,
    validate_runtime_config,
)
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config

from ..spec import MegaMoeSpec
from .storage import can_reuse_backward_dispatch


_STAGES = (6, 8, 12, 18)
HOTSPOT_MIN_ROWS = 131072
HOTSPOT_LOAD_FACTOR = 4


def supports_hotspot_schedule(spec: MegaMoeSpec) -> bool:
    """Keep unmeasured expert shapes on the original interleaved schedule.

    Args:
        spec: Bound shape and hardware specification.

    Returns:
        Whether the shape matches the validated 24-Cube large-expert schedule.
    """
    return (spec.hidden_size, spec.intermediate_size, spec.local_experts, spec.num_cube_cores) == (5120, 1792, 6, 24)


def _copy_with_task_space(original: RuntimeConfigC, extra: int) -> RuntimeConfigC:
    """Copy arrays separately because their offsets change with task capacity."""
    result = allocate_runtime_config(original.task_capacity + extra, original.event_capacity,
                                     original.dynamic_data.dynamic_group_size)
    task_capacity = result.task_capacity
    ctypes.memmove(ctypes.addressof(result), ctypes.addressof(original), ctypes.sizeof(RuntimeConfigC))
    result.task_capacity = task_capacity
    for name, _ in type(original)._fields_:
        source, destination = getattr(original, name), getattr(result, name)
        ctypes.memmove(ctypes.addressof(destination), ctypes.addressof(source), ctypes.sizeof(source))
    result.__dict__.update(original.__dict__)
    return result


def _unused_join_events(original: RuntimeConfigC) -> tuple[int, int]:
    """Leave atomic-write padding after every existing event and handshake."""
    reserved = {int(original.ready_event), int(original.completion_event)}
    reserved.update(index for index, value in enumerate(original.all_event_num_triggers) if value)
    for task in original.all_tasks:
        reserved.add(int(task.trigger_event))
        if task.dependent_event != EVENT_INVALID_ID:
            reserved.add(int(task.dependent_event))
    width = ATOMIC_ADD_VALUE_LEN
    arrival = (max(reserved) + 2 * width - 1) // width * width
    if arrival + 2 * width > original.event_capacity:
        raise ValueError("Insufficient unused event space for the backward W2 phase join.")
    return arrival, arrival + width


def build_hotspot_config(
    original: RuntimeConfigC, topology: TaskSplitValue, num_cube_cores: int,
) -> RuntimeConfigC:
    """Stage GMMs and join W2 without changing original tasks or dependencies.

    Args:
        original: Validated backward graph with one GMM task per expert and Cube.
        topology: Rank-local topology used to validate the resulting image.
        num_cube_cores: Participating Cube workers.

    Returns:
        Independent runtime image with one arrival and wait per Cube after W2.

    Raises:
        ValueError: If the graph is incompatible or has insufficient event space.
    """
    experts = topology.single_rank_expert_num
    cube = original.cube_task_indices[:original.task_index_num[0]]
    keyed = {(original.all_tasks[index].outputs[0].input_position,
              original.all_tasks[index].task_index): index for index in cube}
    expected = {(stage, index) for stage in _STAGES for index in range(experts * num_cube_cores)}
    if set(keyed) != expected or len(cube) != len(expected):
        raise ValueError("Backward W2 phase join requires exactly four per-expert GMM stages.")
    if not can_reuse_backward_dispatch(original, experts, num_cube_cores):
        raise ValueError("Backward W2 phase join requires a receive-reuse-safe original schedule.")
    arrival, sink = _unused_join_events(original)
    result = _copy_with_task_space(original, 2 * num_cube_cores)
    result.all_event_num_triggers[arrival] = num_cube_cores
    result.all_event_num_triggers[sink] = num_cube_cores
    next_task = int(original.task_capacity)
    queue = [keyed[(_STAGES[0], index)] for index in range(experts * num_cube_cores)]
    for dependency, trigger in ((EVENT_INVALID_ID, arrival), (arrival, sink)):
        for core in range(num_cube_cores):
            task = result.all_tasks[next_task]
            task.task_type = TaskType.TASK_BEGIN_TASK_GRAPH
            task.task_aicore_type = original.all_tasks[cube[0]].task_aicore_type
            task.task_index = core
            task.dependent_event = dependency
            task.trigger_event = trigger
            task.profile_desc_id = INVALID_PROFILE_DESC_ID
            task.profile_owner_id = INVALID_PROFILE_OWNER_ID
            queue.append(next_task)
            next_task += 1
    queue.extend(keyed[(stage, index)] for stage in _STAGES[1:] for index in range(experts * num_cube_cores))
    result.task_num = next_task
    result.task_index_num[0] = len(queue)
    for position, task_id in enumerate(queue):
        result.cube_task_indices[position] = task_id
    validate_runtime_config(result, topology, num_cube_cores)
    if not can_reuse_backward_dispatch(result, experts, num_cube_cores):
        raise ValueError("Backward W2 phase join invalidated receive-buffer reuse.")
    return result
