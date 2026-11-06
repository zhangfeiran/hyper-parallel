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
"""Reorder W1 weight-gradient tasks without moving expert storage or communication."""

import importlib

import torch

from hyper_parallel.core.multicore._loader import NativeComponentUnavailableError
from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC, TaskSplitValue
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph

__all__ = []


def reverse_w1_gradient_queue(config: RuntimeConfigC, graph: ComputeGraph, topology: TaskSplitValue) -> None:
    """Reverse the W1 expert groups at their existing Cube queue positions.

    Args:
        config: Ordinary backward runtime without physical guest experts.
        graph: Propagated operators in descriptor construction order.
        topology: The topology used to construct the runtime.

    Note:
        Descriptor identities, events, per-core lane assignments and all other
        task positions are retained. The plan cache applies this once during
        preparation; there is no per-forward queue rewrite or device readback.
    """
    home = topology.single_rank_expert_num
    start = 0
    for operation in graph.topological_sort():
        if operation.name == 'w1_grad':
            if home < 1 or operation.task_num % home:
                raise ValueError('W1 queue permutation requires complete expert task groups')
            lanes = operation.task_num // home
            for index in range(config.task_index_num[0]):
                task = config.cube_task_indices[index]
                if start <= task < start + operation.task_num:
                    expert, lane = divmod(task - start, lanes)
                    config.cube_task_indices[index] = start + (home - 1 - expert) * lanes + lane
            return
        start += operation.task_num


_DEVICE_POLICIES = {'device_w1_asc': 1, 'device_w1_desc': 2, 'device_w1_asc_join': 3, 'device_late_auto': 4,
                    'device_lane_rotate': 5, 'device_w1_asc_join_once': 6, 'device_w1_asc_join_rotate': 7}
_DEVICE_LANE_POLICY = _DEVICE_POLICIES['device_lane_rotate']


def mark_device_gradient_queue(config: RuntimeConfigC, graph: ComputeGraph, mode: str) -> None:
    """Annotate optional native scheduling while retaining the original host queue.

    Args:
        config: Ordinary backward runtime without physical guest experts.
        graph: Propagated graph in descriptor construction order.
        mode: One of the private device policy names.
    """
    policy = _DEVICE_POLICIES[mode]
    spans = {}
    start = 0
    for operation in graph.topological_sort():
        if operation.name in ('w2_grad', 'gate_grad', 'w1_grad'):
            spans[operation.name] = (start, operation.task_num)
        start += operation.task_num
    if policy in (_DEVICE_LANE_POLICY, 7):
        stages = ('w2_grad', 'w1_grad') if policy == _DEVICE_LANE_POLICY else ('w2_grad',)
        for name in stages:
            offset, count = spans[name]
            for index in range(offset, offset + count):
                config.all_tasks[index].extra_value_0 = 0x5A000000 | (_DEVICE_LANE_POLICY << 20)
        if policy == _DEVICE_LANE_POLICY:
            return
    gate_start, gate_count = spans['gate_grad']
    event = config.all_tasks[gate_start + gate_count - 1].trigger_event
    if event >= 1 << 20:
        raise ValueError('Device gradient policies require a completion event below 2**20')
    for name, (offset, count) in spans.items():
        if name == 'w2_grad' or (name == 'gate_grad' and policy != 4):
            continue
        other = spans['w1_grad' if name == 'gate_grad' else 'gate_grad'][0]
        for index in range(offset, offset + count):
            task = config.all_tasks[index]
            if task.inputs[2].dim[0] > 64:
                raise ValueError('Device gradient policies support at most 64 local experts')
            task.extra_value_0 = 0x5A000000 | (policy << 20) | event
            task.tiling_data_offset = other


def require_device_gradient_native(mode: str = 'device_w1_asc') -> None:
    """Fail before launch when an older native payload cannot interpret policy markers."""
    adapter = importlib.import_module('hyper_parallel.core.multicore.torch.ops')
    adapter._load_native()  # pylint: disable=protected-access
    try:
        required = {'device_lane_rotate': 2, 'device_w1_asc_join_once': 3, 'device_w1_asc_join_rotate': 3}.get(mode, 1)
        if torch.ops.hyper_parallel.mega_moe_schedule_version() < required:
            raise RuntimeError('Unsupported native gradient schedule version')
    except (AttributeError, RuntimeError) as error:
        raise NativeComponentUnavailableError(
            'Device gradient scheduling requires a matching rebuilt multicore native payload') from error
