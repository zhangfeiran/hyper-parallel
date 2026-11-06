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
"""Mark ordinary Push dispatch tasks for batched empty-tile signals."""

import importlib

import torch

from hyper_parallel.core.multicore._loader import NativeComponentUnavailableError
from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC, TaskType


EMPTY_DISPATCH_MARKER = 0x40000000


def mark_empty_dispatch(config: RuntimeConfigC) -> None:
    """Keep all descriptors and thresholds, marking only Push dispatch PUTs."""
    for task_id in config.vector_task_indices[:config.task_index_num[1]]:
        task = config.all_tasks[task_id]
        if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL and task.inputs[3].input_position == 4:
            if task.extra_value_0:
                raise ValueError('Empty-dispatch signal marker conflicts with an existing task policy')
            task.extra_value_0 = EMPTY_DISPATCH_MARKER


def require_empty_dispatch_native() -> None:
    """Reject a payload that cannot interpret the empty-dispatch marker."""
    # The native module imports torch_npu; keep CPU graph construction independent.
    ops = importlib.import_module('hyper_parallel.core.multicore.torch.ops')
    ops._load_native()  # pylint: disable=protected-access
    try:
        if torch.ops.hyper_parallel.mega_moe_empty_dispatch_version() != 1:
            raise RuntimeError('Unsupported empty-dispatch signal protocol')
    except (AttributeError, RuntimeError) as error:
        raise NativeComponentUnavailableError('Empty-dispatch signals require a rebuilt multicore payload') from error
