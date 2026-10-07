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
"""Serialize the isolated legacy Gate ABI without importing any NPU family."""

from __future__ import annotations

import struct

from hyper_parallel.core.multicore.runtime.abi import FamilyABI, PipelineTemplate

_INVALID = 0xFFFFFFFF
_PROFILE_STAGE_BASE = 0x30000
_TASK_CAPACITY = 16
_EVENT_CAPACITY = 1024
_WORKER_SLOTS = 48


def _put_fields(image, offset, layout, values):
    for name, value in values.items():
        struct.pack_into("<I", image, offset + layout.offset(name), value)


def gate_runtime_image(abi: FamilyABI, template: PipelineTemplate, *, profiled: bool = False) -> bytes:
    """Emit a Gate descriptor image matching the pinned original builder.

    Args:
        abi: Explicit Gate family ABI; a MoE/MHC schema is rejected.
        template: Pinned forward or backward stage sequence.
        profiled: Enable per-worker cycle records in the serialized header.
    """
    if abi.family != "gate" or template != abi.pipeline(template.name):
        raise ValueError("Gate serialization requires a matching Gate family/template")
    if type(profiled) not in (bool,):
        raise ValueError("profiled must be a boolean")
    header = abi.structure("RuntimeConfigC")
    task = abi.structure("TaskDescC")
    event = abi.structure("EventDescC")
    dynamic = abi.structure("DynamicDataC")
    task_offset = header.size + _EVENT_CAPACITY * 4
    queue_count_offset = task_offset + _TASK_CAPACITY * task.size + _EVENT_CAPACITY * event.size
    vector_queue_offset = queue_count_offset + 16 + _TASK_CAPACITY * 4
    total_bytes = queue_count_offset + 16 + _TASK_CAPACITY * 12 + dynamic.size + 512 * 8 + 8 * 4
    image = bytearray(total_bytes)
    _put_fields(
        image,
        0,
        header,
        {
            "task_num": len(template.logical_stages),
            "num_workers": _WORKER_SLOTS,
            "task_capacity": _TASK_CAPACITY,
            "event_capacity": _EVENT_CAPACITY,
            "cycle_profiling_enabled": int(profiled),
            "aiv_profile_record_capacity": 16,
        },
    )
    struct.pack_into("<4i", image, queue_count_offset, 0, len(template.logical_stages), 0, 0)
    for index, logical_name in enumerate(template.logical_stages):
        _put_fields(
            image,
            task_offset + index * task.size,
            task,
            {
                "task_type": abi.task_id(logical_name),
                "task_aicore_type": 2,
                "trigger_event": _INVALID,
                "dependent_event": _INVALID,
                "tiling_data_offset": index,
                "task_split_num": _WORKER_SLOTS,
                "profile_desc_id": _PROFILE_STAGE_BASE + index,
                "profile_owner_id": _INVALID,
            },
        )
        struct.pack_into("<i", image, vector_queue_offset + index * 4, index)
    return bytes(image)
