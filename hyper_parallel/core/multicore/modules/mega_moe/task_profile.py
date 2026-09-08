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
"""Versioned, bounded task sampling ABI shared with runtime/worker_kernel.h."""

from __future__ import annotations

import os
from typing import Any

MARKER = 0x4D545001
MARKER_OFFSET = 3968
BASE_BYTES = 4096
WORKERS = 48
CAPACITY = 256
RECORD_VALUES = 16
ARENA_BYTES = WORKERS * (CAPACITY + 1) * RECORD_VALUES * 8
FIELDS = (
    "task_id", "task_type", "task_index", "wave", "tiling_position", "dependent_event",
    "trigger_event", "start_cycle", "ready_cycle", "compute_end_cycle", "end_cycle",
    "expert_rows", "output_position", "extra_value_3", "sequence", "reserved",
)


def enabled() -> bool:
    """Enable task sampling only inside an explicitly profiled launch."""
    value = os.getenv("HP_MEGA_MOE_TASK_PROFILE", "0")
    if value not in {"0", "1"}:
        raise ValueError("HP_MEGA_MOE_TASK_PROFILE must be 0 or 1.")
    return value == "1"


def decode(values: list[list[list[int]]], num_cores: int) -> dict[str, Any]:
    """Decode per-worker headers and same-clock task samples after synchronization.

    Strided sampling spans the entire queue and all waves. Counts include
    descriptors skipped by wave policy; these have no execution record. The
    timestamps are local device system cycles, never cross-rank wall clocks.
    """
    if len(values) != WORKERS or not 1 <= num_cores <= WORKERS // 2:
        raise ValueError("invalid task profile worker dimensions")
    workers = []
    for kind, base in (("cube", 0), ("vector", WORKERS // 2)):
        for core in range(num_cores):
            rows = values[base + core]
            if len(rows) != CAPACITY + 1 or any(len(row) != RECORD_VALUES for row in rows):
                raise ValueError("invalid task profile record dimensions")
            seen, recorded, stride, start, end = rows[0][:5]
            if not 0 <= recorded <= CAPACITY or seen < recorded or stride < 1 or end < start:
                raise ValueError("invalid task profile header; check the native payload ABI")
            records = [dict(zip(FIELDS, row)) for row in rows[1:recorded + 1]]
            for record in records:
                times = [record[key] for key in ("start_cycle", "ready_cycle", "compute_end_cycle", "end_cycle")]
                if times != sorted(times):
                    raise ValueError("nonmonotonic same-worker task timestamps")
            workers.append({"kind": kind, "core": core, "seen": seen, "recorded": recorded,
                            "stride": stride, "start_cycle": start, "end_cycle": end, "records": records})
    return {"schema_version": 1, "unit": "system_cycle", "capacity_per_worker": CAPACITY,
            "sampling": "deterministic queue stride; wave-skipped descriptors omitted", "workers": workers}
