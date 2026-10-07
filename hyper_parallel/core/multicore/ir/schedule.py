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
"""Schedule selection metadata; device schedule lowering is a separate stage."""

from __future__ import annotations

from dataclasses import dataclass

from hyper_parallel.core.multicore.ir.program import SourceSpan


@dataclass(frozen=True)
class WorkerPipeline:
    """Ordered stages per row-owning worker, pending family-specific lowering."""

    axis: int = 0
    workers: str = "aiv"

    def __post_init__(self):
        if self.axis != 0 or self.workers != "aiv":
            raise ValueError("V0 WorkerPipeline supports row-axis AIV workers only")


@dataclass(frozen=True)
class TaskDAG:
    """An explicit legacy queue policy, pending family-specific lowering."""

    policy: str

    def __post_init__(self):
        if not isinstance(self.policy, str) or not self.policy:
            raise ValueError("TaskDAG requires an explicit policy name")


@dataclass(frozen=True)
class HardwareSpec:
    """Available AIV workers supplied by the caller; no device probing at compile time."""

    available_aiv_workers: int = 48

    def __post_init__(self):
        if type(self.available_aiv_workers) not in (int,) or not 0 < self.available_aiv_workers < 1 << 32:
            raise ValueError("available_aiv_workers must be a positive uint32 integer")


@dataclass(frozen=True)
class RowPartition:
    """Native launched worker's fixed row interval, including empty tail workers."""

    worker_id: int
    first_row: int
    row_count: int


@dataclass(frozen=True)
class PipelineStage:
    """A logical stage with source provenance, before family-local numeric binding."""

    logical_name: str
    display_name: str
    source_spans: tuple[SourceSpan, ...]
    reason: str = "native_primitive"


@dataclass(frozen=True)
class PipelineScheduleIR:
    """Broadcast descriptor sequence and fixed worker row ownership."""

    stages: tuple[PipelineStage, ...]
    partitions: tuple[RowPartition, ...]
    rows_per_worker: int
    worker_slot_capacity: int = 48
    execution_mode: str = "broadcast_aiv_pipeline"

    def simulate(self) -> tuple[tuple[int, int, int, int], ...]:
        """Enumerate each launched worker's ordered stages without global event waits."""
        return tuple(
            (partition.worker_id, stage, partition.first_row, partition.row_count)
            for partition in self.partitions
            for stage in range(len(self.stages))
        )
