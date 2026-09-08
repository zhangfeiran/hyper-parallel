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
"""Source-based operation work and calibratable roofline service costs.

One microsecond is the time unit, decimal GB/s and TFLOP/s are rates. Defaults
are scenario assumptions, not device specifications. Service excludes waiting;
DAG dependencies and the shared network account for waiting separately.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any

from .workload import Shape


def aligned(value: int, quantum: int) -> int:
    """Round a nonnegative dimension up to the kernel tile quantum."""
    return (value + quantum - 1) // quantum * quantum


@dataclass(frozen=True)
class Hardware:
    """Portable calibration, with explicit source identity and covered shapes."""

    effective_tflops: float = 100.0
    hbm_gbps: float = 800.0
    launch_us: float = 5.0
    descriptor_us: float = 0.1
    vector_workers: int = 24
    route_us: float = 30.0
    planner_us: float = 0.0
    copy_pair_us: float = 0.0
    copy_slot_us: float = 0.0
    replica_gradient_window_us: float = 0.0
    gmm_fixed_us: float = 2.0
    phase_scales: dict[str, float] = field(default_factory=dict)
    gemm_row_scales: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=lambda: {"status": "assumed; no calibration"})

    def __post_init__(self) -> None:
        if not isinstance(self.vector_workers, int) or isinstance(self.vector_workers, bool) or self.vector_workers < 1:
            raise ValueError("vector_workers must be a positive integer")
        for name, value in asdict(self).items():
            if isinstance(value, dict):
                continue
            if not math.isfinite(value) or value < 0 or (name in {"effective_tflops", "hbm_gbps"} and value == 0):
                raise ValueError(f"{name} must be finite and nonnegative (rates must be positive)")
        if any(not math.isfinite(value) or value <= 0 for value in self.phase_scales.values()):
            raise ValueError("phase scales must be finite and positive")
        for points in self.gemm_row_scales.values():
            if not points or any(
                len(point) != 2 or any(not math.isfinite(value) or value <= 0 for value in point) for point in points
            ):
                raise ValueError("GEMM curves require positive finite row/scale pairs")
            if any(left[0] >= right[0] for left, right in pairwise(points)):
                raise ValueError("GEMM curve rows must increase strictly")

    def memory_us(self, size: int) -> float:
        """Return streaming service for explicitly counted HBM bytes."""
        return size / (self.hbm_gbps * 1000)


@dataclass(frozen=True)
class Options:
    """Implementation switches, all uniform across the EP group."""

    inplace_backward: bool = False
    atomic_dw: bool = False
    gradient_receive_slots: int = 8
    linear_copy_runs: bool = True
    optimizer: str = "fp32-main-adamw"
    reserve_gib: float = 8.0
    device_gib: float = 64.0
    external_gib: float = 0.0
    previous_scratch_bytes: int = 0
    previous_replica_slots: int = 0
    plan_storage_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.gradient_receive_slots, int)
            or isinstance(self.gradient_receive_slots, bool)
            or self.gradient_receive_slots < 1
        ):
            raise ValueError("gradient_receive_slots must be a positive integer")
        if self.optimizer not in {"none", "bf16-adamw", "fp32-main-adamw"}:
            raise ValueError("unknown optimizer memory contract")
        for name in (
            "reserve_gib",
            "device_gib",
            "external_gib",
            "previous_scratch_bytes",
            "previous_replica_slots",
            "plan_storage_bytes",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.device_gib == 0:
            raise ValueError("device_gib must be positive")
        if not isinstance(self.previous_replica_slots, int) or isinstance(self.previous_replica_slots, bool):
            raise TypeError("previous_replica_slots must be an integer")


@dataclass(frozen=True)
class Operation:
    """Useful arithmetic, tile-rounded arithmetic, compulsory HBM and service."""

    phase: str
    flops: int
    padded_flops: int
    hbm_bytes: int
    service_us: float


def gemm_curve_key(phase: str, shape: Shape, gradient_bytes: int) -> str:
    """Keep empirical row curves within their matrix dimensions and output dtype."""
    output_bytes = gradient_bytes if phase in {"w1_grad", "w2_grad"} else shape.element_bytes
    return f"{phase}:{shape.hidden}:{shape.intermediate}:{shape.element_bytes}:{output_bytes}"


def _gemm_scale(phase: str, rows: int, shape: Shape, hardware: Hardware, gradient_bytes: int) -> float:
    points = hardware.gemm_row_scales.get(gemm_curve_key(phase, shape, gradient_bytes))
    if not points:
        return hardware.phase_scales.get(phase, 1.0)
    if rows <= points[0][0]:
        return points[0][1]
    for left, right in pairwise(points):
        if rows <= right[0]:
            fraction = math.log(rows / left[0]) / math.log(right[0] / left[0])
            return left[1] + fraction * (right[1] - left[1])
    return points[-1][1]


def gemm(phase: str, groups: list[int], shape: Shape, hardware: Hardware, gradient_bytes: int) -> Operation:
    """Sum expert GEMMs using M/N/K tile rounding and a per-expert fixed cost.

    Tile quanta (16, 16, 16) give a conservative arithmetic floor, not the
    implementation's complete L0/L1 tiling. Effective rates incorporate reuse
    and utilization only within the calibration's recorded shape domain.
    """
    dimensions = {
        "up_proj": lambda rows: (rows, 2 * shape.intermediate, shape.hidden),
        "down_proj": lambda rows: (rows, shape.hidden, shape.intermediate),
        "act_grad": lambda rows: (rows, shape.intermediate, shape.hidden),
        "gate_dx": lambda rows: (rows, shape.hidden, 2 * shape.intermediate),
        "w1_grad": lambda rows: (shape.hidden, 2 * shape.intermediate, rows),
        "w2_grad": lambda rows: (shape.intermediate, shape.hidden, rows),
    }
    if phase not in dimensions:
        raise ValueError(f"unknown GEMM phase {phase}")
    flops = padded = traffic = 0
    duration = 0.0
    for rows in groups:
        if rows < 0:
            raise ValueError("expert rows cannot be negative")
        if not rows:
            continue
        m, n, k = dimensions[phase](rows)
        useful = 2 * m * n * k
        tiled = 2 * aligned(m, 16) * aligned(n, 16) * aligned(k, 16)
        output_bytes = gradient_bytes if phase in {"w1_grad", "w2_grad"} else shape.element_bytes
        size = (m * k + k * n) * shape.element_bytes + m * n * output_bytes
        flops += useful
        padded += tiled
        traffic += size
        service = hardware.gmm_fixed_us + max(tiled / (hardware.effective_tflops * 1e6), hardware.memory_us(size))
        duration += service * _gemm_scale(phase, rows, shape, hardware, gradient_bytes)
    return Operation(phase, flops, padded, traffic, duration)


def vector(phase: str, size: int, hardware: Hardware) -> Operation:
    """Price explicit streaming traffic, allowing a measured per-phase scale."""
    if size < 0:
        raise ValueError("HBM traffic cannot be negative")
    return Operation(
        phase,
        0,
        0,
        size,
        hardware.memory_us(size) * hardware.phase_scales.get(phase, 1.0),
    )


def wave_groups(plan: Any, capacity: int) -> list[list[list[int]]]:
    """Intersect each physical expert's exact rank prefix with each wave."""
    result = []
    groups = [sum(row[index] for row in plan.dispatch_counts) for index in range(plan.physical_num_experts)]
    for wave in range(plan.wave_count(capacity)):
        per_rank = []
        for rank in range(plan.ep_size):
            prefix = 0
            rows = []
            first = rank * plan.physical_experts_per_rank
            for count in groups[first : first + plan.physical_experts_per_rank]:
                rows.append(
                    max(
                        0,
                        min(prefix + count, (wave + 1) * capacity) - max(prefix, wave * capacity),
                    )
                )
                prefix += count
            per_rank.append(rows)
        result.append(per_rank)
    return result
