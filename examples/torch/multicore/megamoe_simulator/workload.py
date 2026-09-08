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
"""Real placement plans, valid Top-K histograms and exact wave byte ledgers."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[4]
PLANNER_PATH = REPOSITORY / "hyper_parallel/core/multicore/modules/mega_moe/placement.py"
Counts = tuple[tuple[int, ...], ...]


@lru_cache(maxsize=1)
def placement_module() -> Any:
    """Execute this checkout's unmodified pure planner without backend imports.

    Importing the root HyperParallel package initializes a framework. Loading
    this framework-independent source directly keeps offline runs NPU-free;
    the source path and digest are included in every report.
    """
    name = "_megamoe_simulator_production_placement"
    spec = importlib.util.spec_from_file_location(name, PLANNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the production planner from {PLANNER_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Shape:
    """Global model shape, held fixed across rank sweeps unless specified."""

    experts: int = 384
    tokens_per_rank: int = 4096
    top_k: int = 8
    hidden: int = 5120
    intermediate: int = 1792
    element_bytes: int = 2
    capacity_factor: float = 1.5

    def __post_init__(self) -> None:
        for name in ("experts", "tokens_per_rank", "top_k", "hidden", "intermediate", "element_bytes"):
            if (not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool)) or getattr(
                self, name
            ) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.top_k > self.experts:
            raise ValueError("top_k must not exceed experts.")
        if not math.isfinite(self.capacity_factor) or self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be finite and positive.")
        if self.capacity < self.tokens_per_rank * self.top_k:
            raise ValueError("receive capacity must cover at least the local routed rows, as required by MegaMoeSpec.")

    @property
    def capacity(self) -> int:
        """Return rank receive capacity, aligned to the runtime's 128-row split."""
        requested = math.ceil(self.tokens_per_rank * self.top_k * self.capacity_factor)
        return ((requested + 127) // 128) * 128

    @property
    def row_bytes(self) -> int:
        """Return one routed hidden/output row's byte size."""
        return self.hidden * self.element_bytes

    @property
    def weight_elements(self) -> tuple[int, int]:
        """Return W1 [H, 2I] and W2 [I, H] elements for one expert."""
        return 2 * self.hidden * self.intermediate, self.hidden * self.intermediate


def validate_counts(counts: Counts, shape: Shape, ranks: int) -> Counts:
    """Check shape, integer conservation and per-token Top-K feasibility."""
    if shape.experts % ranks:
        raise ValueError(f"experts={shape.experts} must be divisible by ranks={ranks}.")
    rows = tuple(tuple(row) for row in counts)
    if len(rows) != ranks or any(len(row) != shape.experts for row in rows):
        raise ValueError(f"Counts must have shape [{ranks}, {shape.experts}].")
    for row in rows:
        if any(
            (not isinstance(count, int) or isinstance(count, bool)) or not 0 <= count <= shape.tokens_per_rank
            for count in row
        ):
            raise ValueError("Counts must be integers in [0, tokens_per_rank]; Top-K cannot duplicate an expert.")
        if sum(row) != shape.tokens_per_rank * shape.top_k:
            raise ValueError("Every source row must contain tokens_per_rank * top_k routes.")
    return rows


def make_counts(
    shape: Shape,
    ranks: int,
    *,
    routing: str = "hotspot",
    seed: int = 42,
    hot_fraction: float = 0.8,
    alpha: float = 0.1,
) -> Counts:
    """Sample unique experts per token or construct reproducible balanced/hot routes."""
    if not 0 <= hot_fraction <= 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("hot_fraction must be in [0, 1] and alpha must be finite and positive.")
    if routing not in {"balanced", "hotspot", "dirichlet"}:
        raise ValueError(f"Unknown routing distribution {routing!r}.")
    generator = np.random.default_rng(seed)
    probabilities = generator.dirichlet(np.full(shape.experts, alpha)) if routing == "dirichlet" else None
    counts = []
    for rank in range(ranks):
        histogram = np.zeros(shape.experts, dtype=np.int64)
        for offset in range(0, shape.tokens_per_rank, 256):
            size = min(256, shape.tokens_per_rank - offset)
            if probabilities is not None:
                # Gumbel Top-K samples without replacement, including extreme skew.
                scores = generator.gumbel(size=(size, shape.experts))
                scores += np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))
                ids = np.argpartition(scores, -shape.top_k, axis=1)[:, -shape.top_k :]
            else:
                tokens = np.arange(offset, offset + size)
                ids = (
                    (tokens[:, None] + rank * shape.tokens_per_rank) * shape.top_k + np.arange(shape.top_k)[None, :]
                ) % shape.experts
                if routing == "hotspot":
                    ids[tokens < int(shape.tokens_per_rank * hot_fraction)] = np.arange(shape.top_k)
            histogram += np.bincount(ids.reshape(-1), minlength=shape.experts)
        counts.append(tuple(int(value) for value in histogram))
    return validate_counts(tuple(counts), shape, ranks)


def build_plan(counts: Counts, slots: int, capacity: int) -> Any:
    """Run the current implementation's capacity-aware placement heuristic."""
    return placement_module().build_constrained_expert_placement(counts, slots, receive_capacity=capacity)


def wave_matrices(plan: Any, capacity: int) -> list[list[list[int]]]:
    """Return [wave][source][destination] rows using expert-major receive offsets.

    Each physical expert is laid out with source ranks in ascending order,
    matching route.py destination prefixes and SliceWaveSegment in the worker.
    Aggregating within a rank pair preserves data bytes but omits tile ordering.
    """
    result = [[[0] * plan.ep_size for _ in range(plan.ep_size)] for _ in range(plan.wave_count(capacity))]
    for target in range(plan.ep_size):
        destination_offset = 0
        first = target * plan.physical_experts_per_rank
        for expert in range(first, first + plan.physical_experts_per_rank):
            for source, row in enumerate(plan.dispatch_counts):
                remaining = row[expert]
                while remaining:
                    wave, offset = divmod(destination_offset, capacity)
                    take = min(remaining, capacity - offset)
                    result[wave][source][target] += take
                    remaining -= take
                    destination_offset += take
    return result


def count_digest(counts: Counts) -> str:
    """Identify the exact source/expert histogram independently of JSON formatting."""
    return hashlib.sha256(np.asarray(counts, dtype="<i8").tobytes()).hexdigest()
