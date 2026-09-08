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
"""Dependency-driven, full-duplex fluid network simulation in microseconds.

Bandwidth is shared with progressive max-min allocation. Propagation latency
delays completion, but does not occupy a link after serialization. All callers
share one fabric, including weight copies, gradients, tokens and collectives.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Fabric:
    """Explicit rank topology and effective, decimal GB/s bandwidths.

    ``inter_gbps`` is the aggregate bandwidth of EACH node in EACH direction.
    It is not independently available to every rank pair. ``intra_gbps`` is
    per-rank bandwidth; ``intra_node_gbps`` bounds its shared local fabric.
    These values are scenario inputs, not advertised hardware specifications.
    """

    node_by_rank: tuple[int, ...]
    inter_gbps: float = 25.0
    inter_latency_us: float = 10.0
    intra_gbps: float = 150.0
    intra_latency_us: float = 1.0
    intra_node_gbps: float = 600.0
    local_gbps: float = 800.0

    def __post_init__(self) -> None:
        if not self.node_by_rank or any(
            (not isinstance(node, int) or isinstance(node, bool)) or node < 0 for node in self.node_by_rank
        ):
            raise ValueError("node_by_rank must contain nonnegative integer node IDs.")
        for name in ("inter_gbps", "intra_gbps", "intra_node_gbps", "local_gbps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("inter_latency_us", "intra_latency_us"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")

    def path(self, source: int, target: int) -> tuple[dict[str, float], float]:
        """Return byte/us capacity constraints and propagation latency."""
        for rank in (source, target):
            if (not isinstance(rank, int) or isinstance(rank, bool)) or not 0 <= rank < len(self.node_by_rank):
                raise ValueError(f"rank {rank} is outside the fabric.")
        if source == target:
            return {f"local:{source}": self.local_gbps * 1000}, 0.0
        source_node, target_node = self.node_by_rank[source], self.node_by_rank[target]
        if source_node != target_node:
            return {
                f"node-tx:{source_node}": self.inter_gbps * 1000,
                f"node-rx:{target_node}": self.inter_gbps * 1000,
            }, self.inter_latency_us
        return {
            f"intra-tx:{source}": self.intra_gbps * 1000,
            f"intra-rx:{target}": self.intra_gbps * 1000,
            f"intra-fabric:{source_node}": self.intra_node_gbps * 1000,
        }, self.intra_latency_us


@dataclass(frozen=True)
class Task:
    """One transfer or timed local operation in a topologically ordered DAG."""

    name: str
    phase: str
    rank: int
    dependencies: tuple[int, ...] = ()
    duration_us: float = 0.0
    target: int | None = None
    size_bytes: int = 0


@dataclass
class SimulationResult:
    """Virtual timestamps, exact submitted bytes and per-link serialization."""

    starts_us: list[float]
    finishes_us: list[float]
    serialization_ends_us: list[float]
    traffic: dict[str, dict[str, int]]
    resource_bytes: dict[str, int]

    @property
    def makespan_us(self) -> float:
        """Return the last visible completion on the virtual clock."""
        return max(self.finishes_us, default=0.0)


def fair_rates(paths: dict[int, dict[str, float]]) -> dict[int, float]:
    """Allocate max-min fair rates without multiplying shared node capacity."""
    capacities = {resource: limit for path in paths.values() for resource, limit in path.items()}
    remaining = set(paths)
    rates = dict.fromkeys(paths, 0.0)
    while remaining:
        users = Counter(resource for task_id in remaining for resource in paths[task_id])
        increment = min(capacities[resource] / count for resource, count in users.items())
        for task_id in remaining:
            rates[task_id] += increment
        for resource, count in users.items():
            capacities[resource] = max(0.0, capacities[resource] - increment * count)
        saturated = {resource for resource in users if capacities[resource] < 1e-7}
        frozen = {task_id for task_id in remaining if saturated.intersection(paths[task_id])}
        if not frozen:
            raise RuntimeError("Bandwidth allocation made no progress.")
        remaining.difference_update(frozen)
    return rates


class EventSimulator:
    """Run dependencies and shared-link transfers without wall-clock sleeps."""

    def __init__(self, fabric: Fabric, tasks: list[Task]) -> None:
        """Validate the DAG and initialize an independent virtual fabric."""
        self.fabric = fabric
        self.tasks = tasks
        self.result = SimulationResult([0.0] * len(tasks), [0.0] * len(tasks), [0.0] * len(tasks), {}, {})
        self.children: dict[int, list[int]] = defaultdict(list)
        self.unsatisfied = []
        self.ready = []
        self.timers: list[tuple[float, int]] = []
        self.paths: dict[int, dict[str, float]] = {}
        self.latencies: dict[int, float] = {}
        self.remaining: dict[int, float] = {}
        self.clock_us = 0.0
        for task_id, task in enumerate(tasks):
            self._validate_task(task_id, task)
            dependencies = set(task.dependencies)
            self.unsatisfied.append(len(dependencies))
            if not dependencies:
                self.ready.append(task_id)
            for dependency in dependencies:
                self.children[dependency].append(task_id)

    def _validate_task(self, task_id: int, task: Task) -> None:
        if any(
            (not isinstance(dep, int) or isinstance(dep, bool)) or not 0 <= dep < task_id for dep in task.dependencies
        ):
            raise ValueError("Task dependencies must refer to earlier tasks.")
        if not math.isfinite(task.duration_us) or task.duration_us < 0:
            raise ValueError("Task duration must be finite and nonnegative.")
        if (not isinstance(task.size_bytes, int) or isinstance(task.size_bytes, bool)) or task.size_bytes < 0:
            raise ValueError("Task bytes must be a nonnegative integer.")
        self.fabric.path(task.rank, task.rank if task.target is None else task.target)
        if task.target is None and task.size_bytes:
            raise ValueError("A byte transfer needs a target rank.")

    def _start_ready(self) -> None:
        for task_id in self.ready:
            task = self.tasks[task_id]
            self.result.starts_us[task_id] = self.clock_us
            if task.target is None:
                heapq.heappush(self.timers, (self.clock_us + task.duration_us, task_id))
                continue
            path, latency = self.fabric.path(task.rank, task.target)
            self.latencies[task_id] = latency + task.duration_us
            if task.size_bytes:
                self.paths[task_id] = path
                self.remaining[task_id] = float(task.size_bytes)
            else:
                self.result.serialization_ends_us[task_id] = self.clock_us
                heapq.heappush(self.timers, (self.clock_us + latency + task.duration_us, task_id))
            traffic = self.result.traffic.setdefault(task.phase, {"local": 0, "intra": 0, "inter": 0, "messages": 0})
            scope = "inter" if self.fabric.node_by_rank[task.rank] != self.fabric.node_by_rank[task.target] else "intra"
            traffic["local" if task.rank == task.target else scope] += task.size_bytes
            traffic["messages"] += 1
            for resource in path:
                self.result.resource_bytes[resource] = self.result.resource_bytes.get(resource, 0) + task.size_bytes
        self.ready.clear()

    def run(self) -> SimulationResult:
        """Advance to the next serialization, completion or dependency event."""
        completed = 0
        while completed < len(self.tasks):
            self._start_ready()
            # Handle a whole timestamp before reallocating the network.
            while self.timers and self.timers[0][0] <= self.clock_us + 1e-8:
                finish, task_id = heapq.heappop(self.timers)
                self.result.finishes_us[task_id] = finish
                if self.tasks[task_id].target is None:
                    self.result.serialization_ends_us[task_id] = finish
                completed += 1
                for child in self.children[task_id]:
                    self.unsatisfied[child] -= 1
                    if not self.unsatisfied[child]:
                        self.ready.append(child)
            if self.ready:
                continue
            rates = fair_rates(self.paths)
            next_transfer = min((self.remaining[key] / rate for key, rate in rates.items()), default=math.inf)
            next_timer = self.timers[0][0] - self.clock_us if self.timers else math.inf
            delta = min(next_transfer, next_timer)
            if math.isinf(delta):
                if completed != len(self.tasks):
                    raise RuntimeError("Task graph cannot make progress.")
                break
            self.clock_us += max(0.0, delta)
            serialized = []
            for task_id, rate in rates.items():
                self.remaining[task_id] -= rate * delta
                if self.remaining[task_id] <= 1e-5:
                    serialized.append(task_id)
            for task_id in serialized:
                self.result.serialization_ends_us[task_id] = self.clock_us
                heapq.heappush(self.timers, (self.clock_us + self.latencies.pop(task_id), task_id))
                del self.paths[task_id]
                del self.remaining[task_id]
        return self.result
