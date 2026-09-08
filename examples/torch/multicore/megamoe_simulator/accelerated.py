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
"""Vectorize the existing fluid network arithmetic for offline model sweeps.

The graph, dependency semantics, rate policy, tolerances, and traffic accounting
are unchanged. The original simulator is retained as the equivalence oracle.
"""

import heapq
import math

import numpy as np

from examples.torch.multicore.megamoe_simulator.network import (
    EventSimulator,
    Fabric,
    SimulationResult,
    Task,
)


class ArrayEventSimulator(EventSimulator):
    """Advance the same events using arrays for active-flow arithmetic."""

    def __init__(self, fabric: Fabric, tasks: list[Task]) -> None:
        """Precompute immutable task-to-resource incidence."""
        super().__init__(fabric, tasks)
        resources = {}
        task_paths = []
        for task in tasks:
            path, _ = fabric.path(task.rank, task.rank if task.target is None else task.target)
            task_paths.append(path)
            for resource, limit in path.items():
                resources.setdefault(resource, limit)
        identifiers = {name: index for index, name in enumerate(resources)}
        self.limits = np.array(list(resources.values()) + [math.inf])
        self.indices = np.full((len(tasks), 3), len(resources), dtype=np.int64)
        self.valid = np.zeros_like(self.indices, dtype=bool)
        for task_id, path in enumerate(task_paths):
            for column, resource in enumerate(path):
                self.indices[task_id, column] = identifiers[resource]
                self.valid[task_id, column] = True
        self.group_indices, self.task_groups = np.unique(self.indices, axis=0, return_inverse=True)
        self.group_valid = self.group_indices != len(resources)
        self.bytes_left = np.array([task.size_bytes for task in tasks], dtype=np.float64)

    def _rates(self, identifiers):
        task_groups = self.task_groups[identifiers]
        counts = np.bincount(task_groups, minlength=len(self.group_indices))
        groups = np.flatnonzero(counts)
        incidence = self.group_indices[groups]
        valid = self.group_valid[groups]
        multiplicity = counts[groups]
        capacities = self.limits.copy()
        rates = np.zeros(len(groups), dtype=np.float64)
        remaining = np.ones(len(groups), dtype=bool)
        while remaining.any():
            selected = incidence[remaining]
            selected_valid = valid[remaining]
            weights = np.broadcast_to(multiplicity[remaining, None], selected.shape)
            users = np.bincount(selected[selected_valid], weights=weights[selected_valid], minlength=len(capacities))
            used = users > 0
            increment = np.min(capacities[used] / users[used])
            rates[remaining] += increment
            capacities[used] = np.maximum(0, capacities[used] - increment * users[used])
            saturated = (capacities < 1e-7) & used
            frozen = remaining & np.any(saturated[incidence] & valid, axis=1)
            if not frozen.any():
                raise RuntimeError("Bandwidth allocation made no progress")
            remaining[frozen] = False
        group_rates = np.zeros(len(self.group_indices))
        group_rates[groups] = rates
        return group_rates[task_groups]

    def run(self) -> SimulationResult:
        """Run identical visible-completion and bandwidth-release events."""
        completed = 0
        while completed < len(self.tasks):
            self._start_ready()
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
            identifiers = np.fromiter(self.paths, dtype=np.int64, count=len(self.paths))
            rates = self._rates(identifiers) if len(identifiers) else np.empty(0)
            next_transfer = np.min(self.bytes_left[identifiers] / rates) if len(rates) else math.inf
            next_timer = self.timers[0][0] - self.clock_us if self.timers else math.inf
            delta = min(next_transfer, next_timer)
            if math.isinf(delta):
                if completed != len(self.tasks):
                    raise RuntimeError("Task graph cannot make progress")
                break
            self.clock_us += max(0.0, delta)
            self.bytes_left[identifiers] -= rates * delta
            serialized = identifiers[self.bytes_left[identifiers] <= 1e-5]
            for identifier in serialized:
                task_id = int(identifier)
                self.result.serialization_ends_us[task_id] = self.clock_us
                heapq.heappush(self.timers, (self.clock_us + self.latencies.pop(task_id), task_id))
                del self.paths[task_id]
                del self.remaining[task_id]
        return self.result
