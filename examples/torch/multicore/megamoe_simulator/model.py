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
"""Rank-stage model of shared replica weights and checkpointed MoE training.

Data volumes and placement come from production contracts. Rank-stage compute,
pair-aggregated token transfers and recursive-doubling collectives are explicit
approximations; this is not an instruction simulator of the AIV/Cube scheduler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .network import Task
from .workload import Shape, wave_matrices


@dataclass(frozen=True)
class Costs:
    """User-calibratable effective local costs; defaults are scenario assumptions."""

    effective_tflops: float = 100.0
    forward_fixed_us: float = 5.0
    backward_fixed_us: float = 10.0
    route_us: float = 5.0
    planner_us: float = 0.0
    optimizer_us: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.effective_tflops) or self.effective_tflops <= 0:
            raise ValueError("effective_tflops must be finite and positive.")
        for name in ("forward_fixed_us", "backward_fixed_us", "route_us", "planner_us", "optimizer_us"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")

    def compute_us(self, shape: Shape, rows: int, backward: bool) -> float:
        """Use GEMM FLOPs plus a calibrated local fixed cost per wave."""
        # Forward W1+W2: 6*H*I FLOPs/row. Backward dX+dW: 12*H*I.
        factor = 12 if backward else 6
        fixed = self.backward_fixed_us if backward else self.forward_fixed_us
        return fixed + rows * factor * shape.hidden * shape.intermediate / (self.effective_tflops * 1e6)


@dataclass(frozen=True)
class Training:
    """Independent layers: all forwards, reverse backwards, then parameter update."""

    layers: int = 2
    steps: int = 2
    checkpoint: bool = True
    pool: str = "shared"
    update_every: int = 1

    def __post_init__(self) -> None:
        if any(
            (not isinstance(value, int) or isinstance(value, bool)) or value < 1 for value in (self.layers, self.steps)
        ):
            raise ValueError("layers and steps must be positive integers.")
        if (not isinstance(self.update_every, int) or isinstance(self.update_every, bool)) or self.update_every < 0:
            raise ValueError("update_every must be a nonnegative integer; zero keeps weights unchanged.")
        if self.pool not in {"shared", "per-layer"}:
            raise ValueError("pool must be shared or per-layer.")


class TrainingGraph:
    """Build one event graph from exact plans and the shared-pool lifetime."""

    def __init__(self, shape: Shape, plans: list[Any], costs: Costs, training: Training) -> None:
        """Bind the production plans, shape and calibrated or assumed costs."""
        if not plans or any(plan.ep_size != plans[0].ep_size for plan in plans):
            raise ValueError("plans must be nonempty and have the same rank count.")
        self.shape, self.plans, self.costs, self.training = shape, plans, costs, training
        self.ranks = plans[0].ep_size
        if self.ranks & (self.ranks - 1):
            raise ValueError("The collective model requires a power-of-two rank count.")
        if any(plan.logical_num_experts != shape.experts for plan in plans):
            raise ValueError("Plan expert count does not match the model shape.")
        self.tasks: list[Task] = []
        self.cache: dict[int, tuple[Any, ...]] = {}
        self.cache_events: list[dict[str, Any]] = []
        self.step_ends: list[list[int]] = []
        self.matrices = {id(plan): wave_matrices(plan, shape.capacity) for plan in plans}

    def _add(
        self,
        name: str,
        phase: str,
        rank: int,
        dependencies: list[int] | tuple[int, ...],
        *,
        duration_us: float = 0.0,
        target: int | None = None,
        size_bytes: int = 0,
    ) -> int:
        task_id = len(self.tasks)
        self.tasks.append(Task(name, phase, rank, tuple(sorted(set(dependencies))), duration_us, target, size_bytes))
        return task_id

    def _local(self, label: str, phase: str, tails: list[list[int]], duration_us: float) -> list[list[int]]:
        return [[self._add(label, phase, rank, deps, duration_us=duration_us)] for rank, deps in enumerate(tails)]

    def _collective(
        self, label: str, phase: str, tails: list[list[int]], size: int, *, gather: bool = False
    ) -> list[list[int]]:
        # A documented topology model, not a claim about HCCL's selected algorithm.
        for round_index in range(self.ranks.bit_length() - 1):
            distance = 1 << round_index
            operations = [
                self._add(
                    f"{label}/round{round_index}",
                    phase,
                    rank,
                    tails[rank] + tails[rank ^ distance],
                    target=rank ^ distance,
                    size_bytes=size * (distance if gather else 1),
                )
                for rank in range(self.ranks)
            ]
            tails = [[operations[rank], operations[rank ^ distance]] for rank in range(self.ranks)]
        return tails

    def _prefetch(self, label: str, plan: Any, layer: int, version: int, tails: list[list[int]]) -> list[list[int]]:
        if not plan.replica_transfers:
            return tails
        tails = self._collective(label, "reuse_agreement", tails, 4)
        pool_id = 0 if self.training.pool == "shared" else layer
        key = (layer, version, plan.slot_to_logical_expert)
        reused = self.cache.get(pool_id) == key
        self.cache_events.append(
            {
                "call": label,
                "layer": layer,
                "version": version,
                "reused": reused,
                "copies": 0 if reused else len(plan.replica_transfers),
            }
        )
        if reused:
            return tails
        self.cache[pool_id] = key
        return self._expert_transfers(label, "weight_prefetch", plan, tails, gradients=False)

    def _expert_transfers(
        self, label: str, phase: str, plan: Any, tails: list[list[int]], *, gradients: bool
    ) -> list[list[int]]:
        result = [list(deps) for deps in tails]
        for transfer in plan.replica_transfers:
            source, target = transfer.owner_rank, transfer.target_rank
            if gradients:
                source, target = target, source
            for index, elements in enumerate(self.shape.weight_elements, 1):
                task = self._add(
                    f"{label}/expert{transfer.logical_expert}/w{index}",
                    phase,
                    source,
                    tails[source] + tails[target],
                    target=target,
                    size_bytes=elements * (4 if gradients else self.shape.element_bytes),
                )
                result[source].append(task)
                result[target].append(task)
        return result

    def _token_transfers(
        self, label: str, phase: str, matrix: list[list[int]], tails: list[list[int]], *, reverse: bool = False
    ) -> tuple[list[list[int]], list[list[int]]]:
        received = [list(deps) for deps in tails]
        sent = [list(deps) for deps in tails]
        for origin, row in enumerate(matrix):
            for destination, count in enumerate(row):
                if not count:
                    continue
                source, target = (destination, origin) if reverse else (origin, destination)
                task = self._add(
                    label, phase, source, tails[source], target=target, size_bytes=count * self.shape.row_bytes
                )
                received[target].append(task)
                sent[source].append(task)
        return received, sent

    def _wave(
        self, label: str, matrix: list[list[int]], tails: list[list[int]], *, backward: bool, multi_wave: bool
    ) -> list[list[int]]:
        if backward and multi_wave:
            received, sent = self._token_transfers(label, "backward_hidden_dispatch", matrix, tails)
            tails = [incoming + outgoing for incoming, outgoing in zip(received, sent)]
            # Current multi-wave backward protects hidden readers before dY can reuse staging.
            tails = self._collective(label, "hidden_reuse_sync", tails, 4)
        phase = "backward_dispatch" if backward else "forward_dispatch"
        received, sent = self._token_transfers(label, phase, matrix, tails)
        computed = []
        for rank, dependencies in enumerate(received):
            rows = sum(source[rank] for source in matrix)
            task = self._add(
                label,
                "backward_compute" if backward else "forward_compute",
                rank,
                dependencies,
                duration_us=self.costs.compute_us(self.shape, rows, backward),
            )
            computed.append([task])
        phase = "backward_combine" if backward else "forward_combine"
        incoming, outgoing = self._token_transfers(label, phase, matrix, computed, reverse=True)
        return [incoming[rank] + outgoing[rank] + sent[rank] for rank in range(self.ranks)]

    def _kernel(self, label: str, plan: Any, tails: list[list[int]], *, backward: bool) -> list[list[int]]:
        matrices = self.matrices[id(plan)]
        for wave, matrix in enumerate(matrices):
            tails = self._wave(f"{label}/wave{wave}", matrix, tails, backward=backward, multi_wave=len(matrices) > 1)
            if wave + 1 < len(matrices):
                tails = self._collective(label, "wave_sync", tails, 4)
        return tails

    def _forward(self, label: str, plan: Any, layer: int, version: int, tails: list[list[int]]) -> list[list[int]]:
        tails = self._collective(label, "route_counts", tails, self.shape.experts * 4, gather=True)
        if plan.replica_slots_per_rank:
            tails = self._local(label, "placement_cpu", tails, self.costs.planner_us)
        weights = self._prefetch(label, plan, layer, version, tails)
        route = self._local(label, "route_prepare", tails, self.costs.route_us)
        return self._kernel(label, plan, [weights[rank] + route[rank] for rank in range(self.ranks)], backward=False)

    def build(self) -> list[Task]:
        """Include checkpoint replay, backward reloads, FP32 dW and version invalidation."""
        if self.tasks:
            raise RuntimeError("A training graph can only be built once.")
        tails: list[list[int]] = [[] for _ in range(self.ranks)]
        for step in range(self.training.steps):
            version = step // self.training.update_every if self.training.update_every else 0
            layer_plans = [
                self.plans[(step * self.training.layers + layer) % len(self.plans)]
                for layer in range(self.training.layers)
            ]
            for layer, plan in enumerate(layer_plans):
                tails = self._forward(f"step{step}/layer{layer}/forward", plan, layer, version, tails)
            for layer in reversed(range(self.training.layers)):
                plan = layer_plans[layer]
                label = f"step{step}/layer{layer}"
                if self.training.checkpoint:
                    tails = self._forward(f"{label}/recompute", plan, layer, version, tails)
                tails = self._prefetch(f"{label}/backward", plan, layer, version, tails)
                tails = self._kernel(f"{label}/backward", plan, tails, backward=True)
                tails = self._expert_transfers(label, "replica_gradient", plan, tails, gradients=True)
            if self.training.update_every and (step + 1) % self.training.update_every == 0:
                tails = self._local(f"step{step}/optimizer", "optimizer", tails, self.costs.optimizer_us)
            # Preserve rank skew within a step; the measurement boundary is global.
            all_ends = [task for deps in tails for task in deps]
            tails = [[self._add(f"step{step}/end", "step_end", rank, all_ends)] for rank in range(self.ranks)]
            self.step_ends.append([deps[0] for deps in tails])
        return self.tasks
