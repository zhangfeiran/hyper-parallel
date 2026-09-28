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
"""Measured execution costs for bounded, deterministic replica quota refinement."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from functools import cached_property
import math

from .capacity import ExpertReplicaConfig, _integer
from .plan import ExpertExecutionPlan


def _nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True)
class ExpertReplicaCostModel:
    """Offline-calibrated millisecond costs for one shape and execution backend.

    Args:
        calibration_version: Explicit schema version; version 2 uses exposed communication
            and separate forward/backward rank maxima. Old calibrations must be regenerated.
        backend: Native HCCL execution or MegaMoe push/pull.
        hidden_size: Calibrated hidden dimension.
        intermediate_size: Calibrated SwiGLU intermediate dimension.
        ep_size: Calibrated expert-parallel group size.
        transport: Calibrated weight/gradient transport mode; native requires p2p.
        forward_ms: Increasing (rows, milliseconds) knots for one expert forward.
        backward_ms: Increasing knots for one expert backward with FP32 partials.
        weight_ms: Full weight-copy cost per replica, charged in each direction.
        gradient_ms: Exposed gradient-return cost per replica after local compute.
        token_ms_per_row: Dispatch/combine cost per remote send or receive row.
        schedule: Validated schedule identity, inferred from backend/transport if omitted.
        gradient_dtype: Weight partial precision; only float32 calibrations are accepted.
        minimum_gain_ms: Minimum predicted worst-rank gain, including measurement
            uncertainty and the budget for extra host decision work.

    Tables start at (0, 0), use nondecreasing times and cover the intended expert
    row counts. Out-of-range plans retain the original policy. Every EP rank
    must use identical calibration data. This is an approximate cost model,
    not a prediction guarantee or runtime calibration/weight cache.
    """

    calibration_version: int
    backend: str
    hidden_size: int
    intermediate_size: int
    ep_size: int
    transport: str
    forward_ms: tuple[tuple[int, float], ...]
    backward_ms: tuple[tuple[int, float], ...]
    weight_ms: float
    gradient_ms: float
    token_ms_per_row: float = 0.0
    minimum_gain_ms: float = 0.0
    schedule: str | None = None
    gradient_dtype: str = "float32"

    def __post_init__(self) -> None:
        """Freeze measured inputs and reject inconsistent calibration contracts."""
        if isinstance(self.calibration_version, bool) or self.calibration_version != 2:
            raise ValueError("Regenerate replica calibration with calibration_version=2")
        expected_schedule = ("native_grouped_v1" if self.backend == "native" else
                             "fixed_queue_w13_first_v1" if self.transport == "shmem_signal_kernel_gradient" else
                             "fixed_queue_v1")
        if self.schedule is not None and self.schedule != expected_schedule:
            raise ValueError("Replica calibration schedule does not match the execution")
        object.__setattr__(self, "schedule", expected_schedule)
        if self.gradient_dtype != "float32":
            raise ValueError("Replica cost calibration requires float32 weight partials")
        if self.backend not in ("native", "push", "pull") or not isinstance(self.transport, str) or not self.transport:
            raise ValueError("Cost model requires native, push or pull and a transport mode")
        if self.backend == "native" and self.transport != "p2p":
            raise ValueError("Native cost models require HCCL p2p")
        for name in ("hidden_size", "intermediate_size", "ep_size"):
            _integer(getattr(self, name), name)
        for name in ("forward_ms", "backward_ms"):
            table = tuple(tuple(point) for point in getattr(self, name))
            if len(table) < 2 or any(len(point) != 2 for point in table) or table[0] != (0, 0):
                raise ValueError(f"{name} must start at (0, 0) and contain at least two knots")
            previous_rows, previous_ms = -1, 0.0
            for rows, duration in table:
                _integer(rows, "calibration rows", 0)
                _nonnegative(duration, "calibration milliseconds")
                if rows <= previous_rows or duration < previous_ms:
                    raise ValueError("Calibration rows must increase and times must not decrease")
                previous_rows, previous_ms = rows, duration
            object.__setattr__(self, name, table)
        for name in ("weight_ms", "gradient_ms", "token_ms_per_row", "minimum_gain_ms"):
            _nonnegative(getattr(self, name), name)

    def validate_execution(self, backend: str, hidden: int, intermediate: int, ep_size: int, transport: str) -> None:
        """Reject accidental reuse on another shape, backend or transport."""
        if (backend, hidden, intermediate, ep_size, transport) != (
                self.backend, self.hidden_size, self.intermediate_size, self.ep_size, self.transport):
            raise ValueError("Replica cost calibration does not match the execution shape/backend/transport")

    @cached_property
    def row_knots(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return interpolation keys without rebuilding them for each candidate."""
        return tuple(point[0] for point in self.forward_ms), tuple(point[0] for point in self.backward_ms)

    def compute_ms(self, rows: int, *, backward: bool = False) -> float:
        """Interpolate within the measured range; never extrapolate silently."""
        _integer(rows, "expert rows", 0)
        table = self.backward_ms if backward else self.forward_ms
        index = bisect_left(self.row_knots[int(backward)], rows)
        if index == len(table):
            raise ValueError("Expert rows exceed the calibrated cost range")
        if table[index][0] == rows:
            return table[index][1]
        left_rows, left_ms = table[index - 1]
        right_rows, right_ms = table[index]
        return left_ms + (right_ms - left_ms) * (rows - left_rows) / (right_rows - left_rows)

    def estimate_ms(self, plan: ExpertExecutionPlan) -> float:
        """Score separate phase bottlenecks, excluding invariant optimizer work.

        This conservative heuristic is not a prediction of an implicit global
        barrier. It does not assume that the same rank is slowest in F and B,
        or credit overlap without a measured schedule-prefix window.
        """
        copies = [{expert: rows for expert, rows in zip(slots, counts) if expert >= 0}
                  for slots, counts in zip(plan.slot_to_logical, plan.destination_counts)]
        return _phase_score(_rank_costs(self, copies, plan.logical_counts, plan.config))


def _rank_costs(model: ExpertReplicaCostModel, copies: list[dict[int, int]],
                counts: tuple[tuple[int, ...], ...], config: ExpertReplicaConfig,
                durations: dict[int, tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    """Return exposed F/B costs, charging owner fan-out and guest fan-in separately."""
    if durations is None:
        durations = {}
    incoming, outgoing = [0] * config.ep_size, [0] * config.ep_size
    for target, experts in enumerate(copies):
        for expert, rows in experts.items():
            owner = expert // config.home_experts
            if rows and owner != target:
                incoming[target] += 1
                outgoing[owner] += 1
    costs = []
    for rank, experts in enumerate(copies):
        forward, backward = [0.0, 0.0], [0.0, 0.0]
        local = 0
        for expert, rows in experts.items():
            guest = int(expert // config.home_experts != rank)
            if rows not in durations:
                durations[rows] = model.compute_ms(rows), model.compute_ms(rows, backward=True)
            forward[guest] += durations[rows][0]
            backward[guest] += durations[rows][1]
            local += min(counts[rank][expert], rows)
        # Without measured link concurrency or queue-prefix windows, neither
        # full duplex nor whole-home compute can hide these transfers safely.
        transfers = incoming[rank] + outgoing[rank]
        ready = transfers * model.weight_ms
        remote_rows = sum(counts[rank]) + sum(experts.values()) - 2 * local
        token_phase = remote_rows * model.token_ms_per_row / 2
        costs.append((sum(forward) + ready + token_phase,
                      sum(backward) + ready + transfers * model.gradient_ms + token_phase))
    return costs


def _phase_score(costs: list[tuple[float, float]]) -> float:
    """Use the same conservative phase objective for reporting and refinement."""
    return max(forward for forward, _ in costs) + max(backward for _, backward in costs)


def refine_replica_quotas(copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...],
                          config: ExpertReplicaConfig, limit: int, model: ExpertReplicaCostModel) -> None:
    """Compare bounded candidates on existing edges; accept only strict worst-rank gains."""
    if model.ep_size != config.ep_size:
        raise ValueError("Replica cost model EP size does not match the plan")
    largest = max(sum(row[expert] for row in counts) for expert in range(config.num_experts))
    if largest > min(model.forward_ms[-1][0], model.backward_ms[-1][0]):
        return
    home = config.home_experts
    edges = sorted((target, expert) for target, row in enumerate(copies)
                   for expert, rows in row.items() if expert // home != target and rows)
    knots = sorted(set(model.row_knots[0] + model.row_knots[1]))
    durations = {}
    # Two deterministic sweeps bound Python work; candidate count depends on
    # calibration knots and existing edges, not the number of routed tokens.
    for _ in range(2):
        changed = False
        for target, expert in edges:
            current = copies[target].get(expert, 0)
            if not current:
                continue
            owner = expert // home
            total = current + copies[owner][expert]
            loads = [sum(row.values()) for row in copies]
            lower = max(0, total + loads[owner] - copies[owner][expert] - limit)
            upper = min(total, limit - loads[target] + current)
            quotas = {current, lower, upper, total // 2}
            quotas.update(value for point in knots for value in (point, total - point))
            best, best_cost = current, _phase_score(_rank_costs(model, copies, counts, config, durations))
            for quota in sorted(value for value in quotas if lower <= value <= upper and value != current):
                copies[target][expert], copies[owner][expert] = quota, total - quota
                cost = _phase_score(_rank_costs(model, copies, counts, config, durations))
                if cost + model.minimum_gain_ms < best_cost:
                    best, best_cost = quota, cost
            copies[target][expert], copies[owner][expert] = best, total - best
            if not best:
                del copies[target][expert]
            changed |= best != current
        if not changed:
            return
