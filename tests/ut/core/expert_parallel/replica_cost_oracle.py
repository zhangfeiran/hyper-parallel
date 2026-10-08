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
"""Frozen full-rank cost/refinement oracle from the pre-optimization policy."""

from __future__ import annotations

from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig


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


def _guest_edges(copies, home):
    """Enumerate the reference policy's sorted active guest edges."""
    return sorted((target, expert) for target, row in enumerate(copies)
                  for expert, rows in row.items() if expert // home != target and rows)


def _quotas(current, lower, upper, total, knots):
    """Enumerate the reference policy's bounded, sorted quota candidates."""
    quotas = {current, lower, upper, total // 2}
    quotas.update(value for point in knots for value in (point, total - point))
    return sorted(value for value in quotas if lower <= value <= upper and value != current)


def _refine_replica_quotas(copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...],
                          config: ExpertReplicaConfig, limit: int, model: ExpertReplicaCostModel) -> None:
    """Compare bounded candidates on existing edges; accept only strict worst-rank gains."""
    if model.ep_size != config.ep_size:
        raise ValueError("Replica cost model EP size does not match the plan")
    largest = max(sum(row[expert] for row in counts) for expert in range(config.num_experts))
    if largest > min(model.forward_ms[-1][0], model.backward_ms[-1][0]):
        return
    home = config.home_experts
    edges = _guest_edges(copies, home)
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
            best, best_cost = current, _phase_score(_rank_costs(model, copies, counts, config, durations))
            for quota in _quotas(current, lower, upper, total, knots):
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
