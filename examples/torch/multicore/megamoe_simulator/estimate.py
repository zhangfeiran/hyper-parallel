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
"""Offline, phase-resolved MegaMoe latency, communication and HBM model.

The rank-stage DAG is a serialization scenario: it waits for all incoming
rank-pair transfers before a rank's compute stage. A hardware kernel pipelines
individual expert tiles. Work/traffic are exact under the selected contracts;
latency is an approximation that must be checked against held-out measurements.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from .accelerated import ArrayEventSimulator
from .cost import (
    Hardware,
    Operation,
    Options,
    gemm,
    gemm_curve_key,
    vector,
    wave_groups,
)
from .memory import memory_ledger
from .model import Costs, Training, TrainingGraph
from .network import Fabric
from .workload import (
    REPOSITORY,
    RUNTIME_CONFIG_PATH,
    Shape,
    build_plan,
    count_digest,
    placement_module,
    runtime_config_module,
    validate_counts,
)


class CostGraph(TrainingGraph):
    """Expand each compute stage into GEMMs, vector traffic and control costs."""

    def __init__(
        self,
        shape: Shape,
        plan: Any,
        hardware: Hardware,
        training: Training,
        options: Options,
    ) -> None:
        """Use the production placement and gradient window schedule."""
        super().__init__(
            shape,
            [plan],
            Costs(route_us=hardware.route_us, planner_us=hardware.planner_us),
            training,
        )
        self.hardware = hardware
        self.options = options
        self.work: dict[int, Operation] = {}
        self.groups = wave_groups(plan, shape.capacity)
        self.wave_index = 0

    def _operation(self, label: str, rank: int, deps: list[int], operation: Operation) -> list[int]:
        task = self._add(label, operation.phase, rank, deps, duration_us=operation.service_us)
        self.work[task] = operation
        return [task]

    def _forward(self, label: str, plan: Any, layer: int, version: int, tails: list[list[int]]) -> list[list[int]]:
        tails = self._collective(label, "route_counts", tails, self.shape.experts * 4, gather=True)
        if plan.replica_slots_per_rank:
            tails = self._local(label, "placement_cpu", tails, self.hardware.planner_us)
        weights = self._prefetch(label, plan, layer, version, tails)
        if plan.replica_transfers:
            pairs = plan.logical_num_experts * plan.physical_num_experts
            scanned = plan.logical_num_experts + plan.physical_num_experts
            duration = (
                self.hardware.copy_slot_us * scanned
                if self.options.linear_copy_runs
                else self.hardware.copy_pair_us * pairs
            )
            tails = self._local(label, "physical_copy_runs_cpu", tails, duration)
        route = self._local(label, "route_prepare", tails, self.hardware.route_us)
        return self._kernel(label, plan, [weights[rank] + route[rank] for rank in range(self.ranks)], backward=False)

    def _expert_transfers(
        self,
        label: str,
        phase: str,
        plan: Any,
        tails: list[list[int]],
        *,
        gradients: bool,
    ) -> list[list[int]]:
        if not gradients:
            return super()._expert_transfers(label, phase, plan, tails, gradients=False)
        # One pure production helper is shared with the real HCCL consumer.
        groups = placement_module()._gradient_transfer_groups(  # pylint: disable=protected-access
            plan, self.options.gradient_receive_slots
        )
        for _, batches in groups:
            for batch in batches:
                participants = {rank for transfer in batch for rank in (transfer.owner_rank, transfer.target_rank)}
                tails = [
                    [
                        self._add(
                            label,
                            "replica_gradient_window",
                            rank,
                            deps,
                            duration_us=self.hardware.replica_gradient_window_us,
                        )
                    ]
                    if rank in participants
                    else deps
                    for rank, deps in enumerate(tails)
                ]
                result = [list(deps) for deps in tails]
                copies = Counter()
                for transfer in batch:
                    source, target = transfer.target_rank, transfer.owner_rank
                    copies[target] += 1
                    for elements in self.shape.weight_elements:
                        task = self._add(
                            label,
                            phase,
                            source,
                            tails[source] + tails[target],
                            target=target,
                            size_bytes=elements * 4,
                        )
                        result[source].append(task)
                        result[target].append(task)
                for owner, count in copies.items():
                    operation = vector(
                        "owner_gradient_add",
                        count * sum(self.shape.weight_elements) * 12,
                        self.hardware,
                    )
                    result[owner] = self._operation(label, owner, result[owner], operation)
                tails = result
        return tails

    def _compute(
        self,
        label: str,
        groups: list[list[int]],
        tails: list[list[int]],
        *,
        backward: bool,
        multi: bool,
    ) -> list[list[int]]:
        gradient_bytes = 4 if self.plans[0].replica_slots_per_rank else self.shape.element_bytes
        result = []
        for rank, deps in enumerate(tails):
            rows = sum(groups[rank])
            size = rows * self.shape.intermediate * self.shape.element_bytes
            if backward:
                phases = [
                    ("act_grad", None),
                    ("w2_grad", None),
                    ("swiglu_grad", size * 7),
                    ("gate_dx", None),
                    ("w1_grad", None),
                ]
                if multi:
                    phases.insert(0, ("activation_recompute", size * 3))
                if multi and self.wave_index > 0 and not self.options.atomic_dw:
                    phases.append(
                        (
                            "wave_dw_accumulate",
                            self.plans[0].physical_experts_per_rank
                            * sum(self.shape.weight_elements)
                            * gradient_bytes
                            * 3,
                        )
                    )
            else:
                phases = [("up_proj", None), ("swiglu", size * 3), ("down_proj", None)]
            for phase, traffic in phases:
                operation = (
                    gemm(phase, groups[rank], self.shape, self.hardware, gradient_bytes)
                    if traffic is None
                    else vector(phase, traffic, self.hardware)
                )
                deps = self._operation(label, rank, deps, operation)
            result.append(deps)
        return result

    def _kernel(self, label: str, plan: Any, tails: list[list[int]], *, backward: bool) -> list[list[int]]:
        matrices = self.matrices[id(plan)]
        tails = self._local(label, "kernel_launch", tails, self.hardware.launch_us)
        tiles = (self.shape.tokens_per_rank + 127) // 128
        directions = 3 if backward else 2
        descriptors = directions * plan.physical_num_experts * tiles
        for wave, matrix in enumerate(matrices):
            self.wave_index = wave
            wave_label = f"{label}/wave{wave}"
            tails = self._local(
                wave_label,
                "descriptor_service",
                tails,
                descriptors / self.hardware.vector_workers * self.hardware.descriptor_us,
            )
            if backward and (len(matrices) > 1 or self.options.inplace_backward):
                incoming, sent = self._token_transfers(wave_label, "backward_hidden_dispatch", matrix, tails)
                tails = [a + b for a, b in zip(incoming, sent)]
                tails = self._collective(wave_label, "hidden_reuse_sync", tails, 4)
            phase = "backward_dispatch" if backward else "forward_dispatch"
            incoming, sent = self._token_transfers(wave_label, phase, matrix, tails)
            computed = self._compute(
                wave_label,
                self.groups[wave],
                incoming,
                backward=backward,
                multi=len(matrices) > 1,
            )
            phase = "backward_combine" if backward else "forward_combine"
            incoming, outgoing = self._token_transfers(wave_label, phase, matrix, computed, reverse=True)
            tails = [incoming[rank] + outgoing[rank] + sent[rank] for rank in range(self.ranks)]
            if wave + 1 < len(matrices):
                tails = self._collective(wave_label, "wave_sync", tails, 4)
        rows = self.shape.tokens_per_rank * self.shape.top_k
        for rank, deps in enumerate(tails):
            # Default forward permute and backward both: output clone/reduce,
            # route score dot products and external dX reduction are local HBM.
            size = rows * self.shape.row_bytes * (5 if backward else 3)
            tails[rank] = self._operation(
                label,
                rank,
                deps,
                vector(
                    "backward_endpoint" if backward else "forward_endpoint",
                    size,
                    self.hardware,
                ),
            )
        return tails

    def build(self) -> list:
        """Include a bandwidth floor for the chosen warmed AdamW contract."""
        if self.options.optimizer != "none":
            per_parameter = 34 if self.options.optimizer == "fp32-main-adamw" else 22
            size = self.training.layers * self.shape.experts // self.ranks * sum(self.shape.weight_elements)
            self.costs = Costs(
                route_us=self.hardware.route_us,
                planner_us=self.hardware.planner_us,
                optimizer_us=self.hardware.memory_us(size * per_parameter),
            )
        return super().build()


def critical_path(tasks: list, result: Any) -> dict[str, Any]:
    """Follow the last finishing dependency in this simulated resource schedule."""
    current = max(range(len(tasks)), key=lambda index: result.finishes_us[index])
    by_phase = Counter()
    path = []
    while True:
        task = tasks[current]
        duration = result.finishes_us[current] - result.starts_us[current]
        by_phase[task.phase] += duration
        path.append({"task": current, "phase": task.phase, "rank": task.rank, "elapsed_us": duration})
        if not task.dependencies:
            break
        current = max(task.dependencies, key=lambda index: result.finishes_us[index])
    return {
        "phase_ms": {phase: value / 1000 for phase, value in by_phase.items()},
        "tasks": list(reversed(path)),
        "scope": "model dependency path; network spans include shared-link contention",
    }


def source_identity() -> dict[str, str]:
    """Identify cost formulas and the runtime sources on which they depend."""
    paths = list((REPOSITORY / "examples/torch/multicore/megamoe_simulator").glob("*.py"))
    paths += [
        REPOSITORY / "hyper_parallel/core/multicore/modules/mega_moe" / f"{name}.py"
        for name in ("placement", "replica", "route", "function", "workspace", "plan", "spec")
    ]
    paths.append(RUNTIME_CONFIG_PATH)
    paths += [
        REPOSITORY / "hyper_parallel/core/multicore/ops/runtime" / name
        for name in ("worker_kernel.h", "runtime_config.hpp")
    ]
    return {str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def predict(
    shape: Shape,
    counts: tuple,
    slots: int,
    fabric: Fabric,
    hardware: Hardware,
    training: Training,
    options: Options,
) -> dict[str, Any]:
    """Predict without loading a framework, touching a device or running kernels."""
    counts = validate_counts(counts, shape, len(fabric.node_by_rank))
    home = build_plan(counts, 0, shape.capacity)
    target = shape.tokens_per_rank * shape.top_k if max(home.destination_loads) > shape.capacity else shape.capacity
    plan = home if max(home.destination_loads) <= shape.capacity else build_plan(counts, slots, target)
    graph = CostGraph(shape, plan, hardware, training, options)
    tasks = graph.build()
    result = ArrayEventSimulator(fabric, tasks).run()
    phase_work: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for task_id, operation in graph.work.items():
        for key in ("flops", "padded_flops", "hbm_bytes", "service_us"):
            phase_work[operation.phase][key] += getattr(operation, key)
    duration_by_rank: dict[str, list[float]] = defaultdict(lambda: [0.0] * plan.ep_size)
    for task_id, task in enumerate(tasks):
        duration_by_rank[task.phase][task.rank] += result.finishes_us[task_id] - result.starts_us[task_id]
    fwd_descriptors = (
        3 * plan.physical_num_experts * ((shape.tokens_per_rank + 127) // 128)
        + 2 * plan.physical_experts_per_rank * 24
        + 48
    )
    step_ends = [max(result.finishes_us[task] for task in ends) / 1000 for ends in graph.step_ends]
    step_durations = [end - start for start, end in zip([0.0] + step_ends[:-1], step_ends)]
    expert_rows = [rows for wave in graph.groups for rank in wave for rows in rank if rows]
    curve_coverage = {}
    for phase in ("up_proj", "down_proj", "act_grad", "gate_dx", "w1_grad", "w2_grad"):
        key = gemm_curve_key(phase, shape, 4 if plan.replica_slots_per_rank else shape.element_bytes)
        domain = hardware.provenance.get("curve_domain", {}).get(key)
        curve_coverage[phase] = {
            "dimension_and_dtype_match": key in hardware.gemm_row_scales,
            "rows_min": min(expert_rows, default=0),
            "rows_max": max(expert_rows, default=0),
            "rows_outside_calibration": sum(
                not domain or rows < domain["rows_min"] or rows > domain["rows_max"] for rows in expert_rows
            ),
        }
    return {
        "schema_version": 1,
        "source_identity": source_identity(),
        "critical_path": critical_path(tasks, result),
        "calibration_coverage": {
            "network": "scenario inputs; no multi-node calibration",
            "rank_count_in_calibration": plan.ep_size in hardware.provenance.get("ranks", []),
            "gemm_dimensions_in_calibration": any(
                (entry["hidden"], entry["intermediate"]) == (shape.hidden, shape.intermediate)
                for entry in hardware.provenance.get("shapes", [])
            ),
            "status": hardware.provenance.get("status", "unverified"),
            "timing_interval": "no statistical interval; use held-out error and bandwidth sensitivity sweeps",
            "gemm_curve_coverage": curve_coverage,
        },
        "prediction_kind": "rank-stage scenario; not measured latency",
        "shape": asdict(shape),
        "training": asdict(training),
        "options": asdict(options),
        "hardware": asdict(hardware),
        "fabric": asdict(fabric),
        "counts_sha256": count_digest(counts),
        "slots": slots,
        "executed_slots": plan.replica_slots_per_rank,
        "copies": len(plan.replica_transfers),
        "wave_count": plan.wave_count(shape.capacity),
        "destination_loads": list(plan.destination_loads),
        "latency_ms": result.makespan_us / 1000,
        "step_ms": step_durations,
        "step_end_ms": step_ends,
        "phase_work_all_ranks": dict(phase_work),
        "phase_rank_max_ms_nonadditive": {name: max(values) / 1000 for name, values in duration_by_rank.items()},
        "traffic_bytes": result.traffic,
        "cache_events": graph.cache_events,
        "memory": memory_ledger(shape, plan, training, options, configured_slots=slots),
        "descriptor_audit": {
            "forward_task_lower_bound": fwd_descriptors,
            "legacy_task_limit": 25600,
            "allocation": "graph-sized task and queue arrays; uint32 byte offsets",
            "event_capacity": runtime_config_module().mega_moe_event_capacity(
                shape.experts + len(counts) * slots, len(counts),
            ),
            "note": (
                "run the meta plan audit for exact buildability and static HBM; "
                "latency needs ABI-matched calibration"
            ),
        },
        "limitations": [
            "rank-stage model omits expert-tile overlap and detailed FIFO head blocking",
            "GEMM/HBM effective rates require calibration within the shape/tiling domain",
            "bulk network model omits individual ACK polling; collectives approximate HCCL selection",
            (
                "replica window startup is calibrated separately; "
                "extrapolation to different peers/topologies is unverified"
            ),
            "planner_us is an explicit cache miss/hit assumption, not an NPU D2H measurement",
        ],
    }
