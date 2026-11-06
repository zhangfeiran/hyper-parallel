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
"""Generate the measured two-expert window from the current task graph."""

from __future__ import annotations
import os
from typing import Any

from hyper_parallel.core.multicore.shmem._runtime import configured_udma_qp_count

from .config import TaskType
from .planned_queue import install_worker_queues


def _reuse_post_dispatch_math(
    cfg: Any, vectors: list[list[int]], combine_start: int, first_stage: tuple[int, int],
) -> None:
    """Borrow drained dispatch lanes for all quantization (1) or the input stage (2)."""
    enabled = os.environ.get("PANGU_MX_REUSE_DISPATCH_WORKERS", "0")
    if enabled not in ("0", "1", "2"):
        raise ValueError("PANGU_MX_REUSE_DISPATCH_WORKERS must be exactly 0, 1 or 2")
    if enabled == "0":
        return
    for lane in range(32):
        math, communication = vectors[2 * lane], vectors[2 * lane + 1]
        if len(math) % 2:
            raise ValueError("Post-dispatch reuse requires paired logical quantization workers")
        boundary = next((index for index, task in enumerate(communication)
                         if task >= combine_start or cfg.all_tasks[task].task_type == 0), len(communication))
        # Preserving the full communication subsequence keeps every peer/QP owner.
        # Borrowing a lane before dispatch drains could introduce remote wait cycles.
        borrowed = math[1::2]
        if enabled == "2":
            start, count = first_stage
            borrowed = [task for task in borrowed if start <= task < start + count]
        moved = set(borrowed)
        vectors[2 * lane] = [task for task in math if task not in moved]
        vectors[2 * lane + 1] = communication[:boundary] + borrowed + communication[boundary:]


def _validate_qp_owners(cfg: Any, spec: Any, vectors: list[list[int]], qps: int) -> None:
    """Reject concurrent producers of one peer/QP without changing task order."""
    owners = {}
    for worker in range(1, 64, 2):
        qp = worker // (64 // qps)
        for tid in vectors[worker]:
            task = cfg.all_tasks[tid]
            if task.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                continue
            peer = task.task_index // (task.task_split_num // spec.ep_size)
            if peer == spec.rank_id:
                continue
            key = peer, qp
            owner = owners.setdefault(key, worker)
            if owner != worker:
                raise ValueError(f"QP owner collision: peer={peer}, qp={qp}, workers={owner}/{worker}")


def _window_stages(graph: Any, experts: int) -> dict[str, tuple[int, int]]:
    """Locate expert-partitioned stages without changing their topological task IDs."""
    stages = {}
    offset = 0
    for op in graph.topological_sort():
        if op.name != "terminate" and op.task_num % experts:
            raise ValueError(f"Window stage {op.name} cannot partition {op.task_num} tasks across experts")
        stages[op.name] = (offset, op.task_num)
        offset += op.task_num
    return stages


def configure_window_queue(cfg: Any, graph: Any, spec: Any, *, forward: bool) -> None:
    """Preserve tasks, dependencies and QP ownership without shape-specific files.

    Args:
        cfg: Mutable host runtime image being assembled.
        graph: Completed task graph with operator-local task ranges.
        spec: Validated physical expert topology and current capacity.
    """
    stages = _window_stages(graph, spec.local_experts)

    def tasks(stage: str, expert: int, lane: int) -> list[int]:
        """Select the task IDs owned by one expert and physical compute lane.

        Args:
            stage: Stage name selected for this expert window.
            expert: Local physical expert slot.
            lane: Physical math worker lane.
        """
        start, count = stages[stage]
        width = count // spec.local_experts
        return list(range(start + expert * width + lane, start + (expert + 1) * width, 32))

    cube_stages = ("up", "down") if forward else ("dw2", "da", "dw1", "dx")
    inputs = ("quant_x", "swiglu_quant") if forward else ("quant_dy", "swiglu_grad_quant")
    front = 1 if forward else 2
    vectors = [[] for _ in range(64)]
    cubes = [[] for _ in range(32)]
    communication = list(cfg.vector_task_indices[: cfg.task_index_num[3]])
    for lane in range(32):
        for stage in cube_stages[:front]:
            cubes[lane].extend(tasks(stage, 0, lane))
        for expert in range(1, spec.local_experts):
            for stage in cube_stages[:front]:
                cubes[lane].extend(tasks(stage, expert, lane))
            for stage in cube_stages[front:]:
                cubes[lane].extend(tasks(stage, expert - 1, lane))
        for stage in cube_stages[front:]:
            cubes[lane].extend(tasks(stage, spec.local_experts - 1, lane))
        vectors[2 * lane + 1] = communication[lane::32]
        vectors[2 * lane].extend(tasks(inputs[0], 0, lane))
        for expert in range(spec.local_experts):
            if expert + 1 < spec.local_experts:
                vectors[2 * lane].extend(tasks(inputs[0], expert + 1, lane))
            vectors[2 * lane].extend(tasks(inputs[1], expert, lane))
    _reuse_post_dispatch_math(cfg, vectors, stages["combine"][0], stages[inputs[0]])
    qps = configured_udma_qp_count()
    if qps < 32:
        _validate_qp_owners(cfg, spec, vectors, qps)
    install_worker_queues(cfg, cubes, vectors)
