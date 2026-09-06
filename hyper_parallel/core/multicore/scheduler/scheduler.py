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
"""
Task-queue reordering passes applied after the main fill loop.

revise_task_queue          — RATR: rank-aware tile reorder for dispatch/combine.
revise_gmm_task_queue_bwd  — Backward GMM1/GMM4 expert interleave in cube_task_indices.
"""
from __future__ import annotations

from hyper_parallel.core.multicore.scheduler.config import (
    RuntimeConfigC,
    TaskSplitValue,
)


def revise_task_queue(cfg: RuntimeConfigC, tsv: TaskSplitValue,
                      dispatch_task_num: int, swiglu_task_num: int,
                      combine_task_num: int | None = None,
                      combine_start: int | None = None) -> None:
    """Reorder dispatch and combine queues based on the local rank.

    Args:
        cfg: Runtime configuration whose Vector queue is reordered.
        tsv: Scheduler dimensions including the local rank id.
        dispatch_task_num: Number of tasks in the dispatch segment.
        swiglu_task_num: Number of tasks in the SwiGLU segment.
        combine_task_num: Optional number of tasks in the combine segment.
        combine_start: Optional first combine task offset. Graphs with route
            work between dispatch and SwiGLU must provide this explicitly.
    """
    temp = list(cfg.vector_task_indices)

    single_rank_expert_num = tsv.single_rank_expert_num
    single_expert_task_num = dispatch_task_num // tsv.all_expert_num
    ep                     = tsv.ep
    rank_id                = tsv.rank_id
    single_rank_task_num   = dispatch_task_num // tsv.ep

    ep_rank = [(i + rank_id) % ep for i in range(ep)]

    # ── dispatch segment ─────────────────────────────────────────────────────
    start = 0
    index = 0
    for j in range(single_rank_expert_num):
        j_v = j * single_expert_task_num
        for k in range(single_expert_task_num):
            k_v = j_v + k
            for i in ep_rank:
                i_v = k_v + i * single_rank_task_num
                cfg.vector_task_indices[start + index] = temp[start + i_v]
                index += 1

    # ── combine segment ──────────────────────────────────────────────────────
    if combine_task_num is None:
        combine_task_num = dispatch_task_num
    single_expert_task_num = combine_task_num // tsv.all_expert_num
    single_rank_task_num = combine_task_num // tsv.ep
    start = (
        dispatch_task_num + swiglu_task_num
        if combine_start is None
        else combine_start
    )
    index = 0
    for j in range(single_rank_expert_num):
        j_v = j * single_expert_task_num
        for k in range(single_expert_task_num):
            k_v = j_v + k
            for i in ep_rank:
                i_v = k_v + i * single_rank_task_num
                cfg.vector_task_indices[start + index] = temp[start + i_v]
                index += 1


def revise_rank_aware_alltoall_segment(
    cfg: RuntimeConfigC,
    tsv: TaskSplitValue,
    *,
    start: int,
    task_num: int,
) -> None:
    """Interleave one AllToAll segment by destination rank.

    Unlike :func:`revise_task_queue`, this helper does not assume that the
    first vector segment is dispatch or infer where Combine starts. This is
    needed by the multi-wave backward graph, which has vector work before and
    between its routed communication segments.

    Args:
        cfg: Runtime configuration whose Vector queue is reordered.
        tsv: Scheduler dimensions including the local rank id.
        start: First queue offset of the AllToAll segment.
        task_num: Number of tasks in the segment.
    """
    if task_num <= 0 or task_num % tsv.all_expert_num:
        raise ValueError(
            "AllToAll task count must be a positive multiple of the expert "
            f"count, got task_num={task_num}, experts={tsv.all_expert_num}."
        )
    single_expert_task_num = task_num // tsv.all_expert_num
    single_rank_task_num = task_num // tsv.ep
    segment = [
        cfg.vector_task_indices[start + index]
        for index in range(task_num)
    ]
    destination_order = [
        (distance + tsv.rank_id) % tsv.ep
        for distance in range(tsv.ep)
    ]
    output_index = 0
    for local_expert in range(tsv.single_rank_expert_num):
        expert_offset = local_expert * single_expert_task_num
        for tile in range(single_expert_task_num):
            within_rank = expert_offset + tile
            for destination in destination_order:
                source_index = within_rank + destination * single_rank_task_num
                cfg.vector_task_indices[start + output_index] = segment[source_index]
                output_index += 1


def revise_gmm_task_queue_bwd(cfg: RuntimeConfigC, tsv: TaskSplitValue,
                               act_grad_task_num: int,
                               num_cube_cores: int = 24) -> None:
    """
    Backward-only: interleave w2_grad and act_grad experts in cube_task_indices.

    Result pattern: [w2_grad exp0, act_grad exp0, w2_grad exp1, act_grad exp1, ...]
    act_grad start offset = 0; w2_grad start offset = act_grad_task_num.

    Args:
        cfg: Runtime configuration whose Cube queue is reordered.
        tsv: Scheduler dimensions for the local expert count.
        act_grad_task_num: Number of activation-gradient tasks.
        num_cube_cores: Number of Cube workers per expert.
    """
    temp          = list(cfg.cube_task_indices)
    expert_single = tsv.single_rank_expert_num
    changes_num   = 2   # two streams: w2_grad (index=1) and act_grad (index=0)

    for i in range(expert_single * changes_num):
        index = 1 - (i % changes_num)   # alternates: 1, 0, 1, 0, ...
        m     = i // changes_num         # expert block: 0, 0, 1, 1, ...
        for j in range(num_cube_cores):
            dst = i * num_cube_cores + j
            src = index * act_grad_task_num + m * num_cube_cores + j
            cfg.cube_task_indices[dst] = temp[src]
