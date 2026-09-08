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
Backward pass: generate RuntimeConfig binary files and tiling binary files.

Usage:
    python gen_runtime_data.py [--tp 4] [--ep 4] [--seq_size 8192]
                               [--all_expert_num 32] [--top_k 8]
                               [--output_dir mega_moe_grad_tp4_ep4_910b]

Outputs (rank-independent):
    <output_dir>/act_grad_tiling.bin         (act_grad, pos 20)
    <output_dir>/gate_grad_tiling.bin        (gate_grad, pos 21)
    <output_dir>/w1_grad_tiling.bin          (w1_grad, pos 22)
    <output_dir>/w2_grad_tiling.bin          (w2_grad, pos 23)
    <output_dir>/swiglu_grad_tiling.bin      (SwiGLU-grad, pos 24)
    <output_dir>/all_event_counters.bin      event counters and ready state
    <output_dir>/gmm_workspace.bin           256 MiB zeros

Outputs (per rank):
    <output_dir>/runtime_config_input_rank_<i>.bin
"""
import argparse
import os

from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import (
    build_backward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.tiling_tables import (
    get_act_grad_tiling_bytes,
    get_gate_grad_tiling_bytes,
    get_swiglu_grad_tiling_bytes,
    get_w1_grad_tiling_bytes,
    get_w2_grad_tiling_bytes,
)
from hyper_parallel.core.multicore.scheduler.builder import allocate_graph_config
from hyper_parallel.core.multicore.scheduler.config import (
    QUEUE_CAPACITY,
    READY_HANDSHAKE_COPY_ACK_VERSION,
    READY_HANDSHAKE_DESTINATION_FENCE_VERSION,
    READY_HANDSHAKE_VERSION,
    RuntimeConfigC,
    TaskSplitValue,
    TaskType,
    configure_ready_handshake,
    event_workspace_bytes,
    init_task_split_value,
    serialize_runtime_config,
)
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph
from hyper_parallel.core.multicore.scheduler.scheduler import (
    revise_gmm_task_queue_bwd,
    revise_rank_aware_alltoall_segment,
    revise_task_queue,
)
from hyper_parallel.core.multicore.tasks.utils import (
    add_dynamic_data,
    add_terminate,
    add_terminate_on_event,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for backward data generation."""
    p = argparse.ArgumentParser()
    p.add_argument('--tp',                type=int, default=4)
    p.add_argument('--ep',                type=int, default=4)
    p.add_argument('--seq_size',          type=int, default=8192)
    p.add_argument('--all_expert_num',    type=int, default=32)
    p.add_argument('--top_k',             type=int, default=8)
    p.add_argument('--hidden_size',       type=int, default=7168,
                   help='Model hidden dimension (e.g. 7168 for Qwen3-235B)')
    p.add_argument('--intermediate_size', type=int, default=2048,
                   help='FFN intermediate dimension after SwiGLU halving (half of up-proj output, e.g. 2048)')
    p.add_argument('--dtype_size',        type=int, default=2,
                   help='Activation bytes per element: bf16=2, fp32=4')
    p.add_argument('--num_cube_cores',    type=int, default=24,
                   help='Number of AI Cube cores on target hardware (910B=24)')
    p.add_argument('--output_dir',        type=str,
                   default='mega_moe_grad_tp4_ep4_910b')
    return p.parse_args()


def _add_backward_termination(
    cfg: RuntimeConfigC,
    graph: ComputeGraph,
    tsv: TaskSplitValue,
    *,
    kernel_multi_wave: bool = False,
) -> None:
    """Append termination after weight gradients and the selected route epilogue."""
    w1_grad_op = graph.get_op("w1_grad")
    w2_grad_op = graph.get_op("w2_grad")
    combine_op = graph.get_op("combine")
    weight_grad_triggers = (
        0
        if kernel_multi_wave
        else w1_grad_op.task_num + w2_grad_op.task_num
    )
    topk_grad_tasks = 0
    if graph.has_op("topk_weight_grad"):
        topk_op = graph.get_op("topk_weight_grad")
        if not topk_op.fill_config.first_wave_only:
            topk_grad_tasks = topk_op.task_num
    has_terminal_epilogue = graph.has_op("unpermute_grad") or graph.has_op(
        "external_output_acquire"
    )
    if topk_grad_tasks or has_terminal_epilogue:
        route_event = (
            tsv.all_event_num - 1 if has_terminal_epilogue else tsv.all_event_num
        )
        cfg.all_event_num_triggers[route_event] = (
            combine_op.task_num + topk_grad_tasks
        )
    if graph.has_op("unpermute_grad"):
        endpoint_tasks = graph.get_op("unpermute_grad").task_num
    elif graph.has_op("external_output_acquire"):
        endpoint_tasks = graph.get_op("external_output_acquire").task_num
    else:
        endpoint_tasks = 0
    if endpoint_tasks:
        add_terminate_on_event(
            cfg,
            tsv,
            tsv.all_event_num,
            weight_grad_triggers + endpoint_tasks,
        )
    elif topk_grad_tasks:
        add_terminate_on_event(
            cfg,
            tsv,
            tsv.all_event_num,
            weight_grad_triggers + combine_op.task_num + topk_grad_tasks,
        )
    else:
        add_terminate(
            cfg,
            tsv,
            weight_grad_triggers + combine_op.task_num // tsv.ep * tsv.ep,
        )


def build_config_for_rank(
    graph: ComputeGraph,
    tsv: TaskSplitValue,
    rank_id: int,
    num_cube_cores: int = 24,
    *,
    kernel_multi_wave: bool = False,
) -> RuntimeConfigC:
    """Build backward RuntimeConfig for a single rank.

    Args:
        graph: Backward compute graph to lower into runtime descriptors.
        tsv: Mutable scheduler counters and model dimensions.
        rank_id: Rank whose task queues are being generated.
        num_cube_cores: Number of Cube workers available to the schedule.
        kernel_multi_wave: Whether to apply the in-kernel multi-wave passes.

    Returns:
        Runtime descriptors and worker queues for the selected rank.
    """
    cfg = allocate_graph_config(graph, tsv)
    cfg.num_workers    = 2 * num_cube_cores   # NUM_WORKERS_VECTOR = 2 × NUM_WORKERS_CUBE
    cfg.queue_capacity = QUEUE_CAPACITY

    # Reset counters
    init_task_split_value(tsv)
    tsv.rank_id = rank_id

    # Fill tasks in topological order (dispatch, act_grad, w2_grad, swiglu_grad, gate_grad, combine, w1_grad)
    vector_segment_starts = {}
    for op in graph.topological_sort():
        vector_segment_starts[op.name] = tsv.pre_vector_task_num
        op.fill_config.fill(cfg, op, tsv)

    # task_num_all = sum of all op tasks + 1 (terminate)
    task_num_all = sum(op.task_num for op in graph.topological_sort()) + 1

    dispatch_op    = graph.get_op("dispatch")
    swiglu_grad_op = graph.get_op("swiglu_grad")
    act_grad_op    = graph.get_op("act_grad")
    combine_op     = graph.get_op("combine")
    dispatch_hidden_op = (
        graph.get_op("dispatch_hidden")
        if graph.has_op("dispatch_hidden")
        else None
    )

    _add_backward_termination(
        cfg,
        graph,
        tsv,
        kernel_multi_wave=kernel_multi_wave,
    )
    if not kernel_multi_wave:
        early_topk_task_num = 0
        if graph.has_op("topk_weight_grad"):
            topk_op = graph.get_op("topk_weight_grad")
            if topk_op.fill_config.first_wave_only:
                early_topk_task_num = topk_op.task_num
        revise_task_queue(
            cfg,
            tsv,
            dispatch_op.task_num,
            swiglu_grad_op.task_num,
            combine_op.task_num,
            combine_start=(
                dispatch_op.task_num
                + early_topk_task_num
                + swiglu_grad_op.task_num
            ),
        )
        revise_gmm_task_queue_bwd(
            cfg,
            tsv,
            act_grad_op.task_num,
            num_cube_cores=num_cube_cores,
        )
    else:
        if dispatch_op.fill_config.destination_wave_fence:
            revise_rank_aware_alltoall_segment(
                cfg,
                tsv,
                start=vector_segment_starts[dispatch_op.name],
                task_num=dispatch_op.task_num,
            )
            if dispatch_hidden_op is not None:
                revise_rank_aware_alltoall_segment(
                    cfg,
                    tsv,
                    start=vector_segment_starts[dispatch_hidden_op.name],
                    task_num=dispatch_hidden_op.task_num,
                )
            revise_rank_aware_alltoall_segment(
                cfg,
                tsv,
                start=vector_segment_starts[combine_op.name],
                task_num=combine_op.task_num,
            )
        topk_task_num = 0
        if graph.has_op("topk_weight_grad"):
            topk_op = graph.get_op("topk_weight_grad")
            if not topk_op.fill_config.first_wave_only:
                topk_task_num = topk_op.task_num
        final_task_num = 0
        for task_index in range(tsv.pre_task_num):
            task = cfg.all_tasks[task_index]
            if task.task_type in {
                TaskType.TASK_UNPERMUTE_REDUCE_GRAD,
                TaskType.TASK_WEIGHTED_UNPERMUTE,
            }:
                task.extra_value_3 = topk_task_num
                final_task_num += 1
        terminate = cfg.all_tasks[tsv.pre_task_num]
        terminate.extra_value_3 = final_task_num or topk_task_num
    add_dynamic_data(cfg, tsv, dynamic_input_position=19)
    ready_version = READY_HANDSHAKE_VERSION
    if (
        kernel_multi_wave
        and graph.has_op("dispatch")
        and graph.get_op("dispatch").fill_config.destination_wave_fence
    ):
        ready_version = READY_HANDSHAKE_DESTINATION_FENCE_VERSION
    elif (
        kernel_multi_wave
        and (
            graph.has_op("release_dy_tiles")
            or (
                graph.has_op("copy_hidden")
                and graph.get_op("copy_hidden").fill_config.route_tile_ack
                and graph.has_op("dispatch")
                and graph.get_op("dispatch").fill_config.wait_for_copy_ack
            )
        )
    ):
        ready_version = READY_HANDSHAKE_COPY_ACK_VERSION
    configure_ready_handshake(cfg, tsv, version=ready_version)

    cfg.task_num = task_num_all
    cfg.atomic_add_values[0] = 1
    return cfg


def write_bin(path: str, data: bytes) -> None:
    """Write binary data to a file, creating parent directories if needed.

    Args:
        path: Destination file path.
        data: Binary payload to write.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    print(f"  wrote {len(data):>10,} bytes → {path}")


def main() -> None:
    """Entry point for backward pass runtime data generation."""
    args = parse_args()
    out  = args.output_dir

    tsv   = TaskSplitValue(
        tp=args.tp, ep=args.ep,
        seq_size=args.seq_size,
        all_expert_num=args.all_expert_num,
        top_k=args.top_k,
    )
    num_groups = tsv.single_rank_expert_num
    graph = build_backward_graph(tsv,
                                 dispatch_sv=128,  act_grad_sv=4096,  w2_grad_sv=4096,
                                 swiglu_sv=128,    gate_grad_sv=4096, w1_grad_sv=4096,
                                 combine_sv=128,
                                 hidden_size=args.hidden_size,
                                 intermediate_size=args.intermediate_size,
                                 dtype_size=args.dtype_size,
                                 num_cube_cores=args.num_cube_cores)
    # Compute task_num for each operator via split-axis propagation
    graph.propagate_splits(tsv)

    dispatch_op    = graph.get_op("dispatch")
    act_grad_op    = graph.get_op("act_grad")
    w2_grad_op     = graph.get_op("w2_grad")
    swiglu_grad_op = graph.get_op("swiglu_grad")
    gate_grad_op   = graph.get_op("gate_grad")
    w1_grad_op     = graph.get_op("w1_grad")
    combine_op     = graph.get_op("combine")

    print(f"[bwd] tp={args.tp} ep={args.ep} seq={args.seq_size} "
          f"E={args.all_expert_num} topk={args.top_k}")
    print(f"      dispatch={dispatch_op.task_num}  act_grad={act_grad_op.task_num}  "
          f"w2_grad={w2_grad_op.task_num}  swiglu_grad={swiglu_grad_op.task_num}  "
          f"gate_grad={gate_grad_op.task_num}  w1_grad={w1_grad_op.task_num}  "
          f"combine={combine_op.task_num}")

    # ── Tiling files (rank-independent) ──────────────────────────────────────
    act_grad_bytes    = get_act_grad_tiling_bytes(act_grad_op.split_value,
                                                  hidden_size=args.hidden_size,
                                                  intermediate_size=args.intermediate_size,
                                                  num_groups=num_groups,
                                                  num_cube_cores=args.num_cube_cores)
    gate_grad_bytes   = get_gate_grad_tiling_bytes(gate_grad_op.split_value,
                                                   hidden_size=args.hidden_size,
                                                   intermediate_size=args.intermediate_size,
                                                   num_groups=num_groups,
                                                   num_cube_cores=args.num_cube_cores)
    w1_grad_bytes     = get_w1_grad_tiling_bytes(w1_grad_op.split_value,
                                                 hidden_size=args.hidden_size,
                                                 intermediate_size=args.intermediate_size,
                                                 num_groups=num_groups,
                                                 num_cube_cores=args.num_cube_cores)
    w2_grad_bytes     = get_w2_grad_tiling_bytes(w2_grad_op.split_value,
                                                 hidden_size=args.hidden_size,
                                                 intermediate_size=args.intermediate_size,
                                                 num_groups=num_groups,
                                                 num_cube_cores=args.num_cube_cores)
    swiglu_grad_bytes = get_swiglu_grad_tiling_bytes(swiglu_grad_op.split_value,
                                                     intermediate_size=args.intermediate_size)

    write_bin(os.path.join(out, 'act_grad_tiling.bin'),   act_grad_bytes)
    write_bin(os.path.join(out, 'gate_grad_tiling.bin'),  gate_grad_bytes)
    write_bin(os.path.join(out, 'w1_grad_tiling.bin'),    w1_grad_bytes)
    write_bin(os.path.join(out, 'w2_grad_tiling.bin'),    w2_grad_bytes)
    write_bin(os.path.join(out, 'swiglu_grad_tiling.bin'), swiglu_grad_bytes)

    # ── Event counters + workspace (rank-independent) ─────────────────────────
    # The persistent tail stores the per-rank ready generation.
    write_bin(os.path.join(out, 'all_event_counters.bin'),
              bytes(event_workspace_bytes(args.ep, tsv.runtime_event_capacity)))
    # gmm_workspace: 256 MiB zeros — kernel-internal scratch buffer
    write_bin(os.path.join(out, 'gmm_workspace.bin'),
              bytes(256 * 1024 * 1024))

    # ── RuntimeConfig files (one per rank) ───────────────────────────────────
    for rank_id in range(args.ep):
        cfg  = build_config_for_rank(graph, tsv, rank_id, num_cube_cores=args.num_cube_cores)
        data = serialize_runtime_config(cfg)
        path = os.path.join(out, f'runtime_config_input_rank_{rank_id}.bin')
        write_bin(path, data)

    print("[bwd] done.")


if __name__ == '__main__':
    main()
