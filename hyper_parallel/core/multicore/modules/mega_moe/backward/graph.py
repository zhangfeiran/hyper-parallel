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
"""MoE-FFN backward compute graph.

反向图把 dispatch 后的梯度拆成两条并行支路：一条计算 act/dW2，另一条经过
SwiGLU 梯度和 gate/dW1，最后 combine 与 dW1 可以并行收尾。该结构会被序列化
为 ``mega_moe_grad`` 使用的 runtime descriptor。
"""

from typing import Optional

from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue
from hyper_parallel.core.multicore.scheduler.graph import (
    ComputeGraph,
    OperatorNode,
    OpType,
    SplitSpec,
    TensorSpec,
)
from hyper_parallel.core.multicore.tasks.alltoall import (
    AllToAllFillConfig,
    AllToAllType,
)
from hyper_parallel.core.multicore.tasks.gmm import GmmFillConfig
from hyper_parallel.core.multicore.tasks.local_accumulate import (
    LocalAccumulateFillConfig,
)
from hyper_parallel.core.multicore.tasks.local_copy import LocalCopyFillConfig
from hyper_parallel.core.multicore.tasks.permute import (
    TopKWeightGradFillConfig,
    UnpermuteReduceFillConfig,
    WeightedUnpermuteFillConfig,
)
from hyper_parallel.core.multicore.tasks.phase_barrier import PhaseBarrierFillConfig
from hyper_parallel.core.multicore.tasks.route_ack import RouteComputeAckFillConfig
from hyper_parallel.core.multicore.tasks.swiglu import SwiGLUFillConfig

# Tiling buffer positions in the C++ runtime descriptor array.
_TILING_POS_ACT_GRAD    = 20
_TILING_POS_GATE_GRAD   = 21
_TILING_POS_W1_GRAD     = 22
_TILING_POS_W2_GRAD     = 23
_TILING_POS_SWIGLU_GRAD = 24

_POS_PERMUTE_SRC_INDEX = 29
_POS_ROUTE_WEIGHT = 30
_POS_REDUCE_ROUTE_INDEX = 31

def _build_bwd_tensor_specs(tsv, hidden_size, intermediate_size, dtype_size):
    """Create all TensorSpec objects for the backward graph."""
    # 这些位置必须和 C++ MegaMoeGrad 的参数 ABI 一致；带 transpose 的权重槽位
    # 表示 GMM 需要的矩阵方向，而不是改变 Python tensor 的实际 storage。
    sre = tsv.single_rank_expert_num
    target = TensorSpec(
        "target", [tsv.per_rank_seq, hidden_size],
        param_position=0, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    target_offset = TensorSpec(
        "target_offset", [tsv.all_expert_num],
        param_position=1, dtype_size=8)
    src = TensorSpec(
        "src", [tsv.per_rank_seq, hidden_size],
        param_position=2, dtype_size=dtype_size, tensor_type=1)
    src_offset = TensorSpec(
        "src_offset", [tsv.all_expert_num],
        param_position=3, dtype_size=8)
    size_d = TensorSpec(
        "size_d", [tsv.all_expert_num],
        param_position=4, dtype_size=4)
    w2_grad_x1 = TensorSpec(
        "w2_grad_x1", [tsv.per_rank_seq, intermediate_size],
        param_position=5, dtype_size=dtype_size,
        tensor_type=1, is_dynamic=True, transpose=True)
    w2_grad_y = TensorSpec(
        "w2_grad_y", [sre, intermediate_size, hidden_size],
        param_position=6, dtype_size=dtype_size, tensor_type=1)
    act_grad_weight = TensorSpec(
        "act_grad_weight", [sre, intermediate_size, hidden_size],
        param_position=7, dtype_size=dtype_size, tensor_type=1, transpose=True)
    act_grad_y = TensorSpec(
        "act_grad_y", [tsv.per_rank_seq, intermediate_size],
        param_position=8, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    swiglu_dy = TensorSpec(
        "swiglu_dy", [tsv.per_rank_seq, intermediate_size * 2],
        param_position=9, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    swiglu_out = TensorSpec(
        "swiglu_out", [tsv.per_rank_seq, intermediate_size * 2],
        param_position=10, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    gate_grad_weight = TensorSpec(
        "gate_grad_weight", [sre, hidden_size, intermediate_size * 2],
        param_position=11, dtype_size=dtype_size, tensor_type=1, transpose=True)
    gate_grad_y = TensorSpec(
        "gate_grad_y", [tsv.per_rank_seq, hidden_size],
        param_position=12, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    combine_out = TensorSpec(
        "combine_out", [tsv.per_rank_seq, hidden_size],
        param_position=13, dtype_size=dtype_size, tensor_type=1)
    target_offset_c = TensorSpec(
        "target_offset_c", [tsv.all_expert_num],
        param_position=14, dtype_size=8)
    src_offset_c = TensorSpec(
        "src_offset_c", [tsv.all_expert_num],
        param_position=15, dtype_size=8)
    size_c = TensorSpec(
        "size_c", [tsv.all_expert_num],
        param_position=16, dtype_size=4)
    w1_grad_x1 = TensorSpec(
        "w1_grad_x1", [tsv.per_rank_seq, hidden_size],
        param_position=17, dtype_size=dtype_size,
        tensor_type=1, is_dynamic=True, transpose=True)
    w1_grad_y = TensorSpec(
        "w1_grad_y", [sre, hidden_size, intermediate_size * 2],
        param_position=18, dtype_size=dtype_size, tensor_type=1)
    glist = TensorSpec(
        "glist", [sre],
        param_position=19, dtype_size=8)
    return (target, target_offset, src, src_offset, size_d,
            w2_grad_x1, w2_grad_y, act_grad_weight, act_grad_y,
            swiglu_dy, swiglu_out, gate_grad_weight, gate_grad_y,
            combine_out, target_offset_c, src_offset_c, size_c,
            w1_grad_x1, w1_grad_y, glist)


def _build_bwd_ops_first(tsv, specs, *, dispatch_sv, act_grad_sv, w2_grad_sv,
                         swiglu_sv, num_cube_cores):
    """Create dispatch, act_grad, w2_grad, swiglu_grad operator nodes."""
    # dispatch 完成后 act_grad 与 w2_grad 共享输入、可并行执行；swiglu_grad
    # 必须等待 act_grad 的输出，因此作为第一段的后继。
    (target, target_offset, src, src_offset, size_d,
     w2_grad_x1, w2_grad_y, act_grad_weight, act_grad_y,
     swiglu_dy, swiglu_out, *_) = specs
    glist = specs[-1]
    dispatch = OperatorNode(
        name="dispatch", op_type=OpType.ALLTOALL,
        inputs=[target_offset, src, src_offset, size_d], outputs=[target],
        param_positions=[1, 2, 3, 4, 0], split_value=dispatch_sv,
        split_spec=SplitSpec(
            split_inputs=None, split_output_dims=[0],
            task_num_fn=lambda tsv: tsv.all_expert_num * (tsv.per_expert_seq_to_other // dispatch_sv),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(moe_type=AllToAllType.DISPATCH, advance="vector",
                                       event_group=tsv.all_expert_num),
    )
    act_grad = OperatorNode(
        name="act_grad", op_type=OpType.GMM,
        inputs=[target, act_grad_weight, glist], outputs=[act_grad_y],
        param_positions=[0, 7, 19, 8], split_value=act_grad_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores:
                _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_ACT_GRAD,
        fill_config=GmmFillConfig(offset_inputs={0}, rank_in_event=True, global_trigger=False,
                                  out_offset=True, advance="cube_only", num_cube_cores=num_cube_cores),
    )
    w2_grad = OperatorNode(
        name="w2_grad", op_type=OpType.GMM,
        inputs=[w2_grad_x1, target, glist], outputs=[w2_grad_y],
        param_positions=[5, 0, 19, 6], split_value=w2_grad_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores:
                _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_W2_GRAD,
        fill_config=GmmFillConfig(offset_inputs={0, 1}, rank_in_event=True,
                                  global_trigger=True, out_offset=False, advance="cube_custom",
                                  event_delta=tsv.single_rank_expert_num, num_cube_cores=num_cube_cores),
    )
    swiglu_grad = OperatorNode(
        name="swiglu_grad", op_type=OpType.SWIGLU_GRAD,
        inputs=[act_grad_y, swiglu_dy], outputs=[swiglu_out],
        param_positions=[8, 9, 10], split_value=swiglu_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv: (tsv.per_expert_seq // swiglu_sv) * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_SWIGLU_GRAD,
        fill_config=SwiGLUFillConfig(),
    )
    return dispatch, act_grad, w2_grad, swiglu_grad


def _build_bwd_ops_second(specs, *, gate_grad_sv, w1_grad_sv, combine_sv,
                          num_cube_cores):
    """Create gate_grad, combine, w1_grad operator nodes."""
    # gate_grad 之后，combine 产生路由 token 梯度，w1_grad 产生第一层权重梯度；
    # 两者互不依赖，调度器可以把它们分配到不同 worker 队列。
    (*_, swiglu_out, gate_grad_weight, gate_grad_y,
     combine_out, target_offset_c, src_offset_c, size_c,
     w1_grad_x1, w1_grad_y, glist) = specs
    gate_grad = OperatorNode(
        name="gate_grad", op_type=OpType.GMM,
        inputs=[swiglu_out, gate_grad_weight, glist], outputs=[gate_grad_y],
        param_positions=[10, 11, 19, 12], split_value=gate_grad_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores:
                _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_GATE_GRAD,
        fill_config=GmmFillConfig(offset_inputs={0}, rank_in_event=False, global_trigger=False,
                                  out_offset=True, advance="cube", num_cube_cores=num_cube_cores),
    )
    combine = OperatorNode(
        name="combine", op_type=OpType.ALLTOALL,
        inputs=[target_offset_c, gate_grad_y, src_offset_c, size_c], outputs=[combine_out],
        param_positions=[14, 12, 15, 16, 13], split_value=combine_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv: tsv.all_expert_num * (tsv.per_expert_seq_to_other // combine_sv),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(
            moe_type=AllToAllType.COMBINE,
            advance="vector_only",
            event_group=1,
        ),
    )
    w1_grad = OperatorNode(
        name="w1_grad", op_type=OpType.GMM,
        inputs=[w1_grad_x1, swiglu_out, glist], outputs=[w1_grad_y],
        param_positions=[17, 10, 19, 18], split_value=w1_grad_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores:
                _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_W1_GRAD,
        fill_config=GmmFillConfig(offset_inputs={0, 1}, rank_in_event=False,
                                  global_trigger=True, out_offset=False, advance="cube_custom",
                                  event_delta=1, num_cube_cores=num_cube_cores),
    )
    return gate_grad, combine, w1_grad


def _append_remote_write_acquire(
    graph,
    tsv,
    combine,
    *,
    output_position,
    hidden_size,
    dtype_size,
    num_cube_cores,
    topk_weight_grad=None,
    source_position=None,
):
    """Acquire peer Combine writes before an external backward consumer."""
    source = None
    inputs = []
    param_positions = [output_position]
    output_rows = 0
    if source_position is not None:
        output_rows = tsv.per_rank_seq
        source = TensorSpec(
            "combine_shmem_staging",
            [output_rows, hidden_size],
            param_position=source_position,
            dtype_size=dtype_size,
            tensor_type=1,
        )
        inputs = [source]
        param_positions = [source_position, output_position]
    combine_out = TensorSpec(
        "combine_acquire_sentinel",
        [output_rows, hidden_size],
        param_position=output_position,
        dtype_size=dtype_size,
        tensor_type=1 if source is not None else 0,
    )
    acquire = OperatorNode(
        name="external_output_acquire",
        op_type=OpType.REMOTE_WRITE_ACQUIRE,
        inputs=inputs,
        outputs=[combine_out],
        param_positions=param_positions,
        split_value=1,
        split_spec=SplitSpec(
            split_inputs=None,
            split_output_dims=[-1],
            # The static scheduler executes two acquire tasks on each active lane.
            task_num_fn=lambda _tsv: 2 * num_cube_cores,
        ),
        tiling_position=-1,
        fill_config=WeightedUnpermuteFillConfig(
            dependent_event=tsv.all_event_num - 1,
            trigger_event=tsv.all_event_num,
            top_k=0,
            acquire_remote_writes=True,
            static_acquire_task_num=num_cube_cores,
        ),
    )
    combine.fill_config.trigger_event = tsv.all_event_num - 1
    graph.add_op(acquire).add_edge(combine, acquire)
    if topk_weight_grad is not None:
        graph.add_edge(topk_weight_grad, acquire)
    return graph


def _build_route_backward_tensors(tsv, hidden_size, dtype_size):
    """Build route-aware tensors that extend the ordinary backward ABI."""
    local_tokens = tsv.seq_size // tsv.tp
    expert_output = TensorSpec(
        "expert_output", [tsv.per_rank_seq, hidden_size],
        param_position=13, dtype_size=dtype_size, tensor_type=1,
    )
    reduce_route_index = TensorSpec(
        "reduce_route_index", [local_tokens, tsv.top_k],
        param_position=_POS_REDUCE_ROUTE_INDEX, dtype_size=4,
    )
    grad_x = TensorSpec(
        "grad_x", [local_tokens, hidden_size],
        param_position=13, dtype_size=dtype_size, tensor_type=1,
    )
    grad_output = TensorSpec(
        "grad_output", [local_tokens, hidden_size],
        param_position=2, dtype_size=dtype_size, tensor_type=1,
    )
    grad_topk_weight = TensorSpec(
        "grad_topk_weight", [local_tokens, tsv.top_k],
        param_position=_POS_ROUTE_WEIGHT, dtype_size=4, tensor_type=1,
    )
    return local_tokens, expert_output, reduce_route_index, grad_x, grad_output, grad_topk_weight


def _build_route_backward_base_graph(
    dispatch,
    act_grad,
    w2_grad,
    swiglu_grad,
    gate_grad,
    combine,
    w1_grad,
    *,
    gate_to_w1=True,
):
    """Connect the shared compute stages of a route-aware backward graph."""
    graph = ComputeGraph()
    graph = (
        graph.add_op(dispatch).add_op(act_grad).add_op(w2_grad)
        .add_op(swiglu_grad).add_op(gate_grad).add_op(combine).add_op(w1_grad)
        .add_edge(dispatch, act_grad).add_edge(dispatch, w2_grad)
        .add_edge(act_grad, swiglu_grad).add_edge(swiglu_grad, gate_grad)
        .add_edge(gate_grad, combine)
        .add_edge(w2_grad, combine)
    )
    if gate_to_w1:
        graph.add_edge(gate_grad, w1_grad)
    return graph


def _append_topk_weight_grad(
    graph,
    tsv,
    combine,
    *,
    enabled,
    local_tokens,
    topk_grad_sv,
    route_completion_event,
    expert_output,
    grad_output,
    reduce_route_index,
    grad_topk_weight,
    first_wave_only=False,
):
    """Optionally append router-weight gradient computation."""
    if not enabled:
        return None
    topk_weight_grad = OperatorNode(
        name="topk_weight_grad",
        op_type=OpType.TOPK_WEIGHT_GRAD,
        inputs=[expert_output, grad_output, reduce_route_index],
        outputs=[grad_topk_weight],
        param_positions=[
            expert_output.param_position,
            grad_output.param_position,
            reduce_route_index.param_position,
            grad_topk_weight.param_position,
        ],
        split_value=topk_grad_sv,
        split_spec=SplitSpec(
            split_inputs=None,
            split_output_dims=[0],
            task_num_fn=lambda _tsv: (
                local_tokens + topk_grad_sv - 1
            ) // topk_grad_sv,
        ),
        tiling_position=-1,
        fill_config=TopKWeightGradFillConfig(
            dependent_event=tsv.all_event_num + 5,
            trigger_event=route_completion_event,
            top_k=tsv.top_k,
            first_wave_only=first_wave_only,
        ),
    )
    graph.add_op(topk_weight_grad)
    if first_wave_only:
        graph.add_edge(graph.get_op("dispatch"), topk_weight_grad)
        graph.add_edge(topk_weight_grad, combine)
    else:
        graph.add_edge(combine, topk_weight_grad)
    return topk_weight_grad


def _append_route_backward_output(
    graph,
    tsv,
    combine,
    *,
    integrated_unpermute,
    local_tokens,
    unpermute_sv,
    hidden_size,
    dtype_size,
    num_cube_cores,
    target,
    reduce_route_index,
    grad_x,
    topk_weight_grad,
    combine_output_position=0,
):
    """Append fused unpermute or an acquire for the external consumer."""
    if not integrated_unpermute:
        return _append_remote_write_acquire(
            graph,
            tsv,
            combine,
            output_position=combine_output_position,
            hidden_size=hidden_size,
            dtype_size=dtype_size,
            num_cube_cores=num_cube_cores,
            topk_weight_grad=topk_weight_grad,
        )
    unpermute_grad = OperatorNode(
        name="unpermute_grad",
        op_type=OpType.UNPERMUTE_REDUCE_GRAD,
        inputs=[target, reduce_route_index],
        outputs=[grad_x],
        param_positions=[
            combine_output_position,
            _POS_REDUCE_ROUTE_INDEX,
            13,
        ],
        split_value=unpermute_sv,
        split_spec=SplitSpec(
            split_inputs=None,
            split_output_dims=[0],
            task_num_fn=lambda _tsv: (
                local_tokens + unpermute_sv - 1
            ) // unpermute_sv,
        ),
        tiling_position=-1,
        fill_config=UnpermuteReduceFillConfig(
            dependent_event=tsv.all_event_num - 1,
            trigger_event=tsv.all_event_num,
            top_k=tsv.top_k,
        ),
    )
    graph.add_op(unpermute_grad).add_edge(combine, unpermute_grad)
    if topk_weight_grad is not None:
        graph.add_edge(topk_weight_grad, unpermute_grad)
    return graph


def build_backward_graph(tsv: TaskSplitValue, *,
                         dispatch_sv:   int = 128,
                         act_grad_sv:   int = 4096,
                         w2_grad_sv:    int = 4096,
                         swiglu_sv:     int = 128,
                         gate_grad_sv:  int = 4096,
                         w1_grad_sv:    int = 4096,
                         combine_sv:    int = 128,
                         hidden_size:       int = 7168,
                         intermediate_size: int = 2048,
                         dtype_size:        int = 2,
                         num_cube_cores:    int = 24) -> ComputeGraph:
    """Build the MoE-FFN backward DAG.

    Execution order:
      dispatch -> {act_grad, w2_grad} -> swiglu_grad -> gate_grad -> {combine, w1_grad}
    act_grad and w2_grad run in parallel (both depend on dispatch).
    combine and w1_grad run in parallel (both depend on gate_grad).

    Param positions (C++ memory slots):
      dispatch:     [target_offset=1, src=2, src_offset=3, size=4 | target=0]
      act_grad:     [x=0, weight=7, glist=19 | y=8]
      w2_grad:      [x1=5, x2=0, glist=19 | y=6]
      swiglu_grad:  [x=8, dy=9 | out=10]
      gate_grad:    [x=10, weight=11, glist=19 | y=12]
      combine:      [target_offset=14, src=12, src_offset=15, size=16 | target=13]
      w1_grad:      [x1=17, x2=10, glist=19 | y=18]

    Args:
        tsv: TaskSplitValue carrying TP/EP/seq partition metadata.
        dispatch_sv / act_grad_sv / w2_grad_sv / swiglu_sv / gate_grad_sv /
        w1_grad_sv / combine_sv: tile sizes per operator.
        hidden_size: model hidden dimension.
        intermediate_size: FFN intermediate dimension after SwiGLU halving.
        dtype_size: bytes per activation element (2=bf16, 4=fp32).
        num_cube_cores: number of AIC cube cores on the target device.

    Returns:
        A fully-connected ComputeGraph ready for propagate_splits().
    """
    # 和前向一样，先构造静态 DAG，再由 propagate_splits 计算每个 operator 的
    # task 数；权重梯度分支通过全局事件加入 kernel 终止条件。
    # Combine writes peer memory. Reserve a final event for the device-side
    # acquire that makes those writes visible to the external ACLNN consumer.
    tsv.extra_event_num = max(tsv.extra_event_num, 1)
    specs = _build_bwd_tensor_specs(tsv, hidden_size, intermediate_size, dtype_size)
    dispatch, act_grad, w2_grad, swiglu_grad = _build_bwd_ops_first(
        tsv, specs,
        dispatch_sv=dispatch_sv, act_grad_sv=act_grad_sv,
        w2_grad_sv=w2_grad_sv, swiglu_sv=swiglu_sv, num_cube_cores=num_cube_cores,
    )
    gate_grad, combine, w1_grad = _build_bwd_ops_second(
        specs,
        gate_grad_sv=gate_grad_sv, w1_grad_sv=w1_grad_sv,
        combine_sv=combine_sv, num_cube_cores=num_cube_cores,
    )
    graph = ComputeGraph()
    (graph.add_op(dispatch).add_op(act_grad).add_op(w2_grad)
          .add_op(swiglu_grad).add_op(gate_grad).add_op(combine).add_op(w1_grad)
          .add_edge(dispatch,    act_grad)
          .add_edge(dispatch,    w2_grad)
          .add_edge(act_grad,    swiglu_grad)
          .add_edge(swiglu_grad, gate_grad)
          .add_edge(gate_grad,   combine)
          .add_edge(gate_grad,   w1_grad))
    _append_remote_write_acquire(
        graph,
        tsv,
        combine,
        output_position=13,
        hidden_size=hidden_size,
        dtype_size=dtype_size,
        num_cube_cores=num_cube_cores,
    )
    return graph


def build_route_backward_graph(
    tsv: TaskSplitValue,
    *,
    integrated_permute: bool = True, integrated_unpermute: bool = True,
    inplace_gate_dx: bool = False,
    dispatch_sv: int = 128, act_grad_sv: int = 4096,
    w2_grad_sv: int = 4096, swiglu_sv: int = 128,
    gate_grad_sv: int = 4096, w1_grad_sv: int = 4096,
    combine_sv: int = 128, unpermute_sv: int = 8,
    topk_grad_sv: int = 32, hidden_size: int = 7168,
    intermediate_size: int = 2048, dtype_size: int = 2,
    num_cube_cores: int = 24,
) -> ComputeGraph:
    """Build backward with independently fused route endpoints.

    ``integrated_permute`` fuses token-order ``dY`` weighting/dispatch and the
    router-weight gradient. ``integrated_unpermute`` fuses the final
    expert-major-to-token-order ``dX`` reduction. At least one endpoint must be
    enabled; the fully external graph is :func:`build_backward_graph`.

    ``inplace_gate_dx`` reuses the dispatch buffer after both dY GMM consumers
    finish. Combine then writes position 13 and leaves the final token-order
    permutation outside the kernel, so the mode requires fused permute and an
    external unpermute endpoint.

    Args:
        tsv: Scheduler dimensions and task counters.
        integrated_permute: Fuse token-order dY routing into the kernel.
        integrated_unpermute: Fuse expert-major dX reduction into the kernel.
        inplace_gate_dx: Reuse dispatch storage for the gate-gradient output.
        dispatch_sv: Dispatch split value in routed rows.
        act_grad_sv: Activation-gradient GMM split value.
        w2_grad_sv: Down-projection weight-gradient GMM split value.
        swiglu_sv: SwiGLU-gradient split value in routed rows.
        gate_grad_sv: Gate-gradient GMM split value.
        w1_grad_sv: Gate/up weight-gradient GMM split value.
        combine_sv: Combine split value in routed rows.
        unpermute_sv: Final unpermute split value in tokens.
        topk_grad_sv: Router-weight gradient split value in tokens.
        hidden_size: Model hidden dimension.
        intermediate_size: Per-expert intermediate dimension.
        dtype_size: Tensor element size in bytes.
        num_cube_cores: Number of Cube workers available to the schedule.

    Returns:
        Route-aware backward compute graph.
    """
    # 两个 endpoint 共用 route-aware 输出布局：combine 先写 expert-major dX 到
    # target buffer。topk_weight_grad 读取保留的 forward expert output；最终
    # unpermute_grad（若启用）再写 token-order dX 到参数 13。
    if not integrated_permute and not integrated_unpermute:
        raise ValueError(
            "build_route_backward_graph requires at least one integrated endpoint"
        )
    if inplace_gate_dx and (not integrated_permute or integrated_unpermute):
        raise ValueError(
            "in-place gate dX requires fused permute and external unpermute."
        )
    tsv.extra_event_num = max(
        tsv.extra_event_num,
        (
            tsv.all_expert_num + tsv.single_rank_expert_num + 2
            if inplace_gate_dx
            else 2
        ),
    )
    specs = _build_bwd_tensor_specs(tsv, hidden_size, intermediate_size, dtype_size)
    dispatch, act_grad, w2_grad, swiglu_grad = _build_bwd_ops_first(
        tsv,
        specs,
        dispatch_sv=dispatch_sv,
        act_grad_sv=act_grad_sv,
        w2_grad_sv=w2_grad_sv,
        swiglu_sv=swiglu_sv,
        num_cube_cores=num_cube_cores,
    )
    if integrated_permute:
        dispatch.fill_config.dependent_event = 0
        dispatch.fill_config.route_index_position = _POS_PERMUTE_SRC_INDEX
        dispatch.fill_config.route_weight_position = _POS_ROUTE_WEIGHT
        dispatch.fill_config.route_top_k = tsv.top_k
        dispatch.fill_config.isolated_route_signals = True
        # Keep the route-dispatch barrier separate from the production ready
        # handshake without shifting all_event_num, which is consumed by
        # globally-signaling weight-gradient GMMs.
        dispatch.fill_config.route_completion_event = tsv.all_event_num + 5
        if inplace_gate_dx:
            # The late unweighted hidden redispatch reuses the same arrival
            # arena. Generation one belongs to dY; generation two belongs to
            # the W1 input and cannot satisfy its consumers prematurely.
            dispatch.fill_config.route_generation_phase = 0
            dispatch.fill_config.route_generation_count = 2

    gate_grad, combine, w1_grad = _build_bwd_ops_second(
        specs,
        gate_grad_sv=gate_grad_sv,
        w1_grad_sv=w1_grad_sv,
        combine_sv=combine_sv,
        num_cube_cores=num_cube_cores,
    )
    target = specs[0]
    (
        local_tokens, expert_output, reduce_route_index,
        grad_x, grad_output, grad_topk_weight,
    ) = _build_route_backward_tensors(tsv, hidden_size, dtype_size)
    if inplace_gate_dx:
        # dY in position 0 is dead after act-grad and dW2. Gate-grad reuses it
        # for expert-major dX; dTopK first consumes the saved expert output in
        # position 13, then Combine safely replaces that slot with routed dX.
        gate_grad.outputs = [target]
        gate_grad.param_positions[-1] = 0
        combine.inputs[1] = target
        combine.outputs = [expert_output]
        combine.param_positions = [14, 0, 15, 16, 13]
    else:
        # Every ordinary route-aware graph keeps expert-major dX in position 0.
        # The fused unpermute consumes it there; the external ACLNN permute-grad
        # path receives the same buffer from the Python wrapper.
        combine.outputs = [target]
        combine.param_positions = [14, 12, 15, 16, 0]
    # Both fused unpermute and the external-consumer acquire use the penultimate
    # event for Combine/route completion and own the final event themselves.
    route_completion_event = tsv.all_event_num - 1
    combine.fill_config.trigger_event = route_completion_event
    graph = _build_route_backward_base_graph(
        dispatch,
        act_grad,
        w2_grad,
        swiglu_grad,
        gate_grad,
        combine,
        w1_grad,
        gate_to_w1=not inplace_gate_dx,
    )
    if inplace_gate_dx:
        graph.add_edge(w2_grad, gate_grad)
        combine.fill_config.additional_dependent_event = tsv.all_event_num - 2
    topk_weight_grad = _append_topk_weight_grad(
        graph, tsv, combine,
        enabled=integrated_permute, local_tokens=local_tokens,
        topk_grad_sv=topk_grad_sv,
        route_completion_event=(
            tsv.all_event_num - 2 if inplace_gate_dx else route_completion_event
        ),
        expert_output=expert_output, grad_output=grad_output,
        reduce_route_index=reduce_route_index, grad_topk_weight=grad_topk_weight,
        first_wave_only=inplace_gate_dx,
    )
    graph = _append_route_backward_output(
        graph, tsv, combine,
        integrated_unpermute=integrated_unpermute,
        local_tokens=local_tokens, unpermute_sv=unpermute_sv,
        hidden_size=hidden_size, dtype_size=dtype_size,
        num_cube_cores=num_cube_cores, target=target,
        reduce_route_index=reduce_route_index, grad_x=grad_x,
        topk_weight_grad=topk_weight_grad,
        combine_output_position=13 if inplace_gate_dx else 0,
    )
    if not inplace_gate_dx:
        return graph

    # Once every local Combine reader and remote write acquire has completed,
    # position 0 is dead. Re-dispatch the original token-order hidden input
    # into that same SHMEM buffer and consume it directly in dW1, avoiding a
    # capacity-sized saved tensor across the autograd boundary.
    phase_barrier = _build_phase_barrier()
    dispatch_hidden, _ = _build_multi_wave_hidden_dispatch_ops(
        tsv,
        specs,
        dispatch_sv=dispatch_sv,
        copy_sv=dispatch_sv,
        hidden_size=hidden_size,
        dtype_size=dtype_size,
        integrated_permute=True,
        hidden_copy_barrier_event=0,
        inplace_wave_backward=True,
        tile_phase_ack=False,
        hidden_ready_event=0,
        align_route_events=True,
    )
    inplace_hidden_for_dw = TensorSpec(
        "one_wave_hidden_for_dw",
        [tsv.per_rank_seq, hidden_size],
        param_position=0,
        dtype_size=dtype_size,
        tensor_type=1,
        is_dynamic=True,
        transpose=True,
    )
    w1_grad.inputs[0] = inplace_hidden_for_dw
    w1_grad.param_positions[0] = 0
    w1_grad.fill_config.rank_in_event = True
    w1_grad.fill_config.dependent_event = None
    graph.add_op(phase_barrier).add_op(dispatch_hidden)
    graph.add_edge(graph.get_op("external_output_acquire"), phase_barrier)
    graph.add_edge(phase_barrier, dispatch_hidden)
    graph.add_edge(dispatch_hidden, w1_grad)
    return graph


def _build_multi_wave_hidden_dispatch_ops(
    tsv,
    specs,
    *,
    dispatch_sv,
    copy_sv,
    hidden_size,
    dtype_size,
    integrated_permute,
    hidden_copy_barrier_event,
    inplace_wave_backward,
    tile_phase_ack,
    hidden_ready_event,
    align_route_events=False,
):
    """Build hidden dispatch followed by copy or arrival-only completion."""
    target, target_offset, _, src_offset, size_d = specs[:5]
    glist = specs[-1]
    forward_input = TensorSpec(
        "forward_input",
        [tsv.per_rank_seq, hidden_size],
        param_position=17,
        dtype_size=dtype_size,
        tensor_type=1,
    )
    hidden_copy = specs[12]

    dispatch_hidden = OperatorNode(
        name="dispatch_hidden",
        op_type=OpType.ALLTOALL,
        inputs=[target_offset, forward_input, src_offset, size_d],
        outputs=[target],
        # Position 17 retains the read-only routed source until every hidden
        # dispatch task has consumed it. Position 9 holds the saved full-route
        # up-projection, while dW1 consumes the bounded copy in position 12.
        param_positions=[1, 17, 3, 4, 0],
        split_value=dispatch_sv,
        split_spec=SplitSpec(
            split_inputs=None,
            split_output_dims=[0],
            task_num_fn=lambda value: value.all_expert_num
            * (value.per_expert_seq_to_other // dispatch_sv),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(
            moe_type=AllToAllType.OTHER,
            advance="vector",
            event_group=tsv.all_expert_num,
            route_generation_phase=1 if inplace_wave_backward else 0,
            route_generation_count=2,
            wait_for_copy_ack=tile_phase_ack,
            expert_copy_ack=tile_phase_ack,
            align_isolated_event_group=align_route_events,
        ),
    )
    if integrated_permute:
        dispatch_hidden.fill_config.dependent_event = 0
        dispatch_hidden.fill_config.route_index_position = _POS_PERMUTE_SRC_INDEX
        dispatch_hidden.fill_config.route_top_k = tsv.top_k
        dispatch_hidden.fill_config.isolated_route_signals = True

    if inplace_wave_backward:
        wait_hidden = OperatorNode(
            name="wait_hidden_arrivals",
            op_type=OpType.LOCAL_COPY,
            inputs=[target],
            outputs=[target],
            param_positions=[0, 0],
            split_value=copy_sv,
            split_spec=SplitSpec(
                split_inputs=[(0, 0)],
                split_output_dims=[0],
                task_num_fn=lambda value:
                (value.per_expert_seq // copy_sv)
                * value.single_rank_expert_num,
            ),
            tiling_position=-1,
            fill_config=LocalCopyFillConfig(
                global_trigger_event=hidden_ready_event,
                route_arrival_only=True,
                route_generation_phase=1,
                route_generation_count=2,
            ),
        )
        return dispatch_hidden, wait_hidden

    copy_inputs = [target, glist]
    copy_positions = [0, 19, 12]
    if integrated_permute:
        # Route-fused dispatch has one cache-line-isolated arrival per source
        # tile. Copy the same source-aligned tile and acknowledge it before the
        # matching dY dispatch is allowed to reuse expert_buffer.
        copy_inputs = [target, specs[15], specs[16]]
        copy_positions = [0, 15, 16, 12]

    copy_hidden = OperatorNode(
        name="copy_hidden",
        op_type=OpType.LOCAL_COPY,
        inputs=copy_inputs,
        outputs=[hidden_copy],
        param_positions=copy_positions,
        split_value=copy_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)],
            split_output_dims=[0],
            task_num_fn=lambda value:
            (value.per_expert_seq // copy_sv) * value.single_rank_expert_num,
        ),
        tiling_position=-1,
        fill_config=LocalCopyFillConfig(
            global_trigger_event=hidden_copy_barrier_event,
            rank_in_event=True,
            route_tile_ack=integrated_permute,
            route_generation_phase=0,
            route_generation_count=2 if integrated_permute else 1,
        ),
    )
    return dispatch_hidden, copy_hidden


def _build_multi_wave_activation_recompute(
    specs,
    *,
    swiglu_sv,
    dependency_event,
    activation_barrier_event,
):
    """Regenerate one rolling activation window from retained GMM1 output."""
    up_proj = specs[9]
    activation = specs[5]
    return OperatorNode(
        name="recompute_activation",
        op_type=OpType.SWIGLU,
        inputs=[up_proj],
        outputs=[activation],
        param_positions=[9, 5],
        split_value=swiglu_sv,
        split_spec=SplitSpec(
            split_inputs=None,
            split_output_dims=[0],
            task_num_fn=lambda value:
            (value.per_expert_seq // swiglu_sv)
            * value.single_rank_expert_num,
        ),
        # Forward SwiGLU tiling follows SwiGLU-grad in the same input tensor.
        tiling_position=_TILING_POS_SWIGLU_GRAD,
        fill_config=SwiGLUFillConfig(
            global_trigger_event=activation_barrier_event,
            dependent_event=dependency_event,
        ),
    )


def _build_weight_grad_accumulator(
    name,
    weight_grad,
    group_list,
    producer,
    *,
    output_position,
    num_vector_cores,
    reuse_retained_scratch,
    atomic_accumulation,
):
    """Build a vector epilogue when dW is not persisted by Matmul itself."""
    return OperatorNode(
        name=name,
        op_type=OpType.LOCAL_ACCUMULATE,
        inputs=[weight_grad, group_list],
        outputs=[weight_grad],
        param_positions=[output_position, 19, output_position],
        split_value=1,
        split_spec=SplitSpec(
            split_inputs=None,
            split_output_dims=[-1],
            task_num_fn=lambda value, cores=num_vector_cores:
            cores * value.single_rank_expert_num,
        ),
        tiling_position=-1,
        fill_config=LocalAccumulateFillConfig(
            producer=producer.fill_config,
            num_vector_cores=num_vector_cores,
            reuse_retained_scratch=reuse_retained_scratch,
            atomic_accumulation=atomic_accumulation,
        ),
    )


def _build_phase_barrier():
    """Build a tensor-free rendezvous shared by Cube and Vector workers."""
    return OperatorNode(
        name="reuse_phase_barrier",
        op_type=OpType.PHASE_BARRIER,
        inputs=[],
        outputs=[],
        param_positions=[],
        split_value=1,
        split_spec=SplitSpec(
            split_inputs=None,
            split_output_dims=[],
            task_num_fn=lambda _value: 1,
        ),
        tiling_position=-1,
        fill_config=PhaseBarrierFillConfig(),
    )


def _build_route_compute_ack(
    tsv,
    activation_producer,
    weight_producer,
    *,
    dispatch_sv,
):
    """Release one expert's symmetric dY after both GMM consumers finish."""
    route_tiles_per_expert = tsv.per_expert_seq_to_other // dispatch_sv
    return OperatorNode(
        name="release_dy_tiles",
        op_type=OpType.LOCAL_COPY,
        inputs=[],
        outputs=[],
        param_positions=[],
        split_value=1,
        split_spec=SplitSpec(
            split_inputs=None,
            split_output_dims=[],
            task_num_fn=lambda value:
            value.single_rank_expert_num * value.ep,
        ),
        tiling_position=-1,
        fill_config=RouteComputeAckFillConfig(
            activation_producer=activation_producer.fill_config,
            weight_producer=weight_producer.fill_config,
            route_tiles_per_expert=route_tiles_per_expert,
        ),
    )


def build_multi_wave_backward_graph(
    tsv: TaskSplitValue,
    *,
    integrated_permute: bool = False,
    integrated_hidden_permute: Optional[bool] = None,
    integrated_unpermute: bool = False,
    inplace_wave_backward: bool = False,
    tile_phase_ack: bool = False,
    reuse_retained_dw_scratch: bool = False,
    atomic_dw_accumulation: bool = False,
    destination_wave_fence: bool = False,
    dispatch_sv: int = 128,
    act_grad_sv: int = 4096,
    w2_grad_sv: int = 4096,
    swiglu_sv: int = 128,
    gate_grad_sv: int = 4096,
    w1_grad_sv: int = 4096,
    combine_sv: int = 128,
    unpermute_sv: int = 8,
    topk_grad_sv: int = 32,
    hidden_size: int = 7168,
    intermediate_size: int = 2048,
    dtype_size: int = 2,
    num_cube_cores: int = 24,
) -> ComputeGraph:
    """Build backward waves with retained GMM1 output and fixed SHMEM reuse.

    Args:
        tsv: Scheduler topology and task-splitting values.
        integrated_permute: Whether to fuse the backward dispatch route endpoint.
        integrated_hidden_permute: Whether the retained W1 input is token-major
            and must be route-gathered during hidden dispatch. Defaults to the
            backward dispatch setting.
        integrated_unpermute: Whether to fuse the final dX route reduction.
        inplace_wave_backward: Whether dY, hidden and dX reuse the same
            symmetric expert buffer without an ordinary-HBM hidden copy.
        tile_phase_ack: Whether receiver-local GMM completion releases each
            source tile instead of using a full-device phase barrier.
        reuse_retained_dw_scratch: Whether later waves may place whole dW
            matrices in the dead first retained up-projection window.
        atomic_dw_accumulation: Whether later waves use Matmul atomic output
            to accumulate directly into the persistent weight gradients.
        destination_wave_fence: Whether next-wave dY dispatch waits only for
            the destination rank whose expert-buffer segment it will overwrite.
        dispatch_sv: Dispatch vector split size.
        act_grad_sv: Activation-gradient GMM split size.
        w2_grad_sv: Second-weight-gradient GMM split size.
        swiglu_sv: SwiGLU and SwiGLU-gradient split size.
        gate_grad_sv: Gate-gradient GMM split size.
        w1_grad_sv: First-weight-gradient GMM split size.
        combine_sv: Combine vector split size.
        unpermute_sv: Token-order reduction split size.
        topk_grad_sv: Router-weight-gradient split size.
        hidden_size: Expert hidden dimension.
        intermediate_size: Expert intermediate dimension.
        dtype_size: Tensor element size in bytes.
        num_cube_cores: Number of Cube workers used by each GMM.

    Returns:
        A backward compute graph whose waves are replayed inside one kernel.
    """
    if integrated_hidden_permute is None:
        integrated_hidden_permute = integrated_permute
    if inplace_wave_backward and not integrated_permute:
        raise ValueError(
            "in-place multi-wave backward requires route-fused dispatch."
        )
    if tile_phase_ack and not inplace_wave_backward:
        raise ValueError(
            "tile phase acknowledgement requires in-place multi-wave backward."
        )
    if destination_wave_fence and not tile_phase_ack:
        raise ValueError(
            "destination wave fencing requires tile phase acknowledgement."
        )
    if reuse_retained_dw_scratch and atomic_dw_accumulation:
        raise ValueError(
            "retained dW scratch and atomic dW accumulation are mutually exclusive."
        )
    # The hidden-preservation prefix adds one all-expert dispatch group and one
    # rank-local expert event group before the ordinary backward schedule. The
    # activation replay adds one further rank-local group. Reserve four final
    # events for activation completion, the hidden copy, the one-time dTopK
    # completion, and Combine completion.
    # Do not use the runtime's high-numbered worker barriers here: event
    # atomics update a cache-line-sized block, so an operator dependency must
    # not share their cache line.
    hidden_copy_event_num = (
        tsv.all_expert_num
        + 6 * tsv.single_rank_expert_num
        + 4
    )
    tsv.extra_event_num = max(tsv.extra_event_num, hidden_copy_event_num)
    activation_barrier_event = tsv.all_event_num - 4
    hidden_ready_event = activation_barrier_event + 1
    topk_completion_event = tsv.all_event_num - 2
    specs = _build_bwd_tensor_specs(tsv, hidden_size, intermediate_size, dtype_size)
    dispatch, act_grad, w2_grad, swiglu_grad = _build_bwd_ops_first(
        tsv,
        specs,
        dispatch_sv=dispatch_sv,
        act_grad_sv=act_grad_sv,
        w2_grad_sv=w2_grad_sv,
        swiglu_sv=swiglu_sv,
        num_cube_cores=num_cube_cores,
    )
    if integrated_permute:
        dispatch.fill_config.route_index_position = _POS_PERMUTE_SRC_INDEX
        dispatch.fill_config.route_weight_position = _POS_ROUTE_WEIGHT
        dispatch.fill_config.route_top_k = tsv.top_k
        dispatch.fill_config.isolated_route_signals = True
        dispatch.fill_config.route_completion_event = tsv.all_event_num + 5
        dispatch.fill_config.wait_for_copy_ack = not inplace_wave_backward
        dispatch.fill_config.destination_wave_fence = destination_wave_fence
    dispatch.fill_config.route_generation_phase = (
        0 if inplace_wave_backward else 1
    )
    dispatch.fill_config.route_generation_count = 2
    dispatch.fill_config.dependent_event = activation_barrier_event

    gate_grad, combine, w1_grad = _build_bwd_ops_second(
        specs,
        gate_grad_sv=gate_grad_sv,
        w1_grad_sv=w1_grad_sv,
        combine_sv=combine_sv,
        num_cube_cores=num_cube_cores,
    )
    if inplace_wave_backward:
        inplace_hidden_for_dw = TensorSpec(
            "inplace_hidden_for_dw",
            [tsv.per_rank_seq, hidden_size],
            param_position=0,
            dtype_size=dtype_size,
            tensor_type=1,
            is_dynamic=True,
            transpose=True,
        )
        w1_grad.inputs[0] = inplace_hidden_for_dw
        w1_grad.param_positions[0] = 0
        w1_grad.fill_config.dependent_event = hidden_ready_event
        gate_grad.outputs = [specs[0]]
        gate_grad.param_positions[-1] = 0
        combine.inputs[1] = specs[0]
        combine.param_positions[1] = 0
    else:
        # Position 12 first stores a local copy of expert input for dW1. Once
        # dW1 has consumed it, gate_grad overwrites the same scratch with dX.
        copied_hidden_for_dw = TensorSpec(
            "copied_hidden_for_dw",
            [tsv.per_rank_seq, hidden_size],
            param_position=12,
            dtype_size=dtype_size,
            tensor_type=1,
            is_dynamic=True,
            transpose=True,
        )
        w1_grad.inputs[0] = copied_hidden_for_dw
        w1_grad.param_positions[0] = 12
    w1_grad.fill_config.global_trigger = False
    w1_grad.fill_config.advance = "cube"
    w1_grad.fill_config.reuse_retained_scratch = reuse_retained_dw_scratch
    w1_grad.fill_config.atomic_accumulation = atomic_dw_accumulation
    w2_grad.fill_config.global_trigger = False
    w2_grad.fill_config.reuse_retained_scratch = reuse_retained_dw_scratch
    w2_grad.fill_config.atomic_accumulation = atomic_dw_accumulation
    # act_grad and w2_grad intentionally retain their legacy overlap. Give the
    # dW branch a disjoint per-expert completion group while keeping
    # swiglu_grad dependent only on act_grad's original group.
    w2_grad.fill_config.advance = "cube_custom"
    w2_grad.fill_config.event_delta = 2 * tsv.single_rank_expert_num
    w2_grad.fill_config.completion_event_offset = tsv.single_rank_expert_num
    combine.outputs = [specs[13]]
    combine.param_positions[-1] = 13

    accumulate_w2 = _build_weight_grad_accumulator(
        "accumulate_w2_grad",
        specs[6],
        specs[19],
        w2_grad,
        output_position=6,
        num_vector_cores=num_cube_cores,
        reuse_retained_scratch=reuse_retained_dw_scratch,
        atomic_accumulation=atomic_dw_accumulation,
    )
    accumulate_w1 = _build_weight_grad_accumulator(
        "accumulate_w1_grad",
        specs[18],
        specs[19],
        w1_grad,
        output_position=18,
        num_vector_cores=num_cube_cores,
        reuse_retained_scratch=reuse_retained_dw_scratch,
        atomic_accumulation=atomic_dw_accumulation,
    )

    dispatch_hidden, copy_hidden = _build_multi_wave_hidden_dispatch_ops(
        tsv,
        specs,
        dispatch_sv=dispatch_sv,
        copy_sv=dispatch_sv,
        hidden_size=hidden_size,
        dtype_size=dtype_size,
        integrated_permute=integrated_hidden_permute,
        hidden_copy_barrier_event=hidden_ready_event,
        inplace_wave_backward=inplace_wave_backward,
        tile_phase_ack=tile_phase_ack,
        hidden_ready_event=hidden_ready_event,
    )
    recompute_activation = _build_multi_wave_activation_recompute(
        specs,
        swiglu_sv=swiglu_sv,
        dependency_event=(0 if inplace_wave_backward else hidden_ready_event),
        activation_barrier_event=activation_barrier_event,
    )

    graph = ComputeGraph()
    if inplace_wave_backward and tile_phase_ack:
        release_dy_tiles = _build_route_compute_ack(
            tsv,
            act_grad,
            w2_grad,
            dispatch_sv=dispatch_sv,
        )
        graph.add_op(recompute_activation)
        graph.add_op(dispatch)
        graph.add_op(act_grad)
        graph.add_op(w2_grad)
        graph.add_op(swiglu_grad)
        graph.add_op(release_dy_tiles)
        (
            graph.add_op(dispatch_hidden)
            .add_op(copy_hidden)
            .add_op(accumulate_w2)
            .add_op(w1_grad)
            .add_op(accumulate_w1)
            .add_op(gate_grad)
            .add_op(combine)
            .add_edge(recompute_activation, dispatch)
            .add_edge(dispatch, act_grad)
            .add_edge(dispatch, w2_grad)
            .add_edge(act_grad, swiglu_grad)
            .add_edge(act_grad, release_dy_tiles)
            .add_edge(w2_grad, release_dy_tiles)
            .add_edge(dispatch_hidden, copy_hidden)
            .add_edge(copy_hidden, accumulate_w2)
            .add_edge(w2_grad, accumulate_w2)
            .add_edge(copy_hidden, w1_grad)
            .add_edge(swiglu_grad, w1_grad)
            .add_edge(w1_grad, accumulate_w1)
            .add_edge(w1_grad, gate_grad)
            .add_edge(gate_grad, combine)
        )
        graph.add_edge(release_dy_tiles, dispatch_hidden)
    elif inplace_wave_backward:
        phase_barrier = _build_phase_barrier()
        (
            graph.add_op(recompute_activation)
            .add_op(dispatch)
            .add_op(act_grad)
            .add_op(w2_grad)
            .add_op(accumulate_w2)
            .add_op(phase_barrier)
            .add_op(dispatch_hidden)
            .add_op(copy_hidden)
            .add_op(swiglu_grad)
            .add_op(w1_grad)
            .add_op(accumulate_w1)
            .add_op(gate_grad)
            .add_op(combine)
            .add_edge(recompute_activation, dispatch)
            .add_edge(dispatch, act_grad)
            .add_edge(dispatch, w2_grad)
            .add_edge(w2_grad, accumulate_w2)
            .add_edge(act_grad, swiglu_grad)
            .add_edge(accumulate_w2, phase_barrier)
            .add_edge(swiglu_grad, phase_barrier)
            .add_edge(phase_barrier, dispatch_hidden)
            .add_edge(dispatch_hidden, copy_hidden)
            .add_edge(copy_hidden, w1_grad)
            .add_edge(swiglu_grad, w1_grad)
            .add_edge(w1_grad, accumulate_w1)
            .add_edge(w1_grad, gate_grad)
            .add_edge(gate_grad, combine)
        )
    else:
        (
            graph.add_op(dispatch_hidden)
            .add_op(copy_hidden)
            .add_op(recompute_activation)
            .add_op(dispatch)
            .add_op(act_grad)
            .add_op(w2_grad)
            .add_op(accumulate_w2)
            .add_op(swiglu_grad)
            .add_op(w1_grad)
            .add_op(accumulate_w1)
            .add_op(gate_grad)
            .add_op(combine)
            .add_edge(dispatch_hidden, copy_hidden)
            .add_edge(copy_hidden, recompute_activation)
            .add_edge(recompute_activation, dispatch)
            .add_edge(dispatch, act_grad)
            .add_edge(dispatch, w2_grad)
            .add_edge(w2_grad, accumulate_w2)
            .add_edge(act_grad, swiglu_grad)
            .add_edge(swiglu_grad, w1_grad)
            .add_edge(w1_grad, accumulate_w1)
            .add_edge(w1_grad, gate_grad)
            .add_edge(gate_grad, combine)
        )

    if not (integrated_permute or integrated_unpermute):
        return _append_remote_write_acquire(
            graph,
            tsv,
            combine,
            output_position=13,
            hidden_size=hidden_size,
            dtype_size=dtype_size,
            num_cube_cores=num_cube_cores,
            source_position=None,
        )

    (
        local_tokens,
        expert_output,
        reduce_route_index,
        grad_x,
        grad_output,
        grad_topk_weight,
    ) = _build_route_backward_tensors(tsv, hidden_size, dtype_size)
    # CANN 9.1 accepts at most 32 custom-op inputs. Position 13 therefore
    # retains the forward expert output only until the first Combine, while
    # dTopK consumes position 31 after the inverse-permutation mapping is dead.
    if integrated_permute:
        grad_topk_weight.param_position = _POS_REDUCE_ROUTE_INDEX
        combine.fill_config.additional_dependent_event = topk_completion_event
    route_completion_event = tsv.all_event_num - 1
    combine.fill_config.trigger_event = route_completion_event
    topk_weight_grad = _append_topk_weight_grad(
        graph,
        tsv,
        combine,
        enabled=integrated_permute,
        local_tokens=local_tokens,
        topk_grad_sv=topk_grad_sv,
        route_completion_event=topk_completion_event,
        expert_output=expert_output,
        grad_output=grad_output,
        reduce_route_index=reduce_route_index,
        grad_topk_weight=grad_topk_weight,
        first_wave_only=True,
    )
    return _append_route_backward_output(
        graph,
        tsv,
        combine,
        integrated_unpermute=integrated_unpermute,
        local_tokens=local_tokens,
        unpermute_sv=unpermute_sv,
        hidden_size=hidden_size,
        dtype_size=dtype_size,
        num_cube_cores=num_cube_cores,
        target=specs[13],
        reduce_route_index=reduce_route_index,
        grad_x=grad_x,
        topk_weight_grad=topk_weight_grad,
        combine_output_position=13,
    )
