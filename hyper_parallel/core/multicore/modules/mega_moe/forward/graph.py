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
"""MoE-FFN forward compute graph.

前向 DAG 的固定顺序是 dispatch -> up projection -> SwiGLU -> down projection
-> combine。这里描述的是 kernel 调度图，不执行实际张量计算；``execution.py`` 会
再根据该图生成 rank-specific runtime descriptor。
"""

from typing import Any

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
from hyper_parallel.core.multicore.tasks.permute import WeightedUnpermuteFillConfig
from hyper_parallel.core.multicore.tasks.swiglu import SwiGLUFillConfig

# Tiling buffer positions in the C++ runtime descriptor array.
_TILING_POS_UP_PROJ = 17
_TILING_POS_SWIGLU  = 18
_TILING_POS_DN_PROJ = 19

_POS_ORIGINAL_X = 22
_POS_PERMUTE_SRC_INDEX = 23
_POS_REDUCE_ROUTE_INDEX = 24
_POS_TOPK_WEIGHTS = 25
_POS_FINAL_OUT = 26


def _build_fwd_tensor_specs(tsv, hidden_size, intermediate_size, dtype_size):
    """Create all TensorSpec objects for the forward graph."""
    # param_position 对应 C++ worker 看到的参数槽位，不能按 Python 调用习惯
    # 随意重排。动态 tensor 的第 0 维由实际 route 的 group_list 决定。
    sre = tsv.single_rank_expert_num
    target = TensorSpec(
        "target", [tsv.per_rank_seq, hidden_size],
        param_position=0, dtype_size=dtype_size, is_dynamic=True)
    target_offset = TensorSpec(
        "target_offset", [tsv.all_expert_num],
        param_position=1, dtype_size=8)
    src = TensorSpec(
        "src", [tsv.per_rank_seq, hidden_size],
        param_position=2, dtype_size=dtype_size)
    src_offset = TensorSpec(
        "src_offset", [tsv.all_expert_num],
        param_position=3, dtype_size=8)
    size_d = TensorSpec(
        "size_d", [tsv.all_expert_num],
        param_position=4, dtype_size=4)
    up_proj_weight = TensorSpec(
        "up_proj_weight", [sre, hidden_size, intermediate_size * 2],
        param_position=5, dtype_size=dtype_size, tensor_type=1)
    up_proj_glist = TensorSpec(
        "up_proj_glist", [sre],
        param_position=6, dtype_size=8)
    up_proj_y = TensorSpec(
        "up_proj_y", [tsv.per_rank_seq, intermediate_size * 2],
        param_position=7, dtype_size=dtype_size, is_dynamic=True)
    swiglu_out = TensorSpec(
        "swiglu_out", [tsv.per_rank_seq, intermediate_size],
        param_position=8, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    down_proj_weight = TensorSpec(
        "down_proj_weight", [sre, intermediate_size, hidden_size],
        param_position=9, dtype_size=dtype_size, tensor_type=1)
    down_proj_glist = TensorSpec(
        "down_proj_glist", [sre],
        param_position=10, dtype_size=8)
    down_proj_y = TensorSpec(
        "down_proj_y", [tsv.per_rank_seq, hidden_size],
        param_position=11, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    combine_out = TensorSpec(
        "combine_out", [tsv.per_rank_seq, hidden_size],
        param_position=12, dtype_size=dtype_size)
    target_offset_c = TensorSpec(
        "target_offset_c", [tsv.all_expert_num],
        param_position=13, dtype_size=8)
    src_offset_c = TensorSpec(
        "src_offset_c", [tsv.all_expert_num],
        param_position=14, dtype_size=8)
    size_c = TensorSpec(
        "size_c", [tsv.all_expert_num],
        param_position=15, dtype_size=4)
    return (target, target_offset, src, src_offset, size_d,
            up_proj_weight, up_proj_glist, up_proj_y, swiglu_out,
            down_proj_weight, down_proj_glist, down_proj_y,
            combine_out, target_offset_c, src_offset_c, size_c)


def _build_fwd_ops(tsv, specs, *, dispatch_sv, up_proj_sv, swiglu_sv,
                   down_proj_sv, combine_sv, num_cube_cores):
    """Create all OperatorNode objects for the forward graph."""
    # AllToAll 使用 vector worker，GMM 使用 cube worker，SwiGLU 再回到 vector
    # worker。fill_config 中的 event_group/advance 决定这些阶段如何流水重叠。
    (target, target_offset, src, src_offset, size_d,
     up_proj_weight, up_proj_glist, up_proj_y, swiglu_out,
     down_proj_weight, down_proj_glist, down_proj_y,
     combine_out, target_offset_c, src_offset_c, size_c) = specs
    dispatch = OperatorNode(
        name="dispatch", op_type=OpType.ALLTOALL,
        inputs=[target_offset, src, src_offset, size_d],
        outputs=[target],
        param_positions=[1, 2, 3, 4, 0],
        split_value=dispatch_sv,
        split_spec=SplitSpec(
            split_inputs=None, split_output_dims=[0],
            task_num_fn=lambda tsv: tsv.all_expert_num * (tsv.per_expert_seq_to_other // dispatch_sv),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(moe_type=AllToAllType.DISPATCH, advance="vector",
                                       event_group=tsv.all_expert_num),
    )
    up_proj = OperatorNode(
        name="up_proj", op_type=OpType.GMM,
        inputs=[target, up_proj_weight, up_proj_glist], outputs=[up_proj_y],
        param_positions=[0, 5, 6, 7], split_value=up_proj_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores:
                _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_UP_PROJ,
        fill_config=GmmFillConfig(offset_inputs={0}, rank_in_event=True, global_trigger=False,
                                  out_offset=True, advance="cube", num_cube_cores=num_cube_cores),
    )
    swiglu = OperatorNode(
        name="swiglu", op_type=OpType.SWIGLU,
        inputs=[up_proj_y], outputs=[swiglu_out],
        param_positions=[7, 8], split_value=swiglu_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv: (tsv.per_expert_seq // swiglu_sv) * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_SWIGLU,
        fill_config=SwiGLUFillConfig(),
    )
    down_proj = OperatorNode(
        name="down_proj", op_type=OpType.GMM,
        inputs=[swiglu_out, down_proj_weight, down_proj_glist], outputs=[down_proj_y],
        param_positions=[8, 9, 10, 11], split_value=down_proj_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores:
                _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_DN_PROJ,
        fill_config=GmmFillConfig(offset_inputs={0}, rank_in_event=False, global_trigger=False,
                                  out_offset=True, advance="cube", num_cube_cores=num_cube_cores),
    )
    combine = OperatorNode(
        name="combine", op_type=OpType.ALLTOALL,
        inputs=[target_offset_c, down_proj_y, src_offset_c, size_c], outputs=[combine_out],
        param_positions=[13, 11, 14, 15, 12], split_value=combine_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv: tsv.all_expert_num * (tsv.per_expert_seq_to_other // combine_sv),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(moe_type=AllToAllType.COMBINE, advance="vector", event_group=1),
    )
    return dispatch, up_proj, swiglu, down_proj, combine


def _append_weighted_unpermute(
    graph,
    tsv,
    specs,
    combine,
    *,
    hidden_size,
    dtype_size,
    unpermute_sv,
):
    """Append the exact FP32 weighted-unpermute epilogue to either forward DAG."""
    if unpermute_sv <= 0:
        raise ValueError("unpermute_sv must be positive")
    local_tokens = tsv.seq_size // tsv.tp
    reduce_route_index = TensorSpec(
        "reduce_route_index",
        [local_tokens, tsv.top_k],
        param_position=_POS_REDUCE_ROUTE_INDEX,
        dtype_size=4,
    )
    topk_weights = TensorSpec(
        "topk_weights",
        [local_tokens, tsv.top_k],
        param_position=_POS_TOPK_WEIGHTS,
        dtype_size=4,
    )
    final_out = TensorSpec(
        "final_out",
        [local_tokens, hidden_size],
        param_position=_POS_FINAL_OUT,
        dtype_size=dtype_size,
    )
    weighted_unpermute = OperatorNode(
        name="weighted_unpermute",
        op_type=OpType.WEIGHTED_UNPERMUTE,
        inputs=[specs[12], reduce_route_index, topk_weights],
        outputs=[final_out],
        param_positions=[
            12,
            _POS_REDUCE_ROUTE_INDEX,
            _POS_TOPK_WEIGHTS,
            _POS_FINAL_OUT,
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
        fill_config=WeightedUnpermuteFillConfig(
            dependent_event=tsv.all_event_num - 1,
            trigger_event=tsv.all_event_num,
            top_k=tsv.top_k,
        ),
    )
    combine.fill_config.trigger_event = tsv.all_event_num - 1
    graph.add_op(weighted_unpermute).add_edge(combine, weighted_unpermute)
    return graph


def _append_remote_write_acquire(
    graph,
    tsv,
    combine,
    *,
    hidden_size,
    dtype_size,
    num_cube_cores,
):
    """Acquire peer Combine writes before an external output consumer runs."""
    combine_out = TensorSpec(
        "combine_acquire_sentinel",
        [0, hidden_size],
        param_position=12,
        dtype_size=dtype_size,
    )
    acquire = OperatorNode(
        name="external_output_acquire",
        op_type=OpType.REMOTE_WRITE_ACQUIRE,
        inputs=[],
        outputs=[combine_out],
        param_positions=[12],
        split_value=1,
        split_spec=SplitSpec(
            split_inputs=None,
            split_output_dims=[-1],
            # One descriptor per active AIV is sufficient: every descriptor
            # performs its acquire before publishing terminal completion.
            task_num_fn=lambda _tsv: num_cube_cores,
        ),
        tiling_position=-1,
        fill_config=WeightedUnpermuteFillConfig(
            dependent_event=tsv.all_event_num - 1,
            trigger_event=tsv.all_event_num,
            top_k=0,
            acquire_remote_writes=True,
        ),
    )
    combine.fill_config.trigger_event = tsv.all_event_num - 1
    graph.add_op(acquire).add_edge(combine, acquire)
    return graph


def build_forward_graph(tsv: TaskSplitValue, *,
                        dispatch_sv:  int = 128,
                        up_proj_sv:   int = 4096,
                        swiglu_sv:    int = 128,
                        down_proj_sv: int = 4096,
                        combine_sv:   int = 128,
                        hidden_size:       int = 7168,
                        intermediate_size: int = 2048,
                        dtype_size:        int = 2,
                        num_cube_cores:    int = 24,
                        integrated_unpermute: bool = False,
                        unpermute_sv: int = 8) -> ComputeGraph:
    """Build the MoE-FFN forward DAG: dispatch -> up_proj -> swiglu -> down_proj -> combine.

    Operator execution order and param_positions (C++ memory slots):
      dispatch:  [target_offset=1, src=2, src_offset=3, size=4 | target=0]
      up_proj:   [x=0, weight=5, glist=6 | y=7]
      swiglu:    [x=7 | out=8]
      down_proj: [x=8, weight=9, glist=10 | y=11]
      combine:   [target_offset=13, src=11, src_offset=14, size=15 | target=12]

    Args:
        tsv: TaskSplitValue carrying TP/EP/seq partition metadata.
        dispatch_sv: tile size for dispatch AllToAll (vector cores).
        up_proj_sv: tile size for up-projection GMM (cube cores).
        swiglu_sv: tile size for SwiGLU (vector cores).
        down_proj_sv: tile size for down-projection GMM (cube cores).
        combine_sv: tile size for combine AllToAll (vector cores).
        hidden_size: model hidden dimension.
        intermediate_size: FFN intermediate dimension after SwiGLU halving.
        dtype_size: bytes per activation element (2=bf16, 4=fp32).
        num_cube_cores: number of AIC cube cores on the target device.

    Returns:
        A fully-connected ComputeGraph ready for propagate_splits().
    """
    # graph 的边只表达依赖关系，实际 task 数要等 propagate_splits(tsv) 后才能
    # 根据 TP/EP、split value 和专家数展开。
    # A persistent epilogue owns remote Combine completion in every mode.
    # Integrated unpermute consumes the data; external consumers need an
    # acquire-only epilogue before the custom-op stream can complete.
    tsv.extra_event_num = max(tsv.extra_event_num, 1)
    specs = _build_fwd_tensor_specs(tsv, hidden_size, intermediate_size, dtype_size)
    dispatch, up_proj, swiglu, down_proj, combine = _build_fwd_ops(
        tsv, specs,
        dispatch_sv=dispatch_sv, up_proj_sv=up_proj_sv, swiglu_sv=swiglu_sv,
        down_proj_sv=down_proj_sv, combine_sv=combine_sv, num_cube_cores=num_cube_cores,
    )
    graph = ComputeGraph()
    (graph.add_op(dispatch).add_op(up_proj).add_op(swiglu).add_op(down_proj).add_op(combine)
          .add_edge(dispatch,  up_proj)
          .add_edge(up_proj,   swiglu)
          .add_edge(swiglu,    down_proj)
          .add_edge(down_proj, combine))
    if integrated_unpermute:
        _append_weighted_unpermute(
            graph,
            tsv,
            specs,
            combine,
            hidden_size=hidden_size,
            dtype_size=dtype_size,
            unpermute_sv=unpermute_sv,
        )
    else:
        _append_remote_write_acquire(
            graph,
            tsv,
            combine,
            hidden_size=hidden_size,
            dtype_size=dtype_size,
            num_cube_cores=num_cube_cores,
        )
    return graph


def build_route_forward_graph(tsv: TaskSplitValue, **kwargs: Any) -> ComputeGraph:
    """Build forward with route-row gather fused into SHMEM dispatch.

    Args:
        tsv: Task split metadata for the forward graph.
        **kwargs: Forward graph dimensions, tiling, and fusion options.

    Returns:
        A route-aware forward graph ready for split propagation.
    """
    # 普通图的 dispatch 输入是已排好 expert-major 的 src；route 图把输入槽位改为
    # 原始 token-order x，并给 AllToAll descriptor 增加 permute_src_index 位置，
    # 由 worker 在发往远端时按索引取行，省掉独立的 routed_tokens materialization。
    integrated_unpermute = kwargs.get("integrated_unpermute", False)
    # Reserve a completion event after the per-expert Combine chain for either
    # the fused reduction or the external-consumer acquire epilogue.
    tsv.extra_event_num = max(tsv.extra_event_num, 1)
    hidden_size = kwargs.get("hidden_size", 7168)
    intermediate_size = kwargs.get("intermediate_size", 2048)
    dtype_size = kwargs.get("dtype_size", 2)
    specs = _build_fwd_tensor_specs(
        tsv, hidden_size, intermediate_size, dtype_size
    )
    dispatch, up_proj, swiglu, down_proj, combine = _build_fwd_ops(
        tsv,
        specs,
        dispatch_sv=kwargs.get("dispatch_sv", 128),
        up_proj_sv=kwargs.get("up_proj_sv", 4096),
        swiglu_sv=kwargs.get("swiglu_sv", 128),
        down_proj_sv=kwargs.get("down_proj_sv", 4096),
        combine_sv=kwargs.get("combine_sv", 128),
        num_cube_cores=kwargs.get("num_cube_cores", 24),
    )
    _, target_offset, _, src_offset, size_d = specs[:5]
    original_x = TensorSpec(
        "original_x",
        [tsv.seq_size // tsv.tp, hidden_size],
        param_position=_POS_ORIGINAL_X,
        dtype_size=dtype_size,
    )
    dispatch.inputs = [target_offset, original_x, src_offset, size_d]
    dispatch.param_positions = [1, _POS_ORIGINAL_X, 3, 4, 0]
    dispatch.fill_config.dependent_event = 0
    dispatch.fill_config.route_index_position = _POS_PERMUTE_SRC_INDEX
    dispatch.fill_config.isolated_route_signals = True

    graph = ComputeGraph()
    (graph.add_op(dispatch).add_op(up_proj).add_op(swiglu).add_op(down_proj)
          .add_op(combine).add_edge(dispatch, up_proj).add_edge(up_proj, swiglu)
          .add_edge(swiglu, down_proj).add_edge(down_proj, combine))
    if integrated_unpermute:
        _append_weighted_unpermute(
            graph,
            tsv,
            specs,
            combine,
            hidden_size=hidden_size,
            dtype_size=dtype_size,
            unpermute_sv=kwargs.get("unpermute_sv", 8),
        )
    else:
        _append_remote_write_acquire(
            graph,
            tsv,
            combine,
            hidden_size=hidden_size,
            dtype_size=dtype_size,
            num_cube_cores=kwargs.get("num_cube_cores", 24),
        )
    return graph
