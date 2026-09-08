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
"""Build sized plan descriptors for managed MegaMoe execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import (
    build_backward_graph,
    build_multi_wave_backward_graph,
    build_route_backward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.tiling_tables import (
    get_act_grad_tiling_bytes,
    get_gate_grad_tiling_bytes,
    get_swiglu_grad_tiling_bytes,
    get_w1_grad_tiling_bytes,
    get_w2_grad_tiling_bytes,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank as build_forward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_forward_graph,
    build_route_forward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.tiling_tables import (
    get_down_proj_tiling_bytes,
    get_swiglu_tiling_bytes,
    get_up_proj_tiling_bytes,
)
from hyper_parallel.core.multicore.scheduler.config import (
    MAX_EVENT_NUM,
    SwiGluTilingDataC,
    TaskSplitValue,
    compact_runtime_enabled,
    serialize_runtime_config,
)
from hyper_parallel.core.multicore.scheduler.graph import OpType

from .spec import MegaMoeSpec

torch = import_module("torch")


_INPLACE_WAVE_BACKWARD_ENV = "HP_MEGA_MOE_INPLACE_WAVE_BACKWARD"
_TILE_PHASE_ACK_ENV = "HP_MEGA_MOE_TILE_PHASE_ACK"
_REUSE_RETAINED_DW_SCRATCH_ENV = "HP_MEGA_MOE_REUSE_RETAINED_DW_SCRATCH"
_ATOMIC_DW_ACCUMULATION_ENV = "HP_MEGA_MOE_ATOMIC_DW_ACCUMULATION"
_DESTINATION_WAVE_FENCE_ENV = "HP_MEGA_MOE_DESTINATION_WAVE_FENCE"


@dataclass(frozen=True)
class MegaMoePlan:
    """Rank-local schedules and tiling tensors for forward and backward."""

    spec: MegaMoeSpec
    fwd_runtime_config: Any
    up_proj_tiling: Any
    swiglu_tiling: Any
    down_proj_tiling: Any
    bwd_runtime_config: Any
    bwd_multi_wave_runtime_config: Any
    act_grad_tiling: Any
    gate_grad_tiling: Any
    w1_grad_tiling: Any
    w2_grad_tiling: Any
    swiglu_grad_tiling: Any
    route_slots: Any | None
    inplace_wave_backward: bool
    reuse_retained_dw_scratch: bool
    atomic_dw_accumulation: bool
    destination_wave_fence: bool
    event_capacity: int = MAX_EVENT_NUM


@dataclass(frozen=True)
class _MegaMoeTilings:
    """Device tiling tensors referenced by the forward and backward graphs."""

    up_proj: Any
    swiglu: Any
    down_proj: Any
    act_grad: Any
    gate_grad: Any
    w1_grad: Any
    w2_grad: Any
    swiglu_grad: Any


def _tensor_from_bytes(data: bytes, device: Any) -> Any:
    """Copy serialized plan data into a device uint8 tensor."""


    array = np.frombuffer(bytearray(data), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(device=device, dtype=torch.uint8)


def _serialize_schedule(config: Any) -> bytes:
    """Select the tensor-table ABI for controlled dense/compact comparisons."""
    return serialize_runtime_config(config, compact=compact_runtime_enabled())


def _resize_swiglu_tiling(
    data: bytes,
    num_cube_cores: int,
    split_value: int,
) -> bytes:
    """Resize worker records and append exact tail-row tilings."""
    baseline_workers = 49
    if len(data) % baseline_workers:
        raise RuntimeError("invalid baseline SwiGLU tiling byte length.")
    struct_size = len(data) // baseline_workers
    if struct_size != len(bytes(SwiGluTilingDataC())):
        raise RuntimeError("invalid baseline SwiGLU tiling struct size.")
    base = SwiGluTilingDataC.from_buffer_copy(data[:struct_size])
    base.baseColLen = min(base.baseColLen, base.colLen)
    worker_tilings = bytes(base) * (2 * num_cube_cores + 1)
    tail_tilings = []
    for rows in range(1, split_value):
        tail = SwiGluTilingDataC.from_buffer_copy(bytes(base))
        tail.rowLen = rows
        tail.baseRowLen = min(tail.baseRowLen, rows)
        tail_tilings.append(bytes(tail))
    return worker_tilings + b"".join(tail_tilings)


def _build_task_values(spec: MegaMoeSpec) -> TaskSplitValue:
    """Translate rank-local token shape into legacy scheduler coordinates.

    Args:
        spec: Validated local-token and expert topology configuration.

    Returns:
        Scheduler values whose single partition is the rank-local input.
    """
    return TaskSplitValue(
        tp=1,
        ep=spec.ep_size,
        seq_size=spec.local_num_tokens,
        all_expert_num=spec.physical_num_experts,
        top_k=spec.top_k,
    )


def _use_token_a2a_offsets(graph: Any) -> None:
    """Keep route metadata in token rows through the custom-op boundary."""
    for op in graph.topological_sort():
        if op.op_type == OpType.ALLTOALL:
            op.fill_config.offsets_in_tokens = True


def _use_fp32_weight_gradients(graph: Any) -> None:
    """Keep dW GMM outputs and later-wave accumulator views in FP32."""
    for op in graph.topological_sort():
        for spec in (*op.inputs, *op.outputs):
            if spec.param_position in (6, 18):
                spec.dtype_size = 4


def _inplace_wave_backward_enabled() -> bool:
    """Return the validated internal selector for the P2.1 A/B experiment."""
    value = os.getenv(_INPLACE_WAVE_BACKWARD_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(
            f"{_INPLACE_WAVE_BACKWARD_ENV} must be 0 or 1, got {value!r}."
        )
    return value == "1"


def _tile_phase_ack_enabled() -> bool:
    """Return the validated internal selector for the P2.2 A/B experiment."""
    value = os.getenv(_TILE_PHASE_ACK_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(
            f"{_TILE_PHASE_ACK_ENV} must be 0 or 1, got {value!r}."
        )
    return value == "1"


def _reuse_retained_dw_scratch_enabled() -> bool:
    """Return the validated internal selector for the P3 HBM experiment."""
    value = os.getenv(_REUSE_RETAINED_DW_SCRATCH_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(
            f"{_REUSE_RETAINED_DW_SCRATCH_ENV} must be 0 or 1, got {value!r}."
        )
    return value == "1"


def _atomic_dw_accumulation_enabled() -> bool:
    """Return the validated internal selector for direct Matmul dW add."""
    value = os.getenv(_ATOMIC_DW_ACCUMULATION_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(
            f"{_ATOMIC_DW_ACCUMULATION_ENV} must be 0 or 1, got {value!r}."
        )
    return value == "1"


def _destination_wave_fence_enabled() -> bool:
    """Return whether each next-wave dispatch waits on its destination rank."""
    value = os.getenv(_DESTINATION_WAVE_FENCE_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(
            f"{_DESTINATION_WAVE_FENCE_ENV} must be 0 or 1, got {value!r}."
        )
    return value == "1"


def _build_forward_schedule(
    spec: MegaMoeSpec,
    task_values: TaskSplitValue,
    device: Any,
) -> tuple[Any, Any]:
    """Build the selected forward graph and its compact device descriptor."""
    builder = (
        build_route_forward_graph
        if spec.forward_permute_fused
        else build_forward_graph
    )
    graph = builder(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        integrated_unpermute=spec.forward_unpermute_fused,
    )
    _use_token_a2a_offsets(graph)
    graph.propagate_splits(task_values)
    data = build_forward_config(
        graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    return graph, _tensor_from_bytes(_serialize_schedule(data), device)


def _build_backward_schedule(
    spec: MegaMoeSpec,
    task_values: TaskSplitValue,
    device: Any,
    *,
    inplace_wave_backward: bool,
    tile_phase_ack: bool,
    reuse_retained_dw_scratch: bool,
    atomic_dw_accumulation: bool,
    destination_wave_fence: bool,
) -> tuple[Any, Any, Any]:
    """Build ordinary and multi-wave backward device descriptors."""
    permute_fused = spec.backward_permute_fused
    unpermute_fused = spec.backward_unpermute_fused
    inplace_gate_dx = inplace_wave_backward and permute_fused
    route_fused = permute_fused or unpermute_fused
    builder = build_route_backward_graph if route_fused else build_backward_graph
    options = (
        {
            "integrated_permute": permute_fused,
            "integrated_unpermute": unpermute_fused and not inplace_gate_dx,
            "inplace_gate_dx": inplace_gate_dx,
        }
        if route_fused
        else {}
    )
    graph = builder(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        **options,
    )
    if spec.replica_slots_per_rank:
        _use_fp32_weight_gradients(graph)
    _use_token_a2a_offsets(graph)
    graph.propagate_splits(task_values)
    data = build_backward_config(
        graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    swiglu_grad = graph.get_op("swiglu_grad")
    swiglu_grad_tiling = _resize_swiglu_tiling(
        get_swiglu_grad_tiling_bytes(
            swiglu_grad.split_value,
            intermediate_size=spec.intermediate_size,
        ),
        spec.num_cube_cores,
        swiglu_grad.split_value,
    )
    multi_wave_graph = build_multi_wave_backward_graph(
        task_values,
        integrated_permute=permute_fused,
        # The retained W1 input is token-major whenever either direction
        # avoided materializing the forward expert-major permutation. This is
        # independent from whether dY itself uses route-fused dispatch.
        integrated_hidden_permute=(
            spec.forward_permute_fused or permute_fused
        ),
        # In-place replay depends on the fused dispatch endpoint. Keep the
        # global opt-in compatible with topologies where route fusion is not
        # available by building their existing non-in-place fallback graph.
        inplace_wave_backward=inplace_gate_dx,
        tile_phase_ack=tile_phase_ack,
        reuse_retained_dw_scratch=reuse_retained_dw_scratch,
        atomic_dw_accumulation=atomic_dw_accumulation,
        destination_wave_fence=destination_wave_fence,
        # The rolling dispatch buffer is overwritten by the next wave, while
        # final dX needs one full routed staging tensor. Keep the final inverse
        # permutation outside the multi-wave kernel so it can consume that
        # staging tensor without an additional full-size copy.
        integrated_unpermute=False,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
    )
    if spec.replica_slots_per_rank:
        _use_fp32_weight_gradients(multi_wave_graph)
    _use_token_a2a_offsets(multi_wave_graph)
    multi_wave_graph.propagate_splits(task_values)
    recompute_activation = multi_wave_graph.get_op("recompute_activation")
    recompute_activation.fill_config.tiling_data_offset = len(
        swiglu_grad_tiling
    )
    multi_wave_data = build_backward_config(
        multi_wave_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
        kernel_multi_wave=True,
    )
    return (
        graph,
        _tensor_from_bytes(_serialize_schedule(data), device),
        _tensor_from_bytes(_serialize_schedule(multi_wave_data), device),
    )


def _build_tilings(
    spec: MegaMoeSpec,
    forward_graph: Any,
    backward_graph: Any,
    device: Any,
) -> _MegaMoeTilings:
    """Materialize all tiling blobs referenced by the two schedules."""
    gmm_options = {
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_groups": spec.physical_local_experts,
        "num_cube_cores": spec.num_cube_cores,
    }
    up_proj = forward_graph.get_op("up_proj")
    swiglu = forward_graph.get_op("swiglu")
    down_proj = forward_graph.get_op("down_proj")
    act_grad = backward_graph.get_op("act_grad")
    gate_grad = backward_graph.get_op("gate_grad")
    w1_grad = backward_graph.get_op("w1_grad")
    w2_grad = backward_graph.get_op("w2_grad")
    swiglu_grad = backward_graph.get_op("swiglu_grad")
    swiglu_grad_tiling = _resize_swiglu_tiling(
        get_swiglu_grad_tiling_bytes(
            swiglu_grad.split_value,
            intermediate_size=spec.intermediate_size,
        ),
        spec.num_cube_cores,
        swiglu_grad.split_value,
    )
    recompute_activation_tiling = _resize_swiglu_tiling(
        get_swiglu_tiling_bytes(
            spec.swiglu_split,
            intermediate_size=spec.intermediate_size,
        ),
        spec.num_cube_cores,
        spec.swiglu_split,
    )
    return _MegaMoeTilings(
        up_proj=_tensor_from_bytes(
            get_up_proj_tiling_bytes(up_proj.split_value, **gmm_options), device
        ),
        swiglu=_tensor_from_bytes(
            _resize_swiglu_tiling(
                get_swiglu_tiling_bytes(
                    swiglu.split_value,
                    intermediate_size=spec.intermediate_size,
                ),
                spec.num_cube_cores,
                swiglu.split_value,
            ),
            device,
        ),
        down_proj=_tensor_from_bytes(
            get_down_proj_tiling_bytes(down_proj.split_value, **gmm_options), device
        ),
        act_grad=_tensor_from_bytes(
            get_act_grad_tiling_bytes(act_grad.split_value, **gmm_options), device
        ),
        gate_grad=_tensor_from_bytes(
            get_gate_grad_tiling_bytes(gate_grad.split_value, **gmm_options), device
        ),
        w1_grad=_tensor_from_bytes(
            get_w1_grad_tiling_bytes(w1_grad.split_value, **gmm_options), device
        ),
        w2_grad=_tensor_from_bytes(
            get_w2_grad_tiling_bytes(w2_grad.split_value, **gmm_options), device
        ),
        swiglu_grad=_tensor_from_bytes(
            swiglu_grad_tiling + recompute_activation_tiling,
            device,
        ),
    )


def _build_route_slots(spec: MegaMoeSpec, device: Any) -> Any | None:
    """Cache route-slot IDs used only by permute-fused routing."""
    if not (spec.forward_permute_fused or spec.backward_permute_fused):
        return None
    return torch.arange(
        spec.routed_slots,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)


def build_mega_moe_plan(
    spec: MegaMoeSpec, device: Any, *, event_capacity: int | None = None,
) -> MegaMoePlan:
    """Build the selected forward/backward route endpoint schedules.

    Args:
        spec: Validated local-token and expert topology configuration.
        device: NPU device receiving serialized descriptors and tiling tensors.
        event_capacity: Shared event layout for a configured plan and its home fallback.

    Returns:
        Rank-local forward and backward plan resources.
    """
    task_values = _build_task_values(spec)
    if event_capacity is not None:
        if event_capacity < task_values.runtime_event_capacity or event_capacity % 16:
            raise ValueError("shared event capacity is insufficient or unaligned")
        task_values.event_capacity = event_capacity
    inplace_wave_backward = _inplace_wave_backward_enabled()
    tile_phase_ack = _tile_phase_ack_enabled()
    reuse_retained_dw_scratch = _reuse_retained_dw_scratch_enabled()
    atomic_dw_accumulation = _atomic_dw_accumulation_enabled()
    destination_wave_fence = _destination_wave_fence_enabled()
    if tile_phase_ack and not inplace_wave_backward:
        raise ValueError(
            f"{_TILE_PHASE_ACK_ENV}=1 requires {_INPLACE_WAVE_BACKWARD_ENV}=1."
        )
    if reuse_retained_dw_scratch and atomic_dw_accumulation:
        raise ValueError(
            f"{_REUSE_RETAINED_DW_SCRATCH_ENV}=1 cannot be combined with "
            f"{_ATOMIC_DW_ACCUMULATION_ENV}=1."
        )
    if destination_wave_fence and not tile_phase_ack:
        raise ValueError(
            f"{_DESTINATION_WAVE_FENCE_ENV}=1 requires {_TILE_PHASE_ACK_ENV}=1."
        )
    forward_graph, forward_config = _build_forward_schedule(
        spec, task_values, device
    )
    backward_graph, backward_config, multi_wave_backward_config = (
        _build_backward_schedule(
            spec,
            task_values,
            device,
            inplace_wave_backward=inplace_wave_backward,
            tile_phase_ack=tile_phase_ack,
            reuse_retained_dw_scratch=reuse_retained_dw_scratch,
            atomic_dw_accumulation=atomic_dw_accumulation,
            destination_wave_fence=destination_wave_fence,
        )
    )
    tilings = _build_tilings(spec, forward_graph, backward_graph, device)
    return MegaMoePlan(
        spec=spec,
        fwd_runtime_config=forward_config,
        up_proj_tiling=tilings.up_proj,
        swiglu_tiling=tilings.swiglu,
        down_proj_tiling=tilings.down_proj,
        bwd_runtime_config=backward_config,
        bwd_multi_wave_runtime_config=multi_wave_backward_config,
        act_grad_tiling=tilings.act_grad,
        gate_grad_tiling=tilings.gate_grad,
        w1_grad_tiling=tilings.w1_grad,
        w2_grad_tiling=tilings.w2_grad,
        swiglu_grad_tiling=tilings.swiglu_grad,
        route_slots=_build_route_slots(spec, device),
        inplace_wave_backward=inplace_wave_backward,
        reuse_retained_dw_scratch=reuse_retained_dw_scratch,
        atomic_dw_accumulation=atomic_dw_accumulation,
        destination_wave_fence=destination_wave_fence,
        event_capacity=task_values.runtime_event_capacity,
    )
