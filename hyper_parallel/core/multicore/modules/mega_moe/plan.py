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
"""Build legacy-ABI plan descriptors for managed MegaMoe execution."""

from __future__ import annotations

from importlib import import_module

from dataclasses import dataclass
from typing import Any

import numpy as np

from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import (
    build_backward_graph,
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
    TaskSplitValue,
    serialize_runtime_config,
)

from .spec import MegaMoeSpec

torch = import_module("torch")


@dataclass(frozen=True)
class MegaMoePlan:
    """Rank-local schedules and tiling tensors for forward and backward."""

    spec: MegaMoeSpec
    fwd_runtime_config: Any
    up_proj_tiling: Any
    swiglu_tiling: Any
    down_proj_tiling: Any
    bwd_runtime_config: Any
    act_grad_tiling: Any
    gate_grad_tiling: Any
    w1_grad_tiling: Any
    w2_grad_tiling: Any
    swiglu_grad_tiling: Any
    route_token_rows: Any | None


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


def _resize_swiglu_tiling(data: bytes, num_cube_cores: int) -> bytes:
    """Resize the baseline 24-core repeated struct for the active workers."""
    baseline_workers = 49
    if len(data) % baseline_workers:
        raise RuntimeError("invalid baseline SwiGLU tiling byte length.")
    struct_size = len(data) // baseline_workers
    return data[:struct_size] * (2 * num_cube_cores + 1)


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
        all_expert_num=spec.num_experts,
        top_k=spec.top_k,
    )


def _build_forward_schedule(
    spec: MegaMoeSpec,
    task_values: TaskSplitValue,
    device: Any,
) -> tuple[Any, Any]:
    """Build the selected forward graph and its compact device descriptor."""
    permute_fused = spec.fusion_mode in {"permute", "both"}
    builder = build_route_forward_graph if permute_fused else build_forward_graph
    graph = builder(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        integrated_unpermute=spec.fusion_mode in {"unpermute", "both"},
    )
    graph.propagate_splits(task_values)
    data = build_forward_config(
        graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    return graph, _tensor_from_bytes(serialize_runtime_config(data), device)


def _build_backward_schedule(
    spec: MegaMoeSpec,
    task_values: TaskSplitValue,
    device: Any,
) -> tuple[Any, Any]:
    """Build the selected backward graph and its compact device descriptor."""
    permute_fused = spec.fusion_mode in {"permute", "both"}
    unpermute_fused = spec.fusion_mode in {"unpermute", "both"}
    route_fused = permute_fused or unpermute_fused
    builder = build_route_backward_graph if route_fused else build_backward_graph
    options = (
        {
            "integrated_permute": permute_fused,
            "integrated_unpermute": unpermute_fused,
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
    graph.propagate_splits(task_values)
    data = build_backward_config(
        graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    return graph, _tensor_from_bytes(serialize_runtime_config(data), device)


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
        "num_groups": spec.local_experts,
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
            _resize_swiglu_tiling(
                get_swiglu_grad_tiling_bytes(
                    swiglu_grad.split_value,
                    intermediate_size=spec.intermediate_size,
                ),
                spec.num_cube_cores,
            ),
            device,
        ),
    )


def _build_route_token_rows(spec: MegaMoeSpec, device: Any) -> Any | None:
    """Cache the token-row payload used only by permute-fused routing."""
    if spec.fusion_mode not in {"permute", "both"}:
        return None
    return torch.arange(
        spec.local_num_tokens,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)


def build_mega_moe_plan(spec: MegaMoeSpec, device: Any) -> MegaMoePlan:
    """Build the selected forward/backward route endpoint schedules.

    Args:
        spec: Validated local-token and expert topology configuration.
        device: NPU device receiving serialized descriptors and tiling tensors.

    Returns:
        Rank-local forward and backward plan resources.
    """
    task_values = _build_task_values(spec)
    forward_graph, forward_config = _build_forward_schedule(
        spec, task_values, device
    )
    backward_graph, backward_config = _build_backward_schedule(
        spec, task_values, device
    )
    tilings = _build_tilings(spec, forward_graph, backward_graph, device)
    return MegaMoePlan(
        spec=spec,
        fwd_runtime_config=forward_config,
        up_proj_tiling=tilings.up_proj,
        swiglu_tiling=tilings.swiglu,
        down_proj_tiling=tilings.down_proj,
        bwd_runtime_config=backward_config,
        act_grad_tiling=tilings.act_grad,
        gate_grad_tiling=tilings.gate_grad,
        w1_grad_tiling=tilings.w1_grad,
        w2_grad_tiling=tilings.w2_grad,
        swiglu_grad_tiling=tilings.swiglu_grad,
        route_token_rows=_build_route_token_rows(spec, device),
    )
