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
"""Build legacy-ABI runtime descriptors for managed MegaMoe execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
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
from hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank as build_forward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_forward_graph,
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

from ._framework import get_torch
from .config import MegaMoeConfig


@dataclass(frozen=True)
class MegaMoeRuntime:
    """Rank-local schedules and tiling tensors for forward and backward."""

    config: MegaMoeConfig
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


def _tensor_from_bytes(data: bytes, device: Any) -> Any:
    """Copy serialized runtime data into a device uint8 tensor."""
    # pylint: disable=C0415
    import numpy as np

    torch = get_torch()

    array = np.frombuffer(bytearray(data), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(device=device, dtype=torch.uint8)


def _resize_swiglu_tiling(data: bytes, num_cube_cores: int) -> bytes:
    """Resize the baseline 24-core repeated struct for the active workers."""
    baseline_workers = 49
    if len(data) % baseline_workers:
        raise RuntimeError("invalid baseline SwiGLU tiling byte length.")
    struct_size = len(data) // baseline_workers
    return data[:struct_size] * (2 * num_cube_cores + 1)


def _build_task_values(config: MegaMoeConfig) -> TaskSplitValue:
    """Translate rank-local token shape into legacy scheduler coordinates.

    Args:
        config: Validated local-token and expert topology configuration.

    Returns:
        Scheduler values whose single partition is the rank-local input.
    """
    return TaskSplitValue(
        tp=1,
        ep=config.ep_size,
        seq_size=config.local_num_tokens,
        all_expert_num=config.num_experts,
        top_k=config.top_k,
    )


def build_mega_moe_runtime(config: MegaMoeConfig, device: Any) -> MegaMoeRuntime:
    """Build the current legacy forward/backward ABI without fusion slots.

    Args:
        config: Validated local-token and expert topology configuration.
        device: NPU device receiving serialized descriptors and tiling tensors.

    Returns:
        Rank-local forward and backward runtime resources.
    """
    task_values = _build_task_values(config)
    forward_graph = build_forward_graph(
        task_values,
        dispatch_sv=config.dispatch_split,
        swiglu_sv=config.swiglu_split,
        combine_sv=config.combine_split,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_cube_cores=config.num_cube_cores,
    )
    forward_graph.propagate_splits(task_values)
    forward_data = build_forward_config(
        forward_graph,
        task_values,
        config.rank_id,
        config.num_cube_cores,
    )

    backward_graph = build_backward_graph(
        task_values,
        dispatch_sv=config.dispatch_split,
        swiglu_sv=config.swiglu_split,
        combine_sv=config.combine_split,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_cube_cores=config.num_cube_cores,
    )
    backward_graph.propagate_splits(task_values)
    backward_data = build_backward_config(
        backward_graph,
        task_values,
        config.rank_id,
        config.num_cube_cores,
    )

    gmm_options = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_groups": config.local_experts,
        "num_cube_cores": config.num_cube_cores,
    }
    up_proj = forward_graph.get_op("up_proj")
    swiglu = forward_graph.get_op("swiglu")
    down_proj = forward_graph.get_op("down_proj")
    act_grad = backward_graph.get_op("act_grad")
    gate_grad = backward_graph.get_op("gate_grad")
    w1_grad = backward_graph.get_op("w1_grad")
    w2_grad = backward_graph.get_op("w2_grad")
    swiglu_grad = backward_graph.get_op("swiglu_grad")
    return MegaMoeRuntime(
        config=config,
        fwd_runtime_config=_tensor_from_bytes(
            serialize_runtime_config(forward_data),
            device,
        ),
        up_proj_tiling=_tensor_from_bytes(
            get_up_proj_tiling_bytes(up_proj.split_value, **gmm_options), device
        ),
        swiglu_tiling=_tensor_from_bytes(
            _resize_swiglu_tiling(
                get_swiglu_tiling_bytes(
                    swiglu.split_value,
                    intermediate_size=config.intermediate_size,
                ),
                config.num_cube_cores,
            ),
            device,
        ),
        down_proj_tiling=_tensor_from_bytes(
            get_down_proj_tiling_bytes(down_proj.split_value, **gmm_options), device
        ),
        bwd_runtime_config=_tensor_from_bytes(
            serialize_runtime_config(backward_data),
            device,
        ),
        act_grad_tiling=_tensor_from_bytes(
            get_act_grad_tiling_bytes(act_grad.split_value, **gmm_options), device
        ),
        gate_grad_tiling=_tensor_from_bytes(
            get_gate_grad_tiling_bytes(gate_grad.split_value, **gmm_options), device
        ),
        w1_grad_tiling=_tensor_from_bytes(
            get_w1_grad_tiling_bytes(w1_grad.split_value, **gmm_options), device
        ),
        w2_grad_tiling=_tensor_from_bytes(
            get_w2_grad_tiling_bytes(w2_grad.split_value, **gmm_options), device
        ),
        swiglu_grad_tiling=_tensor_from_bytes(
            _resize_swiglu_tiling(
                get_swiglu_grad_tiling_bytes(
                    swiglu_grad.split_value,
                    intermediate_size=config.intermediate_size,
                ),
                config.num_cube_cores,
            ),
            device,
        ),
    )
