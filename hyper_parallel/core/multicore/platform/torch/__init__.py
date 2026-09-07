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
hyper_parallel.core.multicore.platform.torch
=============================================
Out-of-tree PyTorch operator registration for MoE-FFN operators.

Registers into the ``hyper_parallel`` PyTorch namespace — does NOT modify
op-plugin or any PyTorch source. The operators are accessible via:

    torch.ops.hyper_parallel.mega_moe(...)
    torch.ops.hyper_parallel.mega_moe_grad(...)

Or via the Python wrappers in this module:

    from hyper_parallel.core.multicore.platform.torch import mega_moe, mega_moe_grad

Forward and backward ACLNN symbols are packaged in one component-owned
``hyper_parallel_multicore_nn`` vendor. Source the packaged ``set_env.bash``
before starting the application or framework Python process so CANN can discover that vendor.
"""

from __future__ import annotations

__all__ = ["mega_moe", "mega_moe_grad"]

import importlib
from functools import lru_cache
from typing import Any

from hyper_parallel.core.multicore._loader import (
    NativeComponentUnavailableError,
    get_multicore_paths,
    preload_vendor_library,
)


@lru_cache(maxsize=1)
def _get_torch():
    """Load torch and register the ABI-specific multicore adapter once."""
    vendor_root, adapter_path = get_multicore_paths("torch")
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    preload_vendor_library(vendor_root)
    try:
        torch.ops.load_library(str(adapter_path))
    except (OSError, RuntimeError) as error:
        raise NativeComponentUnavailableError(
            "[HP-NATIVE-FRAMEWORK-ADAPTER-LOAD-FAILED] component=multicore framework=torch "
            f"library={adapter_path} error={error}. "
            "Check the Python/Torch/torch_npu/CANN version combination and rebuild the Torch adapter."
        ) from error
    return torch


# ---------------------------------------------------------------------------
# Python wrappers — thin pass-through to the registered C++ ops
# ---------------------------------------------------------------------------


def mega_moe(
    dispatch_target: Any,
    dispatch_target_off: Any,
    dispatch_src: Any,
    dispatch_src_off: Any,
    dispatch_size: Any,
    up_proj_weight: Any,
    up_proj_glist: Any,
    up_proj_y: Any,
    swiglu_out: Any,
    down_proj_weight: Any,
    down_proj_glist: Any,
    down_proj_y: Any,
    combine_target: Any,
    combine_target_off: Any,
    combine_src_off: Any,
    combine_size: Any,
    gmm_workspace: Any,
    up_proj_tiling: Any,
    swiglu_tiling: Any,
    down_proj_tiling: Any,
    runtime_config: Any,
    all_event_counters: Any,
    rank_id: int,
    ep: int,
    expert_num: int,
    hidden_size: int,
    seq_size: int,
    *,
    original_x: Any = None,
    permute_src_index: Any = None,
    reduce_route_index: Any = None,
    topk_weights: Any = None,
    final_out: Any = None,
    cycle_profile_workspace: Any = None,
    receive_capacity: int | None = None,
    wave_count: int = 1,
    home_experts: int = 0,
    cycle_profile: int = 0,
) -> None:
    """
    MoE-FFN forward operator.

    Writes in-place to: dispatch_target, up_proj_y, swiglu_out, down_proj_y,
                        combine_target.
    All output tensors must be pre-allocated with correct shapes.

    Parameters
    ----------
    dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off,
    dispatch_size :
        AllToAll dispatch buffers — dispatch_target written in-place.
    up_proj_weight, up_proj_glist :
        Expert weight and cumulative group sizes for GMM1 (up-projection).
    up_proj_y, swiglu_out :
        GMM1 output and SwiGLU output — written in-place.
    down_proj_weight, down_proj_glist, down_proj_y :
        Expert weight, cumulative group sizes, and output for GMM2 (down-projection).
    combine_target, combine_target_off, combine_src_off, combine_size :
        AllToAll combine buffers — combine_target written in-place.
    gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling :
        Pre-computed tiling tensors (from gen_runtime_data.py).
    runtime_config :
        Per-rank runtime config tensor (from gen_runtime_data.py).
    all_event_counters :
        Event synchronization counter tensor.
    rank_id, ep, expert_num, hidden_size, seq_size :
        Topology / shape attributes.
    original_x, permute_src_index, reduce_route_index, topk_weights, final_out :
        Managed route-fusion buffers. Omitted values preserve the legacy
        expert-major, single-wave entry point; unused inputs use typed aliases.
    cycle_profile_workspace, receive_capacity, wave_count, home_experts, cycle_profile :
        Managed kernel workspace and multi-wave/shared-weight attributes.
    """
    torch = _get_torch()
    if topk_weights is None:
        topk_weights = dispatch_size[:1].to(torch.float32)
    torch.ops.hyper_parallel.mega_moe(
        dispatch_target,
        dispatch_target_off,
        dispatch_src,
        dispatch_src_off,
        dispatch_size,
        up_proj_weight,
        up_proj_glist,
        up_proj_y,
        swiglu_out,
        down_proj_weight,
        down_proj_glist,
        down_proj_y,
        combine_target,
        combine_target_off,
        combine_src_off,
        combine_size,
        gmm_workspace,
        up_proj_tiling,
        swiglu_tiling,
        down_proj_tiling,
        runtime_config,
        all_event_counters,
        dispatch_src if original_x is None else original_x,
        dispatch_size if permute_src_index is None else permute_src_index,
        dispatch_size if reduce_route_index is None else reduce_route_index,
        topk_weights,
        combine_target if final_out is None else final_out,
        gmm_workspace if cycle_profile_workspace is None else cycle_profile_workspace,
        rank_id,
        ep,
        expert_num,
        hidden_size,
        seq_size,
        dispatch_target.shape[0] if receive_capacity is None else receive_capacity,
        wave_count,
        home_experts,
        cycle_profile,
    )


def mega_moe_grad(
    dispatch_target: Any,
    dispatch_target_off: Any,
    dy: Any,
    dispatch_src_off: Any,
    dispatch_size: Any,
    hidden: Any,
    hidden_dw: Any,
    w2: Any,
    act_grad_y: Any,
    gate: Any,
    grad_gate: Any,
    w1: Any,
    gate_dx: Any,
    grad_x: Any,
    combine_target_off: Any,
    combine_src_off: Any,
    combine_size: Any,
    permute_out: Any,
    gate_dw: Any,
    group_list: Any,
    act_grad_tiling: Any,
    gate_grad_tiling: Any,
    w1_grad_tiling: Any,
    w2_grad_tiling: Any,
    swiglu_grad_tiling: Any,
    gmm_workspace: Any,
    swiglu_grad_workspace: Any,
    runtime_config: Any,
    all_event_counters: Any,
    rank_id: int,
    ep: int,
    expert_num: int,
    hidden_size: int,
    seq_size: int,
    *,
    permute_src_index: Any = None,
    route_weight: Any = None,
    reduce_route_index: Any = None,
    receive_capacity: int | None = None,
    wave_count: int = 1,
    cycle_profile: int = 0,
    home_experts: int = 0,
) -> None:
    """
    MoE-FFN backward operator.

    Writes in-place to: dispatch_target, hidden_dw, act_grad_y, grad_gate,
                        gate_dx, grad_x, permute_out, gate_dw.
    All output tensors must be pre-allocated with correct shapes.

    Parameters
    ----------
    dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size :
        AllToAll dispatch buffers — dispatch_target written in-place with
        the dispatched gradient.  dy is the source gradient tensor.
    hidden :
        SwiGLU output saved from the forward pass (used by W2-grad, GMM4).
    hidden_dw :
        W2 weight gradient — written in-place.
    w2 :
        W2 weight (= down_proj_weight from forward).
    act_grad_y :
        Activation gradient output from GMM1 bwd (target @ W2.T) — written in-place.
    gate :
        up_proj_y saved from the forward pass (SwiGLU input).
    grad_gate :
        SwiGLU gradient output — written in-place.
    w1 :
        W1 weight (= up_proj_weight from forward).
    gate_dx :
        GMM2 bwd output (grad_gate @ W1.T), before AllToAll combine — written in-place.
    grad_x :
        AllToAll combine output (final activation gradient) — written in-place.
    combine_target_off, combine_src_off, combine_size :
        AllToAll combine buffer descriptors.
    permute_out :
        In-place intermediate buffer for W1-grad (GMM4).
    gate_dw :
        W1 weight gradient — written in-place.
    group_list :
        Cumulative expert token counts ([E] int64).
    act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
    swiglu_grad_tiling :
        Pre-computed tiling tensors (from gen_runtime_data.py bwd).
    gmm_workspace, swiglu_grad_workspace :
        Workspace tensors.
    runtime_config :
        Per-rank runtime config tensor (from gen_runtime_data.py bwd).
    all_event_counters :
        Event synchronization counter tensor.
    rank_id, ep, expert_num, hidden_size, seq_size :
        Topology / shape attributes.
    permute_src_index, route_weight, reduce_route_index :
        Managed route-fusion inputs. Omitted values preserve the legacy
        expert-major entry point and are unused by its descriptor graph.
    receive_capacity, wave_count, cycle_profile, home_experts :
        Managed multi-wave, diagnostic and shared-weight attributes.
    """
    torch = _get_torch()
    if route_weight is None:
        route_weight = dispatch_size[:1].to(torch.float32)
    torch.ops.hyper_parallel.mega_moe_grad(
        dispatch_target,
        dispatch_target_off,
        dy,
        dispatch_src_off,
        dispatch_size,
        hidden,
        hidden_dw,
        w2,
        act_grad_y,
        gate,
        grad_gate,
        w1,
        gate_dx,
        grad_x,
        combine_target_off,
        combine_src_off,
        combine_size,
        permute_out,
        gate_dw,
        group_list,
        act_grad_tiling,
        gate_grad_tiling,
        w1_grad_tiling,
        w2_grad_tiling,
        swiglu_grad_tiling,
        gmm_workspace,
        swiglu_grad_workspace,
        runtime_config,
        all_event_counters,
        dispatch_size if permute_src_index is None else permute_src_index,
        route_weight,
        dispatch_size if reduce_route_index is None else reduce_route_index,
        rank_id,
        ep,
        expert_num,
        hidden_size,
        seq_size,
        dispatch_target.shape[0] if receive_capacity is None else receive_capacity,
        wave_count,
        cycle_profile,
        home_experts,
    )
