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
"""Differentiable execution against the existing low-level MegaMoe ABI."""

from __future__ import annotations

import importlib
from typing import Any

from hyper_parallel.core.multicore.platform import torch as multicore_adapter

from ._framework import get_torch
from .route import RouteMetadata
from .runtime import MegaMoeRuntime
from .workspace import MegaMoeWorkspace

_MEGA_MOE_FUNCTION = None


def _workspace_tensor(tensor: Any | None, name: str) -> Any:
    """Return an initialized workspace tensor or raise an invariant error."""
    if tensor is None:
        raise RuntimeError(f"MegaMoe workspace {name} is not initialized.")
    return tensor


def _barrier() -> None:
    """Synchronize the process group bound to the SHMEM handler."""
    symm = importlib.import_module("hyper_parallel.core.symmetric_memory")

    symm.barrier()


def _allocate_forward_intermediates(
    torch_module: Any,
    config: Any,
    capacity: int,
    routed_tokens: Any,
    dispatch: Any,
) -> tuple[Any, Any, Any]:
    """Allocate forward tensors outside the autograd bridge."""
    up_proj = torch_module.zeros(
        (capacity, config.intermediate_size * 2),
        dtype=routed_tokens.dtype,
        device=routed_tokens.device,
    )
    activation = torch_module.zeros(
        (capacity, config.intermediate_size),
        dtype=routed_tokens.dtype,
        device=routed_tokens.device,
    )
    return up_proj, activation, torch_module.zeros_like(dispatch)


def _allocate_backward_intermediates(
    torch_module: Any,
    config: Any,
    capacity: int,
    grad_output: Any,
    weight1: Any,
    weight2: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Allocate zero-safe backward tensors outside the autograd bridge."""
    grad_weight2 = torch_module.zeros_like(weight2)
    act_grad = torch_module.zeros(
        (capacity, config.intermediate_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    swiglu_grad = torch_module.zeros(
        (capacity, config.intermediate_size * 2),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    gate_dx = torch_module.zeros(
        (capacity, config.hidden_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    # Empty experts must retain exact zero gradients on the baseline kernel.
    grad_weight1 = torch_module.zeros_like(weight1)
    return grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx


def _save_forward_state(
    ctx: Any,
    runtime: MegaMoeRuntime,
    workspace: MegaMoeWorkspace,
    dispatch: Any,
    up_proj: Any,
    activation: Any,
    weight1: Any,
    weight2: Any,
    metadata: RouteMetadata,
) -> None:
    """Save owned forward tensors and route state for backward."""
    ctx.runtime = runtime
    ctx.workspace = workspace
    ctx.save_for_backward(
        dispatch.clone(),
        up_proj,
        activation,
        weight1,
        weight2,
        metadata.group_list,
        metadata.dispatch_src_off,
        metadata.dispatch_target_off,
        metadata.dispatch_size,
        metadata.combine_src_off,
        metadata.combine_target_off,
        metadata.combine_size,
    )


def _get_mega_moe_function() -> Any:
    """Create the Torch autograd.Function lazily inside the Torch backend."""
    # pylint: disable=global-statement
    torch = get_torch()

    global _MEGA_MOE_FUNCTION
    if _MEGA_MOE_FUNCTION is not None:
        return _MEGA_MOE_FUNCTION

    # Torch declares variadic autograd hooks; concrete functions use operator-specific signatures.
    class _MegaMoeFunction(  # pylint: disable=abstract-method,arguments-differ
        torch.autograd.Function
    ):
        """Autograd bridge for the in-place forward/backward custom ops."""

        @staticmethod
        def forward(
            ctx: Any,
            routed_tokens: Any,
            weight1: Any,
            weight2: Any,
            route: RouteMetadata,
            runtime: MegaMoeRuntime,
            workspace: MegaMoeWorkspace,
        ) -> Any:
            """Launch the legacy forward op and save owned backward inputs.

            Args:
                ctx: Autograd context for saved backward state.
                routed_tokens: Expert-major token rows.
                weight1: Local gate and up-projection weights.
                weight2: Local down-projection weights.
                route: Route offsets, sizes and group metadata.
                runtime: Shape-specific forward and backward descriptors.
                workspace: Reusable directional execution buffers.

            Returns:
                Owned expert-major output rows.
            """
            config = runtime.config
            metadata = route
            workspace.ensure(
                config,
                routed_tokens.dtype,
                routed_tokens.device,
                metadata.expert_capacity,
            )
            workspace.claim()
            try:
                dispatch = _workspace_tensor(workspace.expert_buffer, "expert_buffer")
                combine = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
                events = _workspace_tensor(
                    workspace.forward_event_counters,
                    "forward_event_counters",
                )
                gmm_workspace = _workspace_tensor(
                    workspace.gmm_workspace, "gmm_workspace"
                )
                dispatch.zero_()
                combine.zero_()
                events.zero_()
                capacity = workspace.expert_capacity
                up_proj, activation, down_proj = _allocate_forward_intermediates(
                    torch,
                    config,
                    capacity,
                    routed_tokens,
                    dispatch,
                )
                _barrier()
                multicore_adapter.run_mega_moe(
                    dispatch,
                    metadata.dispatch_target_off * config.hidden_size,
                    routed_tokens.contiguous(),
                    metadata.dispatch_src_off * config.hidden_size,
                    metadata.dispatch_size * config.hidden_size,
                    weight1.contiguous(),
                    metadata.group_list,
                    up_proj,
                    activation,
                    weight2.contiguous(),
                    metadata.group_list,
                    down_proj,
                    combine,
                    metadata.combine_target_off * config.hidden_size,
                    metadata.combine_src_off * config.hidden_size,
                    metadata.combine_size * config.hidden_size,
                    gmm_workspace,
                    runtime.up_proj_tiling,
                    runtime.swiglu_tiling,
                    runtime.down_proj_tiling,
                    runtime.fwd_runtime_config,
                    events,
                    config.rank_id,
                    config.ep_size,
                    config.num_experts,
                    config.hidden_size,
                    config.local_num_tokens,
                )
                output = combine.clone()
                _save_forward_state(
                    ctx,
                    runtime,
                    workspace,
                    dispatch,
                    up_proj,
                    activation,
                    weight1,
                    weight2,
                    metadata,
                )
                return output
            finally:
                workspace.release()

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            """Launch the legacy backward op into zero-safe gradient buffers.

            Args:
                ctx: Autograd context populated by ``forward``.
                grad_output: Expert-major output gradient rows.

            Returns:
                Gradients aligned with the differentiable forward inputs.
            """
            runtime = ctx.runtime
            workspace = ctx.workspace
            config = runtime.config
            (
                saved_dispatch,
                up_proj,
                activation,
                weight1,
                weight2,
                group_list,
                dispatch_src_off,
                dispatch_target_off,
                dispatch_size,
                combine_src_off,
                combine_target_off,
                combine_size,
            ) = ctx.saved_tensors
            workspace.claim()
            try:
                dispatch = _workspace_tensor(workspace.expert_buffer, "expert_buffer")
                grad_x = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
                events = _workspace_tensor(
                    workspace.backward_event_counters,
                    "backward_event_counters",
                )
                gmm_workspace = _workspace_tensor(
                    workspace.gmm_workspace, "gmm_workspace"
                )
                swiglu_workspace = _workspace_tensor(
                    workspace.swiglu_grad_workspace,
                    "swiglu_grad_workspace",
                )
                dispatch.zero_()
                grad_x.zero_()
                events.zero_()
                capacity = workspace.expert_capacity
                (
                    grad_weight1,
                    grad_weight2,
                    act_grad,
                    swiglu_grad,
                    gate_dx,
                ) = _allocate_backward_intermediates(
                    torch,
                    config,
                    capacity,
                    grad_output,
                    weight1,
                    weight2,
                )
                _barrier()
                multicore_adapter.run_mega_moe_grad(
                    dispatch,
                    dispatch_target_off * config.hidden_size,
                    grad_output.contiguous(),
                    dispatch_src_off * config.hidden_size,
                    dispatch_size * config.hidden_size,
                    activation,
                    grad_weight2,
                    weight2,
                    act_grad,
                    up_proj,
                    swiglu_grad,
                    weight1,
                    gate_dx,
                    grad_x,
                    combine_target_off * config.hidden_size,
                    combine_src_off * config.hidden_size,
                    combine_size * config.hidden_size,
                    saved_dispatch,
                    grad_weight1,
                    group_list,
                    runtime.act_grad_tiling,
                    runtime.gate_grad_tiling,
                    runtime.w1_grad_tiling,
                    runtime.w2_grad_tiling,
                    runtime.swiglu_grad_tiling,
                    gmm_workspace,
                    swiglu_workspace,
                    runtime.bwd_runtime_config,
                    events,
                    config.rank_id,
                    config.ep_size,
                    config.num_experts,
                    config.hidden_size,
                    config.local_num_tokens,
                )
                return grad_x.clone(), grad_weight1, grad_weight2, None, None, None
            finally:
                workspace.release()

    _MEGA_MOE_FUNCTION = _MegaMoeFunction
    return _MEGA_MOE_FUNCTION


def execute_mega_moe(
    routed_tokens: Any,
    weight1: Any,
    weight2: Any,
    route: RouteMetadata,
    runtime: MegaMoeRuntime,
    workspace: MegaMoeWorkspace,
) -> Any:
    """Execute expert-major rows through the differentiable legacy kernels.

    Args:
        routed_tokens: Expert-major token rows.
        weight1: Local gate and up-projection weights.
        weight2: Local down-projection weights.
        route: Route offsets, sizes and group metadata.
        runtime: Shape-specific forward and backward descriptors.
        workspace: Reusable directional execution buffers.

    Returns:
        Expert-major output rows with independent storage.
    """
    function = _get_mega_moe_function()
    return function.apply(routed_tokens, weight1, weight2, route, runtime, workspace)
