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
"""Differentiable execution for independently selectable route endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyper_parallel.core.multicore.platform import torch as multicore_adapter

from ._framework import get_torch
from .route import (
    PreparedTopKRoute,
    RouteMetadata,
    _build_route_indices,
    restore_topk_output,
)
from .runtime import MegaMoeRuntime
from .workspace import MegaMoeWorkspace

_MEGA_MOE_FUNCTION = None


@dataclass(frozen=True)
class _ForwardBuffers:
    """Forward tensors prepared around the fused multicore launch."""

    dispatch: Any
    combine: Any
    gmm_workspace: Any
    up_proj: Any
    activation: Any
    down_proj: Any
    final_out: Any
    permute_src_index: Any
    events: Any


@dataclass(frozen=True)
class _SavedForwardState:
    """Named view of tensors retained by the autograd forward pass."""

    dispatch: Any
    up_proj: Any
    activation: Any
    weight1: Any
    weight2: Any
    group_list: Any
    dispatch_src_off: Any
    dispatch_target_off: Any
    dispatch_size: Any
    combine_src_off: Any
    combine_target_off: Any
    combine_size: Any
    topk_ids: Any
    topk_weights: Any
    reduce_route_index: Any
    permute_src_index: Any
    expert_output: Any


@dataclass(frozen=True)
class _BackwardRouteState:
    """Route-dependent tensors selected before the backward kernel launch."""

    kernel_grad_output: Any
    permute_src_index: Any
    route_weight: Any
    grad_topk_weights: Any | None
    dispatch: Any
    grad_x: Any
    route_graph: bool


@dataclass(frozen=True)
class _BackwardBuffers:
    """Backward outputs, intermediates and device workspaces."""

    grad_weight1: Any
    grad_weight2: Any
    act_grad: Any
    swiglu_grad: Any
    gate_dx: Any
    gmm_workspace: Any
    swiglu_workspace: Any
    events: Any


def _workspace_tensor(tensor: Any | None, name: str) -> Any:
    """Return an initialized workspace tensor or raise an invariant error."""
    if tensor is None:
        raise RuntimeError(f"MegaMoe workspace {name} is not initialized.")
    return tensor


def _route_tensor(tensor: Any | None, name: str) -> Any:
    """Return a route tensor required by the selected fusion mode."""
    if tensor is None:
        raise RuntimeError(f"MegaMoe fused route is missing {name}.")
    return tensor


def _allocate_forward_intermediates(
    torch_module: Any,
    config: Any,
    capacity: int,
    reference: Any,
) -> tuple[Any, Any, Any]:
    """Allocate forward tensors at the route's actual bounded capacity."""
    up_proj = torch_module.empty(
        (capacity, config.intermediate_size * 2),
        dtype=reference.dtype,
        device=reference.device,
    )
    activation = torch_module.empty(
        (capacity, config.intermediate_size),
        dtype=reference.dtype,
        device=reference.device,
    )
    down_proj = torch_module.empty(
        (capacity, config.hidden_size),
        dtype=reference.dtype,
        device=reference.device,
    )
    return up_proj, activation, down_proj


def _allocate_backward_intermediates(
    torch_module: Any,
    config: Any,
    capacity: int,
    grad_output: Any,
    weight1: Any,
    weight2: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Allocate backward tensors that the device kernel fully initializes."""
    grad_weight2 = torch_module.empty_like(weight2)
    act_grad = torch_module.empty(
        (capacity, config.intermediate_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    swiglu_grad = torch_module.empty(
        (capacity, config.intermediate_size * 2),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    gate_dx = torch_module.empty(
        (capacity, config.hidden_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    grad_weight1 = torch_module.empty_like(weight1)
    return grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx


def _unpermute_grad(
    expert_output: Any,
    grad_output: Any,
    reduce_route_index: Any,
    topk_weights: Any,
) -> tuple[Any, Any]:
    """Differentiate the external weighted-unpermute endpoint with ACLNN."""
    # pylint: disable=C0415
    import torch_npu  # noqa: F401  # pylint: disable=unused-import

    torch = get_torch()
    return torch.ops.npu.npu_moe_token_unpermute_grad(
        expert_output,
        grad_output,
        reduce_route_index.reshape(-1).to(torch.int32),
        topk_weights,
        False,
        None,
    )


def _permute_grad(
    token_template: Any,
    expert_grad: Any,
    topk_ids: Any,
    reduce_route_index: Any,
) -> Any:
    """Differentiate token permute using a shape/dtype template for its output."""
    # pylint: disable=C0415
    import torch_npu  # noqa: F401  # pylint: disable=unused-import

    torch = get_torch()
    return torch.ops.npu.npu_moe_token_permute_grad(
        token_template,
        expert_grad,
        topk_ids,
        reduce_route_index.reshape(-1).to(torch.int32),
        False,
    ).contiguous()


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
    topk_ids: Any,
    topk_weights: Any,
    route: PreparedTopKRoute,
    expert_output: Any,
) -> None:
    """Save owned forward tensors, route indices and static execution state."""
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
        topk_ids,
        topk_weights,
        route.unpermute_mapping,
        (
            route.permute_src_index
            if route.permute_src_index is not None
            else metadata.dispatch_size
        ),
        expert_output,
    )


def _prepare_forward_buffers(
    torch_module: Any,
    runtime: MegaMoeRuntime,
    workspace: MegaMoeWorkspace,
    metadata: RouteMetadata,
    hidden_states: Any,
    route: PreparedTopKRoute,
) -> _ForwardBuffers:
    """Allocate and select tensors consumed by the forward custom operator."""
    config = runtime.config
    capacity = metadata.expert_capacity
    dispatch = _workspace_tensor(
        workspace.expert_buffer,
        "expert_buffer",
    ).narrow(0, 0, capacity)
    up_proj, activation, down_proj = _allocate_forward_intermediates(
        torch_module,
        config,
        capacity,
        hidden_states,
    )
    combine = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
    final_out = (
        torch_module.empty(
            (config.local_num_tokens, config.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if config.forward_unpermute_fused
        else combine.reshape(-1).narrow(0, 0, 1)
    )
    permute_src_index = (
        _route_tensor(route.permute_src_index, "permute_src_index")
        if config.forward_permute_fused
        else route.tokens_per_expert
    )
    return _ForwardBuffers(
        dispatch=dispatch,
        combine=combine,
        gmm_workspace=_workspace_tensor(
            workspace.gmm_workspace,
            "gmm_workspace",
        ),
        up_proj=up_proj,
        activation=activation,
        down_proj=down_proj,
        final_out=final_out,
        permute_src_index=permute_src_index,
        events=workspace.prepare_event_counters(forward=True),
    )


def _launch_forward(
    runtime: MegaMoeRuntime,
    metadata: RouteMetadata,
    hidden_states: Any,
    kernel_input: Any,
    weight1: Any,
    weight2: Any,
    route: PreparedTopKRoute,
    topk_weights: Any,
    buffers: _ForwardBuffers,
) -> None:
    """Launch the forward graph with route-mode-independent ABI arguments."""
    config = runtime.config
    multicore_adapter.run_mega_moe(
        buffers.dispatch,
        metadata.dispatch_target_off * config.hidden_size,
        kernel_input.contiguous(),
        metadata.dispatch_src_off * config.hidden_size,
        metadata.dispatch_size * config.hidden_size,
        weight1.contiguous(),
        metadata.group_list,
        buffers.up_proj,
        buffers.activation,
        weight2.contiguous(),
        metadata.group_list,
        buffers.down_proj,
        buffers.combine,
        metadata.combine_target_off * config.hidden_size,
        metadata.combine_src_off * config.hidden_size,
        metadata.combine_size * config.hidden_size,
        buffers.gmm_workspace,
        runtime.up_proj_tiling,
        runtime.swiglu_tiling,
        runtime.down_proj_tiling,
        runtime.fwd_runtime_config,
        buffers.events,
        hidden_states,
        buffers.permute_src_index,
        route.unpermute_mapping,
        topk_weights,
        buffers.final_out,
        config.rank_id,
        config.ep_size,
        config.num_experts,
        config.hidden_size,
        config.local_num_tokens,
    )


def _load_forward_state(ctx: Any) -> _SavedForwardState:
    """Expose saved tensors by name without repacking device storage."""
    return _SavedForwardState(*ctx.saved_tensors)


def _prepare_fused_backward_route(
    runtime: MegaMoeRuntime,
    saved: _SavedForwardState,
) -> tuple[Any, Any]:
    """Return expert-major route slots and weights for fused dispatch."""
    if runtime.config.forward_permute_fused:
        permute_src_index = saved.permute_src_index
    else:
        route_slots = _route_tensor(runtime.route_slots, "route-slot metadata")
        permute_src_index, _ = _build_route_indices(
            route_slots,
            saved.topk_ids,
        )
    route_weight = saved.topk_weights.detach().reshape(-1).index_select(
        0,
        permute_src_index,
    )
    return permute_src_index, route_weight


def _prepare_backward_route(
    ctx: Any,
    grad_output: Any,
    saved: _SavedForwardState,
) -> _BackwardRouteState:
    """Select external or fused route gradients and their destination buffers."""
    torch = get_torch()
    config = ctx.runtime.config
    workspace = ctx.workspace
    if config.backward_permute_fused:
        kernel_grad_output = grad_output.contiguous()
        permute_src_index, route_weight = _prepare_fused_backward_route(
            ctx.runtime,
            saved,
        )
        grad_topk_weights = None
    else:
        kernel_grad_output, grad_topk_weights = _unpermute_grad(
            saved.expert_output,
            grad_output,
            saved.reduce_route_index,
            saved.topk_weights,
        )
        kernel_grad_output = kernel_grad_output.contiguous()
        # This stable ABI slot has no consumer in the non-permute graph.
        route_weight = saved.topk_weights.reshape(-1).narrow(0, 0, 1)
        permute_src_index = saved.dispatch_size

    capacity = saved.dispatch.shape[0]
    dispatch = _workspace_tensor(
        workspace.expert_buffer,
        "expert_buffer",
    ).narrow(0, 0, capacity)
    route_graph = config.backward_permute_fused or config.backward_unpermute_fused
    if route_graph and config.backward_permute_fused:
        # dTopK reads the saved expert output while Combine writes dispatch.
        grad_x = saved.expert_output
    elif route_graph:
        grad_x = torch.empty(
            (config.local_num_tokens, config.hidden_size),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
    else:
        grad_x = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
    return _BackwardRouteState(
        kernel_grad_output=kernel_grad_output,
        permute_src_index=permute_src_index,
        route_weight=route_weight,
        grad_topk_weights=grad_topk_weights,
        dispatch=dispatch,
        grad_x=grad_x,
        route_graph=route_graph,
    )


def _prepare_backward_buffers(
    runtime: MegaMoeRuntime,
    workspace: MegaMoeWorkspace,
    saved: _SavedForwardState,
    route_state: _BackwardRouteState,
) -> _BackwardBuffers:
    """Allocate backward intermediates and select persistent workspaces."""
    torch = get_torch()
    intermediates = _allocate_backward_intermediates(
        torch,
        runtime.config,
        saved.dispatch.shape[0],
        route_state.kernel_grad_output,
        saved.weight1,
        saved.weight2,
    )
    grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx = intermediates
    return _BackwardBuffers(
        grad_weight1=grad_weight1,
        grad_weight2=grad_weight2,
        act_grad=act_grad,
        swiglu_grad=swiglu_grad,
        gate_dx=gate_dx,
        gmm_workspace=_workspace_tensor(
            workspace.gmm_workspace,
            "gmm_workspace",
        ),
        swiglu_workspace=_workspace_tensor(
            workspace.swiglu_grad_workspace,
            "swiglu_grad_workspace",
        ),
        events=workspace.prepare_event_counters(forward=False),
    )


def _launch_backward(
    ctx: Any,
    saved: _SavedForwardState,
    route_state: _BackwardRouteState,
    buffers: _BackwardBuffers,
) -> None:
    """Launch the backward graph with route-mode-independent ABI arguments."""
    runtime = ctx.runtime
    config = runtime.config
    multicore_adapter.run_mega_moe_grad(
        route_state.dispatch,
        saved.dispatch_target_off * config.hidden_size,
        route_state.kernel_grad_output,
        saved.dispatch_src_off * config.hidden_size,
        saved.dispatch_size * config.hidden_size,
        saved.activation,
        buffers.grad_weight2,
        saved.weight2,
        buffers.act_grad,
        saved.up_proj,
        buffers.swiglu_grad,
        saved.weight1,
        buffers.gate_dx,
        route_state.grad_x,
        saved.combine_target_off * config.hidden_size,
        saved.combine_src_off * config.hidden_size,
        saved.combine_size * config.hidden_size,
        saved.dispatch,
        buffers.grad_weight1,
        saved.group_list,
        runtime.act_grad_tiling,
        runtime.gate_grad_tiling,
        runtime.w1_grad_tiling,
        runtime.w2_grad_tiling,
        runtime.swiglu_grad_tiling,
        buffers.gmm_workspace,
        buffers.swiglu_workspace,
        runtime.bwd_runtime_config,
        buffers.events,
        route_state.permute_src_index,
        route_state.route_weight,
        saved.reduce_route_index,
        config.rank_id,
        config.ep_size,
        config.num_experts,
        config.hidden_size,
        config.local_num_tokens,
    )


def _finish_backward(
    ctx: Any,
    grad_output: Any,
    saved: _SavedForwardState,
    route_state: _BackwardRouteState,
    buffers: _BackwardBuffers,
) -> tuple[Any, ...]:
    """Restore public gradient layouts after the backward custom operator."""
    config = ctx.runtime.config
    grad_topk_weights = route_state.grad_topk_weights
    if config.backward_permute_fused:
        grad_topk_weights = route_state.route_weight.reshape_as(
            saved.topk_weights
        ).to(saved.topk_weights.dtype)
    if config.backward_unpermute_fused:
        grad_hidden = route_state.grad_x.narrow(0, 0, config.local_num_tokens)
    else:
        expert_grad = (
            route_state.dispatch.narrow(0, 0, config.routed_slots)
            if route_state.route_graph
            else route_state.grad_x.narrow(0, 0, config.routed_slots)
        )
        grad_hidden = _permute_grad(
            grad_output,
            expert_grad,
            saved.topk_ids,
            saved.reduce_route_index,
        )
    return (
        grad_hidden,
        None,
        grad_topk_weights,
        buffers.grad_weight1,
        buffers.grad_weight2,
        None,
        None,
        None,
        None,
    )


def _get_mega_moe_function() -> Any:
    """Create the Torch autograd.Function lazily inside the Torch backend."""
    # pylint: disable=global-statement
    torch = get_torch()
    global _MEGA_MOE_FUNCTION
    if _MEGA_MOE_FUNCTION is not None:
        return _MEGA_MOE_FUNCTION

    # Torch declares variadic autograd hooks; concrete functions use operator-specific signatures.
    class _MegaMoeFunction(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
        """Autograd bridge for legacy and independently fused route endpoints."""

        @staticmethod
        def forward(
            ctx: Any,
            hidden_states: Any,
            topk_ids: Any,
            topk_weights: Any,
            weight1: Any,
            weight2: Any,
            routed_tokens: Any | None,
            route: PreparedTopKRoute,
            runtime: MegaMoeRuntime,
            workspace: MegaMoeWorkspace,
        ) -> Any:
            """Launch the selected forward route endpoints and expert graph."""
            config = runtime.config
            metadata = route.metadata
            if metadata is None:
                raise RuntimeError("MegaMoe route metadata is unavailable.")
            kernel_input = (
                hidden_states
                if config.forward_permute_fused
                else _route_tensor(routed_tokens, "routed_tokens")
            )
            workspace.ensure(
                config,
                hidden_states.dtype,
                hidden_states.device,
                metadata.expert_capacity,
            )
            workspace.claim()
            try:
                # Exact route offsets cover every row consumed by GMM and
                # unpermute. Clearing the capacity tail only adds device work.
                buffers = _prepare_forward_buffers(
                    torch,
                    runtime,
                    workspace,
                    metadata,
                    hidden_states,
                    route,
                )
                _launch_forward(
                    runtime,
                    metadata,
                    hidden_states,
                    kernel_input,
                    weight1,
                    weight2,
                    route,
                    topk_weights,
                    buffers,
                )
                expert_output = buffers.combine.clone()
                output = (
                    buffers.final_out
                    if config.forward_unpermute_fused
                    else restore_topk_output(
                        expert_output,
                        route.unpermute_mapping,
                        topk_weights,
                    )
                )
                _save_forward_state(
                    ctx,
                    runtime,
                    workspace,
                    buffers.dispatch,
                    buffers.up_proj,
                    buffers.activation,
                    weight1,
                    weight2,
                    metadata,
                    topk_ids,
                    topk_weights,
                    route,
                    expert_output,
                )
                return output
            finally:
                workspace.release()

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            """Launch the selected backward route endpoints and expert graph."""
            workspace = ctx.workspace
            saved = _load_forward_state(ctx)
            workspace.claim()
            try:
                route_state = _prepare_backward_route(ctx, grad_output, saved)
                buffers = _prepare_backward_buffers(
                    ctx.runtime,
                    workspace,
                    saved,
                    route_state,
                )
                _launch_backward(ctx, saved, route_state, buffers)
                return _finish_backward(
                    ctx,
                    grad_output,
                    saved,
                    route_state,
                    buffers,
                )
            finally:
                workspace.release()

    _MEGA_MOE_FUNCTION = _MegaMoeFunction
    return _MEGA_MOE_FUNCTION


def execute_mega_moe(
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
    weight1: Any,
    weight2: Any,
    route: PreparedTopKRoute,
    runtime: MegaMoeRuntime,
    workspace: MegaMoeWorkspace,
) -> Any:
    """Execute Top-K routing endpoints and expert compute as one autograd unit."""
    function = _get_mega_moe_function()
    return function.apply(
        hidden_states,
        topk_ids,
        topk_weights,
        weight1,
        weight2,
        route.routed_tokens,
        route,
        runtime,
        workspace,
    )
