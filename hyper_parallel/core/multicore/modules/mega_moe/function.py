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

from importlib import import_module

import os
from dataclasses import dataclass
from typing import Any

from hyper_parallel.core.multicore.platform import torch as multicore_adapter

from .route import (
    PreparedTopKRoute,
    RouteMetadata,
    _build_route_indices,
    restore_topk_output,
)
from .plan import MegaMoePlan
from .workspace import MegaMoeWorkspace

torch = import_module("torch")


_MEGA_MOE_FUNCTION = None
_CYCLE_PROFILE_ENV = "HP_MEGA_MOE_KERNEL_CYCLE_PROFILE"


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

    forward_input: Any
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
    reduce_route_index: Any
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
    cycle_profile_enabled: bool


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


def _cycle_profile_enabled() -> bool:
    """Return the validated opt-in state for the diagnostic backward key."""
    value = os.getenv(_CYCLE_PROFILE_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{_CYCLE_PROFILE_ENV} must be 0 or 1, got {value!r}.")
    return value == "1"


def _mark_retained_scratch_modified(
    ctx: Any,
    saved: _SavedForwardState,
) -> None:
    """Make a second backward fail closed after saved storage is reused."""
    if ctx.wave_count <= 1 or not ctx.plan.reuse_retained_dw_scratch:
        return
    torch.autograd.graph.increment_version(saved.up_proj)


def _preserve_forward_input(
    metadata: RouteMetadata,
    dispatch: Any,
    down_proj: Any,
    kernel_input: Any,
    *,
    redispatch_w1_input: bool,
) -> Any:
    """Retain the backward W1 input without allocating another capacity buffer."""
    if metadata.wave_count > 1 or redispatch_w1_input:
        return kernel_input
    # Forward Combine has consumed down-projection output when the custom op
    # returns. Recycle that equally shaped dead storage for the dispatched
    # hidden input that W1-gradient needs, replacing dispatch.clone().
    return down_proj.copy_(dispatch)


def _allocate_forward_intermediates(
    torch_module: Any,
    spec: Any,
    rolling_capacity: int,
    retained_capacity: int,
    reference: Any,
) -> tuple[Any, Any, Any]:
    """Allocate retained GMM1 output and bounded rolling intermediates."""
    up_proj = torch_module.empty(
        (retained_capacity, spec.intermediate_size * 2),
        dtype=reference.dtype,
        device=reference.device,
    )
    activation = torch_module.empty(
        (rolling_capacity, spec.intermediate_size),
        dtype=reference.dtype,
        device=reference.device,
    )
    down_proj = torch_module.empty(
        (rolling_capacity, spec.hidden_size),
        dtype=reference.dtype,
        device=reference.device,
    )
    return up_proj, activation, down_proj


def _allocate_backward_intermediates(
    torch_module: Any,
    spec: Any,
    capacity: int,
    grad_output: Any,
    weight1: Any,
    weight2: Any,
    *,
    multi_wave: bool,
    inplace_backward: bool,
) -> tuple[Any, Any, Any, Any, Any]:
    """Allocate backward tensors that the device kernel fully initializes."""
    allocate_weight_gradient = (
        torch_module.zeros if multi_wave else torch_module.empty
    )
    weight_gradient_dtype = torch_module.float32 if spec.replica_slots_per_rank else weight1.dtype
    grad_weight2 = allocate_weight_gradient(
        tuple(weight2.shape),
        dtype=weight_gradient_dtype,
        device=weight2.device,
    )
    act_grad = torch_module.empty(
        (capacity, spec.intermediate_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    swiglu_grad = torch_module.empty(
        (capacity, spec.intermediate_size * 2),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    gate_dx = (
        None
        if inplace_backward
        else torch_module.empty(
            (capacity, spec.hidden_size),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
    )
    grad_weight1 = allocate_weight_gradient(
        tuple(weight1.shape),
        dtype=weight_gradient_dtype,
        device=weight1.device,
    )
    return grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx


def _unpermute_grad(
    expert_output: Any,
    grad_output: Any,
    reduce_route_index: Any,
    topk_weights: Any,
) -> tuple[Any, Any]:
    """Differentiate the external weighted-unpermute endpoint with ACLNN."""

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

    return torch.ops.npu.npu_moe_token_permute_grad(
        token_template,
        expert_grad,
        topk_ids,
        reduce_route_index.reshape(-1).to(torch.int32),
        False,
    ).contiguous()


def _rebuild_unpermute_mapping(
    route_slots: Any,
    topk_ids: Any,
) -> Any:
    """Rebuild the compact route map after multi-wave dTopK reuses its slot."""
    _, mapping = _build_route_indices(route_slots, topk_ids)
    return mapping


def _save_forward_state(
    ctx: Any,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    forward_input: Any,
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
    ctx.plan = plan
    ctx.workspace = workspace
    ctx.permute_fused = plan.spec.backward_permute_fused
    ctx.inplace_backward = plan.inplace_wave_backward and ctx.permute_fused
    ctx.backward_unpermute_fused = (
        metadata.wave_count == 1
        and plan.spec.backward_unpermute_fused
        and not ctx.inplace_backward
    )
    ctx.expert_capacity = metadata.expert_capacity
    ctx.receive_capacity = metadata.receive_capacity
    ctx.wave_count = metadata.wave_count
    ctx.replay_reduce_route_index = None
    ctx.save_for_backward(
        forward_input,
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
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    metadata: RouteMetadata,
    hidden_states: Any,
    route: PreparedTopKRoute,
) -> _ForwardBuffers:
    """Allocate and select tensors consumed by the forward custom operator."""
    spec = plan.spec
    capacity = metadata.expert_capacity
    dispatch = _workspace_tensor(
        workspace.expert_buffer,
        "expert_buffer",
    ).narrow(0, 0, capacity)
    up_proj, activation, down_proj = _allocate_forward_intermediates(
        torch_module,
        spec,
        capacity,
        metadata.retained_capacity,
        hidden_states,
    )
    combine = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
    final_out = (
        torch_module.empty(
            (spec.local_num_tokens, spec.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if spec.forward_unpermute_fused
        else combine.reshape(-1).narrow(0, 0, 1)
    )
    permute_src_index = (
        _route_tensor(route.permute_src_index, "permute_src_index")
        if spec.forward_permute_fused
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
    plan: MegaMoePlan,
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
    spec = plan.spec
    multicore_adapter.mega_moe(
        buffers.dispatch,
        metadata.dispatch_target_off,
        kernel_input.contiguous(),
        metadata.dispatch_src_off,
        metadata.dispatch_size,
        weight1.contiguous(),
        metadata.group_list,
        buffers.up_proj,
        buffers.activation,
        weight2.contiguous(),
        metadata.group_list,
        buffers.down_proj,
        buffers.combine,
        metadata.combine_target_off,
        metadata.combine_src_off,
        metadata.combine_size,
        buffers.gmm_workspace,
        plan.up_proj_tiling,
        plan.swiglu_tiling,
        plan.down_proj_tiling,
        plan.fwd_runtime_config,
        buffers.events,
        hidden_states,
        buffers.permute_src_index,
        route.unpermute_mapping,
        topk_weights,
        buffers.final_out,
        spec.rank_id,
        spec.ep_size,
        spec.physical_num_experts,
        spec.hidden_size,
        spec.local_num_tokens,
        metadata.receive_capacity,
        metadata.wave_count,
    )


def _load_forward_state(ctx: Any) -> _SavedForwardState:
    """Expose saved tensors by name without repacking device storage."""
    return _SavedForwardState(*ctx.saved_tensors)


def _prepare_fused_backward_route(
    plan: MegaMoePlan,
    saved: _SavedForwardState,
) -> tuple[Any, Any]:
    """Return expert-major route slots and token-major weights for dispatch."""
    if plan.spec.forward_permute_fused:
        permute_src_index = saved.permute_src_index
    else:
        route_slots = _route_tensor(plan.route_slots, "route-slot metadata")
        permute_src_index, _ = _build_route_indices(
            route_slots,
            saved.topk_ids,
        )
    route_weight = saved.topk_weights.detach().reshape(-1)
    return permute_src_index, route_weight


def _prepare_backward_route(
    ctx: Any,
    grad_output: Any,
    saved: _SavedForwardState,
) -> _BackwardRouteState:
    """Select external or fused route gradients and their destination buffers."""
    spec = ctx.plan.spec
    workspace = ctx.workspace
    if spec.backward_permute_fused:
        kernel_grad_output = grad_output.contiguous()
        permute_src_index, route_weight = _prepare_fused_backward_route(
            ctx.plan,
            saved,
        )
        # One-wave dTopK writes its result back into ABI position 30. Keep the
        # autograd-saved token-major weights immutable for retain_graph and
        # deferred replay. Multi-wave dTopK instead reuses position 31 after
        # its compact inverse mapping has been consumed.
        if ctx.wave_count == 1:
            route_weight = route_weight.clone()
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
        permute_src_index = (
            saved.permute_src_index
            if ctx.wave_count > 1 and spec.forward_permute_fused
            else saved.dispatch_size
        )

    capacity = ctx.expert_capacity
    dispatch = _workspace_tensor(
        workspace.expert_buffer,
        "expert_buffer",
    ).narrow(0, 0, capacity)
    route_graph = ctx.permute_fused or ctx.backward_unpermute_fused
    if ctx.wave_count > 1 or (route_graph and ctx.permute_fused):
        # Keep the saved expert output immutable. The common forward/backward
        # path can consume the copy already present in SHMEM. A retained graph
        # or an intervening forward restores it only when the generation shows
        # that the shared buffer has since been overwritten.
        grad_x = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
        if ctx.routed_buffer_generation != workspace.routed_buffer_generation:
            grad_x.copy_(saved.expert_output)
    elif route_graph:
        grad_x = torch.empty(
            (spec.local_num_tokens, spec.hidden_size),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
    else:
        grad_x = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
    reduce_route_index = saved.reduce_route_index
    if ctx.wave_count > 1 and ctx.permute_fused:
        replay_route_index = ctx.replay_reduce_route_index
        if replay_route_index is not None:
            reduce_route_index = replay_route_index
    return _BackwardRouteState(
        kernel_grad_output=kernel_grad_output,
        permute_src_index=permute_src_index,
        route_weight=route_weight,
        grad_topk_weights=grad_topk_weights,
        dispatch=dispatch,
        grad_x=grad_x,
        reduce_route_index=reduce_route_index,
        route_graph=route_graph,
    )


def _prepare_backward_buffers(
    ctx: Any,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    saved: _SavedForwardState,
    route_state: _BackwardRouteState,
) -> _BackwardBuffers:
    """Allocate backward intermediates and select persistent workspaces."""
    intermediates = _allocate_backward_intermediates(
        torch,
        plan.spec,
        ctx.expert_capacity,
        route_state.kernel_grad_output,
        saved.weight1,
        saved.weight2,
        multi_wave=ctx.wave_count > 1,
        inplace_backward=ctx.inplace_backward,
    )
    grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx = intermediates
    if gate_dx is None:
        gate_dx = route_state.dispatch
    cycle_profile_enabled = _cycle_profile_enabled()
    workspace.prepare_cycle_profile(enabled=cycle_profile_enabled)
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
        cycle_profile_enabled=cycle_profile_enabled,
    )


def _launch_backward(
    ctx: Any,
    saved: _SavedForwardState,
    route_state: _BackwardRouteState,
    buffers: _BackwardBuffers,
) -> None:
    """Launch the backward graph with route-mode-independent ABI arguments."""
    plan = ctx.plan
    spec = plan.spec
    multicore_adapter.mega_moe_grad(
        route_state.dispatch,
        saved.dispatch_target_off,
        route_state.kernel_grad_output,
        saved.dispatch_src_off,
        saved.dispatch_size,
        saved.activation,
        buffers.grad_weight2,
        saved.weight2,
        buffers.act_grad,
        saved.up_proj,
        buffers.swiglu_grad,
        saved.weight1,
        buffers.gate_dx,
        route_state.grad_x,
        saved.combine_target_off,
        saved.combine_src_off,
        saved.combine_size,
        saved.forward_input,
        buffers.grad_weight1,
        saved.group_list,
        plan.act_grad_tiling,
        plan.gate_grad_tiling,
        plan.w1_grad_tiling,
        plan.w2_grad_tiling,
        plan.swiglu_grad_tiling,
        buffers.gmm_workspace,
        buffers.swiglu_workspace,
        (
            plan.bwd_multi_wave_runtime_config
            if ctx.wave_count > 1
            else plan.bwd_runtime_config
        ),
        buffers.events,
        route_state.permute_src_index,
        route_state.route_weight,
        route_state.reduce_route_index,
        spec.rank_id,
        spec.ep_size,
        spec.physical_num_experts,
        spec.hidden_size,
        spec.local_num_tokens,
        ctx.receive_capacity,
        ctx.wave_count,
        int(buffers.cycle_profile_enabled),
    )


def _finish_backward(
    ctx: Any,
    grad_output: Any,
    saved: _SavedForwardState,
    route_state: _BackwardRouteState,
    buffers: _BackwardBuffers,
) -> tuple[Any, ...]:
    """Restore public gradient layouts after the backward custom operator."""
    spec = ctx.plan.spec
    grad_topk_weights = route_state.grad_topk_weights
    if ctx.permute_fused:
        if ctx.wave_count > 1:
            route_grad = route_state.reduce_route_index.view(
                torch.float32
            ).reshape_as(saved.topk_weights)
            grad_topk_weights = route_grad.to(saved.topk_weights.dtype)
        else:
            grad_topk_weights = route_state.route_weight.reshape_as(
                saved.topk_weights
            ).to(saved.topk_weights.dtype)
    if ctx.backward_unpermute_fused:
        grad_hidden = route_state.grad_x.narrow(0, 0, spec.local_num_tokens)
    else:
        reduce_route_index = saved.reduce_route_index
        if ctx.permute_fused and ctx.wave_count > 1:
            route_slots = ctx.plan.route_slots
            if route_slots is None:
                raise RuntimeError(
                    "MegaMoe cannot rebuild the multi-wave route mapping."
                )
            reduce_route_index = _rebuild_unpermute_mapping(
                route_slots,
                saved.topk_ids,
            )
            ctx.replay_reduce_route_index = reduce_route_index
        expert_grad = (
            route_state.dispatch.narrow(0, 0, spec.routed_slots)
            if (
                route_state.route_graph
                and ctx.wave_count == 1
                and not ctx.inplace_backward
            )
            else route_state.grad_x.narrow(0, 0, spec.routed_slots)
        )
        grad_hidden = _permute_grad(
            grad_output,
            expert_grad,
            saved.topk_ids,
            reduce_route_index,
        )
    grad_weight1, grad_weight2 = buffers.grad_weight1, buffers.grad_weight2
    if ctx.replica_pending is not None:
        pending = ctx.replica_pending
        grad_weight1, grad_weight2 = pending.manager.reduce_replica_gradients(
            pending.plan, grad_weight1, grad_weight2,
        )
    return (
        grad_hidden,
        None,
        grad_topk_weights,
        grad_weight1,
        grad_weight2,
        None,
        None,
        None,
        None,
    )


def _get_mega_moe_function() -> Any:
    """Create the Torch autograd.Function lazily inside the Torch execution module."""
    # pylint: disable=global-statement
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
            plan: MegaMoePlan,
            workspace: MegaMoeWorkspace,
        ) -> Any:
            """Launch the selected forward route endpoints and expert graph.

            Args:
                ctx: Autograd context used to retain backward state.
                hidden_states: Token-order input activations.
                topk_ids: Selected expert ids for every token.
                topk_weights: Router weights for the selected experts.
                weight1: Gate/up projection weights.
                weight2: Down-projection weights.
                routed_tokens: Optional externally permuted activations.
                route: Prepared routing metadata and mappings.
                plan: Compiled MegaMoe plan configuration.
                workspace: Shared execution workspace.

            Returns:
                Token-order or expert-major output selected by the fusion mode.
            """
            spec = plan.spec
            metadata = route.metadata
            if metadata is None:
                raise RuntimeError("MegaMoe route metadata is unavailable.")
            # The autograd inputs remain logical home weights. Materializing
            # inside forward avoids a BF16 intermediate gradient edge before
            # the FP32 physical partials are reduced to their home owners.
            ctx.replica_pending = route.placement_payload
            if ctx.replica_pending is not None:
                weight1, weight2 = ctx.replica_pending.wait()
            kernel_input = (
                hidden_states
                if spec.forward_permute_fused
                else _route_tensor(routed_tokens, "routed_tokens")
            )
            backward_input = (
                hidden_states if spec.backward_permute_fused else kernel_input
            )
            workspace.ensure(
                spec,
                hidden_states.dtype,
                hidden_states.device,
            )
            workspace.claim()
            try:
                workspace.prepare_gmm_workspace(
                    weight1,
                    weight2,
                    metadata.wave_count,
                    receive_capacity=metadata.receive_capacity,
                    reuse_retained_scratch=(
                        plan.reuse_retained_dw_scratch
                    ),
                    atomic_dw_accumulation=(
                        plan.atomic_dw_accumulation
                    ),
                    weight_gradient_element_size=(4 if spec.replica_slots_per_rank else weight1.element_size()),
                )
                # Exact route offsets cover every row consumed by GMM and
                # unpermute. Clearing the capacity tail only adds device work.
                buffers = _prepare_forward_buffers(
                    torch,
                    plan,
                    workspace,
                    metadata,
                    hidden_states,
                    route,
                )
                _launch_forward(
                    plan,
                    metadata,
                    hidden_states,
                    kernel_input,
                    weight1,
                    weight2,
                    route,
                    topk_weights,
                    buffers,
                )
                ctx.routed_buffer_generation = workspace.mark_routed_buffer_write()
                expert_output = (
                    buffers.down_proj
                    if spec.forward_unpermute_fused and metadata.wave_count == 1
                    else buffers.combine.clone()
                )
                output = (
                    buffers.final_out
                    if spec.forward_unpermute_fused
                    else restore_topk_output(
                        expert_output,
                        route.unpermute_mapping,
                        topk_weights,
                    )
                )
                redispatch_w1_input = (
                    metadata.wave_count == 1
                    and plan.inplace_wave_backward
                    and spec.backward_permute_fused
                )
                if (
                    metadata.wave_count == 1
                    and spec.forward_unpermute_fused
                    and not redispatch_w1_input
                ):
                    # Integrated forward unpermute reuses down_proj as the
                    # expert-output snapshot needed by dTopK.
                    forward_input = buffers.dispatch.clone()
                else:
                    forward_input = _preserve_forward_input(
                        metadata,
                        buffers.dispatch,
                        buffers.down_proj,
                        backward_input,
                        redispatch_w1_input=redispatch_w1_input,
                    )
                _save_forward_state(
                    ctx,
                    plan,
                    workspace,
                    forward_input,
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
            """Launch the selected backward route endpoints and expert graph.

            Args:
                ctx: Autograd context populated by the forward launch.
                grad_output: Gradient of the layer output.

            Returns:
                Gradients aligned with the forward arguments.
            """
            workspace = ctx.workspace
            saved = _load_forward_state(ctx)
            workspace.claim()
            try:
                route_state = _prepare_backward_route(ctx, grad_output, saved)
                buffers = _prepare_backward_buffers(
                    ctx,
                    ctx.plan,
                    workspace,
                    saved,
                    route_state,
                )
                _launch_backward(ctx, saved, route_state, buffers)
                workspace.mark_routed_buffer_write()
                _mark_retained_scratch_modified(ctx, saved)
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
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
) -> Any:
    """Execute Top-K routing endpoints and expert compute as one autograd unit.

    Args:
        hidden_states: Token-major local activations.
        topk_ids: Physical expert IDs selected for every route slot.
        topk_weights: Router weights paired with ``topk_ids``.
        weight1: Rank-local logical home gate/up weights.
        weight2: Rank-local logical home down-projection weights.
        route: Prepared route tensors and count metadata.
        plan: Static forward and backward plan descriptors.
        workspace: Directional buffers shared by this backend instance.

    Returns:
        Token-major local expert output.
    """
    function = _get_mega_moe_function()
    return function.apply(
        hidden_states,
        topk_ids,
        topk_weights,
        weight1,
        weight2,
        route.routed_tokens,
        route,
        plan,
        workspace,
    )
