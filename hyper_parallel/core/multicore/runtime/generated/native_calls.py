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

"""Generated typed launch wrappers; edit runtime/native_calls.json."""

import torch

from hyper_parallel.core.multicore.runtime.bindings import resolve_native_call


def mega_gate_route(
    logits: torch.Tensor,
    text_bias: torch.Tensor,
    vision_bias: torch.Tensor,
    image_mask: torch.Tensor,
    runtime_config: torch.Tensor,
    profile_buffer: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    use_vision_bias: bool,
) -> tuple[torch.Tensor, ...]:
    """Launch the family entry through its verified dispatcher schema.

    Args:
        logits: Native Tensor argument at position 0.
        text_bias: Native Tensor argument at position 1.
        vision_bias: Native Tensor argument at position 2.
        image_mask: Native Tensor argument at position 3.
        runtime_config: Native Tensor argument at position 4.
        profile_buffer: Native Tensor argument at position 5.
        top_k: Native int argument at position 6.
        routed_scaling_factor: Native float argument at position 7.
        use_vision_bias: Native bool argument at position 8.
    """
    return resolve_native_call("mega_gate_route")(
        logits,
        text_bias,
        vision_bias,
        image_mask,
        runtime_config,
        profile_buffer,
        top_k,
        routed_scaling_factor,
        use_vision_bias,
    )


def mega_gate_route_grad(
    logits: torch.Tensor,
    route_scores: torch.Tensor,
    selected_scores: torch.Tensor,
    normalization_denominator: torch.Tensor,
    expert_indices: torch.Tensor,
    grad_routing_weights: torch.Tensor,
    direct_grad_logits: torch.Tensor,
    runtime_config: torch.Tensor,
    profile_buffer: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    has_direct_grad: bool,
) -> torch.Tensor:
    """Launch the family entry through its verified dispatcher schema.

    Args:
        logits: Native Tensor argument at position 0.
        route_scores: Native Tensor argument at position 1.
        selected_scores: Native Tensor argument at position 2.
        normalization_denominator: Native Tensor argument at position 3.
        expert_indices: Native Tensor argument at position 4.
        grad_routing_weights: Native Tensor argument at position 5.
        direct_grad_logits: Native Tensor argument at position 6.
        runtime_config: Native Tensor argument at position 7.
        profile_buffer: Native Tensor argument at position 8.
        top_k: Native int argument at position 9.
        routed_scaling_factor: Native float argument at position 10.
        has_direct_grad: Native bool argument at position 11.
    """
    return resolve_native_call("mega_gate_route_grad")(
        logits,
        route_scores,
        selected_scores,
        normalization_denominator,
        expert_indices,
        grad_routing_weights,
        direct_grad_logits,
        runtime_config,
        profile_buffer,
        top_k,
        routed_scaling_factor,
        has_direct_grad,
    )


def mega_mhc(
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    runtime_config: torch.Tensor,
    all_event_counters: torch.Tensor,
    profile_buffer: torch.Tensor,
    new_residual: torch.Tensor,
    next_pre_mix: torch.Tensor,
    next_post_mix: torch.Tensor,
    next_residual_mix: torch.Tensor,
    block_input: torch.Tensor,
    hc_before_norm: torch.Tensor,
    inv_rms: torch.Tensor,
    sum_out: torch.Tensor,
    norm_out: torch.Tensor,
    mixed_input: torch.Tensor,
    rms_rstd: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
    num_iters: int,
    need_backward: bool,
) -> tuple[torch.Tensor, ...]:
    """Launch the family entry through its verified dispatcher schema.

    Args:
        previous_output: Native Tensor argument at position 0.
        residual: Native Tensor argument at position 1.
        previous_pre_mix: Native Tensor argument at position 2.
        previous_post_mix: Native Tensor argument at position 3.
        previous_residual_mix: Native Tensor argument at position 4.
        phi: Native Tensor argument at position 5.
        alpha: Native Tensor argument at position 6.
        bias: Native Tensor argument at position 7.
        norm_weight: Native Tensor argument at position 8.
        runtime_config: Native Tensor argument at position 9.
        all_event_counters: Native Tensor argument at position 10.
        profile_buffer: Native Tensor argument at position 11.
        new_residual: Native Tensor argument at position 12.
        next_pre_mix: Native Tensor argument at position 13.
        next_post_mix: Native Tensor argument at position 14.
        next_residual_mix: Native Tensor argument at position 15.
        block_input: Native Tensor argument at position 16.
        hc_before_norm: Native Tensor argument at position 17.
        inv_rms: Native Tensor argument at position 18.
        sum_out: Native Tensor argument at position 19.
        norm_out: Native Tensor argument at position 20.
        mixed_input: Native Tensor argument at position 21.
        rms_rstd: Native Tensor argument at position 22.
        hc_eps: Native float argument at position 23.
        norm_eps: Native float argument at position 24.
        num_iters: Native int argument at position 25.
        need_backward: Native bool argument at position 26.
    """
    return resolve_native_call("mega_mhc")(
        previous_output,
        residual,
        previous_pre_mix,
        previous_post_mix,
        previous_residual_mix,
        phi,
        alpha,
        bias,
        norm_weight,
        runtime_config,
        all_event_counters,
        profile_buffer,
        new_residual,
        next_pre_mix,
        next_post_mix,
        next_residual_mix,
        block_input,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        mixed_input,
        rms_rstd,
        hc_eps,
        norm_eps,
        num_iters,
        need_backward,
    )


def mega_mhc_grad(
    grad_hin_placeholder: torch.Tensor,
    grad_h_post: torch.Tensor,
    grad_h_res: torch.Tensor,
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    previous_pre: torch.Tensor,
    hc_before_norm: torch.Tensor,
    inv_rms: torch.Tensor,
    sum_out: torch.Tensor,
    norm_out: torch.Tensor,
    grad_current_pre: torch.Tensor,
    mixed_input: torch.Tensor,
    rms_rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    direct_grad_x: torch.Tensor,
    previous_residual: torch.Tensor,
    previous_output: torch.Tensor,
    previous_post: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    runtime_config: torch.Tensor,
    all_event_counters: torch.Tensor,
    profile_buffer: torch.Tensor,
    grad_residual: torch.Tensor,
    grad_phi: torch.Tensor,
    grad_alpha: torch.Tensor,
    grad_bias: torch.Tensor,
    grad_previous_output: torch.Tensor,
    grad_previous_pre: torch.Tensor,
    grad_previous_post: torch.Tensor,
    grad_previous_residual: torch.Tensor,
    grad_norm_weight: torch.Tensor,
    hc_eps: float,
) -> tuple[torch.Tensor, ...]:
    """Launch the family entry through its verified dispatcher schema.

    Args:
        grad_hin_placeholder: Native Tensor argument at position 0.
        grad_h_post: Native Tensor argument at position 1.
        grad_h_res: Native Tensor argument at position 2.
        x: Native Tensor argument at position 3.
        phi: Native Tensor argument at position 4.
        alpha: Native Tensor argument at position 5.
        bias: Native Tensor argument at position 6.
        previous_pre: Native Tensor argument at position 7.
        hc_before_norm: Native Tensor argument at position 8.
        inv_rms: Native Tensor argument at position 9.
        sum_out: Native Tensor argument at position 10.
        norm_out: Native Tensor argument at position 11.
        grad_current_pre: Native Tensor argument at position 12.
        mixed_input: Native Tensor argument at position 13.
        rms_rstd: Native Tensor argument at position 14.
        norm_weight: Native Tensor argument at position 15.
        direct_grad_x: Native Tensor argument at position 16.
        previous_residual: Native Tensor argument at position 17.
        previous_output: Native Tensor argument at position 18.
        previous_post: Native Tensor argument at position 19.
        previous_residual_mix: Native Tensor argument at position 20.
        runtime_config: Native Tensor argument at position 21.
        all_event_counters: Native Tensor argument at position 22.
        profile_buffer: Native Tensor argument at position 23.
        grad_residual: Native Tensor argument at position 24.
        grad_phi: Native Tensor argument at position 25.
        grad_alpha: Native Tensor argument at position 26.
        grad_bias: Native Tensor argument at position 27.
        grad_previous_output: Native Tensor argument at position 28.
        grad_previous_pre: Native Tensor argument at position 29.
        grad_previous_post: Native Tensor argument at position 30.
        grad_previous_residual: Native Tensor argument at position 31.
        grad_norm_weight: Native Tensor argument at position 32.
        hc_eps: Native float argument at position 33.
    """
    return resolve_native_call("mega_mhc_grad")(
        grad_hin_placeholder,
        grad_h_post,
        grad_h_res,
        x,
        phi,
        alpha,
        bias,
        previous_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        grad_current_pre,
        mixed_input,
        rms_rstd,
        norm_weight,
        direct_grad_x,
        previous_residual,
        previous_output,
        previous_post,
        previous_residual_mix,
        runtime_config,
        all_event_counters,
        profile_buffer,
        grad_residual,
        grad_phi,
        grad_alpha,
        grad_bias,
        grad_previous_output,
        grad_previous_pre,
        grad_previous_post,
        grad_previous_residual,
        grad_norm_weight,
        hc_eps,
    )


def mega_moe(
    dispatch_target: torch.Tensor,
    dispatch_target_off: torch.Tensor,
    dispatch_src: torch.Tensor,
    dispatch_src_off: torch.Tensor,
    dispatch_size: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_glist: torch.Tensor,
    up_proj_y: torch.Tensor,
    swiglu_out: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_glist: torch.Tensor,
    down_proj_y: torch.Tensor,
    combine_target: torch.Tensor,
    combine_target_off: torch.Tensor,
    combine_src_off: torch.Tensor,
    combine_size: torch.Tensor,
    gmm_workspace: torch.Tensor,
    up_proj_tiling: torch.Tensor,
    swiglu_tiling: torch.Tensor,
    down_proj_tiling: torch.Tensor,
    runtime_config: torch.Tensor,
    all_event_counters: torch.Tensor,
    profile_buffer: torch.Tensor,
    rank_id: int,
    ep: int,
    expert_num: int,
    hidden_size: int,
    seq_size: int,
) -> tuple[torch.Tensor, ...]:
    """Launch the family entry through its verified dispatcher schema.

    Args:
        dispatch_target: Native Tensor argument at position 0.
        dispatch_target_off: Native Tensor argument at position 1.
        dispatch_src: Native Tensor argument at position 2.
        dispatch_src_off: Native Tensor argument at position 3.
        dispatch_size: Native Tensor argument at position 4.
        up_proj_weight: Native Tensor argument at position 5.
        up_proj_glist: Native Tensor argument at position 6.
        up_proj_y: Native Tensor argument at position 7.
        swiglu_out: Native Tensor argument at position 8.
        down_proj_weight: Native Tensor argument at position 9.
        down_proj_glist: Native Tensor argument at position 10.
        down_proj_y: Native Tensor argument at position 11.
        combine_target: Native Tensor argument at position 12.
        combine_target_off: Native Tensor argument at position 13.
        combine_src_off: Native Tensor argument at position 14.
        combine_size: Native Tensor argument at position 15.
        gmm_workspace: Native Tensor argument at position 16.
        up_proj_tiling: Native Tensor argument at position 17.
        swiglu_tiling: Native Tensor argument at position 18.
        down_proj_tiling: Native Tensor argument at position 19.
        runtime_config: Native Tensor argument at position 20.
        all_event_counters: Native Tensor argument at position 21.
        profile_buffer: Native Tensor argument at position 22.
        rank_id: Native int argument at position 23.
        ep: Native int argument at position 24.
        expert_num: Native int argument at position 25.
        hidden_size: Native int argument at position 26.
        seq_size: Native int argument at position 27.
    """
    return resolve_native_call("mega_moe")(
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
        profile_buffer,
        rank_id,
        ep,
        expert_num,
        hidden_size,
        seq_size,
    )


def mega_moe_grad(
    dispatch_target: torch.Tensor,
    dispatch_target_off: torch.Tensor,
    dy: torch.Tensor,
    dispatch_src_off: torch.Tensor,
    dispatch_size: torch.Tensor,
    hidden: torch.Tensor,
    hidden_dw: torch.Tensor,
    w2: torch.Tensor,
    act_grad_y: torch.Tensor,
    gate: torch.Tensor,
    grad_gate: torch.Tensor,
    w1: torch.Tensor,
    gate_dx: torch.Tensor,
    grad_x: torch.Tensor,
    combine_target_off: torch.Tensor,
    combine_src_off: torch.Tensor,
    combine_size: torch.Tensor,
    permute_out: torch.Tensor,
    gate_dw: torch.Tensor,
    group_list: torch.Tensor,
    act_grad_tiling: torch.Tensor,
    gate_grad_tiling: torch.Tensor,
    w1_grad_tiling: torch.Tensor,
    w2_grad_tiling: torch.Tensor,
    swiglu_grad_tiling: torch.Tensor,
    gmm_workspace: torch.Tensor,
    swiglu_grad_workspace: torch.Tensor,
    runtime_config: torch.Tensor,
    all_event_counters: torch.Tensor,
    profile_buffer: torch.Tensor,
    rank_id: int,
    ep: int,
    expert_num: int,
    hidden_size: int,
    seq_size: int,
) -> tuple[torch.Tensor, ...]:
    """Launch the family entry through its verified dispatcher schema.

    Args:
        dispatch_target: Native Tensor argument at position 0.
        dispatch_target_off: Native Tensor argument at position 1.
        dy: Native Tensor argument at position 2.
        dispatch_src_off: Native Tensor argument at position 3.
        dispatch_size: Native Tensor argument at position 4.
        hidden: Native Tensor argument at position 5.
        hidden_dw: Native Tensor argument at position 6.
        w2: Native Tensor argument at position 7.
        act_grad_y: Native Tensor argument at position 8.
        gate: Native Tensor argument at position 9.
        grad_gate: Native Tensor argument at position 10.
        w1: Native Tensor argument at position 11.
        gate_dx: Native Tensor argument at position 12.
        grad_x: Native Tensor argument at position 13.
        combine_target_off: Native Tensor argument at position 14.
        combine_src_off: Native Tensor argument at position 15.
        combine_size: Native Tensor argument at position 16.
        permute_out: Native Tensor argument at position 17.
        gate_dw: Native Tensor argument at position 18.
        group_list: Native Tensor argument at position 19.
        act_grad_tiling: Native Tensor argument at position 20.
        gate_grad_tiling: Native Tensor argument at position 21.
        w1_grad_tiling: Native Tensor argument at position 22.
        w2_grad_tiling: Native Tensor argument at position 23.
        swiglu_grad_tiling: Native Tensor argument at position 24.
        gmm_workspace: Native Tensor argument at position 25.
        swiglu_grad_workspace: Native Tensor argument at position 26.
        runtime_config: Native Tensor argument at position 27.
        all_event_counters: Native Tensor argument at position 28.
        profile_buffer: Native Tensor argument at position 29.
        rank_id: Native int argument at position 30.
        ep: Native int argument at position 31.
        expert_num: Native int argument at position 32.
        hidden_size: Native int argument at position 33.
        seq_size: Native int argument at position 34.
    """
    return resolve_native_call("mega_moe_grad")(
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
        profile_buffer,
        rank_id,
        ep,
        expert_num,
        hidden_size,
        seq_size,
    )
