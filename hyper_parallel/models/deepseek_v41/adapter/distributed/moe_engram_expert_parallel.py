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
"""Dispatch DeepSeek-V4.1 MoE experts and Engram lookups over EP."""

from __future__ import annotations

import inspect
from types import MethodType
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    ep_routed_forward,
    require_attrs,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.expert_parallel.hot_replica.admission import reduction_basis_points
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig, _integer
from hyper_parallel.core.expert_parallel.hot_replica.native import (
    combine_native_replicas, dispatch_native_replicas, native_replica_experts,
)
from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute, stable_expert_order
from hyper_parallel.distributed._builder.forward_rewriter import _install_bound_forward
from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import (
    DeepseekV41MegaMoeExperts, _swiglu_limit,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import deepseek_v41_swiglu


def _deepseek_v41_unclamped_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply native FP32 SwiGLU when a nonpositive limit disables clamping."""
    gate, up = gate_up.float().chunk(2, dim=-1)
    return deepseek_v41_swiglu(gate, up, 0.0).to(gate_up.dtype)


def _deepseek_v41_grouped_expert_forward(
        experts: torch.nn.Module,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Run FP32 clamped SwiGLU between two grouped GEMMs.

    The generic EP path has already sorted ``hidden_states`` into expert-major
    order. Routed and shared experts reuse the native FP32 clamp and SwiGLU
    implementation. Nonpositive limits disable clamping; the down projection
    still receives the original hidden-state dtype.
    """
    # The shared primitive imports optional torch_npu; keep CPU/GPU imports lazy.
    from hyper_parallel.components.functional import grouped_matmul  # pylint: disable=C0415

    group_list = torch.cumsum(
        tokens_per_expert.to(device=hidden_states.device, dtype=torch.int64),
        dim=0,
    )
    gate_up = grouped_matmul(  # pylint: disable=not-callable
        hidden_states,
        experts.gate_up_proj.transpose(1, 2),
        bias=None,
        group_list=group_list,
        group_type=0,
        group_list_type=0,
    )
    # Cast the packed projections once; the shared helper's casts are then no-ops.
    gate, up = gate_up.float().chunk(2, dim=-1)
    intermediate = deepseek_v41_swiglu(gate, up, experts.limit)
    return grouped_matmul(  # pylint: disable=not-callable
        intermediate.to(hidden_states.dtype),
        experts.down_proj.transpose(1, 2),
        bias=None,
        group_list=group_list,
        group_type=0,
        group_list_type=0,
    )


def _router(
        module: Any,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the V4.1 learned router's selected experts and weights."""
    output = (
        module.gate(hidden_states)
        if image_mask is None
        else module.gate(hidden_states, image_mask=image_mask)
    )
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise TypeError("DeepSeek-V4.1 gate must return logits, weights, and indices")
    _, weights, indices = output
    return indices, weights


def _routed_and_shared_forward(
        module: Any,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None,
        ep_group: Any,
) -> torch.Tensor:
    """Dispatch routed experts and add the V4.1 shared expert branch."""
    routed = ep_routed_forward(
        module,
        hidden_states,
        router_fn=lambda target_module, target_states: _router(target_module, target_states, image_mask),
        ep_group=ep_group,
    )
    return routed + module.shared_experts(hidden_states)


def _replica_expert_forward(experts: Any, values: torch.Tensor, counts: torch.Tensor,
                            route: ReplicaRoute) -> torch.Tensor:
    """Read current FSDP-unsharded weights inside the experts module boundary."""
    weights = tuple(parameter.to_local() if isinstance(parameter, DTensor) else parameter
                    for parameter in (experts.gate_up_proj, experts.down_proj))
    return native_replica_experts(
        values, weights[0].transpose(1, 2).contiguous(), weights[1].transpose(1, 2).contiguous(),
        counts, route, swiglu_limit=float(experts.limit),
    )


def _replica_routed_forward(module: Any, hidden_states: torch.Tensor, image_mask: torch.Tensor | None,
                            ep_group: Any, config: ExpertReplicaConfig, reduction: float) -> torch.Tensor:
    """Balance logical routes before any large receive activation is allocated."""
    indices, weights = _router(module, hidden_states, image_mask)
    states = hidden_states.reshape(-1, hidden_states.shape[-1])
    order = stable_expert_order(indices.reshape(-1), config.num_experts)
    counts = torch.zeros(config.num_experts, dtype=torch.int64, device=states.device)
    counts.scatter_add_(0, indices.reshape(-1).long(), torch.ones_like(indices.reshape(-1), dtype=torch.int64))
    values = states[torch.div(order, indices.shape[-1], rounding_mode="floor")]
    (values, received), state = dispatch_native_replicas(
        (values, counts), config, ep_group, minimum_replica_rows=0, planner_backend="cpu",
        minimum_load_reduction=reduction,
    )
    output = module.experts(values, received, state.route)
    output = combine_native_replicas(output, state)
    unsorted = torch.empty_like(output)
    unsorted[order] = output
    routed = (unsorted.reshape(*indices.shape, states.shape[-1]) * weights.unsqueeze(-1)).sum(dim=1)
    return routed.to(states.dtype).reshape(hidden_states.shape) + module.shared_experts(hidden_states)


@local_compute
def deepseek_v41_ep_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
        use_grouped_gemm: bool = False,
        replica_slots_per_rank: int = 0,
        replica_minimum_load_reduction: float = 0.0,
) -> Callable:
    """Build the complete V4.1 routed-plus-shared expert forward."""
    del mesh
    require_attrs(module, "gate", "experts", "shared_experts", owner="DeepSeek-V4.1 EP")
    if ep_mesh is None:
        raise ValueError("DeepSeek-V4.1 EP forward requires an active ep_mesh")
    if module.is_hash:
        raise ValueError("DeepSeek-V4.1 validation layers must use learned routing")
    _integer(replica_slots_per_rank, "replica_slots_per_rank", 0)
    reduction_basis_points(replica_minimum_load_reduction)
    if replica_slots_per_rank:
        config = ExpertReplicaConfig(module.experts.num_experts, ep_mesh["ep"].size(), replica_slots_per_rank)
        if any(axis is not None and axis.size() > 1 for axis in (tp_mesh, cp_mesh)):
            raise ValueError("DSV4.1 expert replicas require TP=CP=1")
        _install_bound_forward(module.experts, _replica_expert_forward)
        ep_group = ep_mesh.get_group("ep")

        def replica_compute_fn(module: Any, hidden_states: torch.Tensor,
                               input_ids: torch.Tensor | None = None,
                               image_mask: torch.Tensor | None = None) -> torch.Tensor:
            """Preserve learned routing and shared experts with bounded guest slots."""
            del input_ids
            return _replica_routed_forward(module, hidden_states, image_mask, ep_group, config,
                                           replica_minimum_load_reduction)

        return replica_compute_fn
    apply_gate = (
        module.experts._apply_gate  # pylint: disable=protected-access
        if module.experts.limit > 0 else _deepseek_v41_unclamped_gate
    )
    if use_grouped_gemm:
        module.experts.forward_expert_major = MethodType(
            _deepseek_v41_grouped_expert_forward,
            module.experts,
        )
    ep_group = ep_mesh.get_group("ep")
    bind_local_expert_forward(
        module,
        ep_mesh["ep"].size(),
        use_grouped_gemm=use_grouped_gemm,
        apply_gate=None if use_grouped_gemm else apply_gate,
    )

    if "image_mask" not in inspect.signature(module.forward).parameters:
        def text_compute_fn(
                module: Any,
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Run the text-only source contract without visual routing state."""
            del input_ids
            return _routed_and_shared_forward(module, hidden_states, None, ep_group)

        return text_compute_fn

    def multimodal_compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the multimodal source contract with optional visual routing state."""
        del input_ids
        return _routed_and_shared_forward(module, hidden_states, image_mask, ep_group)

    return multimodal_compute_fn


@local_compute
def deepseek_v41_megamoe_compute_fn(
        *, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> Callable:
    """Bind a declaratively replaced expert module to the full EP mesh.

    The source router, visual routing bias and shared experts retain their
    original semantics. Parameter sharding and FSDP remain framework-owned.
    """
    del mesh
    require_attrs(module, "gate", "experts", "shared_experts", owner="DeepSeek-V4.1 MegaMoe")
    if module.is_hash:
        raise ValueError("DeepSeek-V4.1 MegaMoe requires learned routing")
    if any(axis is not None and axis.size() > 1 for axis in (tp_mesh, cp_mesh)):
        raise ValueError("DSV4.1 MegaMoe requires TP=CP=PP=1")
    if not isinstance(module.experts, DeepseekV41MegaMoeExperts):
        raise TypeError("MegaMoe compute requires the DeepseekV41MegaMoeExperts module replacement")
    if _swiglu_limit(module.shared_experts.limit) != module.experts.swiglu_limit:
        raise ValueError("Routed and shared experts must have the same swiglu_limit")
    ep_group = None if ep_mesh is None else ep_mesh.get_group("ep")
    ep_size = 1 if ep_mesh is None else ep_mesh["ep"].size()
    if dist.is_initialized() and dist.get_world_size(ep_group) != ep_size:
        raise ValueError("MegaMoe EP mesh and process group sizes must match")
    module.experts.configure(ep_group, ep_size, module.gate.top_k)

    def compute_fn(
            module: Any, hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None, image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run original routing and shared experts around the MegaMoe call."""
        del input_ids
        indices, weights = _router(module, hidden_states, image_mask)
        return module.experts(hidden_states, indices, weights) + module.shared_experts(hidden_states)

    return compute_fn


@local_compute
def deepseek_v41_engram_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
) -> Callable:
    """Build the sparse EP-row lookup with CP/TP sequence alignment."""
    del mesh
    require_attrs(
        module,
        "hash_mapping",
        "embed",
        "parallel_forward",
        owner="DeepSeek-V4.1 Engram EP",
    )
    if ep_mesh is None:
        raise ValueError("DeepSeek-V4.1 Engram EP requires an active ep_mesh")
    ep_group = ep_mesh.get_group("ep")
    ep_size = ep_mesh["ep"].size()
    ep_rank = ep_mesh.get_local_rank("ep")
    cp_group = None if cp_mesh is None else cp_mesh.get_group()
    cp_size = 1 if cp_mesh is None else cp_mesh.size()
    cp_rank = 0 if cp_mesh is None else cp_mesh.get_local_rank()
    tp_size = 1 if tp_mesh is None else tp_mesh.size()
    tp_rank = 0 if tp_mesh is None else tp_mesh.get_local_rank()

    def compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            segment_starts: torch.Tensor | None = None,
            token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route hash-row requests and run the exact V4.1 fusion."""
        return module.parallel_forward(
            hidden_states,
            input_ids,
            segment_starts,
            token_mask,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            cp_group=cp_group,
            cp_rank=cp_rank,
            cp_size=cp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    return compute_fn


__all__ = ["deepseek_v41_engram_compute_fn", "deepseek_v41_ep_compute_fn", "deepseek_v41_megamoe_compute_fn"]
