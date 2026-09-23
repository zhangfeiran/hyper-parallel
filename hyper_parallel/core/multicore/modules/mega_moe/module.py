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
"""Model-facing Torch MegaMoe expert module."""

from __future__ import annotations

__all__ = ["MegaMoeExperts"]

import math
import struct
from collections import OrderedDict
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.scheduler.config import MAX_EXPERT_NUM_PER_RANK

from ..module import MulticoreModule
from .function import execute_mega_moe_with_permutation
from .plan import build_mega_moe_plan
from .route import prepare_topk_route, restore_topk_output
from .spec import _COMMUNICATION_SPLIT, _align_capacity, bind_mega_moe_spec
from .workspace import MegaMoeWorkspace, configure_symmetric_heap


def _validate_expert_capacity_factor(expert_capacity_factor: float | None) -> None:
    """Validate the optional bounded receive-capacity multiplier."""
    if expert_capacity_factor is None:
        return
    valid_type = isinstance(expert_capacity_factor, (int, float)) and not isinstance(
        expert_capacity_factor, bool
    )
    try:
        valid_value = valid_type and math.isfinite(expert_capacity_factor)
    except OverflowError:
        valid_value = False
    if not valid_value or expert_capacity_factor < 1.0:
        raise ValueError(
            "expert_capacity_factor must be None or a finite number at least 1.0, "
            f"got {expert_capacity_factor!r}."
        )


def _validate_swiglu_limit(swiglu_limit: float | None) -> None:
    """Validate a positive clamp value that survives float32 serialization."""
    if swiglu_limit is None:
        return
    valid_type = isinstance(swiglu_limit, (int, float)) and not isinstance(
        swiglu_limit, bool
    )
    try:
        encoded_limit = (
            struct.unpack("<f", struct.pack("<f", float(swiglu_limit)))[0]
            if valid_type
            else 0.0
        )
        valid_value = (
            valid_type
            and math.isfinite(swiglu_limit)
            and math.isfinite(encoded_limit)
            and encoded_limit > 0
        )
    except (OverflowError, TypeError, ValueError, struct.error):
        valid_value = False
    if not valid_value:
        raise ValueError(
            "swiglu_limit must be None or a finite positive float32-representable number, "
            f"got {swiglu_limit!r}."
        )


def _create_mega_moe_parameters(
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Create independently initialized local expert parameters."""
    gate_up_weight = torch.nn.Parameter(
        torch.empty((local_experts, hidden_size, intermediate_size * 2))
    )
    down_weight = torch.nn.Parameter(
        torch.empty((local_experts, intermediate_size, hidden_size))
    )
    for expert_index in range(local_experts):
        torch.nn.init.xavier_uniform_(gate_up_weight[expert_index])
        torch.nn.init.xavier_uniform_(down_weight[expert_index])
    return gate_up_weight, down_weight


class _MegaMoeExecutionResources:
    """Own one shape-bound plan and workspace in the shared SHMEM lifecycle."""

    def __init__(
        self,
        specification: dict[str, Any],
        tensor: torch.Tensor,
        *,
        shared: bool,
        active_specifications: tuple[Any, ...],
    ) -> None:
        """Bind resources once to the first NPU tensor."""
        self.spec = bind_mega_moe_spec(specification, tensor)
        self._plans = OrderedDict()
        if dist.is_initialized():
            self._validate_distributed_configuration(active_specifications)
        configure_symmetric_heap(active_specifications, tensor)
        shmem.acquire(self.spec.ep_group)
        try:
            self.plan = (build_mega_moe_plan(self.spec, tensor.device)
                         if self.spec.max_local_num_tokens is None else None)
            self.workspace = MegaMoeWorkspace(shared=shared)
        except Exception:
            shmem.release()
            raise
        self._closed = False

    def _validate_distributed_configuration(self, specifications: tuple[Any, ...]) -> None:
        """Reject different allocation manifests before acquiring symmetric memory."""
        if self.spec.ep_size == 1:
            return
        fields = ("local_num_tokens", "max_local_num_tokens", "hidden_size", "intermediate_size",
                  "num_experts", "top_k", "expert_capacity_factor", "swiglu_limit", "ep_size")
        # BF16 is the sole supported dtype; the literal tags also identify the protocol.
        manifest = ("dynamic_tokens_v1", "bfloat16",
                    tuple(tuple(spec.get(key) for key in fields) for spec in specifications))
        # Resource groups must have the same order as their symmetric allocations.
        manifests = [None] * self.spec.ep_size
        dist.all_gather_object(manifests, manifest, group=self.spec.ep_group)
        if any(value != manifests[0] for value in manifests):
            raise ValueError("MegaMoe resource configurations differ across EP ranks")

    def plan_for_tokens(self, tokens: int, device: Any) -> Any:
        """Cache bounded schedules without changing symmetric workspace capacity."""
        if tokens not in self._plans:
            self._plans[tokens] = build_mega_moe_plan(replace(self.spec, schedule_tokens=tokens), device)
        self._plans.move_to_end(tokens)
        plan = self._plans[tokens]
        while len(self._plans) > 4:
            # Autograd contexts keep in-flight plans alive independently of this cache.
            self._plans.popitem(last=False)
        return plan

    def close(self) -> None:
        """Release the workspace and leave the shared SHMEM lifecycle."""
        if self._closed:
            return
        self.workspace.close()
        shmem.release()
        self._plans.clear()
        self._closed = True


class MegaMoeExperts(MulticoreModule):
    """Execute Router-selected local experts with the Torch MegaMoe kernel.

    Serial model layers can call :meth:`share_execution_resources` before
    their first forward to share one lossless-capacity workspace while keeping
    independent parameters and optimizer state.
    """

    def __init__(
        self,
        *,
        local_num_tokens: int | None = None,
        max_local_num_tokens: int | None = None,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        expert_capacity_factor: float | None = None,
        swiglu_limit: float | None = None,
        ep_size: int = 1,
        ep_group: Any | None = None,
        create_parameters: bool = True,
    ) -> None:
        """Initialize local expert parameters and a lazy execution owner.

        Args:
            local_num_tokens: Static token rows supplied to this rank.
            max_local_num_tokens: Opt-in variable token bound, exclusive with local_num_tokens.
            hidden_size: Input and output hidden dimension.
            intermediate_size: SwiGLU intermediate dimension per expert.
            num_experts: Global routed-expert count.
            top_k: Experts selected for every token.
            expert_capacity_factor: Optional bounded receive-capacity multiplier.
                ``None`` reserves the maximum lossless capacity. A finite value
                of at least 1.0 reserves that multiple of the local routed rows
                and raises a clear error if a route exceeds it.
            swiglu_limit: Optional positive, finite float32-representable clamp
                limit for SwiGLU. The gate branch is clipped at the positive
                limit and the up branch symmetrically. ``None`` preserves the
                legacy unclamped path.
            ep_size: Expert-parallel degree. The current SHMEM path requires it
                to cover the complete Torch distributed world.
            ep_group: Torch expert-parallel process group with the same rank
                ordering as the complete distributed world.
            create_parameters: Allocate owned weights; False requires explicit expert_weights each forward.
        """
        if (local_num_tokens is None) == (max_local_num_tokens is None):
            raise ValueError("Provide exactly one of local_num_tokens and max_local_num_tokens")
        reserved_tokens = local_num_tokens
        if max_local_num_tokens is not None:
            if (not isinstance(max_local_num_tokens, int) or isinstance(max_local_num_tokens, bool)
                    or max_local_num_tokens <= 0):
                raise ValueError("max_local_num_tokens must be a positive integer")
            reserved_tokens = _align_capacity(max_local_num_tokens)
        self._validate_topology(
            local_num_tokens=reserved_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor,
            swiglu_limit=swiglu_limit,
            ep_size=ep_size,
        )
        if max_local_num_tokens is not None and reserved_tokens * max(hidden_size, top_k) > 2**31 - 1:
            raise ValueError("max_local_num_tokens exceeds native INT32 route limits")
        if expert_capacity_factor is not None:
            expert_capacity_factor = float(expert_capacity_factor)
        if swiglu_limit is not None:
            swiglu_limit = float(swiglu_limit)
        specification = {
            "local_num_tokens": reserved_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "expert_capacity_factor": expert_capacity_factor,
            "swiglu_limit": swiglu_limit,
            "ep_size": ep_size,
            "ep_group": ep_group,
        }
        if max_local_num_tokens is not None:
            specification["max_local_num_tokens"] = max_local_num_tokens
        compatibility_key = (
            reserved_tokens,
            max_local_num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            expert_capacity_factor,
            swiglu_limit,
            ep_size,
            id(ep_group),
        )
        super().__init__(
            resource_specification=specification,
            resource_compatibility_key=compatibility_key,
            resource_scope_key=("mega_moe", id(ep_group)),
        )
        self.local_num_tokens = local_num_tokens
        self.max_local_num_tokens = max_local_num_tokens
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.swiglu_limit = swiglu_limit
        self.ep_size = ep_size
        self.local_experts = num_experts // ep_size
        self._ep_group = ep_group
        if create_parameters:
            self.gate_up_weight, self.down_weight = _create_mega_moe_parameters(
                self.local_experts, hidden_size, intermediate_size,
            )
        else:
            self.register_parameter("gate_up_weight", None)
            self.register_parameter("down_weight", None)

    @staticmethod
    def _validate_topology(
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        expert_capacity_factor: float | None,
        swiglu_limit: float | None,
        ep_size: int,
    ) -> None:
        """Validate static shape and topology values before allocation."""
        values = {
            "local_num_tokens": local_num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "ep_size": ep_size,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts})."
            )
        if num_experts % ep_size:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
            )
        if num_experts // ep_size > MAX_EXPERT_NUM_PER_RANK:
            raise ValueError(
                "local expert count cannot exceed the device scratch capacity "
                f"({MAX_EXPERT_NUM_PER_RANK}), got {num_experts // ep_size}."
            )
        if local_num_tokens % _COMMUNICATION_SPLIT:
            raise ValueError(
                "local_num_tokens must be divisible by the fixed communication "
                f"split {_COMMUNICATION_SPLIT}, got {local_num_tokens}."
            )
        _validate_expert_capacity_factor(expert_capacity_factor)
        _validate_swiglu_limit(swiglu_limit)

    def _validate_tensors(self, hidden_states: torch.Tensor, expert_weights: tuple) -> None:
        """Validate activation and parameter metadata before acquiring resources."""
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_npu:
            raise TypeError("MegaMoeExperts requires BF16 NPU hidden states.")
        weight1, weight2 = expert_weights
        if weight1 is None or weight2 is None:
            raise ValueError("Parameterless MegaMoe requires explicit expert_weights")
        expected_weight1 = (
            self.local_experts,
            self.hidden_size,
            self.intermediate_size * 2,
        )
        expected_weight2 = (
            self.local_experts,
            self.intermediate_size,
            self.hidden_size,
        )
        if tuple(weight1.shape) != expected_weight1:
            raise ValueError(
                f"gate_up_weight must have shape {expected_weight1}, got {tuple(weight1.shape)}."
            )
        if tuple(weight2.shape) != expected_weight2:
            raise ValueError(
                f"down_weight must have shape {expected_weight2}, got {tuple(weight2.shape)}."
            )
        if (
            weight1.device != hidden_states.device
            or weight2.device != hidden_states.device
        ):
            raise ValueError(
                "expert weights and hidden states must be on the same NPU."
            )
        if weight1.dtype != hidden_states.dtype or weight2.dtype != hidden_states.dtype:
            raise TypeError("expert weights and hidden states must all use BF16.")

    def _validate_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor | None,
        expert_weights: tuple,
    ) -> torch.Tensor:
        """Validate input shapes and return flattened local token states."""
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states must have at least two dimensions and configured "
                f"hidden size {self.hidden_size}, got {tuple(hidden_states.shape)}."
            )
        hidden_flat = hidden_states.reshape(-1, self.hidden_size)
        if self.max_local_num_tokens is None and hidden_flat.shape[0] != self.local_num_tokens:
            raise ValueError(
                f"MegaMoeExperts expects {self.local_num_tokens} local tokens, "
                f"got {hidden_flat.shape[0]}."
            )
        route_shape = (hidden_flat.shape[0], self.top_k)
        if tuple(topk_ids.shape) != route_shape:
            raise ValueError(
                f"topk_ids must have shape {route_shape}, got {tuple(topk_ids.shape)}."
            )
        if tuple(topk_weights.shape) != route_shape:
            raise ValueError(
                f"topk_weights must have shape {route_shape}, "
                f"got {tuple(topk_weights.shape)}."
            )
        if tokens_per_expert is not None and tuple(tokens_per_expert.shape) != (
            self.num_experts,
        ):
            raise ValueError(
                f"tokens_per_expert must have shape ({self.num_experts},), "
                f"got {tuple(tokens_per_expert.shape)}."
            )
        self._validate_tensors(hidden_flat, expert_weights)
        return hidden_flat

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        tokens_per_expert: torch.Tensor | None = None,
        expert_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Route token states through the configured experts.

        Args:
            hidden_states: Tensor ending in ``hidden_size``.
            topk_ids: Global expert IDs shaped ``[local_num_tokens, top_k]``.
            topk_weights: Router weights with the same shape as ``topk_ids``.
            expert_weights: Current unsharded weights owned by an outer module; never cached by this executor.
            tokens_per_expert: Optional trusted exact global-expert histogram.
                Supplying Router-produced counts skips histogram recomputation.

        Returns:
            Tensor matching the shape and dtype of ``hidden_states``.

        Note:
            ``tokens_per_expert`` must describe the current ``topk_ids`` exactly.
            The steady-state path validates metadata but deliberately does not
            rebuild and compare the histogram.
        """
        weights = ((self.gate_up_weight, self.down_weight) if expert_weights is None else expert_weights)
        hidden_flat = self._validate_forward_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            weights,
        )
        resources = self._get_execution_resources(hidden_flat)
        route_options = {}
        if self.max_local_num_tokens is not None:
            route_options["autograd_mask"] = (
                int(torch.is_grad_enabled()) | (int(hidden_flat.requires_grad) << 1)
                | (int(weights[0].requires_grad) << 2) | (int(weights[1].requires_grad) << 3)
            )
        # The expert autograd bridge consumes permutation gradients before the
        # workspace can be reused, so route preparation needs no separate node.
        with torch.no_grad():
            route = prepare_topk_route(
                hidden_flat,
                topk_ids,
                topk_weights,
                resources.spec,
                tokens_per_expert,
                workspace=resources.workspace,
                **route_options,
            )
        plan = resources.plan if self.max_local_num_tokens is None else resources.plan_for_tokens(
            route.metadata.plan_tokens, hidden_flat.device)
        expert_output = execute_mega_moe_with_permutation(
            hidden_flat,
            topk_ids,
            weights[0],
            weights[1],
            route,
            plan,
            resources.workspace,
        )
        output = restore_topk_output(
            expert_output,
            route.unpermute_mapping,
            topk_weights,
        )
        return output.reshape(hidden_states.shape)

    def close(self) -> None:
        """Reject teardown while the last owner's graphs still need backward."""
        group = self._resource_group
        if not self._resource_closed and group.resources is not None and len(group.members) == 1:
            group.resources.workspace.validate_close()
        super().close()

    def _create_execution_resources(
        self,
        tensor: torch.Tensor,
        *,
        shared: bool,
        active_specifications: tuple[Any, ...],
    ) -> _MegaMoeExecutionResources:
        """Create the plan and workspace owned by this resource group."""
        return _MegaMoeExecutionResources(
            self._resource_group.specification,
            tensor,
            shared=shared,
            active_specifications=active_specifications,
        )
