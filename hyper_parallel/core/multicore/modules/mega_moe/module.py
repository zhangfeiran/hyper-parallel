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
"""Model-facing MegaMoe expert module."""

from __future__ import annotations

__all__ = ["MegaMoeExperts"]

import math
from typing import Any

from hyper_parallel.core.multicore.platform.torch.mega_moe.backend import (
    TorchMegaMoeBackend,
    create_mega_moe_parameters,
)

from ..module import MulticoreModule


class MegaMoeExperts(MulticoreModule):
    """Own local expert weights and a lazily created MegaMoe runtime.

    Serial model layers can call :meth:`share_execution_resources` once after
    construction to use one bounded workspace instead of one workspace per layer.
    """

    def __init__(
        self,
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        expert_capacity_factor: float = 1.5,
        ep_size: int = 1,
        ep_group: Any | None = None,
    ) -> None:
        """Initialize local expert parameters.

        Args:
            local_num_tokens: Token rows supplied to this rank.
            hidden_size: Input and output hidden dimension.
            intermediate_size: SwiGLU intermediate dimension per expert.
            num_experts: Global routed-expert count.
            top_k: Experts selected for every token.
            expert_capacity_factor: Multiplier applied to
                ``local_num_tokens * top_k`` when sizing the maximum rows that
                one EP rank may receive through MegaMoe. The resolved capacity
                is aligned to the kernel communication split; overflowing calls
                use the native fallback.
            ep_size: Expert-parallel degree.
            ep_group: Expert-parallel process group.
        """
        self._validate_topology(
            local_num_tokens=local_num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor,
            ep_size=ep_size,
        )
        expert_capacity_factor = float(expert_capacity_factor)
        specification = {
            "local_num_tokens": local_num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "expert_capacity_factor": expert_capacity_factor,
            "ep_size": ep_size,
            "ep_group": ep_group,
        }
        compatibility_key = (
            local_num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            expert_capacity_factor,
            ep_size,
            id(ep_group),
        )
        super().__init__(
            resource_specification=specification,
            resource_compatibility_key=compatibility_key,
            resource_scope_key=("mega_moe", id(ep_group)),
        )
        self.local_num_tokens = local_num_tokens
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.ep_size = ep_size
        self.local_experts = num_experts // ep_size
        self._ep_group = ep_group
        self.gate_up_weight, self.down_weight = create_mega_moe_parameters(
            self.local_experts,
            hidden_size,
            intermediate_size,
        )

    @staticmethod
    def _validate_topology(
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        expert_capacity_factor: float,
        ep_size: int,
    ) -> None:
        """Validate public shape and topology values before allocation."""
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
        valid_factor_type = isinstance(
            expert_capacity_factor, (int, float)
        ) and not isinstance(expert_capacity_factor, bool)
        try:
            valid_factor_value = valid_factor_type and math.isfinite(
                expert_capacity_factor
            )
        except OverflowError:
            valid_factor_value = False
        if not valid_factor_value or expert_capacity_factor < 1.0:
            raise ValueError(
                "expert_capacity_factor must be a finite number at least 1.0, "
                f"got {expert_capacity_factor!r}."
            )

    def forward(
        self,
        hidden_states: Any,
        topk_ids: Any,
        topk_weights: Any,
        *,
        tokens_per_expert: Any | None = None,
    ) -> Any:
        """Route token states through the configured experts.

        Args:
            hidden_states: Tensor ending in ``hidden_size``.
            topk_ids: Global expert IDs shaped ``[local_tokens, top_k]``.
            topk_weights: Route weights with the same shape as ``topk_ids``.
            tokens_per_expert: Optional trusted exact global-expert histogram.
                When omitted, the histogram is computed from ``topk_ids``.

        Returns:
            Tensor matching the shape and dtype of ``hidden_states``.
        """
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states must have at least two dimensions and configured "
                f"hidden size {self.hidden_size}, got {tuple(hidden_states.shape)}."
            )
        hidden_shape = tuple(hidden_states.shape)
        hidden_flat = hidden_states.reshape(-1, self.hidden_size)
        if hidden_flat.shape[0] != self.local_num_tokens:
            raise ValueError(
                f"MegaMoeExperts expects {self.local_num_tokens} local tokens, "
                f"got {hidden_flat.shape[0]}."
            )
        route_shape = (self.local_num_tokens, self.top_k)
        if tuple(topk_ids.shape) != route_shape:
            raise ValueError(
                f"topk_ids must have shape {route_shape}, got {tuple(topk_ids.shape)}."
            )
        if tuple(topk_weights.shape) != route_shape:
            raise ValueError(
                f"topk_weights must have shape {route_shape}, got {tuple(topk_weights.shape)}."
            )
        if tokens_per_expert is not None and tuple(tokens_per_expert.shape) != (
            self.num_experts,
        ):
            raise ValueError(
                f"tokens_per_expert must have shape ({self.num_experts},), "
                f"got {tuple(tokens_per_expert.shape)}."
            )
        backend = self._get_execution_backend(hidden_flat)
        output = backend.forward(
            hidden_flat,
            topk_ids,
            topk_weights,
            self.gate_up_weight,
            self.down_weight,
            tokens_per_expert=tokens_per_expert,
        )
        return output.reshape(hidden_shape)

    def _create_execution_backend(
        self,
        tensor: Any,
        *,
        shared: bool,
        active_specifications: tuple[Any, ...],
    ) -> Any:
        """Create the MegaMoe runtime and workspace owned by this resource group."""
        return TorchMegaMoeBackend(
            self._resource_group.specification,
            tensor,
            shared=shared,
            active_specifications=active_specifications,
        )
