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
"""Torch execution resources for the core MegaMoe API."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from ._framework import get_torch
from .config import build_config
from .execution import execute_mega_moe
from .native import execute_native_from_permuted_route
from .route import prepare_topk_route, restore_topk_output
from .runtime import build_mega_moe_runtime
from .workspace import MegaMoeWorkspace, configure_symmetric_heap


def create_mega_moe_parameters(
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[Any, Any]:
    """Create independently initialized local expert parameters."""
    torch = get_torch()

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


class TorchMegaMoeBackend:
    """Own one runtime and directional workspace for a resource group."""

    def __init__(
        self,
        specification: Mapping[str, Any],
        tensor: Any,
        *,
        shared: bool,
        active_specifications: tuple[Any, ...],
    ) -> None:
        """Bind runtime resources to the first NPU tensor."""
        self.config = build_config(specification, tensor)
        configure_symmetric_heap(active_specifications, tensor)
        self.runtime = build_mega_moe_runtime(self.config, tensor.device)
        self.workspace = MegaMoeWorkspace(shared=shared)
        symm = importlib.import_module("hyper_parallel.core.symmetric_memory")

        symm.initialize(self.config.ep_group)

    def forward(
        self,
        hidden_states: Any,
        topk_ids: Any,
        topk_weights: Any,
        weight1: Any,
        weight2: Any,
        *,
        tokens_per_expert: Any | None,
    ) -> Any:
        """Execute Top-K routing, legacy kernels and weighted restoration."""
        self._validate_tensors(hidden_states, weight1, weight2)
        route = prepare_topk_route(
            hidden_states,
            topk_ids,
            topk_weights,
            self.runtime,
            tokens_per_expert,
        )
        if route.receive_overflow:
            return execute_native_from_permuted_route(
                route.routed_tokens,
                route.unpermute_mapping,
                topk_weights,
                weight1,
                weight2,
                route.tokens_per_expert,
                route.received_counts,
                self.config,
            )
        if route.metadata is None:
            raise RuntimeError(
                "MegaMoe route data is incomplete without capacity overflow."
            )
        expert_output = execute_mega_moe(
            route.routed_tokens,
            weight1,
            weight2,
            route.metadata,
            self.runtime,
            self.workspace,
        )
        return restore_topk_output(
            expert_output,
            route.unpermute_mapping,
            topk_weights,
        )

    def _validate_tensors(self, hidden_states: Any, weight1: Any, weight2: Any) -> None:
        """Validate dtype, device and local expert parameter shapes."""
        torch = get_torch()

        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_npu:
            raise TypeError("MegaMoeExperts requires BF16 NPU hidden states.")
        expected_weight1 = (
            self.config.local_experts,
            self.config.hidden_size,
            self.config.intermediate_size * 2,
        )
        expected_weight2 = (
            self.config.local_experts,
            self.config.intermediate_size,
            self.config.hidden_size,
        )
        if tuple(weight1.shape) != expected_weight1:
            raise ValueError(f"gate_up_weight must have shape {expected_weight1}.")
        if tuple(weight2.shape) != expected_weight2:
            raise ValueError(f"down_weight must have shape {expected_weight2}.")
        if (
            weight1.device != hidden_states.device
            or weight2.device != hidden_states.device
        ):
            raise ValueError(
                "expert weights and hidden states must be on the same NPU."
            )
        if weight1.dtype != hidden_states.dtype or weight2.dtype != hidden_states.dtype:
            raise TypeError("expert weights and hidden states must all use BF16.")
