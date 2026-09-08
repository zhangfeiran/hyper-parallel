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

import math
from dataclasses import replace
from functools import partial
from importlib import import_module
from typing import Any

from hyper_parallel.platform.torch.symmetric_memory import acquire_symmetric_memory

from ..module import MulticoreModule
from .function import execute_mega_moe
from .plan import build_mega_moe_plan
from .replica import ExpertReplicaManager, SharedExpertWeightLease
from .route import prepare_topk_route
from .spec import _COMMUNICATION_SPLIT, bind_mega_moe_spec
from .workspace import MegaMoeWorkspace, configure_symmetric_heap

torch = import_module("torch")

__all__ = ["MegaMoeExperts"]


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
    """Own one shape-bound plan, SHMEM handle, and workspace."""

    def __init__(
        self,
        specification: dict[str, Any],
        tensor: torch.Tensor,
        *,
        shared: bool,
        active_specifications: tuple[Any, ...],
        pool_cache_size: int = 1,
        compact_physical_weights: bool = False,
        shared_physical_weights: bool = False,
        compact_layout_strategy: str = "eager_on_first_forward",
        replica_grad_receive_slots: int = 8,
    ) -> None:
        """Bind resources once to the first NPU tensor."""
        self.spec = bind_mega_moe_spec(specification, tensor)
        configure_symmetric_heap(active_specifications, tensor)
        self.symmetric_memory = acquire_symmetric_memory(self.spec.ep_group)
        try:
            self.plan = build_mega_moe_plan(self.spec, tensor.device)
            self.home_runtime = (
                build_mega_moe_plan(
                    replace(self.spec, replica_slots_per_rank=0), tensor.device,
                    event_capacity=self.plan.event_capacity,
                )
                if self.spec.replica_slots_per_rank else self.plan
            )
            self.workspace = MegaMoeWorkspace(
                symmetric_memory=self.symmetric_memory,
                shared=shared,
            )
            self.replica_manager = (
                ExpertReplicaManager(
                    self.spec, pool_cache_size=pool_cache_size,
                    compact_physical_weights=compact_physical_weights,
                    compact_layout_strategy=compact_layout_strategy,
                    gradient_receive_slots=replica_grad_receive_slots,
                ) if self.spec.replica_slots_per_rank else None
            )
        except Exception:
            self.symmetric_memory.close()
            raise
        self.shared_physical_weights = shared_physical_weights
        self._closed = False

    def close(self) -> None:
        """Release the workspace and last-owned SHMEM lifecycle."""
        if self._closed:
            return
        self._closed = True
        try:
            self.workspace.close()
        finally:
            self.symmetric_memory.close()


_COMPACT_LAYOUT_STRATEGIES = frozenset(
    {"lazy_on_first_replica", "eager_on_first_forward"}
)


class MegaMoeExperts(MulticoreModule):
    """Execute Router-selected local experts with the Torch MegaMoe kernel.

    Serial model layers can call :meth:`share_execution_resources` before
    their first forward to share one lossless-capacity workspace while keeping
    independent parameters and optimizer state.
    """

    def __init__(
        self,
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        expert_capacity_factor: float | None = None,
        replica_slots_per_rank: int = 0,
        compact_physical_weights: bool = False,
        shared_physical_weights: bool = False,
        compact_layout_strategy: str = "eager_on_first_forward",
        replica_grad_receive_slots: int = 8,
        ep_size: int = 1,
        ep_group: Any | None = None,
    ) -> None:
        """Initialize local expert parameters and a lazy execution owner.

        Args:
            local_num_tokens: Static token rows supplied to this rank.
            hidden_size: Input and output hidden dimension.
            intermediate_size: SwiGLU intermediate dimension per expert.
            num_experts: Global routed-expert count.
            top_k: Experts selected for every token.
            expert_capacity_factor: Optional bounded receive-capacity multiplier.
                ``None`` reserves the maximum lossless capacity. A finite value
                of at least 1.0 reserves that multiple of the local routed rows
                and executes larger routes losslessly in fixed-capacity waves.
            replica_slots_per_rank: Bounded temporary expert-weight slots on
                each EP rank. Zero preserves static home-expert execution.
            replica_grad_receive_slots: Maximum incoming replica gradients
                buffered simultaneously per owner rank. Must be a positive
                integer identical on all EP ranks. The default of eight bounds
                receive storage independently of B and total replica fan-in;
                batches may use fewer slots to balance traffic across peers.
            compact_physical_weights: Store replica suffixes in spare storage
                attached to home parameters instead of snapshotting home rows.
                This reduces HBM under the standard contract that parameters
                are not modified between their forward and backward.
            compact_layout_strategy: Choose whether compact storage is prepared
                before the first forward graph, or delayed until the first
                replica placement without a live home-only graph. Eager layout
                is the default and keeps parameter storage stable after first use.
            shared_physical_weights: Experimental B-slot replica weight and
                gradient pools shared by execution-resource group members.
                Kernels read home parameters directly and prefetch only replicas
                before forward and backward; never rebinds parameters. Requires
                replica slots and excludes compact storage. Extra pool capacity
                is independent of layer count and requires the split-weight ABI.
            ep_size: Expert-parallel degree. The current SHMEM path requires it
                to cover the complete Torch distributed world.
            ep_group: Torch expert-parallel process group with the same rank
                ordering as the complete distributed world.
        """
        self._validate_topology(
            local_num_tokens=local_num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor,
            replica_slots_per_rank=replica_slots_per_rank,
            replica_grad_receive_slots=replica_grad_receive_slots,
            compact_physical_weights=compact_physical_weights,
            shared_physical_weights=shared_physical_weights,
            compact_layout_strategy=compact_layout_strategy,
            ep_size=ep_size,
        )
        if expert_capacity_factor is not None:
            expert_capacity_factor = float(expert_capacity_factor)
        specification = {
            "local_num_tokens": local_num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "expert_capacity_factor": expert_capacity_factor,
            "replica_slots_per_rank": replica_slots_per_rank,
            "replica_grad_receive_slots": replica_grad_receive_slots,
            "compact_physical_weights": compact_physical_weights,
            "shared_physical_weights": shared_physical_weights,
            "compact_layout_strategy": compact_layout_strategy,
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
            replica_slots_per_rank,
            replica_grad_receive_slots,
            compact_physical_weights,
            shared_physical_weights,
            compact_layout_strategy,
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
        self.replica_slots_per_rank = replica_slots_per_rank
        self.replica_grad_receive_slots = replica_grad_receive_slots
        self.compact_physical_weights = compact_physical_weights
        self.shared_physical_weights = shared_physical_weights
        self.compact_layout_strategy = compact_layout_strategy
        self.ep_size = ep_size
        self.local_experts = num_experts // ep_size
        self._ep_group = ep_group
        self.gate_up_weight, self.down_weight = _create_mega_moe_parameters(
            self.local_experts,
            hidden_size,
            intermediate_size,
        )

    @staticmethod
    def _validate_replica_options(
        replica_slots_per_rank: int,
        replica_grad_receive_slots: int,
        compact_physical_weights: bool,
        shared_physical_weights: bool,
        compact_layout_strategy: str,
    ) -> None:
        """Validate public physical-weight storage controls."""
        if (
            not isinstance(replica_grad_receive_slots, int)
            or isinstance(replica_grad_receive_slots, bool)
            or replica_grad_receive_slots <= 0
        ):
            raise ValueError("replica_grad_receive_slots must be a positive integer.")
        if (
            not isinstance(replica_slots_per_rank, int)
            or isinstance(replica_slots_per_rank, bool)
            or replica_slots_per_rank < 0
        ):
            raise ValueError(
                "replica_slots_per_rank must be a nonnegative integer, "
                f"got {replica_slots_per_rank!r}."
            )
        if not isinstance(compact_physical_weights, bool):
            raise ValueError(
                "compact_physical_weights must be a bool, got "
                f"{compact_physical_weights!r}."
            )
        if not isinstance(shared_physical_weights, bool):
            raise TypeError("shared_physical_weights must be a bool.")
        if shared_physical_weights and (compact_physical_weights or not replica_slots_per_rank):
            raise ValueError(
                "shared_physical_weights requires replica slots and excludes compact_physical_weights."
            )
        if (
            not isinstance(compact_layout_strategy, str)
            or compact_layout_strategy not in _COMPACT_LAYOUT_STRATEGIES
        ):
            expected = ", ".join(sorted(_COMPACT_LAYOUT_STRATEGIES))
            raise ValueError(
                "compact_layout_strategy must be one of "
                f"{expected}, got {compact_layout_strategy!r}."
            )

    @staticmethod
    def _validate_topology(
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        expert_capacity_factor: float | None,
        replica_slots_per_rank: int,
        replica_grad_receive_slots: int,
        compact_physical_weights: bool,
        shared_physical_weights: bool,
        compact_layout_strategy: str,
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
        MegaMoeExperts._validate_replica_options(
            replica_slots_per_rank,
            replica_grad_receive_slots,
            compact_physical_weights,
            shared_physical_weights,
            compact_layout_strategy,
        )
        if local_num_tokens % _COMMUNICATION_SPLIT:
            raise ValueError(
                "local_num_tokens must be divisible by the fixed communication "
                f"split {_COMMUNICATION_SPLIT}, got {local_num_tokens}."
            )
        if expert_capacity_factor is None:
            return
        valid_factor_type = isinstance(
            expert_capacity_factor,
            (int, float),
        ) and not isinstance(expert_capacity_factor, bool)
        try:
            valid_factor_value = valid_factor_type and math.isfinite(
                expert_capacity_factor
            )
        except OverflowError:
            valid_factor_value = False
        if not valid_factor_value or expert_capacity_factor < 1.0:
            raise ValueError(
                "expert_capacity_factor must be None or a finite number at least 1.0, "
                f"got {expert_capacity_factor!r}."
            )

    def _validate_tensors(self, hidden_states: torch.Tensor) -> None:
        """Validate activation and parameter metadata before acquiring resources."""
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_npu:
            raise TypeError("MegaMoeExperts requires BF16 NPU hidden states.")
        weight1 = self.gate_up_weight
        weight2 = self.down_weight
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
    ) -> torch.Tensor:
        """Validate input shapes and return flattened local token states."""
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states must have at least two dimensions and configured "
                f"hidden size {self.hidden_size}, got {tuple(hidden_states.shape)}."
            )
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
        self._validate_tensors(hidden_flat)
        return hidden_flat

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route token states through the configured experts.

        Args:
            hidden_states: Tensor ending in ``hidden_size``.
            topk_ids: Global expert IDs shaped ``[local_num_tokens, top_k]``.
            topk_weights: Router weights with the same shape as ``topk_ids``.
            tokens_per_expert: Optional trusted exact global-expert histogram.
                Supplying Router-produced counts skips histogram recomputation.

        Returns:
            Tensor matching the shape and dtype of ``hidden_states``.

        Note:
            ``tokens_per_expert`` must describe the current ``topk_ids`` exactly.
            The steady-state path validates metadata but deliberately does not
            rebuild and compare the histogram.
        """
        hidden_flat = self._validate_forward_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
        )
        resources = self._get_execution_resources(hidden_flat)
        if resources.replica_manager is not None:
            resources.replica_manager.prepare_for_forward(self.gate_up_weight, self.down_weight)
        route_weights = topk_weights.float().contiguous()
        route_ids = topk_ids.to(torch.int32).contiguous()
        route_options: dict[str, Any] = {"workspace": resources.workspace}
        if resources.replica_manager is not None:
            # Size route-event storage for the widest physical plan before
            # a replica-free step is allowed to select the narrower plan.
            resources.workspace.ensure(
                resources.spec,
                hidden_flat.dtype,
                hidden_flat.device,
            )
            route_options["placement_callback"] = partial(
                (
                    resources.replica_manager.start_shared_prefetch
                    if resources.shared_physical_weights
                    else resources.replica_manager.start_prefetch
                ),
                self.gate_up_weight,
                self.down_weight,
            )
            route_options["home_runtime"] = resources.home_runtime
        route = prepare_topk_route(
            hidden_flat,
            route_ids,
            route_weights,
            resources.plan,
            tokens_per_expert,
            **route_options,
        )
        shared_lease = None
        execution_weight1 = self.gate_up_weight
        execution_weight2 = self.down_weight
        if route.placement is not None:
            if resources.replica_manager is None:
                raise RuntimeError("MegaMoe route produced an unexpected replica placement.")
            pending_weights = route.placement_payload
            if pending_weights is None:
                raise RuntimeError("MegaMoe replica placement did not prefetch weights.")
            if isinstance(pending_weights, SharedExpertWeightLease):
                shared_lease = pending_weights
        elif resources.replica_manager is not None:
            execution_weight1, execution_weight2 = (
                resources.replica_manager.materialize_home_weights(self.gate_up_weight, self.down_weight)
            )
        try:
            output = execute_mega_moe(
                hidden_flat,
                route.physical_topk_ids,
                route_weights,
                execution_weight1,
                execution_weight2,
                route,
                route.plan,
                resources.workspace,
            )
            return output.reshape(hidden_states.shape)
        finally:
            if shared_lease is not None:
                shared_lease.release()

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
            pool_cache_size=len(self._resource_group.members),
            compact_physical_weights=self.compact_physical_weights,
            shared_physical_weights=self.shared_physical_weights,
            compact_layout_strategy=self.compact_layout_strategy,
            replica_grad_receive_slots=self.replica_grad_receive_slots,
        )
