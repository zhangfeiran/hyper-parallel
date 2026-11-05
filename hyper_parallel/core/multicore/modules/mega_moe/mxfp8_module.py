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
"""Ascend 950 MXFP8 task graphs with bounded projected expert replicas."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any
from functools import cached_property

import torch
import torch.distributed as dist

from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.multicore import _collective
from hyper_parallel.core.multicore.scheduler.runtime import serialize_runtime_config

from .heap_manager import _get_heap_manager, _workspace_layout
from .module import MegaMoeExperts, _MegaMoeExecutionResources
from .mxfp8_execution import MXFP8Execution
from .mxfp8_diagnostics import prepare_mx_diagnostics
from .mxfp8_function import execute_mxfp8_experts
from .mxfp8_graph import build_mxfp8_runtime
from .mxfp8_weights import MXFP8ExpertWeightBinding
from .spec import bind_mega_moe_spec
from .workspace import MegaMoeWorkspace


def _device_bytes(data: bytes, device: Any) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device=device)


@dataclass(frozen=True)
class _DeviceRuntime:
    runtime: torch.Tensor
    gmm: torch.Tensor
    quant: torch.Tensor
    event_capacity: int
    host: Any
    spec: Any

    @cached_property
    def diagnostics(self) -> Any:
        """Materialize diagnostic metadata only for an explicit capture."""
        return prepare_mx_diagnostics(self.host, self.spec, self.runtime.device)


@dataclass(frozen=True)
class _MXFP8Plan:
    spec: Any
    forward: _DeviceRuntime
    backward: _DeviceRuntime


def _device_runtime(host: Any, device: Any, spec: Any) -> _DeviceRuntime:
    return _DeviceRuntime(
        _device_bytes(serialize_runtime_config(host.config), device),
        _device_bytes(host.gmm_descriptors, device),
        _device_bytes(host.quant_descriptors, device),
        host.config.event_capacity,
        host,
        spec,
    )


class _MXFP8Resources(_MegaMoeExecutionResources):
    """Retain the existing heap restart protocol without building BF16 graphs."""

    def __init__(
        self,
        specification: dict[str, Any],
        tensor: torch.Tensor,
        *,
        shared: bool,
        active_specifications: tuple[Mapping[str, Any], ...],
    ) -> None:
        """Bind validated invocation resources without retaining gathered weight views."""
        self.spec = bind_mega_moe_spec(specification, tensor)
        if self.spec.num_cube_cores != 32:
            raise ValueError("MXFP8 MegaMoE requires the Ascend 950 32-Cube kernel.")
        self.soc_name = ""
        self.rank_nodes = (0,) * self.spec.ep_size
        self._token_capacity = self.spec.local_num_tokens
        self._plans = OrderedDict()
        self.home_spec = self.home_plan = self.last_execution_plan = None
        if dist.is_initialized():
            self._validate_distributed_configuration(active_specifications)
        self.heap_manager = _get_heap_manager(self.spec.ep_group, tensor)
        reservations = tuple((item, _workspace_layout(item, tensor.dtype)) for item in active_specifications)
        added = self.heap_manager.reserve(reservations, allow_resize=True)
        attached = False
        self.plan = self.workspace = None
        try:
            attached = self.heap_manager.attach_runtime()

            def construct() -> None:
                """Bind the reserved symmetric workspace after collective construction agreement."""
                self.workspace = MegaMoeWorkspace(shared=shared)
                self.workspace.capacity_floor = self.heap_manager.bind(self.workspace, self.spec, specification)
                self.workspace.heap_manager = self.heap_manager

            _collective.collective_call(
                construct, stage="MXFP8 resource construction", group=self.heap_manager.root_group
            )
        except Exception:
            self.heap_manager.rollback_reservations(added, self.workspace)
            if attached and not self.heap_manager.workspaces:
                self.heap_manager.detach_runtime()
            raise
        self._closed = self._workspace_closed = False

    def lifecycle_signature(self) -> tuple[Any, ...]:
        """Identify MX precision and scheduling in collective resource lifecycle checks."""
        return (
            *super().lifecycle_signature(),
            "mxfp8_v1",
            self.spec.mx_vector_policy,
            self.spec.mx_poll_policy,
            self.spec.compute_stages,
        )

    def plan_for_capacity(self, capacity: int, device: Any) -> _MXFP8Plan:
        """Cache pointer-free MX plans against the current symmetric receive bound.

        Args:
            capacity: Current symmetric receive-row bound.
            device: Device owning execution and diagnostic storage.
        """
        if capacity not in self._plans:
            spec = replace(self.spec, receive_capacity=capacity)
            self._plans[capacity] = _MXFP8Plan(
                spec,
                _device_runtime(build_mxfp8_runtime(spec, forward=True), device, spec),
                _device_runtime(build_mxfp8_runtime(spec, forward=False), device, spec),
            )
        self._plans.move_to_end(capacity)
        while len(self._plans) > 4:
            self._plans.popitem(last=False)
        return self._plans[capacity]


class MXFP8MegaMoeExperts(MegaMoeExperts):
    """Execute MXFP8 experts while keeping FSDP root Parameters phase-current.

    Args:
        local_num_tokens: Fixed source rows per rank, divisible by transport splits.
        hidden_size: Input width, divisible by 64.
        intermediate_size: Routed expert width, divisible by 64.
        num_experts: Logical global expert count, excluding guest slots.
        top_k: Number of routes selected by the existing Router.
        ep_size: Expert parallel group size.
        ep_group: Raw Torch expert parallel group.
        replica_slots_per_rank: Bounded guest budget. B2 is the recommended candidate.
        replica_min_rows: Minimum useful guest rows for the existing CPU planner.
        initial_capacity_factor: Initial push receive allocation; never a worst-case reservation.
        capacity_growth_factor: Geometric growth factor, bounded by actual route demand.
        dispatch_split: Source transport rows per task.
        combine_split: Return transport rows per task.
        vector_policy: Adopted two-expert lookahead or the legacy control schedule.
        nograd_no_clone: Avoid the intermediate clone for checkpoint forwards.
    """

    def __init__(
        self,
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int = 1,
        ep_group: Any = None,
        replica_slots_per_rank: int = 0,
        replica_min_rows: int = 1024,
        initial_capacity_factor: float | None = None,
        capacity_growth_factor: float | None = None,
        dispatch_split: int = 512,
        combine_split: int = 512,
        vector_policy: str = "window2",
        nograd_no_clone: bool = True,
    ) -> None:
        """Bind validated invocation resources without retaining gathered weight views."""
        if hidden_size % 64 or intermediate_size % 64:
            raise ValueError("MXFP8 hidden and intermediate dimensions must be divisible by 64.")
        if vector_policy not in ("window2", "legacy_odd"):
            raise ValueError("MXFP8 vector_policy must be window2 or legacy_odd.")
        if not isinstance(nograd_no_clone, bool):
            raise ValueError("nograd_no_clone must be a boolean.")
        config = ExpertReplicaConfig(num_experts, ep_size, replica_slots_per_rank)
        self.replica_planner = "cpu"
        self.replica_min_rows = replica_min_rows
        self.replica_target_load = None
        self.replica_minimum_load_reduction = None
        self.replica_transport = "p2p"
        self._cost_model_enabled = False
        specification = {
            "local_num_tokens": local_num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": config.physical_experts,
            "logical_num_experts": num_experts,
            "top_k": top_k,
            "ep_size": ep_size,
            "ep_group": ep_group,
            "expert_weight_layout": "EIO",
            "swiglu_limit": None,
            "dispatch_mode": "push",
            "initial_capacity_factor": initial_capacity_factor,
            "capacity_growth_factor": capacity_growth_factor,
            "dispatch_split": dispatch_split,
            "combine_split": combine_split,
            "replica_slots_per_rank": replica_slots_per_rank,
            "replica_transport": "p2p",
            "auto_tokens": False,
            "compute_stages": 6,
            "mx_vector_policy": vector_policy,
            "mx_poll_policy": "0",
            "_replica_cost_model_enabled": False,
        }
        specification = self._resolve_specification(specification)
        self._initialize_execution_specification(specification, config, False)
        self.nograd_no_clone = nograd_no_clone

    def _create_execution_resources(
        self, tensor: torch.Tensor, *, shared: bool, active_specifications: tuple[Mapping[str, Any], ...]
    ) -> _MXFP8Resources:
        return _MXFP8Resources(
            self._resource_group.specification, tensor, shared=shared, active_specifications=active_specifications
        )

    def _validate_tensors(self, hidden_states: torch.Tensor, expert_weights: tuple) -> None:
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_npu:
            raise TypeError("MXFP8 MegaMoE requires BF16 NPU hidden states.")
        for root, shape in zip(expert_weights, self._weight_shapes):
            MXFP8ExpertWeightBinding(root, shape)

    @property
    def _weight_shapes(self) -> tuple[tuple[int, int, int], ...]:
        return (
            (self.local_experts, self.hidden_size, 2 * self.intermediate_size),
            (self.local_experts, self.intermediate_size, self.hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        expert_weights: tuple[torch.Tensor, torch.Tensor],
        tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use final-route counts and resolve both physical weight axes per phase.

        Args:
            hidden_states: Original token activations with the configured hidden width.
            topk_ids: Final logical expert IDs produced by the Router.
            topk_weights: Differentiable probabilities for the same selected routes.
        """
        flat = self._validate_forward_inputs(hidden_states, topk_ids, topk_weights, tokens_per_expert, expert_weights)
        if flat.shape[0] != self._resource_group.specification["local_num_tokens"]:
            raise ValueError("MXFP8 source rows differ from the configured fixed token count.")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MXFP8 push transport does not support graph capture.")
        resources = self._get_execution_resources(flat)
        replica_route = None
        if self.replica_slots_per_rank:
            mask = (
                int(torch.is_grad_enabled())
                | (int(flat.requires_grad) << 1)
                | (int(expert_weights[0].requires_grad) << 2)
                | (int(expert_weights[1].requires_grad) << 3)
                | (int(topk_weights.requires_grad) << 4)
            )
            with torch.no_grad():
                replica_route = self._prepare_replica_route(topk_ids, resources, flat, mask, tokens_per_expert)
                topk_ids = replica_route.physical_ids
                tokens_per_expert = replica_route.counts_by_source[resources.spec.rank_id]
        execution = MXFP8Execution(
            resources, nograd_no_clone=self.nograd_no_clone, replica_route=replica_route, counts=tokens_per_expert
        )
        return execute_mxfp8_experts(
            hidden_states, *expert_weights, topk_ids, topk_weights, self._weight_shapes, execution
        )
