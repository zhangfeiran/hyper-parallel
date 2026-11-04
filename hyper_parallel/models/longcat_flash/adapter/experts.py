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
"""LongCat real-expert execution with a differentiable local identity branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Native tensor APIs are the project backend contract.
# pylint: disable=forbidden-backend-import
import torch
from torch import nn
from transformers.activations import ACT2FN

from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.multicore.modules.mega_moe.assignments import masked_expert_forward
from hyper_parallel.models.longcat_flash.adapter.layout import ExpertOwnershipPacking
from hyper_parallel.models.longcat_flash.adapter.parameter_grad import real_gate_prefix
from hyper_parallel.models.longcat_flash.adapter.routing import OrderedInputGather
from hyper_parallel.models.replacement import module_replacement


def identity_contribution(hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                          real_experts: int, initial: torch.Tensor | None = None,
                          inputs: tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
    """Accumulate identity slots in expert order with HF's per-contribution cast.

    Args:
        hidden: Token states in original source order.
        ids: Selected global expert IDs.
        weights: Current differentiable routing or expert weights.
        real_experts: Number of parameterized experts.
        initial: Already accumulated real-expert output.
        inputs: Optional storage-sharing identity views with an ordered source backward.
    """
    order = ids.argsort(dim=-1)
    ordered_ids, ordered_weights = ids.gather(1, order), weights.gather(1, order)
    output = torch.zeros_like(hidden) if initial is None else initial
    for slot in range(ids.shape[1]):
        probability = ordered_weights[:, slot] * (ordered_ids[:, slot] >= real_experts)
        source = hidden if inputs is None else inputs[slot]
        output = output + (source * probability[:, None]).to(hidden.dtype)
    return output


def _initial_token_capacity(value: int | None) -> int | None:
    """Validate the optional compact-real reservation owned by the model adapter.

    Args:
        value: Requested initial source rows, or None for the generic default.

    Returns:
        The validated optional row count.
    """
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError("LongCat initial_token_capacity must be a positive integer or None")
    return value


@module_replacement
class LongcatFlashMegaMoeExperts(nn.Module):
    """Retain HF parameter names and execute only real expert assignments."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None, backend: str = "megamoe",
                 dispatch_mode: str = "push", replica_slots_per_rank: int = 0,
                 expert_weight_layout: str = "EOI", swiglu_precision: str = "fused",
                 optimize_tasks: bool = True, routing_backend: str = "torch",
                 direct_weight_gradients: bool = True, initial_token_capacity: int | None = None) -> None:
        """Bind a complete HF expert module before EP and FSDP sharding.

        Args:
            module: Executable HF LongcatFlashExperts with packed SwiGLU weights.
            module_fqn: Framework replacement scope.
            context: Resolved parallel sizes and sequence-parallel setting.
            backend: MegaMoE, native NPU big operators, or explicit eager reference.
            dispatch_mode: MegaMoE push or pull transport.
            replica_slots_per_rank: Optional real-expert guest budget.
            expert_weight_layout: EOI avoids copies; EIO enables current hot replicas.
            direct_weight_gradients: Retain full Gate/Up gradients without clearing the active prefix.
            initial_token_capacity: Optional initial compact-real-row reservation; automatic growth remains enabled.
            routing_backend: Ordered Torch routing, or explicit optional Ascend Triton fusion.
            optimize_tasks: Use exact route bounds and one-expert MegaMoE task addressing.
            swiglu_precision: Native fused arithmetic by default, or explicit HF BF16 intermediate casts.
        """
        super().__init__()
        del module_fqn
        context = context or {}
        self._validate_options(module, context, backend, replica_slots_per_rank, expert_weight_layout)
        self.num_experts = self.num_routed_experts = module.num_routed_experts
        self.zero_expert_num = module.zero_expert_num
        self.total_experts = self.num_experts + self.zero_expert_num
        self.hidden_size, self.intermediate_size = module.hidden_size, module.intermediate_size
        self.act_fn = module.act_fn
        self.identity_expert = module.identity_expert
        if swiglu_precision not in ("hf", "fused"):
            raise ValueError("LongCat swiglu_precision must be hf or fused")
        if not isinstance(optimize_tasks, bool):
            raise ValueError("LongCat optimize_tasks must be boolean")
        if routing_backend not in ("torch", "triton"):
            raise ValueError("LongCat routing_backend must be torch or triton")
        if not isinstance(direct_weight_gradients, bool):
            raise ValueError("LongCat direct_weight_gradients must be boolean")
        self.direct_weight_gradients = all((direct_weight_gradients, optimize_tasks,
                                            expert_weight_layout == "EOI", replica_slots_per_rank == 0))
        self.initial_token_capacity = _initial_token_capacity(initial_token_capacity)
        self.routing_backend = routing_backend
        self.optimize_tasks = optimize_tasks
        self.swiglu_precision = swiglu_precision
        self.backend, self.dispatch_mode = backend, dispatch_mode
        self.replica_slots_per_rank = replica_slots_per_rank
        self.expert_weight_layout = expert_weight_layout
        self.initializer_range = 0.02
        self._hp_reset_after_materialization = True
        self.ep_size = int(context.get("ep_size", 1))
        self._packing = None
        self._packing_checked_version = -1
        if self.ep_size > 1 and self.zero_expert_num and self.num_experts:
            self.register_buffer("_checkpoint_ep_size", torch.tensor([self.ep_size], dtype=torch.int64))
        if self.num_experts:
            self._packing = ExpertOwnershipPacking(self.num_experts, self.zero_expert_num, self.ep_size)
            gate = self._packing.apply(module.gate_up_proj.detach())
            self.gate_up_proj = (module.gate_up_proj if self.ep_size == 1 or self.zero_expert_num == 0 else
                                 nn.Parameter(gate, requires_grad=module.gate_up_proj.requires_grad))
            self.down_proj = module.down_proj
        else:
            self.register_parameter("gate_up_proj", None)
            self.register_parameter("down_proj", None)
        self._executor = None
        self._ep_group = None
        self._configured = False
        self.train(module.training)

    @staticmethod
    def _validate_options(module: nn.Module, context: Mapping[str, Any], backend: str,
                          replica_slots_per_rank: int, expert_weight_layout: str) -> None:
        """Reject incompatible activation, topology and backend options before ownership changes.

        Args:
            module: Source module supplied by the replacement executor.
            context: Resolved model or validation context.
            backend: Declared backend value.
            replica_slots_per_rank: Declared replica slots per rank value.
            expert_weight_layout: Declared expert weight layout value.
        """
        if backend not in ("megamoe", "native", "eager"):
            raise ValueError("LongCat expert backend must be megamoe, native or eager")
        if context.get("pp") or context.get("cp"):
            raise ValueError("LongCat Flash integration currently requires CP=PP=1")
        if context.get("tp"):
            raise ValueError("LongCat expert integration currently requires TP=1")
        if backend != "megamoe" and replica_slots_per_rank:
            raise ValueError("native LongCat reference does not support expert replicas")
        if expert_weight_layout not in ("EOI", "EIO"):
            raise ValueError("expert_weight_layout must be EOI or EIO")
        if replica_slots_per_rank and expert_weight_layout != "EIO":
            raise ValueError("current MegaMoE replicas require expert_weight_layout=EIO")
        if not isinstance(module.act_fn, (nn.SiLU, type(ACT2FN["silu"]))):
            raise ValueError("LongCat MegaMoE requires the source SiLU activation")

    def reset_parameters(self) -> None:
        """Initialize the retained real and unused HF parameter rows."""
        if self.gate_up_proj is not None:
            nn.init.normal_(self.gate_up_proj, std=self.initializer_range)
            nn.init.normal_(self.down_proj, std=self.initializer_range)
        if "_checkpoint_ep_size" in self._buffers:
            self._checkpoint_ep_size.fill_(self.ep_size)

    def make_transforms(self) -> list[WeightConverter]:
        """Declare reversible ownership packing, without dropping unused weights."""
        if self._packing is None or self.ep_size == 1 or self.zero_expert_num == 0:
            return []
        return [WeightConverter(source_patterns="gate_up_proj",
                                target_patterns=["gate_up_proj", "_checkpoint_ep_size"],
                                operations=[self._packing])]

    def configure(self, ep_group: Any = None, ep_size: int = 1, top_k: int | None = None) -> None:
        """Bind the EP group after framework-owned parameter sharding.

        Args:
            ep_group: Explicit expert process group.
            ep_size: Number of ranks in the expert group.
            top_k: Router slot count; execution compacts valid slots independently.
        """
        del top_k
        if self._configured:
            raise RuntimeError("LongCat experts are already configured")
        if ep_size != self.ep_size:
            raise ValueError("EP size differs from checkpoint ownership packing")
        self._ep_group = ep_group
        if self.num_experts and self.backend == "megamoe":
            from hyper_parallel.core.multicore import MegaMoeExperts  # pylint: disable=C0415

            capacity = ({} if self.initial_token_capacity is None
                        else {"initial_token_capacity": self.initial_token_capacity})
            self._executor = MegaMoeExperts(
                hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
                num_experts=self.num_experts, top_k=1, ep_group=ep_group, ep_size=ep_size,
                create_parameters=False, expert_weight_layout=self.expert_weight_layout,
                swiglu_hf_bf16=self.swiglu_precision == "hf",
                route_task_bounds=self.optimize_tasks, single_expert_tasks=self.optimize_tasks,
                gate_up_expert_padding=(self.zero_expert_num
                                        if self.direct_weight_gradients and not self.replica_slots_per_rank else 0),
                dispatch_mode=self.dispatch_mode, replica_slots_per_rank=self.replica_slots_per_rank, **capacity,
            )
        self._configured = True

    def _native_forward(self, hidden: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor,
                        *, expert_weights: tuple) -> torch.Tensor:
        """Execute explicit K=1 assignments through portable differentiable EP.

        Args:
            hidden: Token states in original source order.
            ids: Selected global expert IDs.
            probabilities: K=1 placeholder weights consumed by the native executor.
        """
        del probabilities
        if self.backend == "native" and hidden.device.type == "npu":
            from hyper_parallel.models.longcat_flash.adapter.npu import npu_expert_forward  # pylint: disable=C0415

            return npu_expert_forward(hidden, ids, expert_weights, self._ep_group,
                                      self.ep_size, self.num_experts, self.expert_weight_layout,
                                      activation=self.swiglu_precision)
        from hyper_parallel.models.longcat_flash.adapter.native import native_expert_forward  # pylint: disable=C0415

        return native_expert_forward(hidden, ids, expert_weights, self._ep_group,
                                     self.ep_size, self.num_experts, self.expert_weight_layout)

    def _real_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Select backend storage while retaining the same full HF parameter derivative."""
        gate, down = (value.to_local() if isinstance(value, DTensor) else value
                      for value in (self.gate_up_proj, self.down_proj))
        local = self.num_experts // self.ep_size
        if self.direct_weight_gradients and not self.replica_slots_per_rank:
            gate = gate if self.backend == "megamoe" else real_gate_prefix(gate, local)
        else:
            gate = gate[:local]
        weights = (gate, down)
        if self.expert_weight_layout == "EIO":
            weights = tuple(value.transpose(1, 2).contiguous() for value in weights)
        return weights

    def _validate_checkpoint_packing(self) -> None:
        """Check a newly loaded packing marker once before executing expert matrices."""
        marker = self._buffers.get("_checkpoint_ep_size")
        if marker is not None and marker._version != self._packing_checked_version:
            if int(marker.item()) != self.ep_size:
                raise ValueError("checkpoint EP packing differs; export to HF before changing EP size")
            self._packing_checked_version = marker._version

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """Execute real experts and add weighted identities without changing routing.

        Args:
            hidden_states: Token states in original source order.
            top_k_index: Global IDs for the original Top-K slots.
            top_k_weights: Differentiable original slot weights.
        """
        original_shape = hidden_states.shape
        self._validate_checkpoint_packing()
        hidden = hidden_states.reshape(-1, self.hidden_size)
        if top_k_index.shape != top_k_weights.shape or top_k_index.shape[0] != hidden.shape[0]:
            raise ValueError("LongCat routes must match the input token rows")
        if self.routing_backend == "triton":
            if hidden.device.type != "npu":
                raise ValueError("LongCat Triton routing requires NPU inputs")
            # Triton is an optional dependency, required only for this explicit route backend.
            from hyper_parallel.models.longcat_flash.adapter.fused_routing import (  # pylint: disable=C0415
                fused_expert_forward,
            )

            return fused_expert_forward(self, hidden, top_k_index, top_k_weights).reshape(original_shape)
        route_error = ((top_k_index < 0) | (top_k_index >= self.total_experts)).any()
        order = top_k_index.argsort(dim=-1)
        top_k_index = top_k_index.gather(1, order)
        top_k_weights = top_k_weights.gather(1, order)
        gather = OrderedInputGather(top_k_index.shape[1])
        if not self.num_experts:
            empty = top_k_index.new_empty((0,))
            gather(hidden, empty, empty)
            if bool(route_error):
                raise ValueError("LongCat route contains an invalid expert ID")
            return identity_contribution(hidden, top_k_index, top_k_weights, 0,
                                         inputs=gather.identity_inputs).reshape(original_shape)
        if not self._configured:
            if self.ep_size != 1:
                raise RuntimeError("Configure LongCat EP experts before execution")
            self.configure()
        weights = self._real_weights()
        executor = self._executor if self.backend == "megamoe" else _NativeExecutor(self)
        real = masked_expert_forward(executor, hidden, top_k_index, top_k_weights,
                                     top_k_index < self.num_experts, expert_weights=weights,
                                     ep_group=self._ep_group, route_error=route_error, input_gather=gather)
        return identity_contribution(hidden, top_k_index, top_k_weights,
                                     self.num_experts, initial=real,
                                     inputs=gather.identity_inputs).reshape(original_shape)

    def close_runtime(self) -> None:
        """Release native resources after the final backward."""
        if self._executor is not None:
            self._executor.close()
            self._executor = None
        self._configured = False


class _NativeExecutor:
    """Expose the same packed-assignment contract for the native reference."""

    def __init__(self, module: LongcatFlashMegaMoeExperts) -> None:
        """Bind the selected real-expert implementation.

        Args:
            module: The parameter-owning expert holder.
        """
        self.module = module
        self.num_experts = module.num_experts

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Execute the same K=1 contract as the accelerator backend."""
        return self.module._native_forward(*args, **kwargs)  # pylint: disable=protected-access
