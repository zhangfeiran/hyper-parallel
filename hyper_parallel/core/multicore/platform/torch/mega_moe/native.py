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
"""Differentiable native fallback for bounded MegaMoe routes."""

from __future__ import annotations

__all__ = ["NativeMoeExperts", "create_native_moe", "execute_native_fallback"]

from typing import Any, Protocol, Self

from ._framework import (
    get_torch,
    get_torch_distributed,
    get_torch_distributed_nn_functional,
)

_NATIVE_GROUPED_MATMUL_FUNCTION = None
_NATIVE_MOE_EXPERTS_CLASS = None


class _NativeMoeTopology(Protocol):
    """Shape and EP fields shared by managed and standalone native execution."""

    hidden_size: int
    local_experts: int
    ep_size: int
    ep_group: Any | None


def _get_native_grouped_matmul_function() -> Any:
    """Create the native grouped-matmul autograd adapter lazily."""
    # pylint: disable=global-statement
    torch = get_torch()

    global _NATIVE_GROUPED_MATMUL_FUNCTION
    if _NATIVE_GROUPED_MATMUL_FUNCTION is not None:
        return _NATIVE_GROUPED_MATMUL_FUNCTION

    # Torch declares variadic autograd hooks; concrete functions use operator-specific signatures.
    class _NativeGroupedMatmulFunction(  # pylint: disable=abstract-method,arguments-differ
        torch.autograd.Function
    ):
        """Supply dX and dW for the native grouped-matmul operator."""

        @staticmethod
        def forward(ctx: Any, x: Any, weight: Any, group_list: Any) -> Any:
            """Run grouped matmul and retain its backward inputs."""
            # pylint: disable=C0415
            import torch_npu

            ctx.save_for_backward(x, weight, group_list)
            return torch_npu.npu_grouped_matmul(
                [x],
                [weight],
                bias=None,
                group_list=group_list,
                split_item=3,
                group_type=0,
                group_list_type=0,
            )[0]

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any | None, Any | None, None]:
            """Compute only the requested grouped-matmul gradients."""
            # pylint: disable=C0415
            import torch_npu

            x, weight, group_list = ctx.saved_tensors
            grad_x = None
            grad_weight = None
            if ctx.needs_input_grad[0]:
                grad_x = torch_npu.npu_grouped_matmul(
                    [grad_output],
                    [weight.transpose(1, 2).contiguous()],
                    bias=None,
                    group_list=group_list,
                    split_item=3,
                    group_type=0,
                    group_list_type=0,
                )[0]
            if ctx.needs_input_grad[1]:
                grad_weight = torch_npu.npu_grouped_matmul(
                    [x.transpose(0, 1)],
                    [grad_output],
                    bias=None,
                    group_list=group_list,
                    split_item=3,
                    group_type=2,
                    group_list_type=0,
                )[0]
            return grad_x, grad_weight, None

    _NATIVE_GROUPED_MATMUL_FUNCTION = _NativeGroupedMatmulFunction
    return _NATIVE_GROUPED_MATMUL_FUNCTION


def _grouped_matmul(x: Any, weight: Any, group_list: Any) -> Any:
    """Run native grouped matmul through its differentiable adapter."""
    return _get_native_grouped_matmul_function().apply(x, weight, group_list)


def _all_to_all_rows(
    input_tensor: Any,
    input_splits: list[int],
    output_splits: list[int],
    group: Any | None,
) -> Any:
    """Exchange variable-sized token rows through differentiable HCCL."""
    torch = get_torch()
    dist_func = get_torch_distributed_nn_functional()

    if len(input_splits) == 1:
        return input_tensor
    output = torch.empty(
        (sum(output_splits), *input_tensor.shape[1:]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    return dist_func.all_to_all_single(
        output,
        input_tensor.contiguous(),
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )


def _run_distributed_experts(
    permuted_hidden: Any,
    weight1: Any,
    weight2: Any,
    tokens_per_expert: Any,
    received_counts: Any,
    config: _NativeMoeTopology,
) -> Any:
    """Dispatch and run experts with the RC2 native large-operator path."""
    # pylint: disable=C0415
    import torch_npu

    torch = get_torch()
    input_splits = (
        tokens_per_expert.reshape(config.ep_size, config.local_experts)
        .sum(dim=1)
        .tolist()
    )
    output_splits = received_counts.sum(dim=1).tolist()
    dispatched = _all_to_all_rows(
        permuted_hidden,
        input_splits,
        output_splits,
        config.ep_group,
    )

    source_major_experts = torch.arange(
        config.local_experts,
        dtype=torch.int64,
        device=dispatched.device,
    ).repeat(config.ep_size)
    source_major_experts = torch.repeat_interleave(
        source_major_experts,
        received_counts.reshape(-1).to(torch.int64),
    )
    use_fused_local_permute = dispatched.shape[0] > 0
    if use_fused_local_permute:
        expert_input, unpermute_mapping = torch_npu.npu_moe_token_permute(
            dispatched,
            source_major_experts.unsqueeze(-1),
        )
        permute_indices = None
    else:
        permute_indices = torch.argsort(source_major_experts, stable=True)
        expert_input = dispatched.index_select(0, permute_indices)
        unpermute_mapping = None

    if expert_input.shape[0] > 0:
        group_list = received_counts.sum(dim=0).cumsum(dim=0).to(torch.int64)
        gate_up = _grouped_matmul(expert_input, weight1, group_list)
        activated = torch_npu.npu_swiglu(gate_up, dim=-1)
        expert_output = _grouped_matmul(activated, weight2, group_list)
    else:
        parameter_zero = (weight1.sum() + weight2.sum()).to(expert_input.dtype) * 0
        expert_output = expert_input + parameter_zero

    if use_fused_local_permute:
        if unpermute_mapping is None:
            raise RuntimeError("native local unpermute mapping is missing.")
        rank_major_output = torch_npu.npu_moe_token_unpermute(
            expert_output,
            unpermute_mapping,
        )
    else:
        if permute_indices is None:
            raise RuntimeError("native local permutation indices are missing.")
        rank_major_output = torch.zeros_like(dispatched).scatter_add(
            0,
            permute_indices.unsqueeze(1).expand(-1, config.hidden_size),
            expert_output,
        )
    return _all_to_all_rows(
        rank_major_output,
        output_splits,
        input_splits,
        config.ep_group,
    )


def execute_native_fallback(
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
    weight1: Any,
    weight2: Any,
    tokens_per_expert: Any,
    received_counts: Any,
    config: _NativeMoeTopology,
) -> Any:
    """Execute a lossless expert-parallel fallback for an overflowing route.

    Args:
        hidden_states: Source-side token-major hidden states.
        topk_ids: Token-major global expert IDs.
        topk_weights: Route probabilities matching ``topk_ids``.
        weight1: Local gate and up-projection weights.
        weight2: Local down-projection weights.
        tokens_per_expert: Trusted source counts for every global expert.
        received_counts: Source-major counts for this rank's local experts.
        config: Bound model shape and expert-parallel topology.

    Returns:
        Weighted expert outputs in source token order.
    """
    # pylint: disable=C0415
    import torch_npu

    permuted_hidden, unpermute_mapping = torch_npu.npu_moe_token_permute(
        hidden_states,
        topk_ids,
    )
    return execute_native_from_permuted_route(
        permuted_hidden,
        unpermute_mapping,
        topk_weights,
        weight1,
        weight2,
        tokens_per_expert,
        received_counts,
        config,
    )


def execute_native_from_permuted_route(
    permuted_hidden: Any,
    unpermute_mapping: Any,
    topk_weights: Any,
    weight1: Any,
    weight2: Any,
    tokens_per_expert: Any,
    received_counts: Any,
    config: _NativeMoeTopology,
) -> Any:
    """Execute native experts from a route already permuted by the caller."""
    # pylint: disable=C0415
    import torch_npu

    expert_output = _run_distributed_experts(
        permuted_hidden,
        weight1,
        weight2,
        tokens_per_expert,
        received_counts,
        config,
    )
    return torch_npu.npu_moe_token_unpermute(
        expert_output,
        unpermute_mapping,
        probs=topk_weights.float().contiguous(),
    )


class NativeMoeExperts:
    """Torch native expert module matching the managed expert contract.

    The concrete ``torch.nn.Module`` base is attached lazily on first
    construction so importing the Torch platform package does not eagerly
    import the framework.
    """

    def __new__(
        cls,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int = 1,
        ep_group: Any | None = None,
    ) -> Self:
        """Allocate the lazily materialized ``torch.nn.Module`` subclass."""
        del hidden_size, intermediate_size, num_experts, top_k, ep_size, ep_group
        if cls is NativeMoeExperts:
            return object.__new__(_get_native_moe_experts_class())
        return object.__new__(cls)

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int = 1,
        ep_group: Any | None = None,
    ) -> None:
        """Initialize local experts and validate their EP topology."""
        torch = get_torch()
        dist = get_torch_distributed()
        nn = torch.nn

        super().__init__()
        values = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "ep_size": ep_size,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts})."
            )
        if num_experts % ep_size:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
            )
        if dist.is_initialized() and dist.get_world_size(ep_group) != ep_size:
            group_size = dist.get_world_size(ep_group)
            raise ValueError(
                f"ep_group world size ({group_size}) does not match ep_size ({ep_size})."
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.local_experts = num_experts // ep_size
        self.ep_rank = dist.get_rank(ep_group) if dist.is_initialized() else 0
        self.gate_up_weight = nn.Parameter(
            torch.empty(self.local_experts, hidden_size, intermediate_size * 2)
        )
        self.down_weight = nn.Parameter(
            torch.empty(self.local_experts, intermediate_size, hidden_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize local experts with the managed MegaMoe rule."""
        nn = get_torch().nn

        for expert_index in range(self.local_experts):
            nn.init.xavier_uniform_(self.gate_up_weight[expert_index])
            nn.init.xavier_uniform_(self.down_weight[expert_index])

    def _gather_received_counts(self, tokens_per_expert: Any) -> Any:
        """Collect source counts for the local expert range."""
        torch = get_torch()
        dist = get_torch_distributed()

        if self.ep_size == 1:
            return tokens_per_expert.unsqueeze(0).to(torch.int64)
        gathered = torch.empty(
            self.ep_size * self.num_experts,
            dtype=tokens_per_expert.dtype,
            device=tokens_per_expert.device,
        )
        dist.all_gather_into_tensor(
            gathered,
            tokens_per_expert.contiguous(),
            group=self.ep_group,
        )
        expert_start = self.ep_rank * self.local_experts
        expert_end = expert_start + self.local_experts
        return gathered.view(self.ep_size, self.num_experts)[
            :, expert_start:expert_end
        ].to(torch.int64)

    def forward(
        self,
        hidden_states: Any,
        topk_ids: Any,
        topk_weights: Any,
        *,
        tokens_per_expert: Any | None = None,
    ) -> Any:
        """Execute native token permutation, GMMs and unpermutation."""
        torch = get_torch()

        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states must have at least two dimensions and hidden size "
                f"{self.hidden_size}, got {tuple(hidden_states.shape)}."
            )
        hidden_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_shape[-1])
        expected_route_shape = (hidden_flat.shape[0], self.top_k)
        if tuple(topk_ids.shape) != expected_route_shape:
            raise ValueError(
                f"topk_ids must have shape {expected_route_shape}, got {tuple(topk_ids.shape)}."
            )
        if topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "topk_weights must match topk_ids: "
                f"ids={tuple(topk_ids.shape)}, weights={tuple(topk_weights.shape)}."
            )
        if tokens_per_expert is None:
            tokens_per_expert = torch.bincount(
                topk_ids.reshape(-1).to(torch.int64),
                minlength=self.num_experts,
            ).to(torch.int32)
        elif tuple(tokens_per_expert.shape) != (self.num_experts,):
            raise ValueError(
                f"tokens_per_expert must have shape ({self.num_experts},), "
                f"got {tuple(tokens_per_expert.shape)}."
            )

        output = execute_native_fallback(
            hidden_flat,
            topk_ids,
            topk_weights,
            self.gate_up_weight,
            self.down_weight,
            tokens_per_expert,
            self._gather_received_counts(tokens_per_expert),
            self,
        )
        return output.view(hidden_shape)


def _get_native_moe_experts_class() -> Any:
    """Create the concrete Torch native expert class lazily."""
    # pylint: disable=global-statement
    nn = get_torch().nn

    global _NATIVE_MOE_EXPERTS_CLASS
    if _NATIVE_MOE_EXPERTS_CLASS is None:

        class _TorchNativeMoeExperts(NativeMoeExperts, nn.Module):
            """Concrete torch module for the lazy native implementation."""

        _NATIVE_MOE_EXPERTS_CLASS = _TorchNativeMoeExperts
    return _NATIVE_MOE_EXPERTS_CLASS


def create_native_moe(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    ep_size: int = 1,
    ep_group: Any | None = None,
) -> NativeMoeExperts:
    """Create native experts through the compatibility factory."""
    return NativeMoeExperts(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        ep_size=ep_size,
        ep_group=ep_group,
    )
