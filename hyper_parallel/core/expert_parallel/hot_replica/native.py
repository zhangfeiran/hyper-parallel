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
"""Native grouped-MoE adapter using the shared expert placement contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.utils.communication import differentiable_all_to_all_single

from .capacity import ExpertReplicaConfig
from .planner import build_expert_replica_plan
from .routing import ReplicaRoute, stable_expert_order
from .transport import prefetch_weights, return_gradients


def _npu_ops() -> Any:
    """Load the optional NPU backend only when native NPU experts execute."""
    import torch_npu  # pylint: disable=import-outside-toplevel,unused-import
    # These operators are registered dynamically by the optional extension.
    return torch.ops.npu


def _gmm(inputs: torch.Tensor, weights: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Use native grouped matmul for BF16 forward and input gradients."""
    return _npu_ops().npu_grouped_matmul(
        x=[inputs], weight=[weights], bias=[], group_list=groups,
        split_item=3, group_type=0, group_list_type=0,
    )[0]


def _weight_gradient(inputs: torch.Tensor, gradients: torch.Tensor, route: ReplicaRoute) -> torch.Tensor:
    """Compute FP32 expert partials on native kernels that lack mixed-output GMM.

    CANN's non-quantized grouped-matmul contract ties output dtype to the input
    dtype. Ordinary FP32 matmul avoids an irreversible BF16 rounding of each
    replica partial. Boundaries come from the host plan, with no device readback.
    """
    width = route.plan.config.slots_per_rank
    begin = route.rank * width
    counts = [sum(row[begin + slot] for row in route.plan.dispatch_counts) for slot in range(width)]
    result = torch.zeros((width, inputs.shape[1], gradients.shape[1]), dtype=torch.float32, device=inputs.device)
    inputs, gradients = inputs.float(), gradients.float()
    offset = 0
    for slot, count in enumerate(counts):
        if count:
            torch.mm(inputs[offset:offset + count].T, gradients[offset:offset + count], out=result[slot])
        offset += count
    return result


class _NativeReplicaExperts(torch.autograd.Function):
    """Return gradients to original parameter owners inside the expert backward."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
                counts: torch.Tensor, route: ReplicaRoute) -> torch.Tensor:
        """Prefetch this invocation's weights and execute native grouped SwiGLU."""
        physical1, physical2 = prefetch_weights((weight1, weight2), route)
        groups = counts.to(torch.int64).cumsum(0)
        if inputs.shape[0]:
            up = _gmm(inputs, physical1, groups)
            activation = _npu_ops().npu_swiglu(up)
            output = _gmm(activation, physical2, groups)
        else:
            up = inputs.new_empty((0, weight1.shape[-1]))
            activation = inputs.new_empty((0, weight2.shape[1]))
            output = inputs.clone()
        ctx.route = route
        ctx.save_for_backward(inputs, weight1, weight2, groups, up, activation)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Re-prefetch without retaining guest snapshots and sum FP32 partials."""
        inputs, weight1, weight2, groups, up, activation = ctx.saved_tensors
        physical1, physical2 = prefetch_weights((weight1, weight2), ctx.route)
        if inputs.shape[0]:
            grad_output = grad_output.contiguous()
            grad_activation = _gmm(grad_output, physical2.transpose(-1, -2), groups)
            grad_up = _npu_ops().npu_swiglu_backward(grad_activation, up)
            grad_input = _gmm(grad_up, physical1.transpose(-1, -2), groups)
            grad1 = _weight_gradient(inputs, grad_up, ctx.route)
            grad2 = _weight_gradient(activation, grad_output, ctx.route)
        else:
            grad_input = grad_output.clone()
            grad1 = torch.zeros_like(physical1, dtype=torch.float32)
            grad2 = torch.zeros_like(physical2, dtype=torch.float32)
        grad1, grad2 = return_gradients((grad1, grad2), ctx.route)
        return grad_input, grad1, grad2, None, None


def native_replica_experts(inputs: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
                           counts: torch.Tensor, route: ReplicaRoute) -> torch.Tensor:
    """Execute packed weights [H,D,2I], [H,I,D] with native NPU kernels."""
    if inputs.device.type != "npu" or inputs.dtype != torch.bfloat16:
        raise ValueError("native hot replicas currently require BF16 NPU inputs")
    return _NativeReplicaExperts.apply(inputs, weight1, weight2, counts, route)


@dataclass(frozen=True)
class NativeReplicaDispatch:
    """Invocation-owned all-to-all inverses and physical expert route."""

    route: ReplicaRoute
    send_splits: list[int]
    recv_splits: list[int]
    send_order: torch.Tensor
    recv_order: torch.Tensor


def dispatch_native_replicas(inputs: tuple, config: ExpertReplicaConfig,
                             group: object) -> tuple[tuple, NativeReplicaDispatch]:
    """Dispatch existing native expert-major inputs through shared replicas."""
    values, counts = inputs[:2]
    rank = dist.get_rank(group)
    payload = counts.to(torch.int64).contiguous()
    gathered = [torch.empty_like(payload) for _ in range(config.ep_size)]
    work = dist.all_gather(gathered, payload, group=group, async_op=True)
    work.wait()
    host = torch.stack(gathered).cpu().tolist()
    plan = build_expert_replica_plan(host, config.replica_slots_per_rank)
    physical = plan.physical_to_logical
    runs = sorted((expert, slot, plan.dispatch_counts[rank][slot])
                  for slot, expert in enumerate(physical) if expert >= 0)
    slots = torch.tensor([slot for _, slot, _ in runs], device=values.device)
    repeats = torch.tensor([count for _, _, count in runs], device=values.device)
    physical_ids = torch.repeat_interleave(slots, repeats, output_size=values.shape[0])
    send_order = stable_expert_order(physical_ids, config.physical_experts)
    width = config.slots_per_rank
    send_splits = [sum(plan.dispatch_counts[rank][peer * width:(peer + 1) * width])
                   for peer in range(config.ep_size)]
    received = [row[rank * width:(rank + 1) * width] for row in plan.dispatch_counts]
    recv_splits = [sum(row) for row in received]
    recv_ids = torch.arange(width, device=values.device).repeat(config.ep_size)
    recv_counts = torch.tensor(received, dtype=torch.int64, device=values.device)
    recv_order = stable_expert_order(torch.repeat_interleave(
        recv_ids, recv_counts.flatten(), output_size=sum(recv_splits)), width)
    route = ReplicaRoute(plan, physical_ids, torch.tensor(plan.dispatch_counts, device=values.device), rank, group)
    state = NativeReplicaDispatch(route, send_splits, recv_splits, send_order, recv_order)
    routed = differentiable_all_to_all_single(values[send_order], send_splits, recv_splits, group)[recv_order]
    result = (routed, recv_counts.sum(0))
    if len(inputs) > 2 and inputs[2] is not None:
        scores = differentiable_all_to_all_single(inputs[2][send_order], send_splits, recv_splits, group)[recv_order]
        result += (scores,)
    return result, state


def combine_native_replicas(output: torch.Tensor, state: NativeReplicaDispatch) -> torch.Tensor:
    """Restore original logical expert-major rows before model unpermutation."""
    rank_major = torch.empty_like(output)
    rank_major[state.recv_order] = output
    combined = differentiable_all_to_all_single(
        rank_major, state.recv_splits, state.send_splits, state.route.group)
    result = torch.empty_like(combined)
    result[state.send_order] = combined
    return result
