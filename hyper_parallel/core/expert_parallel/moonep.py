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
"""Native BF16 experts with MoonEP MTE communication on Ascend A3.

The optional MoonEP dependency is built by ``scripts/build_moonep_a3.py``.
Planning runs on the host on every forward. Communication and weight-gradient
reduction use MoonEP's upstream AscendC MTE kernels. One communication/replica
pool is shared by independent layers; autograd saves owned activation copies.
"""

from dataclasses import dataclass
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.core.expert_parallel.moonep_planning import (
    deduplicate_moonep, encode_moonep_routes, plan_moonep,
)


def _dependencies() -> tuple:
    """Load the optional NPU/MoonEP extensions only when an adapter is built."""
    import torch_npu  # pylint: disable=import-outside-toplevel
    from npu_patch import Buffer, MoonEPCommPlan, ShmemRuntime  # pylint: disable=import-outside-toplevel
    from npu_patch import buffer_c  # pylint: disable=import-outside-toplevel
    from npu_patch._C import acl_shmem_barrier_all_on_stream  # pylint: disable=import-outside-toplevel
    return torch_npu, Buffer, MoonEPCommPlan, ShmemRuntime, buffer_c, acl_shmem_barrier_all_on_stream


def _gather(tensor: torch.Tensor, size: int, group: object) -> torch.Tensor:
    values = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(values, tensor.contiguous(), group=group)
    return torch.stack(values)


@dataclass(frozen=True)
class MoonEPRoute:
    """Invocation-owned communication plan and compact native GEMM groups."""

    plan: Any
    counts: tuple
    groups: torch.Tensor
    placement: Any


class MoonEPRuntime:
    """Own one process-wide MoonEP communication and transient expert pool.

    Construct once, share across layers, then call ``initialize`` before the
    first forward and ``close`` before destroying the process group. Other
    SHMEM runtimes must run in separate fresh processes.
    """

    def __init__(self, tokens: int, hidden: int, intermediate: int, experts: int, top_k: int,
                 group: object) -> None:
        """Initialize an A3 SHMEM heap and allocate shared symmetric pools."""
        dependencies = _dependencies()
        self.ops, buffer_class, self.plan_class, self.shmem, self.memory, self.publish_barrier = dependencies
        self.group = group
        self.rank, self.size = dist.get_rank(group), dist.get_world_size(group)
        if min(tokens, hidden, intermediate, experts, top_k) <= 0 or experts % self.size:
            raise ValueError("MoonEP dimensions must be positive and experts divisible by EP")
        self.tokens, self.hidden, self.intermediate = tokens, hidden, intermediate
        self.experts, self.top_k, self.home = experts, top_k, experts // self.size
        self.capacity = tokens * top_k
        self.initialized = False
        self.closed = False
        self.last_route = None
        owned_bytes = self.home * hidden * intermediate * 36 + self.capacity * hidden * 2
        granularity = 256 * 1024**2
        self.reserved_heap_bytes = ((owned_bytes + 512 * 1024**2 + granularity - 1) // granularity) * granularity
        os.environ["ASCEND_MOONEP_A3_HEAP"] = "1"
        self.shmem.init_legacy_a3(group, self.reserved_heap_bytes)
        self.initialized = True
        self.buffer = buffer_class(tokens, hidden, top_k, experts, self.size, B=self.home,
                                   token_padding=1, group=group, use_udma=False,
                                   planning_use_udma=False, explicitly_destroy=True)
        self.weight_home, self.weight_guest, self.grad_home, self.grad_guest = [], [], [], []
        shapes = ((intermediate, hidden), (hidden, intermediate), (intermediate, hidden))
        device = torch.npu.current_device()
        for rows, columns in shapes:
            for collection, dtype in ((self.weight_home, torch.bfloat16), (self.weight_guest, torch.bfloat16),
                                      (self.grad_home, torch.float32), (self.grad_guest, torch.float32)):
                tensor, _, _ = self.memory.create_vmm_physical((self.home, rows, columns), dtype, device)
                collection.append(tensor)
        for home, guest in zip(self.weight_home, self.weight_guest):
            self.memory.attach_local_source_vas(home, home.data_ptr(), guest.data_ptr())
        for home, guest in zip(self.grad_home, self.grad_guest):
            self.memory.attach_local_source_vas(home, home.data_ptr(), guest.data_ptr())
            self.memory.attach_local_source_vas(guest, home.data_ptr(), guest.data_ptr())

    def initialize(self) -> None:
        """Collectively initialize all registered symmetric allocations."""
        if self.closed:
            raise RuntimeError("MoonEP runtime is closed")
        if not self.initialized:
            self.shmem.init_with_buffer(group=self.group, enable_udma=False, qp_count=0)
            self.initialized = True

    @property
    def heap_bytes(self) -> int:
        """Return the entire reserved A3 SHMEM heap, including unused capacity."""
        return self.reserved_heap_bytes

    def route(self, ids: torch.Tensor, counts: torch.Tensor) -> MoonEPRoute:
        """Run online host placement, encoding and rank dedup in the step."""
        if not self.initialized or self.closed:
            raise RuntimeError("MoonEP runtime must be initialized before execution")
        gathered = _gather(counts.to(torch.int64), self.size, self.group)
        host = gathered.cpu().numpy()
        placement = plan_moonep(host, self.capacity)
        packed = encode_moonep_routes(ids.detach().cpu().numpy(), host, placement, self.rank, self.capacity)
        raw = _gather(torch.from_numpy(packed).to(ids.device), self.size, self.group).cpu().numpy()
        encoded, groups, loffs, duplicates = deduplicate_moonep(raw, self.rank, self.top_k, self.capacity)
        def _upload(value):
            return torch.from_numpy(np.ascontiguousarray(value)).to(ids.device)
        plan = self.plan_class(dst=_upload(encoded), experts_to_copy=_upload(placement.experts_to_copy),
                               zero_fill_ranges=torch.zeros((self.experts + self.home, 2),
                                                            dtype=torch.int32, device=ids.device),
                               remote_stats=_upload(placement.remote_stats[self.rank]),
                               dup_groups=_upload(groups), dup_loffs=_upload(loffs), dup_counts=_upload(duplicates),
                               N=self.capacity, R=self.size, E=self.experts, B=self.home,
                               NvS=self.capacity, K=self.top_k)
        compact = tuple(int(value) for value in placement.compact_counts[self.rank])
        route = MoonEPRoute(plan, compact, torch.tensor(compact, device=ids.device).cumsum(0), placement)
        self.last_route = route
        return route

    def prefetch(self, weights: tuple, route: MoonEPRoute) -> None:
        """Publish original parameters before all peers fetch this layer's guests."""
        for target, source in zip(self.weight_home, weights):
            target.copy_(source)
        # The device barrier publishes all home copies before any remote get.
        self.publish_barrier(int(torch.npu.current_stream().npu_stream))
        self.buffer.prefetch_projections(route.plan, self.weight_home)

    def combine(self, route: MoonEPRoute, values: torch.Tensor,
                scores: Optional[torch.Tensor] = None) -> tuple:
        """Publish staged A3 buffers and finish remote reads before reuse.

        A3's legacy heap needs a completion boundary around the upstream
        in-place combine prologue. Buffer's on-stream barrier alone allowed
        nondeterministic results when peers reused shared staging storage.
        """
        self.buffer._ensure_shmem_views()  # pylint: disable=protected-access
        context = self.buffer._ctx  # pylint: disable=protected-access
        hidden = context["hidden_buf_local"]
        hidden.copy_(values)
        routed_scores = None
        if scores is not None:
            context["weights_buf_local"].copy_(scores.view(torch.int32))
            routed_scores = context["weights_buf_local"].view(torch.float32)
        torch.npu.synchronize()
        dist.barrier(group=self.group)
        result = self.buffer.combine(plan=route.plan, hidden_nvsh=hidden, route_weights_nvs=routed_scores,
                                      zero_copy=True, router_weights_zero_copy=scores is not None,
                                      inter_rank_sync=False)
        torch.npu.synchronize()
        dist.barrier(group=self.group)
        return result

    def close(self) -> None:
        """Drain work and release owned pool tensors before SHMEM finalization."""
        if self.closed:
            return
        torch.npu.synchronize()
        self.buffer.destroy()
        allocations = [tensor._vmm_allocation for tensor in  # pylint: disable=protected-access
                       (*self.weight_home, *self.weight_guest, *self.grad_home, *self.grad_guest)]
        for allocation in allocations:
            allocation.destroy()
        self.weight_home.clear()
        self.weight_guest.clear()
        self.grad_home.clear()
        self.grad_guest.clear()
        self.memory.reset_symmetric_descriptors()
        if self.initialized:
            self.shmem.finalize()
        self.last_route = None
        self.closed = True


def _gmm(runtime: MoonEPRuntime, values: torch.Tensor, weights: torch.Tensor,
         groups: torch.Tensor) -> torch.Tensor:
    return runtime.ops.npu_grouped_matmul(x=[values], weight=[weights], bias=[], group_list=groups,
                                          split_item=3, group_type=0, group_list_type=0)[0]


def _split_gmm(runtime: MoonEPRuntime, values: torch.Tensor, home: torch.Tensor, guest: torch.Tensor,
               route: MoonEPRoute, transpose: bool = False) -> torch.Tensor:
    if transpose:
        home, guest = home.transpose(-1, -2), guest.transpose(-1, -2)
    split = sum(route.counts[:runtime.home])
    results = []
    if split:
        results.append(_gmm(runtime, values[:split], home, route.groups[:runtime.home]))
    if split < runtime.capacity:
        results.append(_gmm(runtime, values[split:], guest, route.groups[runtime.home:] - split))
    return torch.cat(results) if len(results) > 1 else results[0]


def _weight_partials(runtime: MoonEPRuntime, inputs: torch.Tensor, gradients: torch.Tensor,
                     route: MoonEPRoute) -> torch.Tensor:
    # Match native's BF16 grouped weight-gradient GEMM before FP32 communication.
    return runtime.ops.npu_grouped_matmul(
        x=[inputs.transpose(0, 1)], weight=[gradients], bias=[], group_list=route.groups,
        split_item=2, group_type=2, group_list_type=0)[0].transpose(1, 2)


def _stage_partials(runtime: MoonEPRuntime, partials: torch.Tensor, route: MoonEPRoute,
                    home: torch.Tensor, guest: torch.Tensor) -> None:
    home.zero_()
    guest.zero_()
    for slot, count in enumerate(route.counts):
        if count:
            output = home[slot] if slot < runtime.home else guest[slot - runtime.home]
            output.copy_(partials[slot])


class _MoonEPExperts(torch.autograd.Function):
    """Own saved activations and use MoonEP for both token backward paths."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, scores: torch.Tensor, ids: torch.Tensor, counts: torch.Tensor,
                weight1: torch.Tensor, weight2: torch.Tensor, weight3: torch.Tensor,
                runtime: MoonEPRuntime) -> torch.Tensor:
        """Plan, communicate and compute with invocation-owned activations."""
        route = runtime.route(ids, counts)
        routed, routed_scores, _, _ = runtime.buffer.dispatch(values, scores.float().contiguous(), plan=route.plan)
        runtime.prefetch((weight1, weight2, weight3), route)
        packed_home = torch.cat((weight1, weight3), dim=1).transpose(1, 2)
        packed_guest = torch.cat((runtime.weight_guest[0], runtime.weight_guest[2]), dim=1).transpose(1, 2)
        up = _split_gmm(runtime, routed, packed_home, packed_guest, route)
        activation = runtime.ops.npu_swiglu(up)
        output = _split_gmm(runtime, activation, weight2.transpose(1, 2),
                            runtime.weight_guest[1].transpose(1, 2), route)
        weighted = (output.float() * routed_scores[:, None]).to(values.dtype)
        combined, _, _ = runtime.combine(route, weighted)
        ctx.runtime, ctx.route = runtime, route
        ctx.save_for_backward(routed, routed_scores, output, up, activation, weight1, weight2, weight3)
        return combined

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Reverse token communication and return FP32 replica partials home."""
        runtime, route = ctx.runtime, ctx.route
        routed, scores, output, up, activation, weight1, weight2, weight3 = ctx.saved_tensors
        grad_rows, _, _, _ = runtime.buffer.dispatch(grad_output.contiguous(), plan=route.plan)
        grad_scores = (grad_rows.float() * output.float()).sum(1)
        grad_experts = (grad_rows.float() * scores[:, None]).to(grad_rows.dtype)
        runtime.prefetch((weight1, weight2, weight3), route)
        grad_activation = _split_gmm(runtime, grad_experts, weight2.transpose(1, 2),
                                     runtime.weight_guest[1].transpose(1, 2), route, transpose=True)
        grad_up = runtime.ops.npu_swiglu_backward(grad_activation, up)
        home = torch.cat((weight1, weight3), dim=1).transpose(1, 2)
        guest = torch.cat((runtime.weight_guest[0], runtime.weight_guest[2]), dim=1).transpose(1, 2)
        grad_routed = _split_gmm(runtime, grad_up, home, guest, route, transpose=True)
        gate_grad, up_grad = _weight_partials(runtime, routed, grad_up, route).chunk(2, dim=1)
        down_grad = _weight_partials(runtime, activation, grad_experts, route)
        for partials, target_home, target_guest in zip(
                (gate_grad, down_grad, up_grad), runtime.grad_home, runtime.grad_guest):
            _stage_partials(runtime, partials, route, target_home, target_guest)
        grad_input, router_gradient, _ = runtime.combine(route, grad_routed, grad_scores.contiguous())
        runtime.buffer.reduce_projection_grads(route.plan, tuple(zip(runtime.grad_home, runtime.grad_guest)))
        # The shared FP32 pools are reused by the next layer's backward.
        gradients = tuple(value.to(weight.dtype) for value, weight in
                          zip(runtime.grad_home, (weight1, weight2, weight3)))
        return grad_input, router_gradient, None, None, *gradients, None


class MoonEPExperts(nn.Module):
    """Native grouped SwiGLU with original local parameters and MoonEP MTE.

    Args:
        weights: Native local ``w1/w2/w3`` BF16 parameters in ``[E/R,out,in]`` order.
        runtime: Shared A3 MoonEP communication runtime.
    """

    def __init__(self, weights: dict[str, torch.Tensor], runtime: MoonEPRuntime) -> None:
        """Keep optimizer-owned weights separate from transient replica buffers."""
        super().__init__()
        self.runtime = runtime
        shapes = {"w1": (runtime.home, runtime.intermediate, runtime.hidden),
                  "w2": (runtime.home, runtime.hidden, runtime.intermediate),
                  "w3": (runtime.home, runtime.intermediate, runtime.hidden)}
        for name in ("w1", "w2", "w3"):
            weight = weights[name]
            if tuple(weight.shape) != shapes[name] or weight.dtype != torch.bfloat16 or weight.device.type != "npu":
                raise ValueError(f"MoonEP {name} must be BF16 NPU weights of shape {shapes[name]}")
            setattr(self, name, weight if isinstance(weight, nn.Parameter) else nn.Parameter(weight))

    def forward(self, values: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor,
                counts: torch.Tensor) -> torch.Tensor:
        """Execute token-major dispatch, native experts and weighted combine."""
        if values.device.type != "npu" or values.dtype != torch.bfloat16:
            raise ValueError("MoonEP A3 currently requires BF16 NPU inputs")
        return _MoonEPExperts.apply(values, scores, ids, counts, self.w1, self.w2, self.w3, self.runtime)
