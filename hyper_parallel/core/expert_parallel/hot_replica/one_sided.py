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
"""Injected one-sided expert transport, independent of a particular SHMEM library."""

from __future__ import annotations

from typing import Any

import torch

from .routing import ReplicaRoute
from .transport import _gradient_accumulators


class OneSidedReplicaTransport:
    """Use owner puts for weights and owner gets for FP32 gradient fan-in.

    The caller owns a symmetric byte inbox of B times the largest FP32 expert
    matrix size on every rank. ``runtime`` supplies put, get and host_barrier;
    its PE numbering must equal the route's group rank numbering. All operations
    and barriers must order the calling stream. The inbox must be exclusively
    leased until return and rebound after any symmetric heap replacement.

    Blocking barriers provide publication and consumption acknowledgement.
    This correctness-first provider is opt-in; it makes no latency claim.
    """

    def __init__(self, runtime: Any, inbox: torch.Tensor) -> None:
        """Bind externally leased symmetric storage and a stream-ordered runtime."""
        if inbox.dtype != torch.uint8 or inbox.ndim != 1 or not inbox.is_contiguous():
            raise ValueError("One-sided replica inbox must be a contiguous uint8 vector")
        self.runtime = runtime
        self.inbox = inbox

    def _view(self, tensor: torch.Tensor, slots: int) -> torch.Tensor:
        elements = slots * tensor[0].numel()
        needed = elements * tensor.element_size()
        if needed > self.inbox.numel() or tensor.device != self.inbox.device:
            raise ValueError("One-sided replica inbox is too small or on the wrong device")
        return self.inbox[:needed].view(tensor.dtype).reshape(slots, *tensor.shape[1:])

    def prefetch(self, weights: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                 route: ReplicaRoute) -> None:
        """Publish only requested owner slices into their destination guest slots."""
        if not route.plan.transfers:
            return
        config = route.plan.config
        for weight, guest in zip(weights, guests):
            inbox = self._view(weight, config.replica_slots_per_rank)
            for transfer in route.plan.transfers:
                if transfer.owner_rank == route.rank:
                    self.runtime.put(inbox[transfer.target_slot - config.home_experts],
                                     weight[transfer.owner_slot].contiguous(), transfer.target_rank)
            self.runtime.host_barrier()
            for transfer in route.plan.transfers:
                if transfer.target_rank == route.rank:
                    slot = transfer.target_slot - config.home_experts
                    guest[slot].copy_(inbox[slot])
            self.runtime.host_barrier()

    def return_gradients(self, gradients: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                         route: ReplicaRoute) -> tuple[torch.Tensor, ...]:
        """Read bounded guest partials and accumulate at the original FP32 owner."""
        result = _gradient_accumulators(gradients, consume=False)
        return self._return_gradients(result, guests, route)

    def return_gradients_owned(self, gradients: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                               route: ReplicaRoute) -> tuple[torch.Tensor, ...]:
        """Consume fresh, exclusive FP32 home buffers using the shared transport contract."""
        result = _gradient_accumulators(gradients, consume=True)
        return self._return_gradients(result, guests, route)

    def _return_gradients(self, result: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                          route: ReplicaRoute) -> tuple[torch.Tensor, ...]:
        config = route.plan.config
        for target in range(config.ep_size):
            transfers = [item for item in route.plan.transfers if item.target_rank == target]
            if not transfers:
                continue
            for gradient, guest in zip(result, guests):
                inbox = self._view(gradient, config.replica_slots_per_rank)
                if route.rank == target:
                    for transfer in transfers:
                        slot = transfer.target_slot - config.home_experts
                        inbox[slot].copy_(guest[slot])
                self.runtime.host_barrier()
                incoming = []
                for transfer in transfers:
                    if transfer.owner_rank == route.rank:
                        value = torch.empty_like(gradient[transfer.owner_slot])
                        self.runtime.get(value, inbox[transfer.target_slot - config.home_experts], target)
                        incoming.append((transfer.owner_slot, value))
                self.runtime.host_barrier()
                for slot, value in incoming:
                    gradient[slot].add_(value)
        return result
