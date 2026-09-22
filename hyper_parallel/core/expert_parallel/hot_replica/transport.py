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
"""Sparse eager weight prefetch and bounded FP32 gradient return."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .routing import ReplicaRoute


def _exchange(operations: list) -> None:
    """Order every tensor consumer after the asynchronous P2P operations."""
    if operations:
        for work in dist.batch_isend_irecv(operations):
            work.wait()


def prefetch_weights(weights: tuple[torch.Tensor, ...], route: ReplicaRoute) -> tuple[torch.Tensor, ...]:
    """Materialize home plus guest weights without registering new parameters.

    This eager transport uses ordinary allocations, so it remains valid across
    push SHMEM heap growth. Saved backward state retains home parameters only;
    backward re-prefetches the invocation's immutable placement.
    """
    config = route.plan.config
    if not config.replica_slots_per_rank:
        return weights
    physical = tuple(weight.new_zeros((config.slots_per_rank, *weight.shape[1:])) for weight in weights)
    for output, weight in zip(physical, weights):
        output[:config.home_experts].copy_(weight)
    operations = []
    for transfer in route.plan.transfers:
        for weight, output in zip(weights, physical):
            if route.rank == transfer.owner_rank:
                peer = dist.get_global_rank(route.group, transfer.target_rank)
                operations.append(dist.P2POp(dist.isend, weight[transfer.owner_slot].contiguous(), peer, route.group))
            elif route.rank == transfer.target_rank:
                peer = dist.get_global_rank(route.group, transfer.owner_rank)
                operations.append(dist.P2POp(dist.irecv, output[transfer.target_slot], peer, route.group))
    _exchange(operations)
    return physical


def return_gradients(gradients: tuple[torch.Tensor, ...], route: ReplicaRoute) -> tuple[torch.Tensor, ...]:
    """Accumulate guest partials in FP32 before returning owner gradients.

    One target rank returns at a time. Each owner needs at most B incoming
    gradient slots, independent of the number of EP ranks or remote replicas.
    All ranks execute rounds in identical order, including ranks with no work.
    """
    home = route.plan.config.home_experts
    result = tuple(gradient[:home].float().clone() for gradient in gradients)
    for target in range(route.plan.config.ep_size):
        operations = []
        incoming = []
        outgoing = []
        for transfer in route.plan.transfers:
            if transfer.target_rank != target:
                continue
            for index, gradient in enumerate(gradients):
                if route.rank == target:
                    value = gradient[transfer.target_slot].float().contiguous()
                    outgoing.append(value)
                    peer = dist.get_global_rank(route.group, transfer.owner_rank)
                    operations.append(dist.P2POp(dist.isend, value, peer, route.group))
                elif route.rank == transfer.owner_rank:
                    value = torch.empty_like(result[index][transfer.owner_slot])
                    incoming.append((index, transfer.owner_slot, value))
                    peer = dist.get_global_rank(route.group, target)
                    operations.append(dist.P2POp(dist.irecv, value, peer, route.group))
        _exchange(operations)
        for index, slot, value in incoming:
            result[index][slot].add_(value)
    return result
