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

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist

from .pool import ReplicaPool, ReplicaPrefetch, replica_pool
from .routing import ReplicaRoute


def _exchange(operations: list) -> None:
    """Order every tensor consumer after the asynchronous P2P operations."""
    if operations:
        for work in dist.batch_isend_irecv(operations):
            work.wait()


@contextmanager
def prefetch_weights(weights: tuple[torch.Tensor, ...], route: ReplicaRoute,
                     *, backward: bool = False, provider: object = None,
                     overlap: bool = False) -> Iterator[ReplicaPool | ReplicaPrefetch]:
    """Borrow home weights and fill only the shared B guest slots.

    Ordinary pool allocations remain valid across push SHMEM heap growth.
    The lease spans all consumers, including gradient return in backward.
    With overlap=True, an enabled provider may defer guest readiness. Call
    wait_weights() before guest reads, or honor weight_ready in a fused kernel.
    """
    provider = route.transport if provider is None else provider
    if provider is not None and callable(getattr(provider, "lease", None)):
        options = {"overlap": True} if overlap and getattr(provider, "overlap_home", False) else {}
        with provider.lease(weights, route, backward=backward, **options) as pool:
            yield pool
        return
    config = route.plan.config
    pool = replica_pool(weights, config.replica_slots_per_rank, route.group)
    with pool.lease(backward=backward):
        if provider is not None:
            provider.prefetch(weights, pool.weights, route)
            yield pool
            return
        operations = []
        for transfer in route.plan.transfers:
            for weight, output in zip(weights, pool.weights):
                if route.rank == transfer.owner_rank:
                    peer = dist.get_global_rank(route.group, transfer.target_rank)
                    operations.append(dist.P2POp(
                        dist.isend, weight[transfer.owner_slot].contiguous(), peer, route.group))
                elif route.rank == transfer.target_rank:
                    peer = dist.get_global_rank(route.group, transfer.owner_rank)
                    guest = output[transfer.target_slot - config.home_experts]
                    operations.append(dist.P2POp(dist.irecv, guest, peer, route.group))
        _exchange(operations)
        yield pool


def return_gradients(gradients: tuple[torch.Tensor, ...], route: ReplicaRoute,
                     guests: tuple[torch.Tensor, ...] | None = None,
                     provider: object = None) -> tuple[torch.Tensor, ...]:
    """Accumulate guest partials in FP32 before returning owner gradients.

    One target rank returns at a time. Each owner needs at most B incoming
    gradient slots, independent of the number of EP ranks or remote replicas.
    All ranks execute rounds in identical order, including ranks with no work.
    """
    provider = route.transport if provider is None else provider
    if provider is not None:
        return provider.return_gradients(gradients, guests, route)
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
                    value = (gradient[transfer.target_slot] if guests is None else
                             guests[index][transfer.target_slot - home]).float().contiguous()
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
