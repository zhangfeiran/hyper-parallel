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
"""Credit-driven one-sided copies into symmetric expert execution storage."""

from __future__ import annotations

from contextlib import contextmanager
import math
import threading
from typing import Any, Iterator

import torch

from .capacity import _integer
from .pool import ReplicaPool
from .routing import ReplicaRoute

_SIGNAL_BYTES = 64
_SIGNAL_CHANNELS = 5
_MAX_EPOCH = 2**31 - 1


def _aligned(size: int) -> int:
    return (size + _SIGNAL_BYTES - 1) // _SIGNAL_BYTES * _SIGNAL_BYTES


def signal_storage_bytes(shapes: tuple[tuple[int, ...], ...], slots: int,
                         ep_size: int, element_size: int) -> int:
    """Size B weight/FP32-gradient slots and cache-line-separated peer signals."""
    _integer(slots, "slots")
    _integer(ep_size, "ep_size")
    _integer(element_size, "element_size")
    if element_size not in (2, 4):
        raise ValueError("Signal transport requires two- or four-byte weights")
    if not shapes or any(not shape for shape in shapes):
        raise ValueError("Signal transport requires nonempty expert matrix shapes")
    for shape in shapes:
        for dim in shape:
            _integer(dim, "expert matrix dimension")
    matrices = sum(_aligned(slots * math.prod(shape) * size) for size in (element_size, 4) for shape in shapes)
    return matrices + _SIGNAL_CHANNELS * ep_size * slots * _SIGNAL_BYTES


class SignalReplicaTransport:
    """Direct symmetric weights/dW with credit, publication and read-completion signals.

    All ranks submit the same sequence of calls on one exclusively leased
    provider. Epochs persist across calls. A receiver grants a slot only after
    its previous stream consumer, so changing the next owner cannot overwrite
    a live expert. Initialization and epoch rollover synchronize collectively;
    steady-state transfers enqueue stream operations without host barriers.
    """

    def __init__(self, runtime: Any, storage: torch.Tensor, slots: int, ep_size: int,
                 *, use_sdma: bool = False) -> None:
        """Bind externally owned, 64-byte-aligned symmetric uint8 storage.

        Args:
            runtime: Stream-ordered put/get, signal/wait and host-barrier provider.
            storage: Symmetric storage kept alive through all remote consumers.
            slots: Guest expert budget per rank.
            ep_size: Number of ranks in the runtime's peer namespace.
            use_sdma: Request runtime put/get with use_sdma=True. The runtime must
                support direct peer mapping and complete copies in stream order.
        """
        if (storage.dtype != torch.uint8 or storage.ndim != 1 or not storage.is_contiguous()
                or storage.data_ptr() % _SIGNAL_BYTES):
            raise ValueError("Signal replica storage must be a cache-line-aligned contiguous uint8 vector")
        _integer(slots, "slots")
        _integer(ep_size, "ep_size")
        self.runtime = runtime
        self._copy_options = {"use_sdma": True} if use_sdma else {}
        self.storage = storage
        self.slots = slots
        self.ep_size = ep_size
        self.pool = None
        self.signals = None
        self.epoch = 0
        self._layout = None
        self._lock = threading.Lock()

    def _bind(self, weights: tuple[torch.Tensor, ...]) -> None:
        if not weights:
            raise ValueError("Signal transport requires nonempty expert weights")
        layout = tuple((tuple(weight.shape[1:]), weight.dtype, weight.device) for weight in weights)
        if self.pool is not None:
            if layout != self._layout:
                raise ValueError("Signal transport matrix layout cannot change without reinitialization")
            return
        if any(weight.device != self.storage.device or weight.dtype != weights[0].dtype for weight in weights):
            raise ValueError("Signal weights must share the symmetric storage device and one dtype")
        shapes = tuple(tuple(weight.shape[1:]) for weight in weights)
        required = signal_storage_bytes(shapes, self.slots, self.ep_size, weights[0].element_size())
        if required > self.storage.numel():
            raise ValueError("Symmetric signal replica storage is too small")
        views = []
        offset = 0
        for dtype, size in ((weights[0].dtype, weights[0].element_size()), (torch.float32, 4)):
            for shape in shapes:
                length = self.slots * math.prod(shape) * size
                views.append(self.storage[offset:offset + length].view(dtype).reshape(self.slots, *shape))
                offset += _aligned(length)
        self.signals = self.storage[offset:required].view(torch.int32).reshape(
            _SIGNAL_CHANNELS, self.ep_size, self.slots, _SIGNAL_BYTES // 4)
        self.signals.zero_()
        self.runtime.host_barrier()
        self.pool = ReplicaPool(tuple(views[:len(weights)]), self.slots, borrow=True)
        self.pool.gradients = tuple(views[len(weights):])
        self._layout = layout

    def _advance(self) -> int:
        if self.epoch == _MAX_EPOCH:
            self.runtime.host_barrier()
            self.signals.zero_()
            self.runtime.host_barrier()
            self.epoch = 0
        self.epoch += 1
        return self.epoch

    def _word(self, channel: int, peer: int, slot: int) -> torch.Tensor:
        return self.signals[channel, peer, slot, :1]

    def _publish(self, channel: int, peer_index: int, slot: int, target: int, epoch: int) -> None:
        self.runtime.signal(self._word(channel, peer_index, slot), epoch, target)

    def _wait(self, channel: int, peer: int, slot: int, epoch: int) -> None:
        self.runtime.wait_signal(self._word(channel, peer, slot), epoch, comparison="ge")

    @contextmanager
    def lease(self, weights: tuple[torch.Tensor, ...], route: ReplicaRoute,
              *, backward: bool = False) -> Iterator[ReplicaPool]:
        """Borrow direct execution views through all kernel and remote consumers."""
        if (route.plan.config.replica_slots_per_rank != self.slots
                or route.plan.config.ep_size != self.ep_size):
            raise ValueError("Signal transport topology does not match the replica route")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent signal replica leases are unsupported")
        try:
            self._bind(weights)
            with self.pool.lease(backward=backward):
                self.prefetch(weights, self.pool.weights, route)
                yield self.pool
        finally:
            self._lock.release()

    def prefetch(self, weights: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                 route: ReplicaRoute) -> None:
        """Grant every incoming credit before waiting on any outgoing transfer."""
        epoch = self._advance()
        home = route.plan.config.home_experts
        transfers = route.plan.transfers
        for item in transfers:
            if route.rank == item.target_rank:
                self._publish(0, route.rank, item.target_slot - home, item.owner_rank, epoch)
        for item in transfers:
            if route.rank == item.owner_rank:
                slot = item.target_slot - home
                self._wait(0, item.target_rank, slot, epoch)
                for weight, guest in zip(weights, guests):
                    self.runtime.put(guest[slot], weight[item.owner_slot].contiguous(),
                                     item.target_rank, **self._copy_options)
                self._publish(1, route.rank, slot, item.target_rank, epoch)
        for item in transfers:
            if route.rank == item.target_rank:
                slot = item.target_slot - home
                self._wait(1, item.owner_rank, slot, epoch)
                self._publish(2, route.rank, slot, item.owner_rank, epoch)
        for item in transfers:
            if route.rank == item.owner_rank:
                self._wait(2, item.target_rank, item.target_slot - home, epoch)

    def return_gradients(self, gradients: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                         route: ReplicaRoute) -> tuple[torch.Tensor, ...]:
        """Read direct FP32 guest outputs; acknowledge before any slot can be reused."""
        epoch = self._advance()
        home = route.plan.config.home_experts
        result = tuple(gradient.float().clone() for gradient in gradients)
        transfers = route.plan.transfers
        for item in transfers:
            if route.rank == item.target_rank:
                self._publish(3, route.rank, item.target_slot - home, item.owner_rank, epoch)
        for target in range(self.ep_size):
            for item in transfers:
                if item.target_rank != target or item.owner_rank != route.rank:
                    continue
                slot = item.target_slot - home
                self._wait(3, target, slot, epoch)
                for output, guest in zip(result, guests):
                    value = torch.empty_like(output[item.owner_slot])
                    self.runtime.get(value, guest[slot], target, **self._copy_options)
                    output[item.owner_slot].add_(value)
                self._publish(4, route.rank, slot, target, epoch)
        for item in transfers:
            if route.rank == item.target_rank:
                self._wait(4, item.owner_rank, item.target_slot - home, epoch)
        return result
