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

from contextlib import contextmanager, nullcontext
import math
import threading
from typing import Any, Iterator

import torch

from .capacity import _integer
from .pool import ReplicaPool, ReplicaPrefetch
from .routing import ReplicaRoute
from .transport import _gradient_accumulators

_SIGNAL_BYTES = 64
_SIGNAL_CHANNELS = 5
_MAX_EPOCH = 2**31 - 1


SIGNAL_TRANSPORT_MODES = (
    "shmem_signal", "shmem_signal_sdma", "shmem_signal_sdma_parallel", "shmem_signal_sdma_bidir",
    "shmem_signal_sdma_overlap", "shmem_signal_sdma_projection",
    "shmem_signal_kernel_gradient",
)


def signal_transport_options(mode: str) -> dict[str, bool]:
    """Resolve a signal transport mode for MegaMoe adapters."""
    if mode not in SIGNAL_TRANSPORT_MODES:
        raise ValueError(f"Unsupported signal replica transport: {mode}")
    return {"use_sdma": mode != "shmem_signal",
            "parallel_prefetch": mode in SIGNAL_TRANSPORT_MODES[2:],
            "parallel_gradients": mode in SIGNAL_TRANSPORT_MODES[3:],
            "overlap_home": mode in SIGNAL_TRANSPORT_MODES[4:],
            "projection_ready": mode in SIGNAL_TRANSPORT_MODES[5:],
            "kernel_gradients": mode == "shmem_signal_kernel_gradient"}


def _aligned(size: int) -> int:
    return (size + _SIGNAL_BYTES - 1) // _SIGNAL_BYTES * _SIGNAL_BYTES


def signal_storage_bytes(shapes: tuple[tuple[int, ...], ...], slots: int,
                         ep_size: int, element_size: int, *, projection_ready: bool = False) -> int:
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
    ready_bytes = len(shapes) * slots * _SIGNAL_BYTES if projection_ready else 0
    return matrices + _SIGNAL_CHANNELS * ep_size * slots * _SIGNAL_BYTES + ready_bytes


class SignalReplicaTransport:
    """Direct symmetric weights/dW with credit, publication and read-completion signals.

    All ranks submit the same sequence of calls on one exclusively leased
    provider. Epochs persist across calls. A receiver grants a slot only after
    its previous stream consumer, so changing the next owner cannot overwrite
    a live expert. Initialization and epoch rollover synchronize collectively;
    steady-state transfers enqueue stream operations without host barriers.
    """

    def __init__(self, runtime: Any, storage: torch.Tensor, slots: int, ep_size: int,
                 *, use_sdma: bool = False, parallel_prefetch: bool = False,
                 parallel_gradients: bool = False, overlap_home: bool = False, projection_ready: bool = False,
                 kernel_gradients: bool = False) -> None:
        """Bind externally owned, 64-byte-aligned symmetric uint8 storage.

        Args:
            runtime: Stream-ordered put/get, signal/wait and host-barrier provider.
            storage: Symmetric storage kept alive through all remote consumers.
            slots: Guest expert budget per rank.
            ep_size: Number of ranks in the runtime's peer namespace.
            use_sdma: Request runtime put/get with use_sdma=True. The runtime must
                support direct peer mapping and complete copies in stream order.
            parallel_prefetch: Enqueue outgoing SDMA weights on one stream per peer.
                Join these streams before releasing the lease.
            parallel_gradients: Read and accumulate different projection matrices on
                separate streams, preserving peer order within each matrix. Cache one
                FP32 expert gradient across projections, independent of slots/peers.
            overlap_home: Permit deferred guest reads when the caller requests overlap.
                Requires parallel SDMA prefetch; ready publication also uses SDMA.
            projection_ready: Publish separate ready words for each weight matrix. Requires
                home overlap. Forward copies matrices in tuple order; backward reverses it.
                Include projection_ready=True when sizing the symmetric storage.
            kernel_gradients: Let a fused consumer perform W2 return on its own workers.
                Requires projection readiness and parallel gradient streams.
        """
        if (storage.dtype != torch.uint8 or storage.ndim != 1 or not storage.is_contiguous()
                or storage.data_ptr() % _SIGNAL_BYTES):
            raise ValueError("Signal replica storage must be a cache-line-aligned contiguous uint8 vector")
        _integer(slots, "slots")
        _integer(ep_size, "ep_size")
        if (parallel_prefetch or parallel_gradients) and not use_sdma:
            raise ValueError("Parallel replica transport requires SDMA copies")
        if overlap_home and not (use_sdma and parallel_prefetch):
            raise ValueError("Home overlap requires parallel SDMA prefetch")
        if projection_ready and not overlap_home:
            raise ValueError("Projection readiness requires home overlap")
        if kernel_gradients and not (projection_ready and parallel_gradients):
            raise ValueError("Early gradients require projection readiness and parallel gradient streams")
        self._projection_ready = projection_ready
        self._kernel_gradients = kernel_gradients
        self.projection_signals = None
        self.runtime = runtime
        self.overlap_home = overlap_home
        self._parallel_prefetch = parallel_prefetch
        self._parallel_gradients = parallel_gradients
        self._gradient_streams = {}
        self._gradient_scratch = {}
        self._streams = {}
        self._copy_options = {"use_sdma": True} if use_sdma else {}
        self.storage = storage
        self.slots = slots
        self.ep_size = ep_size
        self.pool = None
        self.signals = None
        self.epoch = 0
        self._layout = None
        self._lock = threading.Lock()

    @property
    def kernel_gradients(self) -> bool:
        """Allow a fused consumer to own W2 publication, accumulation and acknowledgement."""
        return self._kernel_gradients

    def kernel_gradient_signals(self) -> tuple[int, int, int]:
        """Reserve an epoch and return ready/ACK bases for an active backward lease.

        All ranks must reserve exactly once in the same order. The consumer must
        publish readiness after production, preserve FP32 peer order and wait for
        every remote reader before returning control or reusing guest storage.
        """
        if (not self.kernel_gradients or not self._lock.locked() or self.pool is None
                or self.pool.gradients is None):
            raise ValueError("Kernel gradients require an active projection SDMA lease")
        return self._advance(), self.signals[3].data_ptr(), self.signals[4].data_ptr()

    @property
    def gradient_scratch_bytes(self) -> int:
        """Return cached local FP32 scratch bytes, excluding symmetric execution slots."""
        return sum(tensor.numel() * tensor.element_size() for tensor in self._gradient_scratch.values())

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
        required = signal_storage_bytes(shapes, self.slots, self.ep_size, weights[0].element_size(),
                                        projection_ready=self._projection_ready)
        if required > self.storage.numel():
            raise ValueError("Symmetric signal replica storage is too small")
        views = []
        offset = 0
        for dtype, size in ((weights[0].dtype, weights[0].element_size()), (torch.float32, 4)):
            for shape in shapes:
                length = self.slots * math.prod(shape) * size
                views.append(self.storage[offset:offset + length].view(dtype).reshape(self.slots, *shape))
                offset += _aligned(length)
        signal_end = offset + _SIGNAL_CHANNELS * self.ep_size * self.slots * _SIGNAL_BYTES
        self.signals = self.storage[offset:signal_end].view(torch.int32).reshape(
            _SIGNAL_CHANNELS, self.ep_size, self.slots, _SIGNAL_BYTES // 4)
        self.signals.zero_()
        if self._projection_ready:
            self.projection_signals = self.storage[signal_end:required].view(torch.int32).reshape(
                len(weights), self.slots, _SIGNAL_BYTES // 4)
            self.projection_signals.zero_()
        self.runtime.host_barrier()
        self.pool = ReplicaPool(tuple(views[:len(weights)]), self.slots, borrow=True)
        self.pool.gradients = tuple(views[len(weights):])
        self._layout = layout

    def _advance(self, *, reserve: int = 0) -> int:
        if self.epoch >= _MAX_EPOCH - reserve:
            self.runtime.host_barrier()
            self.signals.zero_()
            if self.projection_signals is not None:
                self.projection_signals.zero_()
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
              *, backward: bool = False, overlap: bool = False) -> Iterator[ReplicaPool | ReplicaPrefetch]:
        """Borrow direct execution views through all kernel and remote consumers."""
        if (route.plan.config.replica_slots_per_rank != self.slots
                or route.plan.config.ep_size != self.ep_size):
            raise ValueError("Signal transport topology does not match the replica route")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent signal replica leases are unsupported")
        try:
            self._bind(weights)
            with self.pool.lease(backward=backward):
                if overlap and self.overlap_home:
                    with self._overlapped_prefetch(weights, route, backward=backward) as prefetched:
                        yield prefetched
                else:
                    self.prefetch(weights, self.pool.weights, route)
                    yield self.pool
        finally:
            self._lock.release()

    @contextmanager
    def _overlapped_prefetch(self, weights: tuple[torch.Tensor, ...],
                             route: ReplicaRoute, *, backward: bool = False) -> Iterator[ReplicaPrefetch]:
        # A fused consumer still polls weight readiness when W2 return reserves its epoch.
        # Roll over before publication, leaving room for both W2 and the W13 tail.
        epoch = self._advance(reserve=2 if backward and self.kernel_gradients else 0)
        home = route.plan.config.home_experts
        incoming = [item for item in route.plan.transfers if item.target_rank == route.rank]
        outgoing = [item for item in route.plan.transfers if item.owner_rank == route.rank]
        for item in incoming:
            self._publish(0, route.rank, item.target_slot - home, item.owner_rank, epoch)
        # No AIV helper may be needed on a copy stream once a fused kernel occupies the device.
        for item in outgoing:
            self._wait(0, item.target_rank, item.target_slot - home, epoch)
        sources = {item.owner_slot: tuple(weight[item.owner_slot].contiguous() for weight in weights)
                   for item in outgoing}
        value = torch.tensor([epoch], dtype=torch.int32, device=weights[0].device)

        def _wait(matrix_index=None):
            for item in incoming:
                slot = item.target_slot - home
                if self._projection_ready:
                    self.runtime.wait_signal(self.projection_signals[matrix_index, slot, :1], epoch, comparison="ge")
                else:
                    self._wait(1, 0, slot, epoch)

        order = tuple(range(len(weights)))
        if self._projection_ready and backward:
            order = order[::-1]

        peers = sorted({item.target_rank for item in outgoing})
        with self._copy_streams(peers, self._streams, weights[0].device) as (backend, streams):
            for item in outgoing:
                stream = streams[item.target_rank]
                with backend.stream(stream):
                    slot = item.target_slot - home
                    value.record_stream(stream)
                    for index in order:
                        source = sources[item.owner_slot][index]
                        source.record_stream(stream)
                        self.runtime.put(self.pool.weights[index][slot], source, item.target_rank,
                                         **self._copy_options)
                        if self._projection_ready:
                            self.runtime.put(self.projection_signals[index, slot, :1], value, item.target_rank,
                                             **self._copy_options)
                    if not self._projection_ready:
                        self.runtime.put(self._word(1, 0, slot), value, item.target_rank, **self._copy_options)
            ready = None if self._projection_ready else (self.signals[1, 0].data_ptr(), epoch)
            projection_ready = ((tuple(value.data_ptr() for value in self.projection_signals), epoch)
                                if self._projection_ready else None)
            prefetched = ReplicaPrefetch(self.pool, _wait, ready, projection_ready=projection_ready)
            try:
                yield prefetched
            finally:
                prefetched.wait_weights()
                for item in incoming:
                    self._publish(2, route.rank, item.target_slot - home, item.owner_rank, epoch)
                for item in outgoing:
                    self._wait(2, item.target_rank, item.target_slot - home, epoch)

    def prefetch(self, weights: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                 route: ReplicaRoute) -> None:
        """Grant every incoming credit before waiting on any outgoing transfer."""
        epoch = self._advance()
        home = route.plan.config.home_experts
        transfers = route.plan.transfers
        for item in transfers:
            if route.rank == item.target_rank:
                self._publish(0, route.rank, item.target_slot - home, item.owner_rank, epoch)
        with self._prefetch_streams(route, weights[0].device) as (backend, streams):
            for item in transfers:
                if route.rank == item.owner_rank:
                    stream = streams.get(item.target_rank)
                    with nullcontext() if stream is None else backend.stream(stream):
                        self._put_weights(weights, guests, item, home, epoch, stream)
            for item in transfers:
                if route.rank == item.target_rank:
                    slot = item.target_slot - home
                    self._wait(1, item.owner_rank, slot, epoch)
                    self._publish(2, route.rank, slot, item.owner_rank, epoch)
            for item in transfers:
                if route.rank == item.owner_rank:
                    self._wait(2, item.target_rank, item.target_slot - home, epoch)

    def _put_weights(self, weights: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                     item: Any, home: int, epoch: int, stream: Any) -> None:
        slot = item.target_slot - home
        self._wait(0, item.target_rank, slot, epoch)
        for weight, guest in zip(weights, guests):
            source = weight[item.owner_slot].contiguous()
            if stream is not None:
                source.record_stream(stream)
            self.runtime.put(guest[slot], source, item.target_rank, **self._copy_options)
        self._publish(1, item.owner_rank, slot, item.target_rank, epoch)

    @contextmanager
    def _prefetch_streams(self, route: ReplicaRoute, device: torch.device) -> Iterator[tuple[Any, dict]]:
        if not self._parallel_prefetch:
            yield None, {}
            return
        peers = sorted({item.target_rank for item in route.plan.transfers if item.owner_rank == route.rank})
        with self._copy_streams(peers, self._streams, device) as state:
            yield state

    @contextmanager
    def _copy_streams(self, keys: list[int], cache: dict[int, Any],
                      device: torch.device) -> Iterator[tuple[Any, dict[int, Any]]]:
        backend = getattr(torch, device.type)
        caller = backend.current_stream(device)
        streams = {}
        try:
            if keys:
                ready = backend.Event()
                # Fork after local producers and incoming publications, avoiding cyclic waits.
                ready.record(caller)
                for key in keys:
                    if key not in cache:
                        cache[key] = backend.Stream(device=device)
                    stream = cache[key]
                    stream.wait_event(ready)
                    streams[key] = stream
            yield backend, streams
        finally:
            # The pool/workspace completion event must include every remote copy consumer.
            for stream in streams.values():
                done = backend.Event()
                done.record(stream)
                caller.wait_event(done)

    def return_gradients(self, gradients: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                         route: ReplicaRoute) -> tuple[torch.Tensor, ...]:
        """Preserve borrowed home gradients while summing direct FP32 guest outputs."""
        result = _gradient_accumulators(gradients, consume=False)
        return self._return_gradients(result, guests, route)

    def return_gradients_owned(self, gradients: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                               route: ReplicaRoute) -> tuple[torch.Tensor, ...]:
        """Consume fresh, exclusive FP32 home buffers using the shared transport contract."""
        result = _gradient_accumulators(gradients, consume=True)
        return self._return_gradients(result, guests, route)

    def _return_gradients(self, result: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                          route: ReplicaRoute) -> tuple[torch.Tensor, ...]:
        epoch = self._advance()
        home = route.plan.config.home_experts
        transfers = route.plan.transfers
        for item in transfers:
            if route.rank == item.target_rank:
                self._publish(3, route.rank, item.target_slot - home, item.owner_rank, epoch)
        if self._parallel_gradients:
            self._return_parallel(result, guests, route, epoch)
        else:
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

    def _return_parallel(self, result: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...],
                         route: ReplicaRoute, epoch: int) -> None:
        owned = sorted((item for item in route.plan.transfers if item.owner_rank == route.rank),
                       key=lambda item: item.target_rank)
        if not owned:
            return
        home = route.plan.config.home_experts
        with self._copy_streams(list(range(len(result))), self._gradient_streams,
                               result[0].device) as (backend, streams):
            for index, (output, guest) in enumerate(zip(result, guests)):
                stream = streams[index]
                with backend.stream(stream):
                    self._accumulate_projection(output, guest, owned, home, epoch, index, stream)
        # A single ACK releases every projection in a guest slot, so join before publishing it.
        for item in owned:
            self._publish(4, route.rank, item.target_slot - home, item.target_rank, epoch)

    def _accumulate_projection(self, output: torch.Tensor, guest: torch.Tensor, owned: list,
                               home: int, epoch: int, index: int, stream: Any) -> None:
        output.record_stream(stream)
        if index not in self._gradient_scratch:
            self._gradient_scratch[index] = torch.empty_like(output[0])
        scratch = self._gradient_scratch[index]
        for item in owned:
            slot = item.target_slot - home
            self._wait(3, item.target_rank, slot, epoch)
            self.runtime.get(scratch, guest[slot], item.target_rank, **self._copy_options)
            output[item.owner_slot].add_(scratch)
