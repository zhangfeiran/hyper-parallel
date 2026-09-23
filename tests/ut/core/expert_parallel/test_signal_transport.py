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
"""Adversarial interleavings for direct symmetric expert slot ownership."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import random
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import (
    ExpertExecutionPlan, ExpertReplicaConfig, build_expert_replica_plan,
)
from hyper_parallel.core.expert_parallel.hot_replica.signal_transport import (
    SignalReplicaTransport, signal_storage_bytes,
)
from tests.common.mark_utils import arg_mark


def _cyclic_plan(ranks: int) -> ExpertExecutionPlan:
    """Give every rank an incoming guest and an outgoing replica in a ring."""
    counts = tuple(tuple(100 if source == expert else 0 for expert in range(ranks)) for source in range(ranks))
    slots = tuple((rank, (rank - 1) % ranks) for rank in range(ranks))
    dispatch = [[0] * (2 * ranks) for _ in range(ranks)]
    for rank in range(ranks):
        dispatch[rank][2 * rank] = 50
        dispatch[rank][2 * ((rank + 1) % ranks) + 1] = 50
    plan = ExpertExecutionPlan(ExpertReplicaConfig(ranks, ranks, 1), counts, slots,
                               tuple(tuple(row) for row in dispatch))
    plan.validate()
    return plan


class _World:
    """Execute independent device streams one operation at a time."""

    def __init__(self, ranks: int, size: int) -> None:
        """Create one symmetric allocation and device queue per simulated rank."""
        self.storage = [torch.empty(size, dtype=torch.uint8) for _ in range(ranks)]
        self.queues = [deque() for _ in range(ranks)]
        self.barriers = [0] * ranks
        self.current = list(range(ranks))

    def remote(self, tensor: torch.Tensor, source: int, target: int) -> torch.Tensor:
        """Translate a symmetric view by its byte offset, preserving shape/dtype."""
        offset = tensor.data_ptr() - self.storage[source].data_ptr()
        size = tensor.numel() * tensor.element_size()
        return self.storage[target][offset:offset + size].view(tensor.dtype).reshape(tensor.shape)

    def drain(self, seed: int) -> None:
        """Pick arbitrary runnable streams, rejecting a global wait cycle."""
        rng = random.Random(seed)
        while any(self.queues):
            ranks = list(range(len(self.queues)))
            rng.shuffle(ranks)
            progress = False
            for rank in ranks:
                queue = self.queues[rank]
                if not queue:
                    continue
                ready, action = queue[0]
                if ready():
                    queue.popleft()
                    action()
                    progress = True
                    break
            if not progress:
                raise AssertionError("all nonempty streams are waiting: protocol deadlock")


class _Runtime:
    """Queue ordinary RMA/signal operations without doing host synchronization."""

    def __init__(self, world: _World, rank: int, *, use_sdma: bool = False) -> None:
        """Select this rank's simulated stream."""
        self.world, self.rank = world, rank
        self.use_sdma = use_sdma

    def enqueue(self, action: Callable) -> None:
        """Preserve the calling device stream's operation order."""
        self.world.queues[self.world.current[self.rank]].append((lambda: True, action))

    def host_barrier(self) -> None:
        """Account for initialization before the simulator starts scheduling."""
        self.world.barriers[self.rank] += 1

    def put(self, destination: torch.Tensor, source: torch.Tensor, target: int, *, use_sdma: bool = False) -> None:
        """Enqueue a remote write whose source remains alive with the operation."""
        if use_sdma != self.use_sdma:
            raise AssertionError("the requested copy engine was not preserved")
        remote = self.world.remote(destination, self.rank, target)
        self.enqueue(lambda: remote.copy_(source))

    def get(self, destination: torch.Tensor, source: torch.Tensor, peer: int, *, use_sdma: bool = False) -> None:
        """Read remote data only after preceding publication waits complete."""
        if use_sdma != self.use_sdma:
            raise AssertionError("the requested copy engine was not preserved")
        remote = self.world.remote(source, self.rank, peer)
        self.enqueue(lambda: destination.copy_(remote))

    def signal(self, word: torch.Tensor, value: int, target: int) -> None:
        """Publish a generation to the corresponding remote cache line."""
        remote = self.world.remote(word, self.rank, target)
        self.enqueue(lambda: remote.fill_(value))

    def wait_signal(self, word: torch.Tensor, value: int, *, comparison: str) -> None:
        """Allow only monotonic, stream-ordered signal waits."""
        if comparison != "ge":
            raise AssertionError("epoch waits must tolerate a newer generation")
        self.world.queues[self.world.current[self.rank]].append((lambda: int(word[0]) >= value, lambda: None))


class _Stream:
    """Keep event waits in the simulated queue, rather than executing them on the host."""

    def __init__(self, world: _World, index: int) -> None:
        """Bind a simulated stream to one device queue."""
        self.world, self.index = world, index

    def wait_event(self, event: _Event) -> None:
        """Wait for the event record operation to execute on another queue."""
        self.world.queues[self.index].append((lambda: event.done, lambda: None))


class _Event:
    """Model a fresh event with a deferred recording operation."""

    def __init__(self) -> None:
        """Start with an event that no stream has completed."""
        self.done = False

    def record(self, stream: _Stream) -> None:
        """Publish completion only when all preceding stream operations have run."""
        stream.world.queues[stream.index].append((lambda: True, self._complete))

    def _complete(self):
        self.done = True


class _Backend:
    """Supply per-rank streams to the CPU protocol simulator."""

    Event = _Event

    def __init__(self, world: _World, rank: int) -> None:
        """Bind backend stream selection to a single rank."""
        self.world, self.rank = world, rank

    def current_stream(self, _device: torch.device) -> _Stream:
        """Return the caller's active queue."""
        return _Stream(self.world, self.world.current[self.rank])

    def Stream(self, *, device: torch.device) -> _Stream:  # pylint: disable=invalid-name
        """Allocate an independent copy queue using the Torch backend contract."""
        del device
        index = len(self.world.queues)
        self.world.queues.append(deque())
        return _Stream(self.world, index)

    @contextmanager
    def stream(self, stream: _Stream) -> Iterator[None]:
        """Switch the runtime's enqueue target only for this host scope."""
        previous = self.world.current[self.rank]
        self.world.current[self.rank] = stream.index
        try:
            yield
        finally:
            self.world.current[self.rank] = previous


@contextmanager
def _queued_math(runtime: _Runtime) -> Iterator[None]:
    """Defer result initialization and accumulation onto the active device queue."""
    add = torch.Tensor.add_

    def _clone(source, *args, **kwargs):
        del args, kwargs
        destination = torch.empty_like(source)
        runtime.enqueue(lambda: destination.copy_(source))
        return destination

    def _accumulate(destination, source):
        runtime.enqueue(lambda: add(destination, source))
        return destination

    with patch.object(torch.Tensor, "clone", _clone), patch.object(torch.Tensor, "add_", _accumulate):
        yield


def _guest_gradient(step: int, rank: int, projection: int) -> float:
    """Use cancellation-sensitive values to expose changes in peer accumulation order."""
    return (0.0, float(2**24), 1.0, -float(2**24))[rank] + step * 32 + projection


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestSignalReplicaTransport(unittest.TestCase):
    """Check changing owners with many calls queued before any device progress."""

    def test_projection_consumer_does_not_wait_for_the_other_matrix(self):
        """Hold the second DMA until the first guest GMM consumes its own ready matrix."""
        ranks = 4
        size = signal_storage_bytes(((2, 2), (2, 1)), 1, ranks, 2, projection_ready=True)
        for seed in range(24):
            world = _World(ranks, size)
            runtimes = [_Runtime(world, rank, use_sdma=True) for rank in range(ranks)]
            providers = [SignalReplicaTransport(runtime, storage, 1, ranks, use_sdma=True,
                                                parallel_prefetch=True, overlap_home=True, projection_ready=True)
                         for runtime, storage in zip(runtimes, world.storage)]
            checked = []
            for step in range(12):
                backward = bool(step % 2)
                first, second = (1, 0) if backward else (0, 1)
                consumed = [SimpleNamespace(done=False) for _ in range(ranks)]
                counts = [[100 if expert == step % ranks else 0 for expert in range(ranks)]] * ranks
                plan = _cyclic_plan(ranks) if step % 3 == 2 else build_expert_replica_plan(counts, 1)
                for rank, provider in enumerate(providers):
                    runtime = runtimes[rank]
                    weights = (torch.full((1, 2, 2), step * 10 + rank, dtype=torch.bfloat16),
                               torch.full((1, 2, 1), step * 10 + rank + 1, dtype=torch.bfloat16))

                    def _put(destination, source, target, *, use_sdma=False,
                             rank=rank, consumed=consumed, second=second):
                        self.assertTrue(use_sdma)
                        self.assertNotEqual(world.current[rank], rank)
                        remote = world.remote(destination, rank, target)
                        delayed = source.dtype == torch.bfloat16 and source.numel() == (4, 2)[second]
                        world.queues[world.current[rank]].append(
                            (lambda: not delayed or consumed[target].done, lambda: remote.copy_(source)))

                    route = SimpleNamespace(plan=plan, rank=rank)
                    with patch.object(torch, "cpu", _Backend(world, rank)), \
                            patch.object(torch.Tensor, "record_stream"), patch.object(runtime, "put", _put), \
                            provider.lease(weights, route, overlap=True, backward=backward) as pool:
                        self.assertIsNone(pool.weight_ready)
                        self.assertEqual(len(pool.projection_ready[0]), 2)
                        self.assertNotEqual(*pool.projection_ready[0])
                        owner = plan.slot_to_logical[rank][1]
                        pool.wait_weights(first)

                        def _consume_first(value=pool.weights[first], expected=step * 10 + owner + first,
                                           owner=owner, state=consumed[rank]):
                            if owner >= 0:
                                torch.testing.assert_close(value, torch.full_like(value, expected))
                                checked.append(True)
                            state.done = True

                        runtime.enqueue(_consume_first)
                        pool.wait_weights(first)
                        pool.wait_weights(second)
                        if owner >= 0:
                            def _consume_second(value=pool.weights[second], expected=step * 10 + owner + second):
                                torch.testing.assert_close(value, torch.full_like(value, expected))
                                checked.append(True)
                            runtime.enqueue(_consume_second)
            world.drain(seed)
            self.assertEqual(len(checked), 80)
            self.assertEqual(world.barriers, [1] * ranks)

    def test_projection_storage_and_rollover_keep_signals_disjoint(self):
        """Zero both publications on rollover without aliasing credit/ACK cache lines."""
        shapes = ((2, 2), (2, 1))
        base_size = signal_storage_bytes(shapes, 1, 2, 2)
        size = signal_storage_bytes(shapes, 1, 2, 2, projection_ready=True)
        self.assertEqual(size - base_size, 128)
        runtime = MagicMock()
        provider = SignalReplicaTransport(runtime, torch.empty(size, dtype=torch.uint8), 1, 2,
                                          use_sdma=True, parallel_prefetch=True, overlap_home=True,
                                          projection_ready=True)
        weights = (torch.ones(1, 2, 2, dtype=torch.bfloat16), torch.ones(1, 2, 1, dtype=torch.bfloat16))
        provider._bind(weights)  # pylint: disable=protected-access
        self.assertGreaterEqual(provider.projection_signals.data_ptr(),
                                provider.signals.data_ptr() + provider.signals.numel() * 4)
        provider.signals.fill_(17)
        provider.projection_signals.fill_(17)
        provider.epoch = 2**31 - 1
        self.assertEqual(provider._advance(), 1)  # pylint: disable=protected-access
        self.assertEqual(int(provider.signals.count_nonzero()), 0)
        self.assertEqual(int(provider.projection_signals.count_nonzero()), 0)
        self.assertEqual(runtime.host_barrier.call_count, 3)

    def test_home_work_can_precede_copy_completion(self):
        """Delay all DMA until home work runs, including ring routes and changing owners."""
        ranks = 4
        size = signal_storage_bytes(((2, 2), (2, 1)), 1, ranks, 2)
        for seed in range(24):
            world = _World(ranks, size)
            runtimes = [_Runtime(world, rank, use_sdma=True) for rank in range(ranks)]
            providers = [SignalReplicaTransport(runtime, storage, 1, ranks, use_sdma=True,
                                                parallel_prefetch=True, overlap_home=True)
                         for runtime, storage in zip(runtimes, world.storage)]
            checked = []
            for step in range(12):
                counts = [[100 if expert == step % ranks else 0 for expert in range(ranks)]] * ranks
                plan = _cyclic_plan(ranks) if step % 3 == 2 else build_expert_replica_plan(counts, 1)
                for rank, provider in enumerate(providers):
                    runtime = runtimes[rank]
                    home_done = SimpleNamespace(done=False)
                    weights = (torch.full((1, 2, 2), step * 10 + rank, dtype=torch.bfloat16),
                               torch.full((1, 2, 1), step * 10 + rank, dtype=torch.bfloat16))

                    def _put(destination, source, target, *, use_sdma=False,
                             rank=rank, home_done=home_done):
                        self.assertTrue(use_sdma)
                        self.assertNotEqual(world.current[rank], rank)
                        remote = world.remote(destination, rank, target)
                        world.queues[world.current[rank]].append(
                            (lambda: home_done.done, lambda: remote.copy_(source)))

                    route = SimpleNamespace(plan=plan, rank=rank)
                    signal, wait_signal = runtime.signal, runtime.wait_signal

                    def _signal(*args, signal=signal, rank=rank, **kwargs):
                        self.assertEqual(world.current[rank], rank)
                        signal(*args, **kwargs)

                    def _wait_signal(*args, wait_signal=wait_signal, rank=rank, **kwargs):
                        self.assertEqual(world.current[rank], rank)
                        wait_signal(*args, **kwargs)

                    with patch.object(torch, "cpu", _Backend(world, rank)), \
                            patch.object(torch.Tensor, "record_stream"), patch.object(runtime, "put", _put), \
                            patch.object(runtime, "signal", _signal), \
                            patch.object(runtime, "wait_signal", _wait_signal), \
                            provider.lease(weights, route, overlap=True) as pool:
                        runtime.enqueue(lambda home_done=home_done: setattr(home_done, "done", True))
                        self.assertIsNotNone(pool.weight_ready)
                        pool.wait_weights()
                        pool.wait_weights()
                        owner = plan.slot_to_logical[rank][1]
                        if owner >= 0:
                            def _consume(values=pool.weights, expected=step * 10 + owner):
                                for value in values:
                                    torch.testing.assert_close(value, torch.full_like(value, expected))
                                checked.append(True)
                            runtime.enqueue(_consume)
            world.drain(seed)
            self.assertEqual(len(checked), 40)
            self.assertEqual(world.barriers, [1] * ranks)

    def test_owner_rotation_never_overwrites_live_slots(self):
        """Credits protect a slot even when an unrelated next owner runs ahead."""
        ranks = 4
        size = signal_storage_bytes(((2, 2), (2, 1)), 1, ranks, 2)
        for seed in range(72):
            use_sdma = seed >= 24
            parallel = seed >= 48
            world = _World(ranks, size)
            runtimes = [_Runtime(world, rank, use_sdma=use_sdma) for rank in range(ranks)]
            providers = [SignalReplicaTransport(runtime, storage, 1, ranks, use_sdma=use_sdma,
                                                parallel_prefetch=parallel)
                         for runtime, storage in zip(runtimes, world.storage)]
            checked = []
            for step in range(12):
                counts = [[0] * ranks for _ in range(ranks)]
                for row in counts:
                    row[step % ranks] = 100
                plan = _cyclic_plan(ranks) if step % 3 == 2 else build_expert_replica_plan(counts, 1)
                for rank, provider in enumerate(providers):
                    weights = (torch.full((1, 2, 2), step * 10 + rank, dtype=torch.bfloat16),
                               torch.full((1, 2, 1), step * 10 + rank, dtype=torch.bfloat16))
                    route = SimpleNamespace(plan=plan, rank=rank)
                    if parallel:
                        values = tuple(weight.clone() for weight in weights)
                        for weight in weights:
                            weight.zero_()
                        def _prepare(weights=weights, values=values):
                            for weight, value in zip(weights, values):
                                weight.copy_(value)
                        runtimes[rank].enqueue(_prepare)
                    with patch.object(torch, "cpu", _Backend(world, rank)), \
                            patch.object(torch.Tensor, "record_stream"), provider.lease(weights, route) as pool:
                        owner = plan.slot_to_logical[rank][1]
                        expected = step * 10 + owner
                        if owner >= 0:
                            def _consume(values=pool.weights, expected=expected):
                                for value in values:
                                    torch.testing.assert_close(value, torch.full_like(value, expected))
                                checked.append(expected)
                            runtimes[rank].enqueue(_consume)
            world.drain(seed)
            self.assertEqual(len(checked), 40)
            self.assertEqual(len(world.queues), ranks * ranks if parallel else ranks)
            self.assertEqual(world.barriers, [1] * ranks)
            self.assertTrue(all(provider.epoch == 12 for provider in providers))

    def test_parallel_gradients_preserve_order_and_slot_reuse(self):
        """All matrix reads must finish before ACK allows a different owner to reuse the slot."""
        ranks = 4
        size = signal_storage_bytes(((2, 2), (2, 1)), 1, ranks, 2)
        for seed in range(24):
            world = _World(ranks, size)
            runtimes = [_Runtime(world, rank, use_sdma=True) for rank in range(ranks)]
            providers = [SignalReplicaTransport(runtime, storage, 1, ranks, use_sdma=True,
                                                parallel_prefetch=seed >= 12, parallel_gradients=True)
                         for runtime, storage in zip(runtimes, world.storage)]
            checked = []
            scratch_allocations = []
            empty_like = torch.empty_like

            def _allocate(tensor, **kwargs):
                result = empty_like(tensor, **kwargs)
                if tensor.ndim == 2:
                    scratch_allocations.append(result.data_ptr())
                return result

            for step in range(12):
                counts = [[100 if expert == step % ranks else 0 for expert in range(ranks)]] * ranks
                plan = _cyclic_plan(ranks) if step % 3 == 2 else build_expert_replica_plan(counts, 1)
                for rank, provider in enumerate(providers):
                    runtime = runtimes[rank]
                    route = SimpleNamespace(plan=plan, rank=rank)
                    weights = (torch.ones(1, 2, 2, dtype=torch.bfloat16),
                               torch.ones(1, 2, 1, dtype=torch.bfloat16))
                    home = tuple(torch.full_like(weight, step + 0.5, dtype=torch.float32) for weight in weights)
                    expected = tuple(value.clone() for value in home)
                    owned = sorted((item for item in plan.transfers if item.owner_rank == rank),
                                   key=lambda item: item.target_rank)
                    for item in owned:
                        for index, value in enumerate(expected):
                            value[item.owner_slot].add_(_guest_gradient(step, item.target_rank, index))
                    with patch.object(torch, "cpu", _Backend(world, rank)), \
                            patch.object(torch.Tensor, "record_stream"), patch.object(torch, "empty_like", _allocate), \
                            _queued_math(runtime), provider.lease(weights, route) as pool:
                        def _produce(guests=pool.gradients, step=step, rank=rank):
                            for index, guest in enumerate(guests):
                                guest.fill_(_guest_gradient(step, rank, index))
                        runtime.enqueue(_produce)
                        result = provider.return_gradients(home, pool.gradients, route)

                        def _consume(result=result, expected=expected):
                            for value, reference in zip(result, expected):
                                torch.testing.assert_close(value, reference, rtol=0, atol=0)
                            checked.append(True)
                        runtime.enqueue(_consume)
            world.drain(seed)
            self.assertEqual(len(checked), 48)
            self.assertEqual(len(scratch_allocations), 2 * ranks)
            self.assertEqual([provider.gradient_scratch_bytes for provider in providers], [24] * ranks)
            self.assertEqual(world.barriers, [1] * ranks)

    def test_parallel_gradients_without_remote_replicas_allocate_no_scratch(self):
        """Balanced routes without fan-in must not retain an unused expert-sized cache."""
        runtime = MagicMock()
        size = signal_storage_bytes(((2, 2),), 1, 2, 4)
        provider = SignalReplicaTransport(runtime, torch.empty(size, dtype=torch.uint8), 1, 2,
                                          use_sdma=True, parallel_gradients=True)
        route = SimpleNamespace(plan=build_expert_replica_plan([[1, 0], [0, 1]], 1), rank=0)
        weights = (torch.ones(1, 2, 2),)
        with provider.lease(weights, route) as pool:
            actual = provider.return_gradients(weights, pool.gradients, route)
        torch.testing.assert_close(actual[0], weights[0], rtol=0, atol=0)
        self.assertEqual(provider.gradient_scratch_bytes, 0)
        self.assertFalse(runtime.get.called)

    def test_gradient_get_acknowledges_after_all_projections(self):
        """The owner sums FP32 values and acknowledges only after both gets."""
        runtime = MagicMock()
        runtime.get.side_effect = lambda dst, _src, peer: dst.fill_(0.001 * peer)
        size = signal_storage_bytes(((2, 2), (2, 1)), 1, 4, 2)
        provider = SignalReplicaTransport(runtime, torch.empty(size, dtype=torch.uint8), 1, 4)
        weights = (torch.ones(1, 2, 2, dtype=torch.bfloat16), torch.ones(1, 2, 1, dtype=torch.bfloat16))
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        route = SimpleNamespace(plan=plan, rank=0)
        with provider.lease(weights, route, backward=True) as pool:
            runtime.reset_mock()
            home = tuple(value.float() for value in weights)
            result = provider.return_gradients(home, pool.gradients, route)
            for actual, expected in zip(result, home):
                torch.testing.assert_close(actual, expected + 0.006)
            calls = [call[0] for call in runtime.mock_calls]
            self.assertEqual(calls, ["wait_signal", "get", "get", "signal"] * 3)
            self.assertFalse(runtime.host_barrier.called)
            for value in pool.weights + pool.gradients:
                self.assertGreaterEqual(value.data_ptr(), provider.storage.data_ptr())
                self.assertLess(value.data_ptr(), provider.storage.data_ptr() + provider.storage.numel())

    def test_sdma_get_keeps_gradient_acknowledgement(self):
        """Selecting DMA preserves FP32 fan-in and acknowledgement after every read."""
        runtime = MagicMock()
        runtime.get.side_effect = lambda dst, _src, _peer, **_kw: dst.fill_(0.125)
        size = signal_storage_bytes(((2, 2),), 1, 2, 2)
        provider = SignalReplicaTransport(runtime, torch.empty(size, dtype=torch.uint8), 1, 2, use_sdma=True)
        plan = build_expert_replica_plan([[100, 0], [100, 0]], 1)
        route = SimpleNamespace(plan=plan, rank=0)
        weights = (torch.ones(1, 2, 2, dtype=torch.bfloat16),)
        with provider.lease(weights, route, backward=True) as pool:
            self.assertTrue(all(call.kwargs == {"use_sdma": True} for call in runtime.put.call_args_list))
            runtime.reset_mock()
            result = provider.return_gradients(weights, pool.gradients, route)
            torch.testing.assert_close(result[0], torch.full((1, 2, 2), 1.125))
            self.assertEqual([call[0] for call in runtime.mock_calls], ["wait_signal", "get", "signal"])
            self.assertEqual(runtime.get.call_args.kwargs, {"use_sdma": True})

    def test_parallel_prefetch_requires_sdma(self):
        """Reject a parallel copy request that cannot use the selected runtime engine."""
        runtime = MagicMock()
        with self.assertRaisesRegex(ValueError, "requires SDMA"):
            SignalReplicaTransport(runtime, torch.empty(64, dtype=torch.uint8), 1, 2, parallel_prefetch=True)
        with self.assertRaisesRegex(ValueError, "requires SDMA"):
            SignalReplicaTransport(runtime, torch.empty(64, dtype=torch.uint8), 1, 2, parallel_gradients=True)
        with self.assertRaisesRegex(ValueError, "requires parallel SDMA"):
            SignalReplicaTransport(runtime, torch.empty(64, dtype=torch.uint8), 1, 2,
                                   use_sdma=True, overlap_home=True)
        with self.assertRaisesRegex(ValueError, "requires home overlap"):
            SignalReplicaTransport(runtime, torch.empty(64, dtype=torch.uint8), 1, 2, projection_ready=True)
        self.assertFalse(runtime.host_barrier.called)

    def test_invalid_layout_fails_before_collectives(self):
        """Reject malformed public sizing arguments and insufficient storage early."""
        for shapes, slots, size, element in ((((),), 1, 2, 2), (((2, 2),), True, 2, 2),
                                            (((2, 2),), 1, 1.5, 2), (((2, 2),), 1, 2, 1),
                                            (((2, 0),), 1, 2, 2)):
            with self.assertRaises(ValueError):
                signal_storage_bytes(shapes, slots, size, element)
        runtime = MagicMock()
        provider = SignalReplicaTransport(runtime, torch.empty(64, dtype=torch.uint8), 1, 2)
        route = SimpleNamespace(plan=build_expert_replica_plan([[1, 0], [0, 1]], 1), rank=0)
        with self.assertRaisesRegex(ValueError, "too small"):
            with provider.lease((torch.ones(1, 2, 2),), route):
                self.fail("undersized storage unexpectedly accepted")
        self.assertFalse(runtime.host_barrier.called)

    def test_rollover_resets_only_after_quiescence(self):
        """A bounded integer epoch resets collectively instead of wrapping stale signals."""
        runtime = MagicMock()
        size = signal_storage_bytes(((2, 2),), 1, 2, 2)
        provider = SignalReplicaTransport(runtime, torch.empty(size, dtype=torch.uint8), 1, 2)
        provider.epoch = 2**31 - 1
        route = SimpleNamespace(plan=build_expert_replica_plan([[1, 0], [0, 1]], 1), rank=0)
        with provider.lease((torch.ones(1, 2, 2, dtype=torch.bfloat16),), route):
            self.assertEqual(provider.epoch, 1)
            self.assertEqual(runtime.host_barrier.call_count, 3)
            self.assertEqual(int(provider.signals.count_nonzero()), 0)
            with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                with provider.lease((torch.ones(1, 2, 2, dtype=torch.bfloat16),), route):
                    self.fail("nested lease unexpectedly succeeded")
