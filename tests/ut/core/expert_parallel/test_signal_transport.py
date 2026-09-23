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

from collections import deque
from collections.abc import Callable
import random
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

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
        self.world.queues[self.rank].append((lambda: True, action))

    def host_barrier(self) -> None:
        """Account for initialization before the simulator starts scheduling."""
        self.world.barriers[self.rank] += 1

    def put(self, destination: torch.Tensor, source: torch.Tensor, target: int, *, use_sdma: bool = False) -> None:
        """Enqueue a remote write whose source remains alive with the operation."""
        if use_sdma != self.use_sdma:
            raise AssertionError("the requested copy engine was not preserved")
        remote = self.world.remote(destination, self.rank, target)
        self.enqueue(lambda: remote.copy_(source))

    def signal(self, word: torch.Tensor, value: int, target: int) -> None:
        """Publish a generation to the corresponding remote cache line."""
        remote = self.world.remote(word, self.rank, target)
        self.enqueue(lambda: remote.fill_(value))

    def wait_signal(self, word: torch.Tensor, value: int, *, comparison: str) -> None:
        """Allow only monotonic, stream-ordered signal waits."""
        if comparison != "ge":
            raise AssertionError("epoch waits must tolerate a newer generation")
        self.world.queues[self.rank].append((lambda: int(word[0]) >= value, lambda: None))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestSignalReplicaTransport(unittest.TestCase):
    """Check changing owners with many calls queued before any device progress."""

    def test_owner_rotation_never_overwrites_live_slots(self):
        """Credits protect a slot even when an unrelated next owner runs ahead."""
        ranks = 4
        size = signal_storage_bytes(((2, 2), (2, 1)), 1, ranks, 2)
        for seed in range(48):
            use_sdma = seed >= 24
            world = _World(ranks, size)
            runtimes = [_Runtime(world, rank, use_sdma=use_sdma) for rank in range(ranks)]
            providers = [SignalReplicaTransport(runtime, storage, 1, ranks, use_sdma=use_sdma)
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
                    with provider.lease(weights, route) as pool:
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
            self.assertEqual(world.barriers, [1] * ranks)
            self.assertTrue(all(provider.epoch == 12 for provider in providers))

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
