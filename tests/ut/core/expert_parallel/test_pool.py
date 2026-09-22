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
"""Guest storage ownership, reuse and stream ordering tests."""

import gc
import unittest
from unittest.mock import MagicMock, patch
import weakref

import torch

from hyper_parallel.core.expert_parallel.hot_replica import build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.pool import ReplicaPool, replica_pool
from tests.common.mark_utils import arg_mark


class _Group:
    """Weak-referenceable process-group stand-in."""


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestReplicaPool(unittest.TestCase):
    """Exercise the pool independently of accelerator and collective setup."""

    def test_layer_reuse_and_group_lifetime(self):
        """Different layers reuse B storage without pinning parameters or groups."""
        group = _Group()
        owner = torch.ones(6, 8, 16)
        owner_ref = weakref.ref(owner)
        pool = replica_pool((owner,), 1, group)
        self.assertEqual(pool.weights[0].shape, (1, 8, 16))
        self.assertIs(pool, replica_pool((owner.clone(),), 1, group))
        self.assertIsNot(pool, replica_pool((owner,), 2, group))
        del owner
        self.assertIsNone(owner_ref())
        pool_ref = weakref.ref(pool)
        del pool, group
        gc.collect()
        self.assertIsNone(pool_ref())

    def test_lease_reuse_and_exception_cleanup(self):
        """Nested claims fail; failures release ownership; backward zeros guests."""
        pool = ReplicaPool((torch.ones(6, 8, 16),), 2)
        with self.assertRaisesRegex(ValueError, "consumer"):
            with pool.lease(backward=True):
                pool.gradients[0].fill_(12)
                with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                    with pool.lease():
                        self.fail("nested claim unexpectedly succeeded")
                raise ValueError("consumer failed")
        previous = pool.gradients[0]
        with pool.lease(backward=True):
            self.assertIs(pool.gradients[0], previous)
            torch.testing.assert_close(previous, torch.zeros_like(previous))
            self.assertEqual(previous.dtype, torch.float32)

    def test_stream_waits_before_reuse(self):
        """A later stream must wait for the previous lease's final consumer."""
        device = torch.device("cuda", 0)
        tensor = MagicMock(device=device)
        backend = MagicMock()
        streams = [MagicMock(), MagicMock()]
        backend.current_stream.side_effect = streams
        pool = ReplicaPool((torch.ones(1, 1, 1),), 1)
        pool.weights = (tensor,)
        with patch.object(torch, "cuda", backend):
            with pool.lease():
                pass
            event = backend.Event.return_value
            with pool.lease():
                streams[1].wait_event.assert_called_once_with(event)
        self.assertEqual(tensor.record_stream.call_count, 2)
        self.assertEqual(event.record.call_count, 2)

    def test_stream_lookup_failure_releases_host_lock(self):
        """An unavailable device stream cannot permanently strand the pool lock."""
        pool = ReplicaPool((torch.ones(1, 1, 1),), 1)
        original = pool.weights
        pool.weights = (MagicMock(device=torch.device("cuda", 0)),)
        with patch.object(torch.cuda, "current_stream", side_effect=RuntimeError("device unavailable")):
            with self.assertRaisesRegex(RuntimeError, "device unavailable"):
                with pool.lease():
                    self.fail("stream lookup unexpectedly succeeded")
        pool.weights = original
        with pool.lease():
            self.assertIs(pool.weights, original)

    def test_source_affinity_is_optimal_for_fixed_placement(self):
        """Local row count reaches the sum of per-rank supply/quota minima."""
        counts = [[60, 0, 0, 0], [0, 0, 0, 0], [50, 0, 0, 0], [90, 0, 0, 0]]
        plan = build_expert_replica_plan(counts, 1)
        width = plan.config.slots_per_rank
        actual = 0
        optimum = 0
        for rank, slots in enumerate(plan.slot_to_logical):
            for slot, expert in enumerate(slots):
                if expert < 0:
                    continue
                physical = rank * width + slot
                quota = sum(row[physical] for row in plan.dispatch_counts)
                optimum += min(counts[rank][expert], quota)
                actual += plan.dispatch_counts[rank][physical]
        self.assertEqual(actual, optimum)
        self.assertEqual(actual, 150)
