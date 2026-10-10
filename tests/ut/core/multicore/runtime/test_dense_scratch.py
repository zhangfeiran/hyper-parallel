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
"""CPU protocol coverage for bounded stream-owned dense kernel scratch."""

import gc
import unittest
from dataclasses import dataclass

import torch

from hyper_parallel.core.multicore.runtime.dense_scratch import DenseScratchPool, shared_dense_scratch
from tests.common.mark_utils import arg_mark


@dataclass(frozen=True)
class _Stream:
    """Separate wrappers that compare equal for one simulated native stream."""

    identity: int


class _Owner:
    """Weakly registerable executable stand-in without framework initialization."""


class TestDenseScratch(unittest.TestCase):
    """Exercise storage ownership without treating CPU execution as device ordering proof."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_reuse_refreshes_addresses_and_joins(self):
        """Feature: Repeated launches on one stream.
        Description: Reuse completed scratch with different tensor addresses and stale event/overflow words.
        Expectation: Storage stays stable while every invocation receives current pointers and reset joins.
        """
        pool = DenseScratchPool(torch.device("cpu"), 3, 32, 64)
        with pool.lease(_Stream(1)) as first:
            first.prepare((101, 102, 103))
            first.events.fill_(7)
            first.overflow.fill_(9)
            storage = tuple(value.data_ptr() for value in
                            (first.pointers, first.events, first.workspace, first.overflow))
        with pool.lease(_Stream(1)) as second:
            second.prepare((201, 202, 203))
            self.assertIs(first, second)
            self.assertEqual(tuple(value.data_ptr() for value in
                                   (second.pointers, second.events, second.workspace, second.overflow)), storage)
            self.assertEqual(second.pointers.tolist(), [201, 202, 203])
            self.assertEqual(torch.count_nonzero(second.events).item(), 0)
            self.assertEqual(torch.count_nonzero(second.overflow).item(), 0)
        self.assertEqual(pool.statistics()["allocations"], 1)
        self.assertEqual(pool.statistics()["reuses"], 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_streams_and_overlapping_host_calls_keep_private_storage(self):
        """Feature: Concurrent scratch ownership.
        Description: Hold a lease while another stream and another host invocation on the same stream launch.
        Expectation: Neither invocation can reset or overwrite the held lease's metadata.
        """
        pool = DenseScratchPool(torch.device("cpu"), 3, 32, 64)
        with pool.lease(_Stream(1)) as first:
            first.prepare((101, 102, 103))
            with pool.lease(_Stream(2)) as other, pool.lease(_Stream(1)) as overlapping:
                other.prepare((201, 202, 203))
                overlapping.prepare((301, 302, 303))
                self.assertEqual(first.pointers.tolist(), [101, 102, 103])
                self.assertEqual(len({first.workspace.data_ptr(), other.workspace.data_ptr(),
                                      overlapping.workspace.data_ptr()}), 3)
            self.assertEqual(first.pointers.tolist(), [101, 102, 103])
        with pool.lease(_Stream(1)) as reused:
            self.assertIs(reused, first)
        self.assertEqual(pool.statistics()["cached_streams"], 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_extra_streams_do_not_grow_persistent_cache(self):
        """Feature: Bounded stream cache.
        Description: Use many distinct stream identities after filling a one-slot cache.
        Expectation: Calls remain supported while persistent bytes stay at the configured bound.
        """
        pool = DenseScratchPool(torch.device("cpu"), 3, 32, 64, max_cached_streams=1)
        with pool.lease(_Stream(1)) as cached:
            cached.prepare((101, 102, 103))
        for identity in range(2, 10):
            with pool.lease(_Stream(identity)) as private:
                private.prepare((identity, identity + 1, identity + 2))
                self.assertIsNot(private, cached)
        statistics = pool.statistics()
        self.assertEqual(statistics["cached_streams"], 1)
        self.assertEqual(statistics["cached_bytes"], statistics["bytes_per_slot"])
        with pool.lease(_Stream(1)) as reused:
            self.assertIs(reused, cached)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_close_preserves_held_lease_and_rejects_new_calls(self):
        """Feature: Close during a retained launch lease.
        Description: Close the cache while one invocation still owns its buffers.
        Expectation: Held storage remains usable, cached bytes are released and new leases fail.
        """
        pool = DenseScratchPool(torch.device("cpu"), 3, 32, 64)
        with pool.lease(_Stream(1)) as held:
            held.prepare((101, 102, 103))
            pool.close()
            held.prepare((201, 202, 203))
            self.assertEqual(held.pointers.tolist(), [201, 202, 203])
            self.assertEqual(pool.statistics()["cached_bytes"], 0)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                with pool.lease(_Stream(2)):
                    pass
        pool.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_failed_enqueue_releases_the_host_lease(self):
        """Feature: Failed launch cleanup.
        Description: Raise after preparing a cached lease, then submit another invocation.
        Expectation: The next call reuses and refreshes that slot without retaining a busy host reservation.
        """
        pool = DenseScratchPool(torch.device("cpu"), 3, 32, 64)
        with self.assertRaisesRegex(RuntimeError, "synthetic launch failure"):
            with pool.lease(_Stream(1)) as first:
                first.prepare((101, 102, 103))
                raise RuntimeError("synthetic launch failure")
        with pool.lease(_Stream(1)) as second:
            second.prepare((201, 202, 203))
            self.assertIs(first, second)
            self.assertEqual(second.pointers.tolist(), [201, 202, 203])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_private_policy_and_pointer_count_admission(self):
        """Feature: Private scratch and descriptor bounds.
        Description: Disable caching and submit an address list inconsistent with the descriptor.
        Expectation: Private calls stay isolated and mismatched pointers are rejected before enqueue.
        """
        pool = DenseScratchPool(torch.device("cpu"), 3, 32, 64, max_cached_streams=0)
        with pool.lease(_Stream(1)) as first, pool.lease(_Stream(1)) as second:
            self.assertIsNot(first, second)
            with self.assertRaisesRegex(ValueError, "pointer count"):
                first.prepare((101,))
        self.assertEqual(pool.statistics()["cached_bytes"], 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_shared_shapes_grow_without_invalidating_queued_storage(self):
        """Feature: Shared scratch across layers and shapes.
        Description: Grow descriptor storage, then return to a small shape while holding old Tensor references.
        Expectation: SDK workspace is reused, old storage survives and views retain exact descriptor lengths.
        """
        owners = (_Owner(), _Owner())
        pool = shared_dense_scratch(torch.device("cpu"), 3, 32, 256, owners[0])
        peer = shared_dense_scratch(torch.device("cpu"), 6, 64, 256, owners[1])
        try:
            self.assertIs(pool, peer)
            with pool.lease(_Stream(1), pointers=3, event_elements=32) as first:
                first.prepare((101, 102, 103))
                first.events.fill_(7)
                old_pointers, old_events = first.pointers, first.events
                workspace = first.workspace.data_ptr()
            with peer.lease(_Stream(1), pointers=6, event_elements=64) as large:
                large.prepare((201, 202, 203, 204, 205, 206))
                self.assertEqual(large.workspace.data_ptr(), workspace)
                self.assertEqual(old_pointers.tolist(), [101, 102, 103])
                self.assertEqual(old_events.tolist(), [7] * 32)
            high_watermark = pool.statistics()["cached_bytes"]
            with pool.lease(_Stream(1), pointers=3, event_elements=32) as small:
                small.prepare((301, 302, 303))
                self.assertEqual(small.pointers.numel(), 3)
                self.assertEqual(small.events.numel(), 32)
                self.assertEqual(small.workspace.data_ptr(), workspace)
            self.assertEqual(pool.statistics()["cached_bytes"], high_watermark)
            self.assertEqual(pool.statistics()["allocations"], 1)
            pool.release_owner(owners[0])
            with peer.lease(_Stream(1), pointers=6, event_elements=64) as continued:
                continued.prepare((401, 402, 403, 404, 405, 406))
                self.assertEqual(continued.workspace.data_ptr(), workspace)
        finally:
            for owner in owners:
                pool.release_owner(owner)
        self.assertEqual(pool.statistics()["cached_bytes"], 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with peer.lease(_Stream(1)):
                pass

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_dead_owner_does_not_keep_shared_cache_alive(self):
        """Feature: Weak executable registration.
        Description: Drop one owner without close, then explicitly close the last peer.
        Expectation: A dead executable cannot keep persistent scratch allocated.
        """
        first, second = _Owner(), _Owner()
        pool = shared_dense_scratch(torch.device("cpu"), 3, 32, 512, first)
        peer = shared_dense_scratch(torch.device("cpu"), 3, 32, 512, second)
        try:
            with pool.lease(_Stream(1)) as buffers:
                buffers.prepare((101, 102, 103))
            del first
            gc.collect()
            peer.release_owner(second)
            self.assertEqual(pool.statistics()["cached_bytes"], 0)
            replacement_owner = _Owner()
            replacement = shared_dense_scratch(torch.device("cpu"), 3, 32, 512, replacement_owner)
            self.assertIsNot(replacement, pool)
            replacement.release_owner(replacement_owner)
        finally:
            pool.close()
