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
"""CPU-native mocks for shared budgets, ownership, collective identity and leases."""

import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.shmem import _api, _lifecycle
from hyper_parallel.core.multicore.shmem.consumer import (
    SharedShmemRoot,
    ShmemConsumerSpec,
    allocation_budget,
)


class SharedRootFixture(unittest.TestCase):
    """Isolate Python SHMEM lifecycle with CPU-native allocation and mocked NPU events."""

    def setUp(self) -> None:
        """Prepare isolated lifecycle state and fake NPU stream events."""
        self.addCleanup(patch.stopall)
        for name, value in (("_users", 0), ("_registry_token", None), ("_heap_generation", 0),
                            ("_shutdown_failed", False), ("_root_group", None), ("_root_uses_distributed", None),
                            ("_root_size", None), ("_root_ranks", None)):
            patch.object(_lifecycle, name, value).start()
        self.native = Mock()
        self.created = {}
        self.native._empty.side_effect = self._allocate
        self.dist = Mock()
        self.dist.is_initialized.return_value = False
        patch.object(_lifecycle, "_torch_modules", return_value=(torch, self.dist)).start()
        patch.object(_lifecycle, "_load_native", return_value=self.native).start()
        patch.object(_api, "_load_native", return_value=self.native).start()
        patch.object(_api, "_torch_modules", return_value=(torch, self.dist)).start()
        patch("hyper_parallel.core.multicore.shmem.consumer.dist.is_initialized", return_value=False).start()
        self.event = patch.object(torch.npu, "Event").start().return_value
        self.stream = patch.object(torch.npu, "current_stream").start().return_value
        patch.object(torch.npu, "current_device", return_value=0).start()
        patch.object(torch.npu, "synchronize").start()
        patch.object(torch.npu, "is_current_stream_capturing", return_value=False).start()
        self.root = SharedShmemRoot("npu:0")
        self.consumers = [self.root.reserve(ShmemConsumerSpec(name, name, 4096, (name,))) for name in ("moe", "dsa")]
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        if self.root.lease_owner is not None:
            self.root.release_lease(self.root.lease_owner)
        for consumer in self.consumers:
            if consumer.bound:
                with consumer.access():
                    for pointer in tuple(consumer.allocations):
                        self.native._free.side_effect = None
                        shmem.free(self.created[pointer])
                consumer.close()
        self.root.close()

    def _allocate(self, shape, dtype, _alignment):
        tensor = torch.empty(shape, dtype=dtype)
        self.created[tensor.data_ptr()] = tensor
        return tensor


class TestSharedShmemRoot(SharedRootFixture):
    """Run the real Python lifecycle and registry against CPU-native allocations."""

    def test_complete_registry_freezes_before_one_native_initialize(self):
        """Two consumers share one reference and include both byte budgets before initialization."""
        for consumer in self.consumers:
            consumer.bind()
        self.native._initialize.assert_called_once_with(0, 1, b"", 2 * 1024**2)
        self.assertEqual(_lifecycle._reference_count(), 1)
        self.assertEqual(self.root.generation, 1)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            self.root.reserve(ShmemConsumerSpec("late", "dsa", 128, ()))

    def test_unmanaged_owner_cannot_join_or_release(self):
        """Bare lifecycle calls cannot alter the shared registry's sole native reference."""
        self.consumers[0].bind()
        with self.assertRaisesRegex(RuntimeError, "registered consumers"):
            shmem.acquire()
        with self.assertRaisesRegex(RuntimeError, "only the managed shared root"):
            shmem.release()
        self.assertEqual(_lifecycle._reference_count(), 1)

    def test_unscoped_allocations_and_communication_rejected(self):
        """A consumer scope is required for allocation, and an execution lease for one-sided work."""
        consumer = self.consumers[0]
        consumer.bind()
        with self.assertRaisesRegex(RuntimeError, "scope or lease"):
            shmem.empty((16,), dtype=torch.uint8)
        with consumer.access(), self.assertRaisesRegex(RuntimeError, "execution lease"):
            shmem.get(torch.empty(1), torch.empty(1), 0)
        self.native._empty.assert_not_called()
        self.native._get.assert_not_called()

    def test_allocation_bill_and_cross_consumer_free(self):
        """Budget accounting follows native success and retains failed frees for retry."""
        first, second = self.consumers
        first.bind()
        second.bind()
        with first.access():
            tensor = shmem.empty((8,), dtype=torch.float32, alignment=512)
        bill = allocation_budget((8,), torch.float32, 512)
        self.assertEqual(first.allocated_bytes, bill)
        with second.access(), self.assertRaisesRegex(RuntimeError, "not owned"):
            shmem.free(tensor)
        self.native._free.side_effect = RuntimeError("free failed")
        with first.access(), self.assertRaisesRegex(RuntimeError, "free failed"):
            shmem.free(tensor)
        self.assertEqual(first.allocated_bytes, bill)
        self.native._free.side_effect = None
        with first.access():
            shmem.free(tensor)
        self.assertEqual(first.allocated_bytes, 0)

    def test_per_consumer_overflow_precedes_native_allocation(self):
        """A large global heap cannot hide overflow of an individual consumer's budget."""
        consumer = self.consumers[0]
        consumer.bind()
        with consumer.access(), self.assertRaisesRegex(RuntimeError, "declared byte budget"):
            shmem.empty((4096,), dtype=torch.uint8, alignment=512)
        self.native._empty.assert_not_called()

    def test_serial_root_lease_and_cross_stream_completion(self):
        """Another consumer is rejected until release, then waits on the previous event."""
        first, second = self.consumers
        first.bind()
        second.bind()
        self.assertEqual(first.claim(), 1)
        with self.assertRaisesRegex(RuntimeError, "concurrent consumer"):
            second.claim()
        with self.assertRaisesRegex(RuntimeError, "already active"):
            first.claim()
        first.release()
        self.event.record.assert_called_once_with(self.stream)
        second.claim()
        self.stream.wait_event.assert_called_with(self.event)
        second.release()

    def test_active_lease_forbids_allocation_free_and_close(self):
        """No allocation can be destroyed or introduced while native work may use the arena."""
        consumer = self.consumers[0]
        consumer.bind()
        with consumer.access():
            tensor = shmem.empty((8,), dtype=torch.float32)
        consumer.claim()
        with self.assertRaisesRegex(RuntimeError, "before claiming"):
            shmem.empty((8,), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "active execution lease"):
            shmem.free(tensor)
        with self.assertRaisesRegex(RuntimeError, "no active lease"):
            consumer.close()
        consumer.release()
        with consumer.access():
            shmem.free(tensor)

    def test_rebuild_and_stale_generation_rejected(self):
        """Frozen shared roots cannot be reinitialized by any legacy heap manager."""
        consumer = self.consumers[0]
        consumer.bind()
        with self.assertRaisesRegex(RuntimeError, "cross-consumer"):
            _lifecycle._reinitialize(4 * 1024**2)
        self.native._shutdown.assert_not_called()
        with patch.object(_lifecycle, "_heap_generation", 2), self.assertRaisesRegex(RuntimeError, "generation"):
            consumer.claim()

    def test_close_requires_closed_consumers_then_releases_once(self):
        """Root close owns final collective shutdown and is idempotent."""
        for consumer in self.consumers:
            consumer.bind()
        with self.assertRaisesRegex(RuntimeError, "all consumers closed"):
            self.root.close()
        for consumer in self.consumers:
            consumer.close()
        self.root.close()
        self.root.close()
        self.native._shutdown.assert_called_once()
        self.assertEqual(_lifecycle._reference_count(), 0)
        self.assertIsNone(_lifecycle._registry_token)

    def test_insufficient_whole_root_budget_rejected_before_native(self):
        """Aggregate byte reservations must fit the whole-root budget, regardless of allocation timing."""
        self.root.heap_size_bytes = 4096
        with self.assertRaisesRegex(RuntimeError, "too small"):
            self.consumers[0].bind()
        self.native._initialize.assert_not_called()
        self.assertEqual(self.root.state, "planning")

    def test_peer_declaration_difference_rejected(self):
        """Consumer order/shape differences converge before vendor initialization."""
        self.root.members = (0, 1)
        def _disagree(peers, value, **_):
            peers[:] = [value, (("different layout",), None)]
        with patch("hyper_parallel.core.multicore.shmem.consumer.dist.all_gather_object", side_effect=_disagree), \
             self.assertRaisesRegex(RuntimeError, "differs across ranks"):
            self.consumers[0].bind()
        self.native._initialize.assert_not_called()

    def test_environment_limit_and_invalid_declarations(self):
        """Honor the existing heap environment and reject invalid byte/page declarations."""
        with patch.dict("os.environ", {"HYPER_PARALLEL_SHMEM_HEAP_SIZE": str(4 * 1024**2)}):
            self.assertEqual(SharedShmemRoot("npu:0").heap_size_bytes, 4 * 1024**2)
        for size in (0, -1, True, 4096):
            with self.subTest(size=size), self.assertRaises(ValueError):
                SharedShmemRoot("npu:0", heap_size_bytes=size)
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.root.reserve(ShmemConsumerSpec("moe", "moe", 32, ()))

    def test_device_or_mutable_layout_declarations_rejected(self):
        """Object collectives carry only immutable host metadata, never device tensors."""
        for layout in ((torch.ones(1),), ([1, 2],), ({"offset": 1},)):
            with self.subTest(layout=layout), self.assertRaisesRegex(TypeError, "host declaration"):
                ShmemConsumerSpec("invalid", "dsa", 32, layout)

    def test_current_device_drift_rejected_before_claim(self):
        """An executor cannot record root events or use addresses on another current device."""
        consumer = self.consumers[0]
        consumer.bind()
        with patch.object(torch.npu, "current_device", return_value=1), \
             self.assertRaisesRegex(RuntimeError, "prepared NPU"):
            consumer.claim()
        self.assertIsNone(self.root.lease_owner)
