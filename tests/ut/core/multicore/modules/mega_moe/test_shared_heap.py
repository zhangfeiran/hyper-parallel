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
"""MoE-specific layout preservation in the generic frozen root registry."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.shared_heap import (
    FixedMegaMoeHeap,
    reserve_mega_moe_consumer,
)
from hyper_parallel.core.multicore.modules.mega_moe.workspace import MegaMoeWorkspace
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture


class TestSharedMegaMoeHeap(SharedRootFixture):
    """Require a declared no-growth shape while sharing root leases across modules."""

    def _module(self, **kwargs):
        module = MegaMoeExperts(local_num_tokens=128, hidden_size=128, intermediate_size=128,
                                num_experts=2, top_k=2, create_parameters=False, **kwargs)
        self.addCleanup(module.close)
        return module

    def test_shared_layers_reserve_one_consumer_after_grouping(self):
        """Serial layers keep one allocation layout and independent parameter ownership."""
        modules = (self._module(dispatch_mode="pull"), self._module(dispatch_mode="pull"))
        MegaMoeExperts.share_execution_resources(modules)
        for module in modules:
            module.bind_shmem_root(self.root, "moe-layers", dtype=torch.bfloat16)
        self.assertEqual(len(self.root.consumers), 3)
        self.assertIs(modules[0]._resource_group.shmem_binding, modules[1]._resource_group.shmem_binding)

    def test_grouping_after_reservation_and_rebinding_rejected(self):
        """A prepared byte declaration cannot silently change execution resource ownership."""
        first, second = self._module(dispatch_mode="pull"), self._module(dispatch_mode="pull")
        first.bind_shmem_root(self.root, "moe-layer", dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "before reserving"):
            MegaMoeExperts.share_execution_resources((first, second))
        with self.assertRaisesRegex(RuntimeError, "different shared"):
            first.bind_shmem_root(self.root, "other", dtype=torch.bfloat16)

    def test_push_growth_rejected_before_any_registry_mutation(self):
        """Default geometric push growth is unsuitable for first-version coexistence."""
        module = self._module(ep_size=2, initial_capacity_factor=1.25)
        self.root.members = (0, 1)
        with patch("hyper_parallel.core.multicore.modules.mega_moe.shared_heap.root_members", return_value=(0, 1)), \
             self.assertRaisesRegex(ValueError, "lossless receive capacity"):
            module.bind_shmem_root(self.root, "moe-push", dtype=torch.bfloat16)
        self.assertNotIn("moe-push", self.root.consumers)

    def test_full_bound_push_reservation_and_fixed_overflow(self):
        """Preallocation admits push without changing the legacy layout or enabling rebuild."""
        module = self._module(ep_size=2, initial_capacity_factor=2)
        self.root.members = (0, 1)
        with patch("hyper_parallel.core.multicore.modules.mega_moe.shared_heap.root_members", return_value=(0, 1)):
            consumer = reserve_mega_moe_consumer(self.root, "moe-push", module._resource_group.specification,
                                                torch.bfloat16)
        resource = SimpleNamespace(workspace=SimpleNamespace(capacity_floor=512))
        FixedMegaMoeHeap.ensure_capacity(resource, 512)
        with self.assertRaisesRegex(RuntimeError, "fixed"):
            FixedMegaMoeHeap.ensure_capacity(resource, 513)
        self.assertEqual(consumer.specification.kind, "mega_moe")

    def test_workspace_lease_orders_root_and_other_consumers(self):
        """The unchanged MoE buffers now claim the same root-wide lease as DSA."""
        consumer = self.consumers[0]
        workspace = MegaMoeWorkspace(shared=False)
        workspace.heap_manager = FixedMegaMoeHeap(consumer)
        workspace.root_consumer = consumer
        workspace.completion_event = Mock()
        workspace.device = torch.device("npu:0")
        self.consumers[1].bind()
        workspace.claim()
        self.assertIs(self.root.lease_owner, consumer)
        with self.assertRaisesRegex(RuntimeError, "concurrent consumer"):
            self.consumers[1].claim()
        workspace.release()
        self.assertIsNone(self.root.lease_owner)
        workspace.wait_for_reuse()
        self.stream.wait_event.assert_called()
