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
"""One-sided publication, consumption and FP32 fan-in contracts."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import torch

from hyper_parallel.core.expert_parallel.hot_replica import build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.one_sided import OneSidedReplicaTransport
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestOneSidedReplicaTransport(unittest.TestCase):
    """Use deterministic remote memory contents without a device runtime."""

    def test_owner_publishes_only_requested_slices(self):
        """No full home copy; each guest is published before the acknowledgement."""
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        runtime = MagicMock()
        provider = OneSidedReplicaTransport(runtime, torch.empty(16, dtype=torch.uint8))
        home = torch.full((1, 2, 2), 2.0, dtype=torch.bfloat16)
        provider.prefetch((home,), (torch.empty_like(home),), SimpleNamespace(plan=plan, rank=0))
        self.assertEqual(runtime.put.call_count, 3)
        self.assertEqual(runtime.host_barrier.call_count, 2)
        for call in runtime.put.call_args_list:
            self.assertEqual(call.args[1].data_ptr(), home.data_ptr())
            self.assertEqual(call.args[1].numel(), 4)

    def test_guest_waits_for_publication_then_acknowledges_copy(self):
        """The producer may reuse its inbox only after the guest copy consumes it."""
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        inbox = torch.empty(16, dtype=torch.uint8)
        guest = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
        calls = []

        def barrier() -> None:
            """Emulate producer publication and check the consumer acknowledgement."""
            if not calls:
                inbox[:8].view(torch.bfloat16).fill_(7)
            else:
                torch.testing.assert_close(guest, torch.full_like(guest, 7))
            calls.append("barrier")

        runtime = SimpleNamespace(host_barrier=barrier)
        provider = OneSidedReplicaTransport(runtime, inbox)
        provider.prefetch((torch.empty_like(guest),), (guest,), SimpleNamespace(plan=plan, rank=1))
        self.assertEqual(len(calls), 2)

    def test_owner_pulls_fp32_without_remote_atomics(self):
        """High fan-in uses one B inbox and preserves small FP32 increments."""
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        runtime = MagicMock()
        runtime.get.side_effect = lambda dst, _src, _rank: dst.fill_(0.001)
        provider = OneSidedReplicaTransport(runtime, torch.empty(16, dtype=torch.uint8))
        home = torch.ones(1, 2, 2, dtype=torch.float32)
        result, = provider.return_gradients((home,), (torch.zeros_like(home),), SimpleNamespace(plan=plan, rank=0))
        torch.testing.assert_close(result, home + 0.003)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(runtime.get.call_count, 3)
        self.assertEqual(runtime.host_barrier.call_count, 6)
        self.assertFalse(runtime.put.called)
        torch.testing.assert_close(home, torch.ones_like(home))

    def test_owned_fp32_return_reuses_the_callers_storage(self):
        """The owned path sums into fresh home storage and leaves guest gradients intact."""
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        runtime = MagicMock()
        runtime.get.side_effect = lambda dst, _src, _rank: dst.fill_(0.125)
        provider = OneSidedReplicaTransport(runtime, torch.empty(16, dtype=torch.uint8))
        home, guest = torch.ones(1, 2, 2), torch.full((1, 2, 2), 7.0)
        result, = provider.return_gradients_owned((home,), (guest,), SimpleNamespace(plan=plan, rank=0))
        self.assertIs(result, home)
        torch.testing.assert_close(result, torch.full_like(home, 1.375))
        torch.testing.assert_close(guest, torch.full_like(guest, 7.0))
        self.assertEqual(runtime.get.call_count, 3)
        self.assertEqual(runtime.host_barrier.call_count, 6)

    def test_inbox_size_is_checked_before_publication(self):
        """Reject an undersized externally supplied inbox before starting a put."""
        runtime = MagicMock()
        provider = OneSidedReplicaTransport(runtime, torch.empty(1, dtype=torch.uint8))
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        with self.assertRaisesRegex(ValueError, "too small"):
            provider.prefetch((torch.ones(1, 2, 2),), (torch.ones(1, 2, 2),),
                              SimpleNamespace(plan=plan, rank=0))
        self.assertFalse(runtime.put.called)
