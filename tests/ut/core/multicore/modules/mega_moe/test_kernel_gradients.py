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
"""W2 metadata ordering, ownership and fallback tests without device execution."""

import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import build_expert_replica_plan
from hyper_parallel.core.multicore.modules.mega_moe.kernel_gradients import prepare_kernel_gradient_return


class TestKernelGradientReturn(unittest.TestCase):
    """Retain per-call completion storage and encode only a verified ready dependency."""

    def test_metadata_matches_physical_slots_and_ordered_peers(self):
        """Every ready event belongs to its physical expert; records follow FP32 peer order."""
        placement = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        plan = SimpleNamespace(replica_w2_events=(32, 64), spec=SimpleNamespace(num_cube_cores=20))
        retained = []
        for rank in range(4):
            route = SimpleNamespace(plan=placement, rank=rank)
            provider = SimpleNamespace(kernel_gradients=True,
                                       kernel_gradient_signals=Mock(return_value=(7, 4096, 8192)))
            gradient, guest = torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)
            with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream") as record:
                result = prepare_kernel_gradient_return(plan, route, provider, gradient, guest)
            retained.append(result)
            self.assertEqual(record.call_count, 2)
            provider.kernel_gradient_signals.assert_called_once_with()
            raw = bytes(result.metadata.tolist())
            values = struct.unpack("<" + "Q" * (len(raw) // 8), raw)
            incoming = [t for t in placement.transfers if t.target_rank == rank]
            owned = sorted((t for t in placement.transfers if t.owner_rank == rank), key=lambda t: t.target_rank)
            self.assertEqual(values[:12], (1, rank, 1, 7, 32, gradient.data_ptr(), guest.data_ptr(),
                                          4096, 8192, result.completion.data_ptr(), len(incoming), len(owned)))
            expected = []
            for transfer in incoming:
                expected.extend((transfer.owner_rank, transfer.target_slot - 1, 64))
            for transfer in owned:
                expected.extend((transfer.target_rank, transfer.target_slot - 1, transfer.owner_slot, 32))
            self.assertEqual(values[12:], tuple(expected))
            torch.testing.assert_close(result.completion, torch.zeros(20, 16, dtype=torch.int32))
        self.assertEqual(len({r.completion.data_ptr() for r in retained}), 4)

    def test_unknown_schedule_or_disabled_provider_does_not_advance_epoch(self):
        """Fallback must not consume protocol state or allocate kernel metadata."""
        placement = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        route = SimpleNamespace(plan=placement, rank=0)
        provider = SimpleNamespace(kernel_gradients=True, kernel_gradient_signals=Mock())
        plan = SimpleNamespace(replica_w2_events=())
        gradient, guest = torch.ones(1, 4, 8), torch.ones(1, 4, 8)
        self.assertIsNone(prepare_kernel_gradient_return(plan, route, provider, gradient, guest))
        plan.replica_w2_events = (32, 64)
        provider.kernel_gradients = False
        self.assertIsNone(prepare_kernel_gradient_return(plan, route, provider, gradient, guest))
        provider.kernel_gradient_signals.assert_not_called()

    def test_invalid_owner_buffer_fails_before_protocol_advance(self):
        """The fused path cannot silently cast or mutate an autograd-owned gradient."""
        placement = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        route = SimpleNamespace(plan=placement, rank=0)
        plan = SimpleNamespace(replica_w2_events=(32, 64))
        provider = SimpleNamespace(kernel_gradients=True, kernel_gradient_signals=Mock())
        for gradient in (torch.ones(1, 4, 8, dtype=torch.bfloat16), torch.ones(1, 4, 8, requires_grad=True),
                         torch.ones(1, 8, 4).transpose(1, 2)):
            with self.assertRaisesRegex(ValueError, "detached contiguous FP32"):
                prepare_kernel_gradient_return(plan, route, provider, gradient, torch.ones(1, 4, 8))
        provider.kernel_gradient_signals.assert_not_called()
