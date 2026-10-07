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
"""Keep payload accounting separate from shared storage reservations."""

import unittest
from unittest.mock import patch

import torch

from tests.common.mark_utils import arg_mark
from tests.torch.expert_parallel.hot_replica_measurements import HostMeasurements, _payload_bytes


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestReplicaMeasurements(unittest.TestCase):
    """Pool tensor views must not each claim the whole SHMEM allocation."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_light_host_observation_adds_no_device_events_or_synchronize(self):
        """
        Feature: Host-only observation
        Description: Observe a span with NPU events and synchronization forbidden.
        Expectation: Record host time without device events or synchronization.
        """
        with patch.object(torch.npu, "Event", side_effect=AssertionError("Unexpected device event")), \
                patch.object(torch.npu, "synchronize", side_effect=AssertionError("Unexpected synchronization")):
            with HostMeasurements(device_intervals=False) as host:
                with host.span("host_only"):
                    pass
        self.assertEqual(len(host.records), 1)
        self.assertEqual(host.records[0]["stage"], "host_only")
        self.assertGreaterEqual(host.records[0]["host_ms"], 0)
        self.assertNotIn("stream_interval_ms", host.records[0])
        self.assertNotIn("events", host.records[0])

    def test_shared_weight_gradient_backing_and_padding(self):
        """Count BF16/FP32 payload independently of common padded storage."""
        backing = torch.empty(128, dtype=torch.uint8)
        weights = (backing[:16].view(torch.bfloat16), backing[16:32].view(torch.bfloat16))
        gradients = (backing[32:64].view(torch.float32), backing[64:96].view(torch.float32))
        self.assertEqual(weights[0].untyped_storage().nbytes(), 128)
        self.assertEqual(_payload_bytes(weights), 32)
        self.assertEqual(_payload_bytes(gradients), 64)
        self.assertEqual(_payload_bytes(weights + gradients), 96)

    def test_duplicate_tensor_views_are_not_counted_twice(self):
        """Repeated references count once while disjoint projections both count."""
        values = torch.empty(24, dtype=torch.float32)
        self.assertEqual(_payload_bytes((values[:8], values[:8], values[8:16])), 64)
        self.assertEqual(_payload_bytes(()), 0)
