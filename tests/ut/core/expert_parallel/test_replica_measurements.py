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
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import signal_transport
from hyper_parallel.core.expert_parallel.hot_replica import routing
from hyper_parallel.core.multicore.modules.mega_moe import function
from tests.common.mark_utils import arg_mark
from tests.torch.expert_parallel.hot_replica_measurements import HostMeasurements, _payload_bytes


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestReplicaMeasurements(unittest.TestCase):
    """Pool tensor views must not each claim the whole SHMEM allocation."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_count_observation_preserves_flags_and_adds_no_synchronization(self) -> None:
        """
        Feature: Scoped validation/count observation
        Description: Observe strided invalid and duplicate routes with device events and synchronization forbidden.
        Expectation: Preserve independent int64 payloads, restore operations, and nest only count subintervals.
        """
        ids = torch.tensor([[0, 0], [-1, 4], [1, 3]]).T.contiguous().T
        sort = torch.Tensor.sort
        with (patch.object(torch.npu, "Event", side_effect=AssertionError("Unexpected device event")),
              patch.object(torch.npu, "synchronize", side_effect=AssertionError("Unexpected synchronization"))):
            with HostMeasurements(device_intervals=False, count_details=True) as host:
                result = routing._replica_count_payload(ids, 4)
                with host.span("outside_counts"):
                    ids.sort()
        torch.testing.assert_close(result, torch.tensor([3, 1, 0, 2, 1], dtype=torch.int64))
        self.assertIs(torch.Tensor.sort, sort)
        children = [record for record in host.records if record["stage"].startswith("count_")]
        self.assertTrue(children)
        self.assertTrue(all(record["parent"] == "validation_and_counts" for record in children))
        self.assertTrue(all("stream_interval_ms" not in record for record in host.records))
        self.assertEqual(sum(record["stage"] == "count_topk_sort" for record in children), 1)

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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_shared_weight_gradient_backing_and_padding(self) -> None:
        """
        Feature: Shared storage accounting
        Description: Count BF16 weights and FP32 gradients sharing padded backing storage.
        Expectation: Report tensor payload bytes independently from the full shared storage.
        """
        backing = torch.empty(128, dtype=torch.uint8)
        weights = (backing[:16].view(torch.bfloat16), backing[16:32].view(torch.bfloat16))
        gradients = (backing[32:64].view(torch.float32), backing[64:96].view(torch.float32))
        self.assertEqual(weights[0].untyped_storage().nbytes(), 128)
        self.assertEqual(_payload_bytes(weights), 32)
        self.assertEqual(_payload_bytes(gradients), 64)
        self.assertEqual(_payload_bytes(weights + gradients), 96)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_duplicate_tensor_views_are_not_counted_twice(self) -> None:
        """
        Feature: Repeated tensor view accounting
        Description: Count duplicate and disjoint slices from one tensor's backing storage.
        Expectation: Count repeated slices once while retaining disjoint payloads.
        """
        values = torch.empty(24, dtype=torch.float32)
        self.assertEqual(_payload_bytes((values[:8], values[:8], values[8:16])), 64)
        self.assertEqual(_payload_bytes(()), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_prefetch_observes_async_epoch_and_actual_saved_generation_reuse(self) -> None:
        """
        Feature: Prefetch diagnostics after saved-forward reuse
        Description: Observe fresh and reused views through an async epoch uploader without device events.
        Expectation: Report the provider's actual generation match and preserve upload bytes without synchronization.
        """
        token = object()
        current = [None]
        data = bytes((17, 0, 0, 1))

        @contextmanager
        def _prefetch(*_args, **_options):
            signal_transport._async_device_bytes(data, torch.device("cpu"))
            yield SimpleNamespace(weight_generation=current[0])

        weights = (torch.empty(1), torch.empty(1))
        with (patch.object(function, "prefetch_weights", _prefetch),
              patch.object(torch.npu, "Event", side_effect=AssertionError("Unexpected device event")),
              patch.object(torch.npu, "synchronize", side_effect=AssertionError("Unexpected synchronization"))):
            with HostMeasurements(device_intervals=False, prefetch_details=True) as host:
                for generation in (object(), token, None):
                    current[0] = generation
                    with function.prefetch_weights(weights, object(), backward=True, reuse_generation=token):
                        pass
        prefetches = [record for record in host.records if record["stage"] == "weight_prefetch_enqueue"]
        self.assertEqual([record["weights_reused"] for record in prefetches], [False, True, False])
        self.assertEqual(sum(record["stage"] == "prefetch_epoch_upload" for record in host.records), 3)
        self.assertTrue(all("stream_interval_ms" not in record for record in host.records))
