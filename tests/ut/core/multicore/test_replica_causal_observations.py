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
"""Verify protocol identity, bounded capture, and independent clock semantics."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_parallel.core.expert_parallel.hot_replica._observer import (
    _get_replica_observer, _set_replica_observer,
)
from hyper_parallel.core.multicore.diagnostics.profiler.replica import ReplicaHostCapture
from hyper_parallel.core.multicore.diagnostics.profiler.cycle_trace import (
    CORE_HEADER, PROFILE_RECORD, _replica_causal_args,
    _calculate_event_driven_profile_layout,
)
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config
from tests.ut.core.multicore.test_mega_kernel_cycle_trace import _parse, _profile_buffer


class TestReplicaCausalObservations(unittest.TestCase):
    """Observe transitions without assuming remote clocks or changing execution."""

    def tearDown(self) -> None:
        """Release the exclusive bridge after every test, including failures."""
        _set_replica_observer(None)

    def test_all_protocol_events_keep_exact_identity_and_local_clock(self) -> None:
        """Decode both projections and retain enough identity for causal pairing."""
        for projection in (0, 1):
            for kind in range(1, 12):
                with self.subTest(projection=projection, kind=kind):
                    buffer = _profile_buffer()
                    PROFILE_RECORD.pack_into(buffer, CORE_HEADER.size, 110, 160,
                                             0x11000 + projection * 0x100 + kind,
                                             (63 << 14) | (23 << 8) | 61, 10003, (62 << 16) | 11)
                    trace = _parse(buffer, detailed_task_names=True)
                    event = next(item for item in trace["traceEvents"] if item["ph"] == "X")
                    args = event["args"]
                    self.assertEqual(args["epoch"], 10003)
                    self.assertEqual(args["band"], 61)
                    self.assertEqual(args["peer_ep_rank"], 63)
                    self.assertEqual(args["producer_core"], 23)
                    self.assertEqual(args["ep_local_rank"], 62)
                    self.assertEqual(args["physical_expert"], 11)
                    self.assertEqual(args["projection"], "W13" if projection else "W2")
                    self.assertEqual(args["clock_domain"], "npu_system_counter:rank3:device7")
                    self.assertNotIn("task_number", args)
                    self.assertEqual(args["producer_has_no_tiles"], kind == 9)
                    if kind == 6:
                        self.assertEqual(args["task_stage"], "ReplicaReturnsMerged")

    def test_worker_completion_has_a_distinct_boundary(self) -> None:
        """Completion must not be decoded as task zero's compute stage."""
        for descriptor, expected in ((0x1001E, "WorkerTasksComplete"), (0x1001F, "WorkerComplete")):
            trace = _parse(_profile_buffer(desc_id=descriptor))
            event = next(item for item in trace["traceEvents"] if item["ph"] == "X")
            self.assertEqual(event["name"], expected)
            self.assertEqual(event["args"]["task_stage"], expected)

    def test_projection_wide_sentinels_and_unknown_descriptors(self) -> None:
        """Do not invent an expert, producer core, or band for global completion."""
        record = {"desc_id": 0x11006, "task_id": (63 << 8) | 255,
                  "stage_task_index": 7, "owner_id": (2 << 16) | 65535}
        parsed = _replica_causal_args(record)
        for key in ("band", "producer_core", "physical_expert"):
            self.assertIsNone(parsed[key])
        for descriptor in (0x10008, 0x11201, 0x1100C):
            self.assertEqual(_replica_causal_args({**record, "desc_id": descriptor}), {})

    def test_ep64_worst_worker_storage_includes_both_projections(self) -> None:
        """Budget every head transition, return, probe, ACK, and final marker."""
        runtime = allocate_runtime_config(0)
        runtime.num_workers = 48
        copies = 63 * 6
        gradient = 2 * copies * (5 * 16 + 3) + 3
        producer = 2 * 12 * 64 + 2 * 6
        layout = _calculate_event_driven_profile_layout(
            runtime, replica_gradient_records=gradient, producer_records=producer,
        )
        self.assertGreaterEqual(layout.aiv_record_capacity, gradient + 2)
        self.assertGreaterEqual(layout.aic_record_capacity, producer + 2)

    def test_inactive_capture_never_reads_lazy_route(self) -> None:
        """Disabled capture must not inspect a histogram, transfer, or Tensor."""
        capture = ReplicaHostCapture(lambda: False, lambda: 7)
        self.assertIsNone(capture.begin_lease(object(), False))
        self.assertIsNone(capture.begin("disabled"))
        _set_replica_observer(capture)
        self.assertIsNone(_get_replica_observer())
        self.assertEqual(capture.take()["events"], [])

    def test_host_limit_dropped_count_and_nested_lease_restore(self) -> None:
        """Bound primitive metadata and restore the outer lease on exceptions."""
        capture = ReplicaHostCapture(lambda: True, lambda: 8, limit=4)
        edge = SimpleNamespace(logical_expert=3, owner_rank=0, owner_slot=3, target_rank=2, target_slot=6)
        route = SimpleNamespace(rank=0, plan=SimpleNamespace(
            transfers=(edge,), config=SimpleNamespace(ep_size=64, home_experts=6)))
        outer = capture.begin_lease(route, False, 123)
        with patch("hyper_parallel.core.multicore.diagnostics.profiler.replica.is_recomputing", return_value=True):
            inner = capture.begin_lease(route, False, 123)
        self.assertNotEqual(outer[1], inner[1])
        capture.end_lease(inner)
        self.assertEqual(capture.current_lease, outer[1])
        capture.end_lease(outer)
        self.assertIsNone(capture.current_lease)
        capture.event("overflow")
        facts = capture.take()
        self.assertEqual(len(facts["events"]), 4)
        self.assertEqual(facts["dropped_events"], 1)
        self.assertEqual(facts["events"][1]["metadata"]["phase"], "recompute")
        self.assertEqual(facts["events"][0]["metadata"]["transfers"][0]["logical_expert"], 3)
        self.assertIn("not SDMA completion", facts["scope"])
        self.assertEqual(capture.take()["events"], [])

    def test_bridge_visible_to_autograd_thread_without_sharing_lease(self) -> None:
        """Process visibility must not leak thread-local kernel identity."""
        capture = ReplicaHostCapture(lambda: True, lambda: 1)
        _set_replica_observer(capture)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            _set_replica_observer(capture)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(lambda: (_get_replica_observer() is capture, capture.current_lease)).result()
        self.assertEqual(result, (True, None))
        for limit in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                ReplicaHostCapture(lambda: True, lambda: 1, limit=limit)
