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
"""Precision and attribution guards for the fixed-workload timeline analyzer."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.torch.multicore import analyze_mega_moe_gradient_timeline as timeline


class TestGradientTimelineAnalysis(unittest.TestCase):
    """Do not round epoch timestamps or associate work from another thread."""

    def test_decimal_epoch_timestamps_keep_nanosecond_differences(self) -> None:
        """A 20 ns event separation must survive a microsecond timestamp near the current epoch."""
        before = timeline._nanoseconds("1788799322106487.140")
        after = timeline._nanoseconds("1788799322106487.160")
        self.assertEqual(before, 1788799322106487140)
        self.assertEqual(after - before, 20)
        self.assertEqual(timeline._nanoseconds(0.02), 20)

    def test_scope_containment_requires_exact_process_and_thread(self) -> None:
        """Nearby timestamps alone cannot attribute an asynchronous device launch."""
        scope = {"pid": 7, "tid": 8, "start_ns": 100, "end_ns": 200}
        for pid, tid, timestamp, expected in (
            (7, 8, 100, True), (7, 8, 200, True), (7, 8, 99, False), (7, 8, 201, False),
            (9, 8, 150, False), (7, 9, 150, False),
        ):
            with self.subTest(pid=pid, tid=tid, timestamp=timestamp):
                self.assertEqual(timeline._contains(scope, {"pid": pid, "tid": tid, "start_ns": timestamp}), expected)

    def test_fixed_workload_analyzer_rejects_incomplete_or_reordered_ranks(self) -> None:
        """The per-layer neighborhood comparison requires the complete fixed benchmark contract."""
        for layers, ranks in ((8, range(8)), (16, range(7)), (16, reversed(range(8)))):
            with self.subTest(layers=layers, ranks=str(ranks)):
                payload = {"configuration": {"layers": layers},
                           "diagnostic_trace": {"per_rank": [{"rank": rank} for rank in ranks]}}
                with patch.object(sys, "argv", ["analyzer", "result.json"]), \
                        patch.object(Path, "read_text", return_value=json.dumps(payload)), \
                        self.assertRaisesRegex(ValueError, "sixteen layers and all eight ranks"):
                    timeline.main()

    def test_stream_dependencies_separate_receive_wait_from_caller_join(self) -> None:
        """Only independent receives may use a different stream, with an explicit owner join."""
        scopes = {name: {'pid': 7, 'tid': 8, 'start_ns': start, 'end_ns': end}
                  for name, start, end in (('reduction', 100, 400), ('launch', 120, 150), ('wait', 200, 220))}
        native = {'pid': 9, 'tid': 47}
        for independent in (False, True):
            with self.subTest(independent=independent):
                stream = 44 if independent else 47
                record = {'pid': 9, 'tid': stream, 'name': 'EVENT_RECORD'}
                wait = {'pid': 9, 'tid': stream, 'name': 'EVENT_WAIT'}
                join = {'pid': 9, 'tid': 47, 'name': 'EVENT_WAIT'}
                joined = [(record, {'pid': 7, 'tid': 8, 'start_ns': 130}),
                          (wait, {'pid': 7, 'tid': 8, 'start_ns': 210})]
                if independent:
                    joined.append((join, {'pid': 7, 'tid': 8, 'start_ns': 300}))
                records, barriers, caller = timeline._communication_events(
                    joined, native, scopes, receive_only=independent,
                )
                self.assertEqual(records, [record])
                self.assertEqual(barriers, [wait])
                self.assertEqual(caller, [join] if independent else [wait])
                if independent:
                    with self.assertRaisesRegex(ValueError, 'producer dependency'):
                        timeline._communication_events(joined, native, scopes, receive_only=False)
                    with self.assertRaisesRegex(ValueError, 'caller-stream join'):
                        timeline._communication_events(joined[:-1], native, scopes, receive_only=True)

    def test_stream_order_requires_native_and_communication_before_owner(self) -> None:
        """Early receive readiness must not relax native production or final owner dependencies."""
        row = {'independent_receive': True, 'native_end_ns': 100, 'ready_ns': 20,
               'p2p_wait_start_ns': 30, 'p2p_wait_end_ns': 150,
               'wait_start_ns': 110, 'wait_end_ns': 160, 'owner_start_ns': 170}
        timeline._validate_stream_order(row)
        for key, value in (('independent_receive', False), ('p2p_wait_start_ns', 10),
                           ('wait_start_ns', 90), ('wait_end_ns', 140), ('owner_start_ns', 155)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                timeline._validate_stream_order(row | {key: value})
