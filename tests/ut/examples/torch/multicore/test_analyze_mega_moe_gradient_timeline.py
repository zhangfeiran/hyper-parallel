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
