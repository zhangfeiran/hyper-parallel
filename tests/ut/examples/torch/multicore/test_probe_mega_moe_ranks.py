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
"""Probe cleanup contracts using only short-lived owned CPU subprocesses."""

import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from examples.torch.multicore.probe_mega_moe_ranks import (
    _cleanup_workers,
    _process_start,
)


class TestRankProbeCleanup(unittest.TestCase):
    """Check separate worker groups and protect against stale PID records."""

    def test_cleanup_requires_matching_worker_start_time(self):
        for correct_start in (False, True):
            with (
                self.subTest(correct_start=correct_start),
                tempfile.TemporaryDirectory() as directory,
                subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
                ) as child,
            ):
                try:
                    start = _process_start(child.pid)
                    self.assertIsNotNone(start)
                    record = {"pid": child.pid, "process_start": start if correct_start else "stale"}
                    (Path(directory) / "rank0.json").write_text(json.dumps(record), encoding="utf-8")
                    _cleanup_workers(Path(directory))
                    if correct_start:
                        self.assertEqual(child.wait(timeout=5), -signal.SIGKILL)
                    else:
                        self.assertIsNone(child.poll())
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=5)
