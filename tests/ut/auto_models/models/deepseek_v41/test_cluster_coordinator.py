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
"""Check peer failure propagation and cleanup using disposable CPU processes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.training_demo.deepseek_v41 import coordinate_deepseek_v41_cluster as coordinator


class TestCaseCoordinator(unittest.TestCase):
    """A failed peer must not leave another node blocked inside training collectives."""

    def test_peer_failure_stops_workers_and_allows_next_case(self) -> None:
        original_popen = subprocess.Popen

        def spawn(command: list, **kwargs: object) -> subprocess.Popen:
            """Replace the hardware sampler while keeping real worker processes."""
            if 'examples.training_demo.deepseek_v41.monitor_deepseek_v41_hbm' in command:
                command = [sys.executable, '-c', 'import time; time.sleep(60)']
            return original_popen(command, **kwargs)

        with TemporaryDirectory() as directory:
            peers = [Path(directory) / str(rank) for rank in range(2)]
            commands = [[sys.executable, '-c', 'import time; time.sleep(0.5); raise SystemExit(7)'],
                        [sys.executable, '-c', 'import time; time.sleep(60)']]
            with patch.object(coordinator, '_snapshot'), patch.object(coordinator.subprocess, 'Popen', spawn):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(coordinator.run_case, command, folder, peers, 1, 1, 5)
                               for command, folder in zip(commands, peers)]
                    records = [future.result(timeout=20) for future in futures]
            self.assertEqual(records[0], records[1])
            self.assertTrue(all(record['cleanup_ok'] for record in records[0]))
            self.assertEqual(records[0][0]['training_exit_status'], 7)
            self.assertTrue(records[0][1]['peer_failed'])
            for folder in peers:
                group = int((folder / 'launcher.pid').read_text())
                self.assertFalse(coordinator._group_alive(group))

    def test_cleanup_removes_only_the_case_process_group(self) -> None:
        with subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                              start_new_session=True) as foreign, subprocess.Popen(
                [sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True) as owned:
            try:
                self.assertTrue(coordinator._stop_group(owned.pid))
                owned.wait(timeout=5)
                self.assertIsNone(foreign.poll())
            finally:
                foreign.terminate()
                foreign.wait(timeout=5)
                if owned.poll() is None:
                    owned.kill()
                    owned.wait(timeout=5)

    def test_missing_or_unclean_peer_cannot_advance(self) -> None:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'Timed out'):
                coordinator.wait_for_cleanup([folder], 0.01)
            coordinator.write_json(folder / 'case_complete.json', {'cleanup_ok': False})
            with self.assertRaisesRegex(RuntimeError, 'did not clean up'):
                coordinator.wait_for_cleanup([folder], 1)

    def test_orphan_workers_are_cleaned_after_launcher_exit(self) -> None:
        with TemporaryDirectory() as directory:
            child_file = Path(directory) / 'child.pid'
            command = [sys.executable, '-c',
                       'import pathlib, subprocess, sys; '
                       'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); '
                       'pathlib.Path(sys.argv[1]).write_text(str(child.pid))', str(child_file)]
            with subprocess.Popen(command, start_new_session=True, stdout=subprocess.DEVNULL) as launcher:
                try:
                    self.assertEqual(launcher.wait(timeout=5), 0)
                    self.assertTrue(child_file.exists())
                    self.assertTrue(coordinator._group_alive(launcher.pid))
                    self.assertTrue(coordinator._stop_group(launcher.pid))
                    self.assertFalse(coordinator._group_alive(launcher.pid))
                finally:
                    coordinator._stop_group(launcher.pid)
