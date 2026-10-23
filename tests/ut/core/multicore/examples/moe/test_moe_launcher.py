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

"""Exercise MoE launcher device selection without starting distributed workers."""

import json
import os
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.common.mark_utils import arg_mark


class TestMoeLauncher(unittest.TestCase):
    """Record the actual shell launcher arguments with a device-free interpreter."""

    def setUp(self) -> None:
        """Provide empty activation scripts and a recording Python executable."""
        self.root = Path(__file__).resolve().parents[6]
        resources = ExitStack()
        self.addCleanup(resources.close)
        self.directory = Path(resources.enter_context(tempfile.TemporaryDirectory()))
        self.capture = self.directory / "capture.json"
        env_file = self.directory / "set_env.sh"
        env_file.write_text("", encoding="utf-8")
        interpreter = self.directory / "python"
        interpreter.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
if sys.argv[1] == '-c':
    print(os.environ['HP_LAUNCHER_TEST_IMPORT'])
else:
    Path(os.environ['HP_LAUNCHER_TEST_CAPTURE']).write_text(json.dumps({
        'args': sys.argv[1:], 'cwd': os.getcwd(), 'visible': os.environ['ASCEND_RT_VISIBLE_DEVICES']}))
''', encoding="utf-8")
        interpreter.chmod(0o755)
        self.env = dict(os.environ)
        for name in ("HP_ROOT", "NNODES", "NPROC_PER_NODE", "NODE_RANK", "ASCEND_RT_VISIBLE_DEVICES"):
            self.env.pop(name, None)
        self.env.update(PYTHON_BIN=str(interpreter), CANN_ENV_FILE=str(env_file), NATIVE_ENV_FILE=str(env_file),
                        SHMEM_LIB_DIR=str(self.directory), HP_LAUNCHER_TEST_CAPTURE=str(self.capture),
                        HP_LAUNCHER_TEST_IMPORT=str(self.root / "hyper_parallel/__init__.py"))

    def _run(self, overrides: dict[str, str]) -> subprocess.CompletedProcess:
        """Execute environment and argument assembly without a real framework process."""
        self.capture.unlink(missing_ok=True)
        return subprocess.run(
            ["bash", str(self.root / "hyper_parallel/core/multicore/examples/moe/run_benchmark.sh"),
             "--dispatch-mode", "pull"],
            env={**self.env, **overrides}, cwd=self.directory, capture_output=True, text=True, check=False,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_ep16_and_explicit_device_allocations(self) -> None:
        """Feature: Configurable launcher topology.
        Description: Run default EP8, automatic EP16 and an explicitly reordered multi-node allocation.
        Expectation: The real shell exports matching devices, preserves arguments and uses the checkout root.
        """
        for overrides, ranks, visible in (
            ({}, 8, ",".join(map(str, range(8)))),
            ({"NPROC_PER_NODE": "16"}, 16, ",".join(map(str, range(16)))),
            ({"NNODES": "2", "NODE_RANK": "1", "NPROC_PER_NODE": "4",
              "ASCEND_RT_VISIBLE_DEVICES": "7,5,3,1"}, 4, "7,5,3,1"),
        ):
            with self.subTest(overrides=overrides):
                result = self._run(overrides)
                self.assertEqual(result.returncode, 0, msg=f"Launcher failed: {result.stderr}")
                captured = json.loads(self.capture.read_text(encoding="utf-8"))
                self.assertEqual(captured["visible"], visible)
                self.assertIn(f"--nproc_per_node={ranks}", captured["args"])
                self.assertIn(f"--nnodes={overrides.get('NNODES', '1')}", captured["args"])
                self.assertIn(f"--node_rank={overrides.get('NODE_RANK', '0')}", captured["args"])
                self.assertEqual(captured["args"][-2:], ["--dispatch-mode", "pull"])
                self.assertEqual(captured["cwd"], str(self.root))


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_device_allocations_fail_before_worker_launch(self) -> None:
        """Feature: Explicit device allocation validation.
        Description: Request too few, duplicate or malformed logical device identifiers.
        Expectation: Each invalid allocation fails before the recording worker is invoked.
        """
        for devices, message in (("0,1,2,3,4,5,6,7", "fewer devices"),
                                 (",".join(["0"] * 16), "duplicate"), ("0,1,", "comma-separated"),
                                 ("0,00", "comma-separated")):
            with self.subTest(devices=devices):
                result = self._run({"NPROC_PER_NODE": "16", "ASCEND_RT_VISIBLE_DEVICES": devices})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.capture.exists())
