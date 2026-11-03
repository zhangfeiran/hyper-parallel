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
"""Compile the actual ready publisher with observable device primitive stubs."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.ut.core.multicore.test_gradient_return_schedule import _definition


class TestGradientBandProducer(unittest.TestCase):
    """Check final-output fence, per-core empty bands, and the disabled path."""

    def test_ready_is_published_after_the_real_fence(self) -> None:
        """An empty core contribution is distinct from a completed nonempty band."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("The host producer test requires g++")
        runtime = Path(__file__).resolve().parents[4] / "hyper_parallel/core/multicore/ops/runtime"
        encoding = (runtime / "replica_trace.h").read_text()
        source = (runtime / "gradient_band.h").read_text()
        native = _definition(encoding, "enum class ReplicaTraceEvent") + ";\n"
        native += _definition(encoding, "__aicore__ inline void RecordReplica") + "\n"
        native += _definition(source, "class GradientBandProducer") + ";\n"
        fixture = Path(__file__).with_name("gradient_band_producer.cpp")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "native_producer.h").write_text(native)
            binary = work / "producer"
            build = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(work), str(fixture),
                                    "-o", str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Verified producer fence, empty contribution, and disabled observations", result.stdout)
