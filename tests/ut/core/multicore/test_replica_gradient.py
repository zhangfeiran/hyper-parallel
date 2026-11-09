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
"""Compile actual mapped and UDMA FP32 gradient copies with guarded host buffers."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.common.mark_utils import arg_mark
from tests.ut.core.multicore.test_gradient_return_schedule import _definition


class TestReplicaGradient(unittest.TestCase):
    """Preserve ordered FP32 results, tails and neighboring elements for each architecture."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_fp32_return_tiles_keep_complete_coverage(self) -> None:
        """Feature: Bounded FP32 replica return arithmetic.
        Description: Compile actual native loops for mapped and staged copies, tails and ordered peers.
        Expectation: Every element is assigned once per peer, results remain exact, and buffers stay bounded.
        """
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("The host gradient copy test requires g++")
        repository = Path(__file__).resolve().parents[4]
        source = repository / "hyper_parallel/core/multicore/ops/runtime/replica_gradient.h"
        function = _definition(source.read_text(), "__aicore__ inline void AddReplicaGradient(")
        udma = (source.parent / "replica_gradient_udma.h").read_text()
        function += "\n" + _definition(udma, "__aicore__ inline uint32_t ReplicaGradientChunkSize")
        function += "\n" + _definition(udma, "__aicore__ inline void ValidateReplicaGradientUdmaLayout")
        function += "\n" + _definition(udma, "__aicore__ inline void AddReplicaGradientUdma(")
        fixture = Path(__file__).with_name("replica_gradient.cpp")
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "native_gradient.h").write_text(function + "\n")
            for architecture, udma_mode, tile in ((2201, 0, 4096), (3510, 0, 4096), (3510, 1, 8192)):
                with self.subTest(architecture=architecture, udma=udma_mode):
                    binary = work / str(architecture)
                    build = subprocess.run(
                        [compiler, "-std=c++17", "-O0", "-fno-fast-math", "-ffp-contract=off",
                         f"-D__NPU_ARCH__={architecture}", f"-DHP_EXPECTED_TILE={tile}",
                         f"-DHP_UDMA={udma_mode}",
                         "-I", str(work), str(fixture), "-o", str(binary)],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(build.returncode, 0, f"expected compile=0 actual={build.stderr}")
                    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0,
                                     f"expected exact coverage/order actual={result.stdout} {result.stderr}")
                    self.assertIn("Verified 11 guarded ordered FP32 gradient shapes", result.stdout,
                                  f"expected 11 complete shapes actual={result.stdout}")
