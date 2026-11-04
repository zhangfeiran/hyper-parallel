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
"""Compile real token-pull functions against a CPU transfer and credit model."""

from contextlib import ExitStack
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from tests.common.mark_utils import arg_mark


_ROOT = Path(__file__).resolve().parents[4]
_OPS = _ROOT / "hyper_parallel/core/multicore/ops"


class TestIndexedPull(unittest.TestCase):
    """Check data, logical DMA calls, UB reuse and credits without NPU hardware."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compile the unchanged production function bodies once per test class."""
        with ExitStack() as resources:
            directory = Path(resources.enter_context(tempfile.TemporaryDirectory()))
            cls.addClassCleanup(resources.pop_all().close)
        source = (_OPS / "runtime/indexed_pull.h").read_text()
        begin = source.index("  template <bool Count>\n  __aicore__ inline void AddInputTileInterval")
        end = source.index("  template <typename T>\n  __aicore__ inline void ExecuteTokenPull", begin)
        fragment = source[begin:end]
        tile_header = (_OPS / "bf16_gmm/tile_ready.h").read_text()
        matched = re.search(r"INPUT_TILE_ROWS\s*=\s*(\d+)", tile_header)
        if matched is None or int(matched[1]) != 128:
            raise ValueError("The independent CPU tile-credit reference requires the 128-row contract")
        header = ("namespace MulticoreBf16 { constexpr int64_t INPUT_TILE_ROWS = 128; }\n"
                  "namespace MulticoreRuntime {\n" + fragment + "}\n")
        (directory / "production_token_pull.h").write_text(header)
        probe = Path(__file__).with_name("indexed_pull_cpu_probe.cpp")
        cls.executables = {}
        for architecture in (3510, 2201):
            executable = directory / f"indexed-pull-probe-{architecture}"
            built = subprocess.run(
                ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", f"-I{directory}",
                 f"-D__NPU_ARCH__={architecture}", str(probe), "-o", str(executable)],
                capture_output=True, text=True, check=False,
            )
            if built.returncode:
                raise RuntimeError(f"expected_compile_exit=0 actual={built.returncode}: {built.stderr}")
            cls.executables[architecture] = executable

    def run_probe(self, suite: str, architecture: int = 3510) -> dict[str, object]:
        """Run a deterministic CPU model and preserve a useful failure message."""
        result = subprocess.run([str(self.executables[architecture]), suite], capture_output=True, text=True, check=False)
        self.assertEqual(
            result.returncode, 0,
            f"expected_exit=0 actual={result.returncode} suite={suite}: {result.stderr}",
        )
        return json.loads(result.stdout)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_full_row_payload_uses_one_dma_per_wanted_token(self) -> None:
        """Feature: Peer token ingress on Ascend950.
        Description: Pull complete H5120 rows through the production transfer function.
        Expectation: Each wanted token uses one payload DMA with exact receiver bytes.
        """
        result = self.run_probe("geometry")
        self.assertEqual(result["cases"], 1, f"expected_cases=1 actual={result['cases']}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_other_hidden_sizes_preserve_the_existing_transfer_geometry(self) -> None:
        """Feature: Peer token ingress fallback.
        Description: Pull rows at four hidden sizes outside H5120.
        Expectation: The existing chunk geometry and exact row contents are preserved.
        """
        result = self.run_probe("fallback")
        self.assertEqual(result["cases"], 4, f"expected_cases=4 actual={result['cases']}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_other_architectures_preserve_the_existing_transfer_geometry(self) -> None:
        """Feature: Architecture-specific token-pull geometry.
        Description: Compile and run the production function for a non-3510 target.
        Expectation: H5120 retains the existing two-transfer geometry on that target.
        """
        result = self.run_probe("geometry", architecture=2201)
        self.assertEqual(result["cases"], 1, f"expected_cases=1 actual={result['cases']}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_route_rows_credits_and_ub_lifetime_across_metadata_boundaries(self) -> None:
        """Feature: Peer token and tile credit publication.
        Description: Exercise metadata boundaries, empty ranks, hot routes and reversed ordinals.
        Expectation: All rows and credits are exact without reuse of pending UB payloads.
        """
        result = self.run_probe("routing")
        self.assertEqual(result["cases"], 136, f"expected_cases=136 actual={result['cases']}")
