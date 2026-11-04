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
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Hardware-independent contracts for Multicore build targets and private SHMEM SDKs."""

"""Host-compiled pointer and shape contracts for the A5 BF16 GMM worker."""

from pathlib import Path
import subprocess
import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[4]


class TestBf16GmmProblem(unittest.TestCase):
    """Check all six matrix roles without CANN or accelerator initialization."""

    def test_expert_slices_cover_six_roles_empty_tails_and_large_offsets(self) -> None:
        """Compile the device-independent helper against independent slice equations."""
        source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include "bf16_gmm/problem.h"
int main() {
  using MulticoreBf16::MakeProblem;
  struct Shape { uint64_t m, n, k; bool trans_a; };
  const std::array<Shape, 6> shapes = {{{4096,3584,5120,false}, {4096,5120,1792,false},
      {4096,1792,5120,false}, {4096,5120,3584,false}, {5120,3584,4096,true}, {1792,5120,4096,true}}};
  for (const auto& shape : shapes) {
    uint64_t start = uint64_t{1} << 20;
    for (uint64_t expert = 0; expert < 8; ++expert) {
      const std::array<uint64_t, 5> lengths = {0, 1, 127, 2048, 2305};
      const uint64_t rows = lengths[expert % lengths.size()];
      const auto problem = MakeProblem(shape.m, shape.n, shape.k, expert, start, rows, shape.trans_a);
      assert(problem.n == shape.n);
      if (shape.trans_a) {
        assert(problem.m == shape.m && problem.k == rows);
        assert(problem.a == start * shape.m && problem.b == start * shape.n);
        assert(problem.c == expert * shape.m * shape.n);
      } else {
        assert(problem.m == rows && problem.k == shape.k);
        assert(problem.a == start * shape.k && problem.b == expert * shape.k * shape.n);
        assert(problem.c == start * shape.n);
      }
      start += rows;
    }
  }
  const auto up = MulticoreBf16::SplitTail(2048, 3584, 256, 256, 32);
  assert(up.m == 2 && up.n == 1);
  const auto full = MulticoreBf16::SplitTail(2048, 5120, 256, 256, 32);
  assert(full.m == 1 && full.n == 1);
  const auto small = MulticoreBf16::SplitTail(1, 1, 256, 256, 32);
  assert(small.m == 1 && small.n == 1);
  for (uint64_t rows = 1; rows < 8192; rows += 127) {
    const uint64_t tiles = ((rows + 255) / 256) * 14;
    const auto split = MulticoreBf16::SplitTail(rows, 3584, 256, 256, 32);
    const uint64_t tail = tiles <= 32 ? 0 : tiles % 32;
    if (tail == 0) assert(split.m == 1 && split.n == 1);
    else {
      assert(split.m * split.n * tail <= 32);
      assert((split.m + 1) * split.n * tail > 32);
    }
  }
  return 0;
}
'''
        include = _ROOT / "hyper_parallel/core/multicore/ops"
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "bf16-gmm-problem"
            built = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{include}",
                 "-x", "c++", "-", "-o", str(executable)],
                input=source, capture_output=True, text=True, check=False,
            )
            self.assertEqual(built.returncode, 0, f"expected=0 actual={built.returncode}: {built.stderr}")
            result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
