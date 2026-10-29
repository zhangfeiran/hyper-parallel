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
"""Host-compiled coverage and load oracles for cross-expert Cube rotation."""
from pathlib import Path
import subprocess
import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[4]


class TestBf16GmmLogicalLane(unittest.TestCase):
    """Keep tile ownership bijective and avoid repeated partial-wave owners."""

    def test_rotation_matches_flat_group_schedule_and_balances_partial_waves(self) -> None:
        """Compile the real helper and compare tile owners against a flat work list."""
        source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#include "bf16_gmm/logical_lane.h"
int main() {
  for (uint32_t cubes : {16U, 24U, 32U}) {
    for (uint32_t experts : {1U, 3U, 8U, 16U}) {
      for (uint64_t tiles : {0U, 1U, 31U, 32U, 33U, 56U, 112U, 128U, 140U, 152U, 160U, 280U}) {
        std::vector<uint64_t> loads(cubes), expected_loads(cubes);
        for (uint32_t expert = 0; expert < experts; ++expert) {
          std::vector<uint32_t> visits(tiles);
          for (uint32_t worker = 0; worker < cubes; ++worker) {
            uint32_t logical = MulticoreBf16::RotateLogicalCube(worker, expert, tiles, cubes);
            assert(logical < cubes);
            for (uint64_t tile = logical; tile < tiles; tile += cubes) {
              ++visits[tile];
              ++loads[worker];
              assert(worker == (expert * tiles + tile) % cubes);
            }
          }
          for (auto count : visits) assert(count == 1);
        }
        for (uint64_t tile = 0; tile < experts * tiles; ++tile) ++expected_loads[tile % cubes];
        assert(loads == expected_loads);
      }
    }
  }
  std::array<uint64_t, 32> old_loads{}, new_loads{};
  for (uint32_t expert = 0; expert < 8; ++expert) {
    for (uint32_t worker = 0; worker < 32; ++worker) {
      for (uint64_t tile = worker; tile < 56; tile += 32) ++old_loads[worker];
      for (uint64_t tile = MulticoreBf16::RotateLogicalCube(worker, expert, 56, 32); tile < 56; tile += 32)
        ++new_loads[worker];
    }
  }
  assert(*std::max_element(old_loads.begin(), old_loads.end()) == 16);
  assert(*std::min_element(new_loads.begin(), new_loads.end()) == 14);
  assert(*std::max_element(new_loads.begin(), new_loads.end()) == 14);
  // Nonuniform and empty groups still visit each local tile once.
  for (uint32_t expert = 0; expert < 8; ++expert) {
    uint64_t tiles = std::array<uint64_t,8>{0,1,7,31,33,56,112,129}[expert];
    std::vector<uint32_t> visits(tiles);
    for (uint32_t worker = 0; worker < 32; ++worker)
      for (uint64_t tile = MulticoreBf16::RotateLogicalCube(worker, expert, tiles, 32); tile < tiles; tile += 32)
        ++visits[tile];
    for (auto count : visits) assert(count == 1);
  }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary)/"bf16-gmm-rotation"
            built = subprocess.run(
                ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 f"-I{_ROOT / 'hyper_parallel/core/multicore/ops'}", "-x", "c++", "-", "-o", str(executable)],
                input=source, capture_output=True, text=True, check=False,
            )
            self.assertEqual(built.returncode, 0, f"expected=0 actual={built.returncode}: {built.stderr}")
            result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
