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
"""CPU oracle for BF16-typed clamp scalars and boundary gradients."""
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import torch


_ROOT = Path(__file__).resolve().parents[4]


class TestBf16Clamp(unittest.TestCase):
    """Check the device helper against Torch BF16 scalar conversion and autograd."""

    def test_rounding_and_clamp_gradient_boundaries(self) -> None:
        """Preserve disabled limits and match inclusive BF16 clamp boundaries."""
        source = r'''
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include "runtime/bf16_clamp.h"
int main() {
  for (uint32_t bits : {0U, 0xbf800000U, 1U, 0x3dcccccdU, 0x3e000000U,
                        0x40e00000U, 0x3f808000U, 0x3f818000U}) {
    union { float value; uint32_t bits; } input{}, output{};
    input.bits = bits;
    output.value = MulticoreRuntime::Bf16ClampLimit(input.value);
    std::printf("%u %u\n", bits, output.bits);
  }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary)/'bf16-clamp'
            result = subprocess.run(
                ['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                 f'-I{_ROOT / "hyper_parallel/core/multicore/ops"}', '-x', 'c++', '-', '-o', str(executable)],
                input=source, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
            result = subprocess.run([str(executable)], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
        converted = {}
        for line in result.stdout.splitlines():
            before, after = map(int, line.split())
            value = struct.unpack('<f', struct.pack('<I', before))[0]
            rounded = torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).float().item()
            expected = value if value <= 0 or rounded == 0 else rounded
            expected_bits = struct.unpack('<I', struct.pack('<f', expected))[0]
            self.assertEqual(after, expected_bits, f"input={value} expected={expected_bits} actual={after}")
            converted[before] = struct.unpack('<f', struct.pack('<I', after))[0]
        bound = converted[0x3dcccccd]
        inputs = torch.tensor([-1, -.1006, -.1, -.0996, 0, .0996, .1, .1006, 1],
                              dtype=torch.bfloat16, requires_grad=True)
        inputs.clamp(min=-.1, max=.1).float().sum().backward()
        actual = ((inputs.detach().float() >= -bound) & (inputs.detach().float() <= bound)).float()
        torch.testing.assert_close(actual, inputs.grad.float(), atol=0, rtol=0)
        boundary = torch.tensor(.1, dtype=torch.bfloat16).float().item()
        self.assertGreater(boundary, .1, f"expected>{.1} actual={boundary}")
        self.assertLessEqual(boundary, bound, f"expected<={bound} actual={boundary}")
