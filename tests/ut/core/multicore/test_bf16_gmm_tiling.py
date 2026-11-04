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
"""CPU contracts for the private A5 tiling serializer and record preservation."""
import struct
import unittest

from hyper_parallel.core.multicore.modules.mega_moe.bf16_gmm_tiling import retile_a5_bf16
from hyper_parallel.core.multicore.modules.mega_moe.forward import tiling_tables as forward
from hyper_parallel.core.multicore.modules.mega_moe.backward import tiling_tables as backward


class TestBf16GmmTiling(unittest.TestCase):
    """Check real producers, hardware resource bounds and unsupported layouts."""

    def test_six_producers_preserve_every_record_metadata(self) -> None:
        getters = (forward.get_up_proj_tiling_bytes, forward.get_down_proj_tiling_bytes,
                   backward.get_act_grad_tiling_bytes, backward.get_gate_grad_tiling_bytes,
                   backward.get_w1_grad_tiling_bytes, backward.get_w2_grad_tiling_bytes)
        allowed = {6, 8, 9, 10, 11, 12, 13, 14, 19, 20, 26, 27, 30, 31, 32}
        for getter in getters:
            with self.subTest(role=getter.__name__):
                original = getter(4096, hidden_size=5120, intermediate_size=1792,
                                  num_groups=8, num_cube_cores=32)
                result = retile_a5_bf16(original, 32)
                self.assertEqual(len(result), len(original), f"expected={len(original)} actual={len(result)}")
                for offset in range(0, len(original), 2016):
                    old = struct.unpack_from('<504I', original, offset)
                    new = struct.unpack_from('<504I', result, offset)
                    start = old.index(393216, 350)-19
                    changed = {index-start for index, pair in enumerate(zip(old, new)) if pair[0] != pair[1]}
                    self.assertLessEqual(changed, allowed, f"allowed={allowed} actual={changed}")
                    shape = (new[start+8], new[start+9], new[start+10])
                    self.assertEqual(shape, (256, 256, 64), f"expected={(256,256,64)} actual={shape}")
                    bm, bn, bk = shape
                    l0a, l0b, l0c = bm*bk*2*2, bn*bk*2*2, bm*bn*4
                    l1 = (bm+bn)*bk*new[start+26]*2*2
                    self.assertLessEqual(l0a, 65536, f"limit={65536} actual={l0a}")
                    self.assertLessEqual(l0b, 65536, f"limit={65536} actual={l0b}")
                    self.assertLessEqual(l0c, 262144, f"limit={262144} actual={l0c}")
                    self.assertLessEqual(l1, 524288, f"limit={524288} actual={l1}")

    def test_invalid_record_and_architecture_fail_before_retile(self) -> None:
        original = forward.get_up_proj_tiling_bytes(4096, num_cube_cores=32)
        for data, cores in ((original, 24), (original[:-4], 32), (bytes(len(original)), 32)):
            with self.subTest(size=len(data), cores=cores), self.assertRaises(ValueError):
                retile_a5_bf16(data, cores)
        damaged = bytearray(original)
        damaged[-2016:] = bytes(2016)
        with self.assertRaises(ValueError):
            retile_a5_bf16(bytes(damaged), 32)
