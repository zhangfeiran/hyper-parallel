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
"""Integer oracle and owned-buffer checks for the independent count launch."""

import ctypes
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import count_kernel
from tests.common.mark_utils import arg_mark


class TestReplicaCountKernel(unittest.TestCase):
    """Compile the production row algorithm on CPU, mocking only its NPU launch boundary."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_unsupported_inputs_keep_torch_without_loading_a_library(self) -> None:
        """
        Feature: Optional fused counts boundary
        Description: Offer CPU, unsupported TopK and unsupported expert ranges to the dispatcher.
        Expectation: Return None before querying hardware or loading native code.
        """
        npu = SimpleNamespace(type="npu")
        cases = ((torch.zeros(1, 2, dtype=torch.int64), 24),
                 (SimpleNamespace(device=npu, dtype=torch.int32, ndim=2, shape=(1, 8)), 24),
                 (SimpleNamespace(device=npu, dtype=torch.int64, ndim=1, shape=(8,)), 24),
                 (SimpleNamespace(device=npu, dtype=torch.int64, ndim=2, shape=(1, 9)), 24),
                 (SimpleNamespace(device=npu, dtype=torch.int64, ndim=2, shape=(1, 8)), 257))
        with patch.object(count_kernel, "_counts_available", side_effect=AssertionError("Unexpected device query")):
            for ids, experts in cases:
                with self.subTest(experts=experts, shape=ids.shape):
                    self.assertIsNone(count_kernel.try_count_payload(ids, experts))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_optional_binary_and_supported_dispatch(self) -> None:
        """
        Feature: Supported count dispatch
        Description: Toggle binary availability for an eligible input without touching a real device.
        Expectation: Retain fallback when unbuilt and return the actual launch output when available.
        """
        ids = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.int64, ndim=2, shape=(128, 8))
        expected = object()
        for available in (False, True):
            with self.subTest(available=available):
                with (patch.object(count_kernel, "_counts_available", return_value=available),
                      patch.object(count_kernel, "launch_route_counts", return_value=expected) as launch):
                    result = count_kernel.try_count_payload(ids, 24)
                self.assertIs(result, expected if available else None)
                self.assertEqual(launch.call_count, int(available))

    @classmethod
    def setUpClass(cls) -> None:
        """Build the integer kernel's host wrapper without CANN or an NPU."""
        header = Path(count_kernel.__file__).parent / "_device_kernel" / "count_algorithm.h"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "counts.cpp"
            source.write_text(
                '#include "count_algorithm.h"\n'
                'extern "C" int launch_replica_counts(void*, const int64_t* ids, int64_t* partials, '
                'int64_t* result, int64_t rows, int topk, int experts, int stride) {\n'
                'const int64_t per_block=(rows+31)/32;\n'
                'for(int block=0;block<32;++block) {\n'
                'auto counts=partials+block*stride; for(int e=0;e<=experts;++e) counts[e]=0;\n'
                'for(int64_t row=block*per_block;row<(block+1)*per_block && row<rows;++row) '
                'hot::CountRow(ids+row*topk,counts,topk,experts); }\n'
                'for(int e=0;e<=experts;++e) { result[e]=0; '
                'for(int b=0;b<32;++b) result[e]+=partials[b*stride+e]; }\n'
                'result[experts]=result[experts]!=0; return 0; }\n'
            )
            subprocess.run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', '-I', str(header.parent),
                            str(source), '-o', str(root / 'counts.so')], check=True, capture_output=True)
            cls.library = ctypes.CDLL(str(root / 'counts.so'))
            cls.library.launch_replica_counts.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] + [ctypes.c_int] * 3
            cls.library.launch_replica_counts.restype = ctypes.c_int

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_partial_count_and_flags_match_scalar_oracle(self) -> None:
        """
        Feature: Fused integer count contract
        Description: Count strided, retained, empty and extreme-ID rows through the host launch ABI.
        Expectation: Match an independent scalar histogram and duplicate/range flag with owned int64 outputs.
        """
        generator = torch.Generator().manual_seed(1923)
        retained = []
        stream = Mock(npu_stream=0)
        with (patch.object(count_kernel, "_validate_input"),
              patch.object(count_kernel, "_count_library", return_value=self.library),
              patch.object(torch.npu, "device"), patch.object(torch.npu, "current_stream", return_value=stream),
              patch.object(torch.Tensor, "record_stream") as record):
            for experts in (1, 24, 256):
                for tokens in (0, 1, 33, 129):
                    for topk in (1, 2, 8):
                        ids = torch.randint(-2, experts + 2, (tokens, topk), generator=generator)
                        if tokens:
                            ids[0, 0] = torch.iinfo(torch.int64).min
                            ids[-1, -1] = torch.iinfo(torch.int64).max
                        ids = ids.T.contiguous().T
                        expected = [0] * (experts + 1)
                        for row in ids.tolist():
                            expected[-1] |= len(row) != len(set(row))
                            for value in row:
                                expected[-1] |= not 0 <= value < experts
                                expected[min(max(value, 0), experts - 1)] += 1
                        result = count_kernel.launch_route_counts(ids, experts)
                        retained.append((result, expected))
                        self.assertEqual(result.dtype, torch.int64)
            self.assertEqual(record.call_count, 3 * len(retained))
        self.assertEqual(len({result.data_ptr() for result, _ in retained}), len(retained))
        for result, expected in reversed(retained):
            torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.int64), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_launch_errors_are_reported_after_recording_lifetimes(self) -> None:
        """
        Feature: Count launch error propagation
        Description: Inject an ACL launch error while mocking the NPU boundary.
        Expectation: Raise explicitly after recording all source, scratch and output lifetimes.
        """
        library = Mock(launch_replica_counts=Mock(return_value=7))
        with (patch.object(count_kernel, "_validate_input"),
              patch.object(count_kernel, "_count_library", return_value=library),
              patch.object(torch.npu, "device"), patch.object(torch.npu, "current_stream"),
              patch.object(torch.Tensor, "record_stream") as record):
            with self.assertRaisesRegex(RuntimeError, "ACL error 7"):
                count_kernel.launch_route_counts(torch.zeros(1, 1, dtype=torch.int64), 24)
        self.assertEqual(record.call_count, 3)
