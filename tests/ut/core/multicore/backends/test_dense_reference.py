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
"""以真实 SDK 查询快照和独立编译器结果检查访问参数、覆盖和查询准入。"""

from __future__ import annotations

import copy
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hyper_parallel.core.multicore._build import query_dense_reference
from hyper_parallel.core.multicore.backends.dense_reference import ACCESS, DenseReferenceTiler, reference_access
from hyper_parallel.core.multicore.backends.dense_tile import DenseSdkTiler, bind_dense_tiles
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy, compile_dense_tiles
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from hyper_parallel.core.multicore.runtime.dense import DenseSpec
from tests.common.mark_utils import arg_mark

ROOT = Path(__file__).resolve().parents[5]
FIXTURE = Path(__file__).parent / "fixtures/dense_reference_910b3.json"
PROBE = """
#define __aicore__
#include "dense_access.h"
#include <iostream>
int main() {
  using namespace HyperParallelDense;
  DenseMatrixAccess access;
  uint32_t rows, columns, contracted, row, column;
  std::cin >> access.m_span >> access.n_span >> access.m_extent >> access.n_extent
           >> access.k_chunk >> access.close_shift >> rows >> columns >> contracted >> row >> column;
  const auto state = MatrixKState(access, rows, columns, contracted, row, column);
  std::cout << state << '\\n';
  for (uint32_t offset = 0; offset < contracted; ++offset) {
    std::cout << MatrixKOffset(offset, contracted, state) << ' ';
  }
  std::cout << '\\n';
  const auto span = access.m_extent >= access.n_extent ? access.m_span : access.n_span;
  const auto end = access.m_extent >= access.n_extent ? rows : columns;
  for (uint32_t first = 0; first < end;) {
    const auto next = MatrixSegmentEnd(first, span, end);
    std::cout << first << ' ' << next << ' ';
    first = next;
  }
}
"""


class TestDenseReference(unittest.TestCase):
    """查询快照只证明主机字段，C++ 探针检查实际设备所用的纯整数函数。"""

    @classmethod
    def setUpClass(cls) -> None:
        """一次编译实际设备辅助函数，并保留可在临时目录执行的主机二进制。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "probe.cpp"
            source.write_text(PROBE, encoding="utf-8")
            binary = root / "probe"
            subprocess.run([shutil.which("g++") or "/usr/bin/g++", "-std=c++17", "-O2", str(source),
                            "-I" + str(ROOT / "hyper_parallel/core/multicore/ops/dense"), "-o", str(binary)],
                           check=True, capture_output=True, text=True)
            cls.binary_data = binary.read_bytes()
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))["results"]

    def _probe(self, access, shape, row, column):
        payload = " ".join(str(value) for value in (*access, *shape[:3], row, column))
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "probe"
            binary.write_bytes(self.binary_data)
            binary.chmod(0o700)
            output = subprocess.check_output([str(binary)], input=payload, text=True).splitlines()
        return int(output[0]), tuple(map(int, output[1].split())), tuple(map(int, output[2].split()))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_query_snapshot_admits_only_proven_template(self):
        """Feature: SDK 访问参数。
        Description: 对八种实际查询快照检查参数与能力边界。
        Expectation: 已证明的 NN 模板准入，NT 和满载模板保留明确的禁用原因。
        """
        expected = {0: (224, 256, 5, 4, 256, 0), 4: (80, 128, 2, 8, 512, 0)}
        for index, result in enumerate(self.fixture):
            with self.subTest(index=index):
                access, reason = reference_access(result, tuple(result["shape"]), 20)
                if index in expected:
                    self.assertEqual(ACCESS.unpack(access), expected[index])
                    self.assertEqual(reason, "sdk_k_block_access")
                else:
                    self.assertEqual(access, bytes(24))
                    expected_reason = ("native_matmul_v3_sequential" if index in (2, 3)
                                       else "unsupported_reference_template")
                    self.assertEqual(reason, expected_reason)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_device_mapping_matches_compiler_golden_orders(self):
        """Feature: K 访问顺序。
        Description: 编译真实设备辅助函数，对照 SDK 编译器生成的逆序、旋转和尾部顺序。
        Expectation: 每个输出分区的 K 元素精确对应独立编译器顺序。
        """
        access = (224, 256, 5, 4, 512, 0)
        orders = ((0, 1, 2, 3, 4, 5, 6, 7), (7, 6, 5, 4, 3, 2, 1, 0),
                  (4, 5, 6, 7, 0, 1, 2, 3), (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
        for index, order in enumerate(orders):
            with self.subTest(index=index):
                _, mapping, _ = self._probe(access, (1024, 1024, 4096, False), index * 224, 0)
                expected = tuple(element for block in order for element in range(block * 512, (block + 1) * 512))
                self.assertEqual(mapping, expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_each_partition_covers_k_and_output_without_overlap(self):
        """Feature: 通用分区覆盖。
        Description: 枚举 M/N 轴、轴平局、超过 17 的轴及显式禁用情况。
        Expectation: 每个 K 元素访问一次，输出片段连续覆盖原矩阵范围。
        """
        for m_extent, n_extent in ((5, 4), (4, 5), (4, 4), (2, 10), (1, 20), (20, 1), (2, 3)):
            for close in (0, 1):
                shape = (m_extent * 16 - 7, n_extent * 32 - 5, 256, False)
                access = (16, 32, m_extent, n_extent, 16, close)
                for index in range(max(m_extent, n_extent)):
                    row, column = (index * 16, 0) if m_extent >= n_extent else (0, index * 32)
                    with self.subTest(m=m_extent, n=n_extent, close=close, index=index):
                        state, mapping, segments = self._probe(access, shape, row, column)
                        self.assertEqual(sorted(mapping), list(range(256)))
                        end = shape[0] if m_extent >= n_extent else shape[1]
                        covered = tuple(value for begin, stop in zip(segments[::2], segments[1::2])
                                        for value in range(begin, stop))
                        self.assertEqual(covered, tuple(range(end)))
                        if close:
                            self.assertEqual(state, 0)
                        if max(m_extent, n_extent) == 20 and index == 1 and not close:
                            self.assertNotEqual(state, 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_invalid_fields_and_unsupported_grid_never_generate_access(self):
        """Feature: 元数据准入。
        Description: 损坏 shape、原始字段长度、原始维度、模板及硬件分区。
        Expectation: 损坏记录拒绝，未证明的硬件分区保留顺序访问。
        """
        for failure in ("shape", "size", "origin", "key", "grid", "family", "choice"):
            result = copy.deepcopy(self.fixture[0])
            shape = tuple(result["shape"])
            if failure == "shape":
                result["shape"][0] += 1
            elif failure == "size":
                result["run_info"]["tiling_data"] = "00"
            elif failure == "origin":
                raw = bytearray.fromhex(result["run_info"]["tiling_data"])
                struct.pack_into("<i", raw, 4, shape[0] + 1)
                result["run_info"]["tiling_data"] = raw.hex()
            elif failure == "key":
                result["run_info"]["tiling_key"] = -1
            elif failure == "family":
                result["native_family"] = "MatMulV3"
            elif failure == "choice":
                result["v3_with_split_k_false_true"] = []
            with self.subTest(failure=failure):
                if failure == "grid":
                    access, reason = reference_access(result, shape, 16)
                    self.assertEqual((access, reason), (bytes(24), "unsupported_reference_partition"))
                else:
                    with self.assertRaises(ValueError):
                        reference_access(result, shape, 20)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_child_query_preserves_target_and_execution_flags(self):
        """Feature: 查询进程隔离。
        Description: 模拟子进程返回有效字段并修改目标身份。
        Expectation: 使用明确解释器和脚本，不导入 TBE；身份漂移不能通过准入。
        """
        tiler = DenseReferenceTiler(Path("/sdk"), Path("/cache/reference.so"), "Ascend910B3", 20, False)
        result = self.fixture[0]
        shape = tuple(result["shape"])
        record = {"format_version": 2, "operator": "native_dense_matmul", "scope": "sdk_host_query",
                  "soc": "Ascend910B3", "cube_workers": 20, "deterministic": False, "results": [result]}
        process = SimpleNamespace(returncode=0, stdout=json.dumps(record), stderr="")
        with patch("hyper_parallel.core.multicore.backends.dense_reference.subprocess.run",
                   return_value=process) as run:
            self.assertEqual(tiler.query((shape,)), record)
            command = run.call_args.args[0]
            self.assertTrue(command[1].endswith("_build/query_dense_reference.py"))
            self.assertFalse(json.loads(run.call_args.kwargs["input"])["deterministic"])
            record["soc"] = "other"
            process.stdout = json.dumps(record)
            with self.assertRaisesRegex(ValueError, "身份"):
                tiler.query((shape,))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_bank_binds_reference_access_to_each_matrix_slot(self):
        """Feature: tiling bank 版本 2。
        Description: 将两个真实参考分区绑定到完整 FFN 的矩阵和激活描述符。
        Expectation: 固定偏移、零激活槽和形状顺序正确；截断 bank 不能导出有效清单。
        """
        dense = dense_ffn.plan(DenseSpec({"T": 1024, "H": 1024, "PackedI": 8192, "I": 4096}),
                               intermediate_size=4096)
        plan = compile_dense_tiles(dense, DenseTilePolicy(cube_workers=20))
        binding = bind_dense_tiles(plan)
        record = {"results": [self.fixture[2], self.fixture[0]]}
        tiler = DenseSdkTiler.__new__(DenseSdkTiler)
        tiler.soc, tiler.cube_workers, tiler.tiling_bytes, tiler.workspace_bytes = "test-target", 20, 200, 8
        tiler.reference = SimpleNamespace(query=lambda shapes: record)

        def _generate(_soc, _rows, _tile, _columns, _contracted, _transpose, block, capacity):
            block.raw = bytes([7]) * capacity
            return 0

        tiler.generate = _generate
        bank = tiler.bank(plan, binding)
        self.assertEqual(len(bank), 3 * 224)
        self.assertEqual(bank[224:448], bytes(224))
        self.assertEqual(bank[200:224], bytes(24))
        self.assertEqual(ACCESS.unpack(bank[648:672]), (224, 256, 5, 4, 256, 0))
        self.assertEqual(tiler.export_manifest(binding, bank)["reference"], record)
        with self.assertRaisesRegex(ValueError, "ABI"):
            tiler.export_manifest(binding, bank[:-1])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_sdk_dependency_is_loaded_before_bridge_without_rpath(self):
        """Feature: SDK 主机库加载。
        Description: 模拟桥接库没有 RPATH，要求其 SDK 依赖先按绝对路径加载。
        Expectation: 不依赖额外的 OPP 库搜索路径，并保持全部库句柄有效。
        """
        loaded = []
        initializer = SimpleNamespace(hyper_parallel_dense_compile_platform=Mock(return_value=0))
        symbols = SimpleNamespace(TbeLoadSoAndSaveToRegistry=Mock(), DoOpTilingForCompile=Mock())

        def _load_library(name):
            if name == "/cache/reference.so" and not loaded[0].endswith("/libophost_comm_legacy.so"):
                raise OSError("缺少 SDK 依赖")
            loaded.append(name)
            return initializer if name == "/cache/reference.so" else symbols

        request = {"cann_root": "/sdk", "library": "/cache/reference.so", "soc": "Ascend910B3",
                   "cube_workers": 20}
        with patch.object(query_dense_reference.ctypes, "CDLL", side_effect=_load_library):
            _, libraries = query_dense_reference._query_library(request)
        self.assertTrue(loaded[0].startswith("/sdk/opp/built-in/"))
        self.assertTrue(loaded[0].endswith("/libophost_comm_legacy.so"))
        self.assertEqual(loaded[1], "/cache/reference.so")
        self.assertEqual(len(libraries), 5)
