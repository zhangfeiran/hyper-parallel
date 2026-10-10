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
"""原始 NK 数值来源、三/四输入完整反向及主机补算的结构门禁。"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from hyper_parallel.core.multicore.backends.dense_backward import bind_dense_backward_tiles
from hyper_parallel.core.multicore.backends.dense_linear import bind_dense_linear_tiles
from hyper_parallel.core.multicore.compiler.dense_backward import compile_dense_backward
from hyper_parallel.core.multicore.compiler.dense_backward_tiles import compile_dense_backward_tiles
from hyper_parallel.core.multicore.compiler.dense_tile import compile_dense_tiles
from hyper_parallel.core.multicore.examples.mega_ffn import source_contract
from hyper_parallel.core.multicore.examples.mega_ffn.source_contract import resident_source_entry
from hyper_parallel.core.multicore.frontend.examples.common_ffn import common_ffn
from hyper_parallel.core.multicore.frontend.examples.linear_ffn import linear_ffn
from hyper_parallel.core.multicore.modules.mega_ffn.module import MegaFFN
from tests.common.mark_utils import arg_mark


def _packed():
    path = Path(__file__).parent / "backends/fixtures/dense_source_execution_910b3.json"
    return json.loads(path.read_text())["entry"]


def _synthetic(baseline):
    program = common_ffn if baseline == "common" else linear_ffn
    forward = compile_dense_tiles(MegaFFN(17, 19, program=program, create_parameters=False)._plan(7))
    backward = compile_dense_backward_tiles(compile_dense_backward(forward.dense), forward.policy)

    def _tiling(binding):
        return {"matrix_references": [value.export_manifest() if value is not None else None
                                      for value in binding.matrix_references],
                "access_reasons": ["native_matmul_v3_sequential" if value is not None else "no_matrix"
                                   for value in binding.matrix_references],
                "required_capabilities": list(binding.required_capabilities)}

    return json.loads(json.dumps({"scope": "真实 CPU 计划与模拟调用计数，仅用于结构门禁测试", "native": {
        "execution_mode": "resident_dense_tile_candidate", "backward_execution_mode": "resident_dense_backward",
        "tiling": {**_tiling(bind_dense_linear_tiles(forward)), "direction": "forward"},
        "backward_runtime": {"native_calls": 1, "host_calls": 0, "unbound_capabilities": [], "native_payloads": [
            {"execution_mode": "resident_backward_candidate", "plan": {
                "input_mask": [True] * (4 if baseline == "common" else 3),
                                                                        "output_mask": [True]},
             "tiling": _tiling(bind_dense_backward_tiles(backward))}]}}}))


class TestSourceContract(unittest.TestCase):
    """结构门禁必须区分对应数值语义，并严格拒绝部分梯度及未证明的矩阵访问。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_actual_packed_and_common_coordinate_snapshots(self):
        """Feature: 当前原始数值来源的真实设备记录。
        Description: 读取实际 packed/common 常驻组件清单切片，不修改调用计数或准入原因。
        Expectation: 各自完整三/四输入记录通过结构检查，交换基线则拒绝。
        """
        path = Path(__file__).parent / "backends/fixtures/dense_source_coordinates_execution_910b3.json"
        records = json.loads(path.read_text(encoding="utf-8"))["entries"]
        for baseline, entry in records.items():
            with self.subTest(baseline=baseline):
                self.assertTrue(resident_source_entry(entry, baseline))
                self.assertFalse(resident_source_entry(entry, "packed" if baseline == "common" else "common"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_source_reference_coordinates_keep_complete_packed_backward(self):
        """Feature: 原始 NK 执行记录。
        Description: 先拒绝旧方向快照，再使用真实 CPU 计划与模拟调用计数检查完整契约。
        Expectation: 完整 packed 记录可通过结构检查；主机补算、缺失数值来源或部分梯度均拒绝。
        """
        self.assertFalse(resident_source_entry(_packed(), "packed"))
        entry = _synthetic("packed")
        self.assertTrue(resident_source_entry(entry, "packed"))
        self.assertFalse(resident_source_entry(entry, "common"))
        for kind in ("host", "boolean_calls", "partial", "integer_mask", "missing_weight",
                     "transpose", "coordinates", "reason"):
            with self.subTest(kind=kind):
                corrupt = deepcopy(entry)
                runtime = corrupt["native"]["backward_runtime"]
                payload = runtime["native_payloads"][0]
                if kind == "host":
                    runtime["host_calls"] = 1
                elif kind == "boolean_calls":
                    runtime["native_calls"] = True
                elif kind == "partial":
                    payload["plan"]["input_mask"][1] = False
                elif kind == "integer_mask":
                    payload["plan"]["input_mask"][1] = 1
                else:
                    tiling = payload["tiling"]
                    index = next(index for index, value in enumerate(tiling["matrix_references"])
                                 if value is not None and value["source"] == "dense.linear.weight_grad")
                    if kind == "missing_weight":
                        tiling["matrix_references"][index] = None
                        tiling["access_reasons"][index] = "no_matrix"
                    elif kind == "transpose":
                        tiling["matrix_references"][index]["matrix"][3:] = [False, True]
                    elif kind == "coordinates":
                        tiling["matrix_references"][index]["coordinates_transposed"] = False
                    else:
                        tiling["access_reasons"][index] = "unsupported_reference_template"
                self.assertFalse(resident_source_entry(corrupt, "packed"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_common_four_input_schema_requires_complete_gradient_coverage(self):
        """Feature: common 四输入结构。
        Description: 使用规范 CPU 前后向计划与模拟计数，分别删除输入梯度或改成 packed 掩码。
        Expectation: 只有全部四输入、三个投影和分步激活的记录能通过 common 结构检查。
        """
        entry = _synthetic("common")
        self.assertTrue(resident_source_entry(entry, "common"))
        self.assertFalse(resident_source_entry(entry, "packed"))
        for mask in ([True, True, True], [True, True, False, True]):
            with self.subTest(mask=mask):
                corrupt = deepcopy(entry)
                corrupt["native"]["backward_runtime"]["native_payloads"][0]["plan"]["input_mask"] = mask
                self.assertFalse(resident_source_entry(corrupt, "common"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_supervisor_checker_loads_without_framework_imports(self):
        """Feature: 门禁监督进程轻量导入。
        Description: 按实际脚本路径在新进程加载检查器，并检查已导入模块。
        Expectation: 不加载 Torch、TorchNPU 或 HyperParallel 包。
        """
        script = ("import importlib.util, sys\n"
                  "spec = importlib.util.spec_from_file_location('_source_check', sys.argv[1])\n"
                  "module = importlib.util.module_from_spec(spec)\n"
                  "spec.loader.exec_module(module)\n"
                  "assert not {'torch', 'torch_npu', 'hyper_parallel'}.intersection(sys.modules)\n")
        subprocess.run([sys.executable, "-c", script, str(Path(source_contract.__file__).resolve())], check=True)
