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
"""Source-NK 新进程完整训练与性能分析器的合成证据测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hyper_parallel.core.multicore.examples.mega_ffn import source_acceptance
from tests.common.mark_utils import arg_mark
from tests.ut.core.multicore.test_dense_acceptance import _suite
from tests.ut.core.multicore.test_source_contract import _synthetic


def _source_suite(root):
    suite = _suite(root)
    for job in suite["jobs"]:
        path = Path(job["result"])
        record = json.loads(path.read_text(encoding="utf-8"))
        record["scope"] = "全部数值和计时均为 CPU 合成证据，仅用于门禁逻辑测试"
        record["identity"].update(source_weight_layout="nk", numerical_contract=job["baseline"],
                                   deterministic_algorithms=True)
        if job["mode"] == "validate":
            for row in record["steps"]:
                for kind, check in row["checks"].items():
                    check.update(atol=0.002 if kind in ("logits", "loss", "gradients", "parameters") else (
                        0 if kind == "step" else 1e-8), rtol=0 if kind == "step" else 0.02)
                row["checks"]["step"]["actual_update_count_passed"] = True
        else:
            record.update(peak_allocated_bytes=1024, peak_reserved_bytes=2048)
        if job["backend"] == "mega_ffn":
            payloads = {f"layers.{index}.mlp": [_synthetic(job["baseline"])] for index in range(2)}
            if job["mode"] == "validate":
                record["native_payloads"]["candidate"] = payloads
            else:
                record["native_payloads"] = payloads
        path.write_text(json.dumps(record), encoding="utf-8")
    return suite


class TestSourceAcceptance(unittest.TestCase):
    """不缩减两个原始基线、八项状态、全部设备梯度或完整训练步统计范围。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_complete_synthetic_common_and_packed_suite_covers_source_contracts(self):
        """Feature: 两种原始数值契约的完整验收。
        Description: 分析 52 个合成新进程记录，每个基线具有独立 100 步和三块完整 ABBA。
        Expectation: 所有基线、完整梯度、实际计数及干净对照共同满足时才放行。
        """
        with tempfile.TemporaryDirectory() as directory:
            suite = _source_suite(Path(directory))
            self.assertEqual(len(suite["jobs"]), 52)
            result = source_acceptance.analyze_suite(suite)
        self.assertTrue(result["formal_original_training_accepted"])
        self.assertTrue(result["performance_accepted"])
        self.assertEqual(set(result["identities"]), {"common", "packed"})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_relaxed_tolerances_fake_step_counts_and_partial_source_masks_fail(self):
        """Feature: 完整数值和设备范围。
        Description: 放宽主参数阈值、伪造相同步数、删减 common 梯度或添加主机补算。
        Expectation: 看似正确的轨迹和计时均不能放行错误范围。
        """
        for failure in ("tolerance", "counter", "mask", "host", "coordinates"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                suite = _source_suite(Path(directory))
                path = Path(suite["jobs"][1]["result"])
                record = json.loads(path.read_text(encoding="utf-8"))
                if failure == "tolerance":
                    record["steps"][0]["checks"]["main"]["atol"] = 0.002
                elif failure == "counter":
                    record["steps"][0]["checks"]["step"]["actual_update_count_passed"] = False
                else:
                    runtime = record["native_payloads"]["candidate"]["layers.0.mlp"][0]["native"]["backward_runtime"]
                    if failure == "host":
                        runtime["host_calls"] = 1
                    elif failure == "mask":
                        runtime["native_payloads"][0]["plan"]["input_mask"] = [True, True, True]
                    else:
                        references = runtime["native_payloads"][0]["tiling"]["matrix_references"]
                        next(value for value in references if value and value["source"] ==
                             "dense.linear.weight_grad")["coordinates_transposed"] = False
                path.write_text(json.dumps(record), encoding="utf-8")
                result = source_acceptance.analyze_suite(suite)
                self.assertFalse(result["formal_original_training_accepted"])
                self.assertFalse(result["performance_accepted"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_missing_hbm_wrong_source_and_cross_contract_workload_drift_are_rejected(self):
        """Feature: 真实来源、HBM 与固定工作量身份。
        Description: 删除峰值内存，关闭确定性，或单独改变 packed 原始模型的层数配置。
        Expectation: 缺失必要证据或对照配置漂移立即拒绝，不计算收益。
        """
        for failure in ("hbm", "source", "drift"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                suite = _source_suite(Path(directory))
                job = suite["jobs"][2] if failure != "drift" else suite["jobs"][26]
                path = Path(job["result"])
                record = json.loads(path.read_text(encoding="utf-8"))
                if failure == "hbm":
                    record.pop("peak_allocated_bytes")
                elif failure == "source":
                    record["identity"]["deterministic_algorithms"] = False
                else:
                    record["workload"]["config"]["num_layers"] = 1
                path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    source_acceptance.analyze_suite(suite)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_supervisor_imports_without_frameworks(self):
        """Feature: 设备等待时的轻量监督入口。
        Description: 在独立进程按脚本路径导入 Source-NK 监督入口及其来源检查器。
        Expectation: Torch、TorchNPU 和 HyperParallel 包根均不加载。
        """
        script = ("import importlib.util, sys\n"
                  "spec = importlib.util.spec_from_file_location('_source_supervisor', sys.argv[1])\n"
                  "module = importlib.util.module_from_spec(spec)\n"
                  "spec.loader.exec_module(module)\n"
                  "assert not {'torch', 'torch_npu', 'hyper_parallel'}.intersection(sys.modules)\n")
        subprocess.run([sys.executable, "-c", script, source_acceptance.__file__], check=True)
