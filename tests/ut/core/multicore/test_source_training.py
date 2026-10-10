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
"""Source-NK 独立完整模型与真实参数组 AdamW 历史的 CPU 验证。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.components.optim import Float16OptimizerWithFloat16Params
from hyper_parallel.core.multicore.examples.mega_ffn import source_training
from hyper_parallel.core.multicore.examples.mega_ffn.qwen_dense_model import QwenDenseConfig, QwenDenseModel
from hyper_parallel.core.multicore.examples.mega_ffn.source_layout import build_source_model
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from tests.common.mark_utils import arg_mark
from tests.ut.core.multicore.test_qwen_dense_model import _cpu_attention, _cpu_norm, _cpu_rotary


def _workload(candidate, recompute=False):
    config = QwenDenseConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_seq_len=8)
    torch.manual_seed(17)
    model = (build_source_model(config, "common", recompute=recompute) if candidate else
             QwenDenseModel(config, "common", recompute=recompute)).to(dtype=torch.bfloat16)
    leaf = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    optimizer = Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": leaf}), model)
    return source_training.SourceTrainingWorkload(model, optimizer)


class TestSourceTraining(unittest.TestCase):
    """完整八项状态、独立优化器所有权和实际更新计数必须同时成立。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    @patch.object(DefaultDeviceType, "_default_device_type", "cpu")
    def test_full_common_model_uses_real_group_adamw_and_eight_independent_states(self):
        """Feature: 正式 Source-NK 训练入口。
        Description: 完整双层模型独立训练两步，保留真实 fused AdamW、FP32 主参数及重计算。
        Expectation: 八项逐值一致，全部组计数实际为二，矩与主参数不共享存储。
        """
        attention = "hyper_parallel.core.multicore.examples.mega_moe.qwen_moe_model.run_qwen3_moe_flash_attention"
        with (patch(attention, side_effect=_cpu_attention),
              patch("hyper_parallel.components.modules.rms_norm.rms_norm", side_effect=_cpu_norm),
              patch("hyper_parallel.core.multicore.examples.mega_moe.qwen_moe_model.apply_rotary_pos_emb",
                    side_effect=_cpu_rotary)):
            for recompute in (False, True):
                with self.subTest(recompute=recompute):
                    reference, candidate = _workload(False, recompute), _workload(True, recompute)
                    tokens = torch.arange(16).view(2, 8)
                    for index in range(2):
                        left, right = reference.step(tokens, True), candidate.step(tokens, True)
                        row = source_training.trajectory_checks(reference, candidate, left, right, index + 1)
                        self.assertTrue(row["passed"], row)
                        self.assertEqual(set(row["checks"]), {"loss", "logits", "gradients",
                                                              *source_training.STATE_KINDS})
                        for check in row["checks"].values():
                            self.assertEqual(check["max_abs"], 0)
                    for kind in ("main", "exp_avg", "exp_avg_sq"):
                        expected, actual = reference.state(kind), candidate.state(kind)
                        for name, expected_value in expected.items():
                            self.assertEqual(actual[name].dtype, torch.float32)
                            self.assertNotEqual(actual[name].data_ptr(), expected_value.data_ptr())
                    self.assertTrue(all(value.item() == 2 for value in candidate.state("step").values()))
                    self.assertTrue(all("step" not in state
                                        for leaf in candidate.optimizer.chained_optimizers
                                        for state in leaf.state.values()))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_missing_moments_duplicate_ownership_and_wrong_shared_steps_cannot_pass(self):
        """Feature: 真实优化器证据完整性。
        Description: 删除更新后的矩、重复注册主参数，并使两条轨迹保持相同但错误的计数。
        Expectation: 不用零值补齐缺失矩，相同错误计数仍然无法通过应有更新次数检查。
        """
        reference, actual = _workload(False), _workload(True)
        for workload in (reference, actual):
            workload.optimizer.chained_optimizers[0].param_groups[0]["step"] = 1
            with self.assertRaisesRegex(ValueError, "missing after an update"):
                workload.state("exp_avg")
        main_parameter = next(iter(actual.optimizer.optimizer_param_by_model_param.values()))
        group = actual.optimizer.chained_optimizers[0].param_groups[0]
        group["params"].append(main_parameter)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            actual.state("step")
        group["params"].pop()
        for workload in (reference, actual):
            workload.optimizer.chained_optimizers[0].param_groups[0]["step"] = 0
        gradients = {name: torch.zeros_like(value) for name, value in reference.state("parameters").items()}
        step = (torch.zeros(()), torch.zeros(1), gradients)
        row = source_training.trajectory_checks(reference, actual, step, step, 1)
        self.assertFalse(row["passed"])
        self.assertFalse(row["checks"]["step"]["actual_update_count_passed"])
        self.assertEqual(row["checks"]["step"]["max_abs"], 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_initial_mismatch_and_candidate_build_failure_close_existing_workloads(self):
        """Feature: 初始状态和错误路径所有权。
        Description: 单独改变 FP32 主参数，再模拟第二个模型构造失败。
        Expectation: 原始证据保留失败状态，已创建模型始终关闭。
        """
        reference, candidate = _workload(False), _workload(True)
        with torch.no_grad():
            next(iter(candidate.optimizer.optimizer_param_by_model_param.values())).add_(1)
        with tempfile.TemporaryDirectory() as directory:
            args = source_training.parse_args(["--mode", "validate", "--baseline", "common", "--steps", "2",
                                               "--output", str(Path(directory) / "result.json")])
            with (patch.object(source_training, "_build", side_effect=[reference, candidate]),
                  patch.object(source_training, "_identity", return_value={"identity": {"scope": "CPU only"}}),
                  patch.object(reference.model, "close") as close_reference,
                  patch.object(candidate.model, "close") as close_candidate):
                with self.assertRaisesRegex(RuntimeError, "initial"):
                    source_training.validate(args, reference.model.config, torch.zeros(2, 8, dtype=torch.int64))
                close_reference.assert_called_once()
                close_candidate.assert_called_once()
            record = json.loads(args.output.read_text())
            self.assertTrue(record["initial"]["parameters"]["passed"])
            self.assertFalse(record["initial"]["main"]["passed"])
            self.assertFalse(record["complete"])
            self.assertFalse(record["formal_original_training_accepted"])
            with (patch.object(source_training, "_build", side_effect=[reference, ValueError("new model failed")]),
                  patch.object(reference.model, "close") as close_reference):
                with self.assertRaisesRegex(ValueError, "new model"):
                    source_training.validate(args, reference.model.config, torch.zeros(2, 8, dtype=torch.int64))
                close_reference.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_source_cli_rejects_mismatched_original_contracts(self):
        """Feature: 原始数值契约身份。
        Description: 混用 common/packed 原生对照，或将旧 KN 候选当作原始基线。
        Expectation: 在设备初始化前拒绝，合法配置完整保留缓存根与模型参数。
        """
        for arguments in (("--baseline", "common", "--backend", "packed"), ("--baseline", "mega_ffn")):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                source_training.parse_args(["--mode", "validate", "--output", "unused.json", *arguments])
        args = source_training.parse_args(["--mode", "validate", "--baseline", "common", "--cache-root", "/tmp/cache",
                                           "--output", "unused.json"])
        self.assertEqual(args.cache_root, Path("/tmp/cache"))
        self.assertEqual(args.baseline, "common")
