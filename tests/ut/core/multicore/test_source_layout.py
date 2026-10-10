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
"""Source-NK 完整模型替换及优化器状态映射的 CPU 验证。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.core.multicore.examples.mega_ffn.qwen_dense_model import QwenDenseConfig, QwenDenseModel
from hyper_parallel.core.multicore.examples.mega_ffn.source_layout import build_source_model, source_layout_tensors
from tests.common.mark_utils import arg_mark
from tests.ut.core.multicore.test_qwen_dense_model import _cpu_attention, _cpu_norm, _cpu_rotary


def _config():
    return QwenDenseConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, max_seq_len=8)


def _parameters(model):
    return {name: value.detach() for name, value in model.named_parameters()}


class TestSourceLayout(unittest.TestCase):
    """独立原始模型与候选必须使用同一逻辑参数，并保留完整状态和检查点信息。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_original_full_model_initialization_and_checkpoint_mapping(self):
        """Feature: 显式 Source-NK 模型接入。
        Description: 按相同种子分别构造原始模型与候选，再通过正式检查点接口恢复源布局。
        Expectation: 所有逻辑参数逐值一致，三个投影形状完整，候选保留各自数值程序。
        """
        for baseline in ("common", "packed"):
            with self.subTest(baseline=baseline):
                torch.manual_seed(17)
                source = QwenDenseModel(_config(), baseline)
                torch.manual_seed(17)
                candidate = build_source_model(_config(), baseline)
                expected = source_layout_tensors(source, _parameters(source))
                actual = source_layout_tensors(candidate, _parameters(candidate))
                self.assertEqual(set(actual), set(expected))
                for name, value in expected.items():
                    torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
                for layer in candidate.layers:
                    self.assertEqual(layer.mlp.weight_layout, "nk")
                    self.assertEqual(len(layer.mlp.weight_names), 3 if baseline == "common" else 2)
                recovered = revert_weight_conversion(candidate, candidate.state_dict())
                self.assertEqual(set(recovered), set(expected))
                for name, value in expected.items():
                    torch.testing.assert_close(recovered[name], value, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    @patch.object(DefaultDeviceType, "_default_device_type", "cpu")
    def test_independent_complete_cpu_common_training_preserves_source_state(self):
        """Feature: 完整 common 训练图接入。
        Description: 独立执行注意力、残差、FFN、损失和两次 SGD，覆盖重计算。
        Expectation: 每一步输出、全部梯度和更新后的参数与原始 common 基线逐值一致。
        """
        path = "hyper_parallel.core.multicore.examples.mega_moe.qwen_moe_model.run_qwen3_moe_flash_attention"
        with (patch(path, side_effect=_cpu_attention),
              patch("hyper_parallel.components.modules.rms_norm.rms_norm", side_effect=_cpu_norm),
              patch("hyper_parallel.core.multicore.examples.mega_moe.qwen_moe_model.apply_rotary_pos_emb",
                    side_effect=_cpu_rotary)):
            for recompute in (False, True):
                with self.subTest(recompute=recompute):
                    torch.manual_seed(17)
                    source = QwenDenseModel(_config(), "common", recompute=recompute).to(dtype=torch.bfloat16)
                    torch.manual_seed(17)
                    candidate = build_source_model(_config(), "common", recompute=recompute).to(dtype=torch.bfloat16)
                    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-3) for model in (source, candidate)]
                    tokens = torch.arange(16).view(2, 8)
                    for _step in range(2):
                        results = []
                        for model, optimizer in zip((source, candidate), optimizers):
                            optimizer.zero_grad(set_to_none=True)
                            result = model(tokens, tokens)
                            result["loss"].backward()
                            gradients = source_layout_tensors(model, {name: value.grad
                                                                     for name, value in model.named_parameters()})
                            optimizer.step()
                            results.append((result, gradients, source_layout_tensors(model, _parameters(model))))
                        reference, actual = results[0], results[1]
                        for name in ("logits", "loss"):
                            torch.testing.assert_close(reference[0][name], actual[0][name], rtol=0, atol=0)
                        for index in (1, 2):
                            self.assertEqual(set(reference[index]), set(actual[index]))
                            for name, value in reference[index].items():
                                torch.testing.assert_close(value, actual[index][name], rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_scalar_steps_are_complete_and_mapping_uses_matrix_views(self):
        """Feature: 优化器状态完整性。
        Description: 展开打包参数计数，检查矩阵映射为视图，再注入缺失键、错误形状和旧 KN 布局。
        Expectation: 三投影计数完整，矩阵不复制，无法证明源布局的状态严格拒绝。
        """
        model = build_source_model(_config(), "packed")
        steps = {name: torch.tensor(7) for name, _ in model.named_parameters()}
        mapped = source_layout_tensors(model, steps)
        self.assertEqual(mapped["layers.0.mlp.gate_proj.weight"].item(), 7)
        self.assertEqual(mapped["layers.0.mlp.up_proj.weight"].item(), 7)
        values = _parameters(model)
        mapped = source_layout_tensors(model, values)
        self.assertEqual(mapped["layers.0.mlp.gate_proj.weight"].untyped_storage().data_ptr(),
                         values["layers.0.mlp.gate_up"].untyped_storage().data_ptr())
        with self.assertRaisesRegex(ValueError, "every named"):
            source_layout_tensors(model, {})
        with self.assertRaisesRegex(ValueError, "invalid tensor shape"):
            source_layout_tensors(model, dict(values, **{"layers.0.mlp.gate_up": torch.empty(1)}))
        legacy = QwenDenseModel(_config(), "mega_ffn")
        with self.assertRaisesRegex(ValueError, "original NK"):
            source_layout_tensors(legacy, _parameters(legacy))
        with self.assertRaisesRegex(ValueError, "common or packed"):
            build_source_model(_config(), "mega_ffn")
