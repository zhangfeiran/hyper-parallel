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
"""CPU-only construction and input contracts for the optional full Qwen example."""

import importlib
import unittest
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from examples.torch.multicore.mega_moe.qwen_moe_config import QwenMoeConfig

# Only this model integration file requires the example's optional packages;
# configuration and namespace tests run without either dependency installed.
pytest.importorskip('transformers')
pytest.importorskip('torch_npu')

qwen_model = importlib.import_module('examples.torch.multicore.mega_moe.qwen_moe_model')
qwen_benchmark = importlib.import_module('examples.torch.multicore.mega_moe.qwen_moe_benchmark')


class _PositionRecorder(torch.nn.Module):
    """Inspect decoder inputs without invoking any NPU operator."""

    def __init__(self) -> None:
        """Retain only the last decoder input positions."""
        super().__init__()
        self.positions = None

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Keep the hidden states intact while retaining the positions for an assertion."""
        self.positions = positions
        return hidden


class TestQwenMoeModel(unittest.TestCase):
    """Exercise the actual CPU model factory without distributed setup or native execution."""

    @staticmethod
    def _config() -> QwenMoeConfig:
        return replace(QwenMoeConfig(), hidden_size=64, intermediate_size=32, num_layers=2,
                       num_attention_heads=2, num_key_value_heads=1, shared_expert_intermediate_size=32,
                       max_seq_len=128, local_num_tokens=128, vocab_size=64,
                       replica_slots_per_rank=1, shared_physical_weights=True)

    def test_layers_share_resources_but_not_parameters(self) -> None:
        """The second decoder must join the same B resource group without aliasing home weights."""
        model = qwen_model.QwenMoeModel(self._config())
        first, second = (layer.mlp.experts for layer in model.layers)
        self.assertTrue(first.shared_physical_weights)
        self.assertTrue(second.shared_physical_weights)
        self.assertIs(first._resource_group, second._resource_group)
        self.assertNotEqual(first.gate_up_weight.data_ptr(), second.gate_up_weight.data_ptr())
        self.assertNotEqual(first.down_weight.data_ptr(), second.down_weight.data_ptr())
        model.close()

    def test_rotary_positions_keep_batch_and_sequence_axes(self) -> None:
        """RoPE must receive [batch, sequence], including more than one batch row."""
        model = qwen_model.QwenMoeModel(self._config())
        decoder = _PositionRecorder()
        with patch.object(model, 'layers', torch.nn.ModuleList([decoder])), \
                patch.object(model, 'norm', torch.nn.Identity()):
            output = model(torch.zeros((2, 64), dtype=torch.long))
        self.assertEqual(tuple(output['logits'].shape), (2, 64, 64))
        self.assertEqual(tuple(decoder.positions.shape), (2, 64))
        torch.testing.assert_close(decoder.positions, torch.arange(64).expand(2, -1))
        model.close()


class TestExpertUpdateProbe(unittest.TestCase):
    """Keep optimizer validation meaningful when skew leaves an expert or owner cold."""

    def test_later_expert_update_is_visible(self) -> None:
        """A cold first expert cannot hide a later expert's parameter update."""
        parameter = torch.zeros((2, 5000))
        before = qwen_benchmark._expert_update_sample(parameter).clone()
        parameter[1, 0] = 1
        gradients = {'experts.0.gate_up_weight': torch.ones_like(parameter)}
        with patch.object(qwen_benchmark, '_all_ranks_true', side_effect=lambda value, _: value), \
                patch.object(qwen_benchmark, '_rank_max', side_effect=lambda value, _: value):
            updated = qwen_benchmark._expert_parameters_updated([(parameter, before)], gradients, torch.device('cpu'))
        self.assertTrue(updated)
        self.assertEqual(tuple(before.shape), (2, 4096))

    def test_cold_owner_needs_a_global_update(self) -> None:
        """An unchanged cold owner passes only when another owner actually updates."""
        parameter = torch.zeros((2, 4))
        probes = [(parameter, parameter.clone())]
        gradients = {'experts.0.gate_up_weight': torch.zeros_like(parameter),
                     'dense.weight': torch.ones_like(parameter)}
        for global_update in (0.0, 1.0):
            with self.subTest(global_update=global_update), \
                    patch.object(qwen_benchmark, '_all_ranks_true', side_effect=lambda value, _: value), \
                    patch.object(qwen_benchmark, '_rank_max', return_value=global_update) as rank_max:
                updated = qwen_benchmark._expert_parameters_updated(probes, gradients, torch.device('cpu'))
                self.assertEqual(updated, bool(global_update))
                rank_max.assert_called_once_with(0.0, torch.device('cpu'))

    def test_active_owner_without_update_still_fails(self) -> None:
        """Another owner's update cannot mask an unchanged active owner."""
        parameter = torch.zeros((2, 4))
        gradients = {'experts.0.gate_up_weight': torch.ones_like(parameter)}
        with patch.object(qwen_benchmark, '_all_ranks_true', side_effect=lambda value, _: value), \
                patch.object(qwen_benchmark, '_rank_max', return_value=1.0) as rank_max:
            updated = qwen_benchmark._expert_parameters_updated(
                [(parameter, parameter.clone())], gradients, torch.device('cpu'),
            )
        self.assertFalse(updated)
        rank_max.assert_called_once_with(0.0, torch.device('cpu'))
