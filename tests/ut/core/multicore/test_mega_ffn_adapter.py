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
"""Existing declarative replacement/checkpoint integration for dense FFN."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)
from tests.common.mark_utils import arg_mark


class _SourceFFN(nn.Module):
    def __init__(self) -> None:
        """Create independent bias-free source projection parameters."""
        super().__init__()
        self.gate_proj = nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(6, 8, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the source projection graph.

        Args:
            x: CPU BF16 input tokens.
        """
        gate, up = self.gate_proj(x).float(), self.up_proj(x).float()
        return self.down_proj((torch.nn.functional.silu(gate) * up).to(x.dtype))


class TestMegaFFNAdapter(unittest.TestCase):
    """Exercise actual replacement and reversible weight conversion surfaces."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_existing_replacement_context_selects_dense_backend(self):
        """Feature: Dense backend configuration through the existing replacement API.
        Description: Supply a typed resident configuration and then an invalid execution context.
        Expectation: The adapter retains the configuration while CPU references stay explicit.
        """
        source = _SourceFFN()
        execution = DenseExecutionConfig(backend="resident_tiles", soc="Ascend910B3")
        model = nn.Module()
        model.mlp = source
        plan = compile_module_replacements(model, [ModuleReplacementSpec(("mlp",), MegaFFNAdapter, _SourceFFN)])
        apply_module_replacements(model, plan, weights_mapping=[], context={"mega_ffn_execution": execution})
        self.assertIs(model.mlp.execution, execution)
        x = torch.randn(7, 8, dtype=torch.bfloat16)
        torch.testing.assert_close(model.mlp(x), source(x), rtol=0.02, atol=0.002)
        with self.assertRaisesRegex(ValueError, "DenseExecutionConfig"):
            MegaFFNAdapter(module=source, context={"mega_ffn_execution": {"backend": "resident_tiles"}})
        with self.assertRaisesRegex(ValueError, "backend"):
            DenseExecutionConfig(backend="unknown")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_declarative_replacement_preserves_aliases_and_forward(self):
        """Feature: Formal FFN replacement surface.
        Description: Replace an aliased MLP using the existing declarative plan executor.
        Expectation: Both aliases see one adapter with two packed parameters and equivalent outputs.
        """
        source = _SourceFFN().eval()
        model = nn.Module()
        model.mlp, model.alias = source, source
        x = torch.randn(2, 7, 8, dtype=torch.bfloat16)
        expected = source(x)
        plan = compile_module_replacements(model, [ModuleReplacementSpec(("mlp",), MegaFFNAdapter, _SourceFFN)])
        replacement, mapping = apply_module_replacements(model, plan, weights_mapping=[])
        self.assertIs(replacement.mlp, replacement.alias)
        self.assertIsInstance(replacement.mlp, MegaFFNAdapter)
        self.assertFalse(replacement.mlp.training)
        self.assertEqual(len(list(replacement.parameters())), 2)
        self.assertEqual(len(mapping), 2)
        torch.testing.assert_close(replacement.mlp(x), expected, rtol=0.02, atol=0.002)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_checkpoint_conversion_roundtrips_original_projection_weights(self):
        """Feature: Dense checkpoint compatibility.
        Description: Convert and reverse-convert both packed projection checkpoint layouts.
        Expectation: Original gate, up and down weights are restored exactly.
        """
        source = _SourceFFN()
        adapter = MegaFFNAdapter(module=source)
        gate_up, down = adapter.make_transforms()
        gate_up.collected_tensors["gate_proj.weight"] = [source.gate_proj.weight]
        gate_up.collected_tensors["up_proj.weight"] = [source.up_proj.weight]
        packed = gate_up.convert("gate_up", model=adapter)
        torch.testing.assert_close(packed["gate_up"], adapter.gate_up, rtol=0, atol=0)
        inverse = gate_up.reverse_transform()
        inverse.collected_tensors["gate_up"] = [packed["gate_up"]]
        restored = inverse.convert("gate_up", model=adapter)
        torch.testing.assert_close(restored["gate_proj.weight"], source.gate_proj.weight, rtol=0, atol=0)
        torch.testing.assert_close(restored["up_proj.weight"], source.up_proj.weight, rtol=0, atol=0)
        down.collected_tensors["down_proj.weight"] = [source.down_proj.weight]
        converted = down.convert("down", model=adapter)
        inverse = down.reverse_transform()
        inverse.collected_tensors["down"] = [converted["down"]]
        restored = inverse.convert("down", model=adapter)
        torch.testing.assert_close(restored["down_proj.weight"], source.down_proj.weight, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_replacement_rejects_tied_weights_and_mixed_training_policy(self):
        """Feature: Source FFN training invariants.
        Description: Attempt to pack tied projections or gate/up with different requires_grad.
        Expectation: Unsupported parameter ownership is rejected before replacement.
        """
        source = _SourceFFN()
        source.up_proj.weight = source.gate_proj.weight
        with self.assertRaisesRegex(ValueError, "distinct"):
            MegaFFNAdapter(module=source)
        source = _SourceFFN()
        source.up_proj.weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "training policies"):
            MegaFFNAdapter(module=source)
