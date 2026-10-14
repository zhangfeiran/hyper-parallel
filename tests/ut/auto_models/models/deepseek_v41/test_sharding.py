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
"""Keep variable-image visual execution inside one root FSDP invocation."""

import unittest

from torch import nn

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models.deepseek_v41.adapter.policies.sharding import (
    get_fsdp_excluded_subtrees,
    get_fsdp_execution_order,
    get_fsdp_wrap_modules,
)


class TestVisualFSDPPolicy(unittest.TestCase):
    """Visual parameters must remain root-owned when image counts differ by rank."""

    def test_visual_loop_has_no_nested_collective_boundaries(self) -> None:
        """Repeated visual forwards must not issue rank-dependent FSDP collectives."""
        model = nn.Module()
        model.model = nn.Module()
        model.model.gradient_checkpointing = False
        model.model.layers = nn.ModuleList([nn.Module()])
        layer = model.model.layers[0]
        layer.engram = nn.Module()
        layer.engram.embed = nn.Embedding(8, 4)
        layer.engram.wkv = nn.Linear(4, 4)
        model.model.vision = nn.Module()
        model.model.vision.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        model.model.aligner = nn.Module()
        model.model.aligner.w1 = nn.Linear(4, 4)
        model.model.aligner.w2 = nn.Linear(4, 4)

        declared = get_fsdp_wrap_modules(model)
        self.assertEqual(declared, ("model.layers.0.engram.embed", "model.layers.0.engram.wkv"))
        excluded = set(get_fsdp_excluded_subtrees(model))
        self.assertEqual(excluded, {"model.vision", "model.aligner"})
        automatic, _ = FSDP2Manager._find_transformer_block_modules(  # pylint: disable=protected-access
            model, excluded_subtree_module_fqns=excluded | set(declared),
        )
        self.assertEqual([unit.fqn for unit in automatic], ["model.layers.0"])
        selected = ("model.layers.0", *declared)
        self.assertEqual(get_fsdp_execution_order(model, tuple(reversed(selected))), selected)
