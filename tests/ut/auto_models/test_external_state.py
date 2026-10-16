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
"""Host table and variable-image FSDP ownership boundaries."""

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.deepseek_v41.adapter.policies.sharding import (
    get_fsdp_wrap_modules, get_fsdp_excluded_subtrees, get_fsdp_execution_order,
)


class TestExternalStateOwnership(unittest.TestCase):
    """Keep CPU tables and per-image parameters under their selected owners."""

    def test_fsdp_excludes_host_table_and_keeps_dense_projection(self) -> None:
        """The owner table stays CPU-owned while the Engram projection remains an FSDP unit."""
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        layer = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([layer])
        layer.engram = torch.nn.Module()
        layer.engram.embed = HostEngramTable(
            source_weight=torch.nn.Parameter(torch.empty((16, 2), device="meta")), logical_rows=15,
        )
        layer.engram.wkv = torch.nn.Linear(2, 2)
        self.assertEqual(get_fsdp_wrap_modules(model), ("model.layers.0.engram.wkv",))
        layer.engram.embed = torch.nn.Embedding(16, 2)
        self.assertEqual(get_fsdp_wrap_modules(model),
                         ("model.layers.0.engram.embed", "model.layers.0.engram.wkv"))

    def test_variable_image_recipe_keeps_visual_collectives_outside_image_loop(self) -> None:
        """Selecting root ownership removes every per-image FSDP unit and prefetch target."""
        model = torch.nn.Module()
        model.config = SimpleNamespace(v41_root_owned_vision=False)
        model.model = torch.nn.Module()
        model.model.vision = torch.nn.Module()
        model.model.vision.patch_embed = torch.nn.Linear(2, 2)
        model.model.vision.norm = torch.nn.LayerNorm(2)
        model.model.vision.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        model.model.aligner = torch.nn.Linear(2, 2)
        wrapped = get_fsdp_wrap_modules(model)
        self.assertIn("model.vision.blocks.0", wrapped)
        self.assertIn("model.aligner", wrapped)
        model.config.v41_root_owned_vision = True
        self.assertEqual(get_fsdp_wrap_modules(model), ())
        self.assertEqual(get_fsdp_excluded_subtrees(model), ("model.vision", "model.aligner"))
        self.assertEqual(get_fsdp_execution_order(model, ()), ())
