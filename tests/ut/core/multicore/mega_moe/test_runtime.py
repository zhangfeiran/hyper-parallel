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
"""Unit tests for MegaMoe runtime construction."""

import unittest

import torch

from hyper_parallel.core.multicore.platform.torch.mega_moe.config import MegaMoeConfig
from hyper_parallel.core.multicore.platform.torch.mega_moe.runtime import (
    _build_route_slots,
    _build_task_values,
)


class TestMegaMoeRuntime(unittest.TestCase):
    """Validate translation from the public local-token contract."""

    def test_local_tokens_use_one_legacy_scheduler_partition(self):
        """Preserve routed dimensions without exposing TP in MegaMoe config."""
        config = MegaMoeConfig(
            local_num_tokens=4096,
            hidden_size=5120,
            intermediate_size=1792,
            num_experts=32,
            top_k=8,
            receive_capacity=40960,
            ep_size=4,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
        )

        task_values = _build_task_values(config)

        self.assertEqual(task_values.tp, 1)
        self.assertEqual(task_values.seq_size, config.local_num_tokens)
        self.assertEqual(
            task_values.per_rank_seq, config.local_num_tokens * config.top_k
        )
        self.assertEqual(task_values.per_expert_seq_to_other, config.local_num_tokens)

    def test_permute_fusion_caches_each_route_slot(self):
        """Represent every Top-K slot exactly once for expert-major routing."""
        config = MegaMoeConfig(
            local_num_tokens=4,
            hidden_size=8,
            intermediate_size=4,
            num_experts=4,
            top_k=2,
            receive_capacity=128,
            ep_size=1,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
            forward_fusion_mode="permute",
        )

        route_slots = _build_route_slots(config, "cpu")

        self.assertEqual(tuple(route_slots.shape), (config.routed_slots, 1))
        self.assertEqual(route_slots.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                route_slots.reshape(-1),
                torch.arange(config.routed_slots, dtype=torch.float32),
            )
        )
