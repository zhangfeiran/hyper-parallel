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
"""Backend-free contracts for the full Qwen benchmark's shared-B configuration."""

import unittest
from dataclasses import replace

from examples.torch.multicore.mega_moe.qwen_moe_config import QwenMoeConfig, make_config, parse_args


class TestQwenMoeConfiguration(unittest.TestCase):
    """Keep defaults stable while validating shared storage before distributed initialization."""

    def test_defaults_preserve_original_workload(self) -> None:
        """Opting into shared-B or a smaller workload must remain explicit."""
        config = make_config(parse_args([]))
        self.assertEqual(config, QwenMoeConfig())
        self.assertFalse(config.shared_physical_weights)
        self.assertEqual((config.num_layers, config.hidden_size, config.local_num_tokens), (8, 5120, 4096))

    def test_shared_small_shape_reaches_complete_configuration(self) -> None:
        """Every shape and routing option must survive the CLI-to-model boundary."""
        config = make_config(parse_args([
            '--shared-physical-weights', '--replica-slots-per-rank', '1', '--num-layers', '2',
            '--hidden-size', '1024', '--intermediate-size', '256', '--vocab-size', '2048',
            '--sequence-length', '1024', '--expert-capacity-factor', '1.5', '--routing', 'dirichlet',
            '--router-load-alpha', '0.1',
        ]))
        self.assertTrue(config.shared_physical_weights)
        self.assertFalse(config.compact_physical_weights)
        self.assertEqual((config.num_layers, config.hidden_size, config.intermediate_size), (2, 1024, 256))
        self.assertEqual((config.local_num_tokens, config.max_seq_len, config.vocab_size), (1024, 1024, 2048))
        self.assertEqual((config.replica_slots_per_rank, config.expert_capacity_factor), (1, 1.5))
        self.assertEqual((config.routing, config.router_load_alpha), ('dirichlet', 0.1))

    def test_shared_rejects_missing_slots_compact_and_non_boolean(self) -> None:
        """The example must reject the same invalid storage modes as the underlying module."""
        for options in ({'shared_physical_weights': True},
                        {'shared_physical_weights': True, 'replica_slots_per_rank': 1,
                         'compact_physical_weights': True},
                        {'shared_physical_weights': 1, 'replica_slots_per_rank': 1}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'shared_physical_weights'):
                replace(QwenMoeConfig(), **options)

    def test_invalid_cli_shapes_and_storage_fail_before_runtime(self) -> None:
        """Malformed dimensions must not initialize a distributed group first."""
        for arguments in (['--shared-physical-weights'], ['--hidden-size', '0'], ['--num-layers', '0'],
                          ['--hidden-size', '1000'], ['--sequence-length', '1000']):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                parse_args(arguments)
