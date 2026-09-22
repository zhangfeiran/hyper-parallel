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
"""CPU contracts for the synthetic VLM workload."""
import unittest

import torch

from examples.training_demo.deepseek_v41_random_input import randomize_sample
from tests.common.mark_utils import arg_mark


class TestRandomVLMInput(unittest.TestCase):
    """Verify determinism and unchanged geometry, weights RNG and loss density."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_content_changes_without_geometry_or_rng_changes(self):
        """Feature: random inputs. Description: randomize padding. Expectation: stable shapes and RNG."""
        sample = {"input_ids": torch.tensor([2, 99, 99, 7, 2]),
                  "labels": torch.tensor([-100, -100, -100, 7, -100]),
                  "token_types": torch.tensor([-1, 0, 3, -1, -1]),
                  "attention_mask": torch.tensor([1, 1, 1, 1, 0]),
                  "pixel_values": torch.zeros(2, 3, 14, 14, dtype=torch.bfloat16),
                  "image_token_starts": torch.tensor([1])}
        state = torch.random.get_rng_state().clone()
        pool = torch.arange(3, 50)
        first = randomize_sample(sample, pool, 17)
        again = randomize_sample(sample, pool, 17)
        for key in sample:
            torch.testing.assert_close(first[key], again[key], rtol=0, atol=0)
            self.assertEqual(first[key].shape, sample[key].shape)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        self.assertEqual(first["input_ids"][1:3].tolist(), [99, 99])
        self.assertTrue(first["attention_mask"].eq(1).all())
        self.assertEqual(first["labels"].ne(-100).sum().item(), 1)
        self.assertTrue(first["input_ids"][[0, 3, 4]].ge(3).all())
        self.assertTrue(first["pixel_values"].abs().le(1).all())
        self.assertFalse(first["pixel_values"].eq(0).all())
        self.assertIs(first["image_token_starts"], sample["image_token_starts"])
        self.assertEqual(sample["input_ids"].tolist(), [2, 99, 99, 7, 2])
