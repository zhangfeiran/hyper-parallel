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
"""Verify the fused-compatible vision attention layout and eager math parity."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional

from hyper_parallel.models.deepseek_v41 import vision


class TestVisionAttention(unittest.TestCase):
    """Compare full-image attention and gradients against explicit dense math."""

    def test_batched_head_layout_matches_dense_attention_and_gradients(self) -> None:
        """Four-dimensional SDPA must preserve one-image outputs and all gradients."""
        torch.manual_seed(314)
        module = vision.DeepseekV41VisionAttention(16, 2).double()
        reference = copy.deepcopy(module)
        inputs = torch.randn(12, 16, dtype=torch.float64, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        angles = torch.randn(12, 1, 4, dtype=torch.float64)
        cosine, sine = angles.cos(), angles.sin()
        with patch.object(vision.functional, "scaled_dot_product_attention",
                          wraps=functional.scaled_dot_product_attention) as attention:
            actual = module(inputs, cosine, sine)
        self.assertEqual(tuple(attention.call_args.args[0].shape), (1, 2, 12, 8))
        self.assertFalse(attention.call_args.kwargs["is_causal"])

        query, key, value = (part.reshape(12, 2, 8) for part in reference.wqkv(reference_inputs).chunk(3, -1))
        query = vision._apply_vision_rotary(query, cosine, sine)  # pylint: disable=protected-access
        key = vision._apply_vision_rotary(key, cosine, sine)  # pylint: disable=protected-access
        scores = torch.einsum("thd,shd->hts", query, key) / 8**0.5
        expected = reference.wo(torch.einsum("hts,shd->thd", scores.softmax(-1), value).reshape(12, 16))
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=1e-9, atol=1e-9)
        for (name, parameter), (reference_name, reference_parameter) in zip(
                module.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, reference_name)
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-9, atol=1e-9)
