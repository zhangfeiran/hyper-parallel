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
"""Source rotary values, independent FP64 Jacobians and retained tensor versions."""

import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.modules.mega_dsa.source_rope import (
    source_attention_rope_with_fp32_vjp,
)


def _rotate(value, cos, sin, *, interleaved):
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if interleaved:
        cos = cos[..., :cos.shape[-1]//2].repeat_interleave(2, -1)
        sin = sin[..., :sin.shape[-1]//2].repeat_interleave(2, -1)
        rotated = torch.stack((-value[..., 1::2], value[..., 0::2]), -1).flatten(-2)
    else:
        first, second = value.chunk(2, -1)
        rotated = torch.cat((-second, first), -1)
    output = value*cos.unsqueeze(2)+rotated*sin.unsqueeze(2)
    if interleaved:
        output = torch.cat((output[..., 0::2], output[..., 1::2]), -1)
    return output


def _source(query, key, embeddings, *, interleaved):
    return tuple(_rotate(value, *embeddings, interleaved=interleaved) for value in (query, key))


class TestSourceRope(unittest.TestCase):
    """Compare custom VJPs to an independently differentiated FP64 forward."""

    def test_layout_broadcast_empty_and_checkpoint_vjps(self):
        """Unequal sin halves catch the incorrect shortcut of multiplying after rotation transpose."""
        torch.manual_seed(519)
        for interleaved in (False, True):
            for batch, tokens, embedding_batch in ((1, 3, 1), (2, 3, 1), (2, 3, 2), (1, 0, 1)):
                for replay in ('plain', 'retained', 'checkpoint'):
                    with self.subTest(layout=interleaved, geometry=(batch, tokens, embedding_batch),
                                      replay=replay):
                        query = torch.randn(batch, tokens, 4, 8).bfloat16().float().requires_grad_()
                        key = torch.randn(batch, tokens, 1, 8).bfloat16().float().requires_grad_()
                        cos = torch.randn(embedding_batch, tokens, 8).bfloat16()
                        sin = torch.randn_like(cos)
                        query_upstream = torch.randn_like(query)*1.003
                        key_upstream = torch.randn_like(key)*.997
                        reference = tuple(value.detach().double().requires_grad_() for value in (query, key))
                        expected = tuple(_rotate(value, cos.double(), sin.double(), interleaved=interleaved)
                                         for value in reference)
                        expected_gradients = torch.autograd.grad(expected, reference, (query_upstream.double(),
                                                                                       key_upstream.double()))

                        def _operation(query_value, key_value, embeddings=(cos, sin), layout=interleaved):
                            return source_attention_rope_with_fp32_vjp(query_value, key_value, embeddings,
                                                                       interleaved=layout)

                        with patch('hyper_parallel.core.multicore.modules.mega_dsa.source_rope._apply_attention_rope',
                                   side_effect=_source):
                            with torch.autocast('cpu', dtype=torch.bfloat16):
                                actual = (checkpoint(_operation, query, key, use_reentrant=False)
                                          if replay == 'checkpoint' else _operation(query, key))
                            for result, source in zip(actual, _source(query.bfloat16(), key.bfloat16(),
                                                                      (cos, sin), interleaved=interleaved)):
                                torch.testing.assert_close(result, source.float(), rtol=0, atol=0)
                            gradients = torch.autograd.grad(actual, (query, key), (query_upstream, key_upstream),
                                                            retain_graph=replay == 'retained')
                            for result, expected_gradient in zip(gradients, expected_gradients):
                                torch.testing.assert_close(result.double(), expected_gradient, rtol=2e-6, atol=2e-6)
                            if replay == 'retained':
                                second = torch.autograd.grad(actual, (query, key), (query_upstream, key_upstream))
                                for first, repeated in zip(gradients, second):
                                    torch.testing.assert_close(first, repeated, rtol=0, atol=0)

    def test_versions_and_boundary_admission(self):
        """Mutation of an original frequency or activation invalidates delayed backward."""
        query = torch.ones(1, 3, 4, 8, requires_grad=True)
        key = torch.ones(1, 3, 1, 8, requires_grad=True)
        cos, sin = torch.ones(3, 8), torch.zeros(3, 8)
        self.assertIs(source_attention_rope_with_fp32_vjp(query, key, None, interleaved=False)[0], query)
        for embeddings in ((cos.requires_grad_(), sin), (cos.detach()[:, :4], sin[:, :4])):
            with self.assertRaises(ValueError):
                source_attention_rope_with_fp32_vjp(query, key, embeddings, interleaved=False)
        cos = cos.detach()
        for mutated in (cos, query):
            with patch('hyper_parallel.core.multicore.modules.mega_dsa.source_rope._apply_attention_rope',
                       side_effect=_source):
                output = source_attention_rope_with_fp32_vjp(query, key, (cos, sin), interleaved=False)
            with torch.no_grad():
                mutated.add_(1)
            with self.assertRaises(RuntimeError):
                torch.autograd.grad(output[0].sum(), query)
        with self.assertRaises(ValueError):
            source_attention_rope_with_fp32_vjp(query.bfloat16(), key, None, interleaved=False)
