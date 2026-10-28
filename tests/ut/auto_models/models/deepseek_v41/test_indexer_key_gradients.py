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
"""Validate compressed Indexer gradients against an autograd objective."""

import unittest

import torch

from hyper_parallel.components.modules.shared_compressed_dsa_attention import shared_compressed_indexer_kl_loss


def _reference_loss(
        inputs: list[torch.Tensor],
        teacher: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Differentiate the complete selected-key objective without manual gradients."""
    query, key, weights = inputs
    attention_query, attention_key, indices, sinks = teacher
    batch_indices = torch.arange(query.shape[0]).view(-1, 1, 1)
    valid = indices >= 0
    selected_key = key[batch_indices, indices.clamp_min(0)].float()
    dots = torch.einsum("bsid,bskd->bsik", query.float(), selected_key).relu()
    scores = (dots * weights.float().unsqueeze(-1)).sum(dim=2).masked_fill(~valid, -1e9)

    selected_teacher = attention_key[batch_indices, indices.clamp_min(0)].float()
    teacher_scores = torch.einsum("bhsd,bskd->bhsk", attention_query.float(), selected_teacher) * 0.5
    teacher_scores = teacher_scores.masked_fill(~valid.unsqueeze(1), -1e9)
    sink_scores = sinks.view(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[1], -1)
    target = torch.cat((teacher_scores, sink_scores), dim=-1).softmax(dim=-1)
    target = target[..., :-1].masked_fill(~valid.unsqueeze(1), 0).sum(dim=1)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    row_loss = (
        target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - scores.log_softmax(-1))
    ).sum(-1)
    return row_loss[valid.any(-1)].sum() * (0.1 / (query.shape[0] * query.shape[1]))


class TestIndexerKeyGradients(unittest.TestCase):
    """Keep repeated keys, invalid positions and batches numerically independent."""

    def test_chunked_gradients_match_autograd_with_repeated_and_invalid_keys(self) -> None:
        """Compare loss and all three gradients across layouts and chunk boundaries."""
        for batch_size in (1, 2):
            for noncontiguous in (False, True):
                for chunk_size in (1, 2, 4):
                    with self.subTest(batch=batch_size, noncontiguous=noncontiguous, chunk=chunk_size):
                        generator = torch.Generator().manual_seed(41)
                        query = torch.randn(batch_size, 6, 2, 3, generator=generator)
                        key = torch.randn(batch_size, 4, 3, generator=generator)
                        if noncontiguous:
                            key = key.transpose(1, 2).contiguous().transpose(1, 2)
                        weights = torch.randn(batch_size, 6, 2, generator=generator)
                        indices = torch.tensor([
                            [-1, -1, -1], [-1, -1, -1], [0, 0, 1],
                            [2, -1, 2], [3, 1, 1], [0, 3, -1],
                        ]).unsqueeze(0).expand(batch_size, -1, -1).clone()
                        if batch_size == 2:
                            indices[1] = indices[1].flip(0)
                        teacher = (
                            torch.randn(batch_size, 2, 6, 4, generator=generator),
                            torch.randn(batch_size, 4, 4, generator=generator),
                            indices,
                            torch.randn(2, generator=generator),
                        )
                        expected_inputs = [value.clone().requires_grad_() for value in (query, key, weights)]
                        actual_inputs = [value.clone().requires_grad_() for value in (query, key, weights)]

                        expected_loss = _reference_loss(expected_inputs, teacher)
                        expected_loss.backward()
                        actual_loss = shared_compressed_indexer_kl_loss(
                            *actual_inputs, *teacher,
                            attention_scale=0.5, loss_coeff=0.1, query_chunk_size=chunk_size,
                        )
                        actual_loss.backward()

                        torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=2e-6)
                        for actual, expected in zip(actual_inputs, expected_inputs):
                            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6)
