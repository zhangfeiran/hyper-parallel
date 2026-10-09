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
"""Bounded CPU training oracle checked against the expanded selected-KV reference."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch.nn import functional

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    _index_inputs, _main_inputs, _masked_distribution, _selection,
)


def _dense_chunk(inputs, indices, meta, normalization, coefficient, attention_scale):
    main = _main_inputs(inputs[:4], meta)
    index = _index_inputs(*inputs[4:], meta)
    offsets, valid = _selection(indices, meta)
    query, key, query_rope, key_rope = main
    padded_key = torch.cat((key, key.new_zeros((1, key.shape[-1]))))
    padded_rope = torch.cat((key_rope, key_rope.new_zeros((1, key_rope.shape[-1]))))
    scores = (query @ padded_key.T + query_rope @ padded_rope.T) * attention_scale
    addresses = offsets[:, None].expand(-1, query.shape[1], -1)
    selected = scores.gather(2, addresses)
    probability, _ = _masked_distribution(selected, valid[:, None])
    # Scatter probabilities, rather than expanded C512 keys, keeps the CPU KV VJP
    # in matrix multiplies while preserving every selected slot and its original order.
    weights = torch.zeros_like(scores).scatter_add(2, addresses, probability)
    output = weights @ padded_key
    index_query, index_key, merge_weight = index
    padded_index = torch.cat((index_key, index_key.new_zeros((1, index_key.shape[-1]))))
    index_scores = (index_query @ padded_index.T).gather(
        2, offsets[:, None].expand(-1, index_query.shape[1], -1))
    logits = (index_scores.relu() * merge_weight[..., None]).sum(1)
    _, log_probability = _masked_distribution(logits, valid)
    teacher = probability.detach().sum(1)
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1)
    loss = functional.kl_div(log_probability, teacher, reduction="sum")
    return output, loss * (coefficient * normalization.local_sum_scale)


def training_reference(
    inputs: tuple[torch.Tensor, ...], cotangent: torch.Tensor, indices: torch.Tensor,
    normalization: DsaLossNormalization, coefficient: float, mode: str, auxiliary: float,
    lengths: tuple[int, ...], *, attention_scale: float, query_chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """Evaluate every query and seven FP32 VJPs with bounded query-chunk storage.

    The dense score matrices are filtered by the original sparse selection.
    Teacher probabilities are detached, and full key leaves accumulate every
    chunk's contribution. No query sampling or device/native calls are used.
    The companion unit test compares this representation with the independent
    expanded selected-KV oracle, including signed weights and packed masking.
    """
    if not isinstance(query_chunk, int) or isinstance(query_chunk, bool) or query_chunk < 1:
        raise ValueError("query_chunk must be a positive integer")
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and nonnegative")
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("attention_scale must be finite and positive")
    meta = DsaBatchMeta.packed(lengths)
    if normalization.global_valid_queries != meta.global_valid_queries:
        raise ValueError("normalization must use the complete packed query count")
    full = tuple(tensor.detach().float().cpu().requires_grad_() for tensor in inputs)
    cotangent = cotangent.float().cpu()
    outputs, losses = [], []
    query_gradients = {field: [] for field in (0, 2, 4, 6)}
    for start in range(0, sum(lengths), query_chunk):
        end = min(start + query_chunk, sum(lengths))
        local_meta = replace(meta, q_global_ids=meta.q_global_ids[start:end])
        local = tuple(tensor if field in (1, 3, 5) else tensor[start:end].detach().requires_grad_()
                      for field, tensor in enumerate(full))
        output, loss = _dense_chunk(local, indices[start:end], local_meta,
                                   normalization, coefficient, attention_scale)
        if mode == "kl_only":
            objective = loss * auxiliary
        else:
            objective = (output * cotangent[start:end]).sum()
            if mode != "lm_only":
                objective = objective + loss * auxiliary
        objective.backward()
        outputs.append(output.detach())
        losses.append(loss.detach())
        for field, gradients in query_gradients.items():
            gradients.append(local[field].grad)
    gradients = tuple(tensor.grad if field in (1, 3, 5) else
                      None if query_gradients[field][0] is None else torch.cat(query_gradients[field])
                      for field, tensor in enumerate(full))
    return torch.cat(outputs), torch.stack(losses).sum(), gradients
