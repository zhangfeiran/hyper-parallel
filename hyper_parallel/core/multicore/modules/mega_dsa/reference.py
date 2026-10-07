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
"""Small CPU FP32/FP64 DSA oracle, deliberately unsuitable for training dispatch."""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)


@dataclass(frozen=True)
class DsaReferenceStats:
    """Natural-exponent softmax statistics in oracle-only [Tq, H] layout."""

    maximum: torch.Tensor
    denominator: torch.Tensor

    @property
    def lse(self) -> torch.Tensor:
        """Return natural-log LSE; empty rows have negative infinity."""
        return self.maximum + self.denominator.log()


def _working_tensors(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if any(tensor.device.type != "cpu" or not tensor.is_floating_point() for tensor in tensors):
        raise ValueError("the small DSA oracle requires floating-point CPU tensors")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("oracle inputs must have the same floating-point dtype")
    dtype = torch.float64 if tensors[0].dtype == torch.float64 else torch.float32
    return tuple(tensor.to(dtype) for tensor in tensors)


def _selection(indices: torch.Tensor, meta: DsaBatchMeta) -> tuple[torch.Tensor, torch.Tensor]:
    if indices.device.type != "cpu" or indices.dtype != torch.int32:
        raise ValueError("topk_indices must be an int32 CPU tensor")
    if indices.ndim != 2 or indices.shape[0] != len(meta.q_global_ids) or indices.shape[1] < 1:
        raise ValueError("topk_indices must have shape [local queries, positive K]")
    storage = {token: offset for offset, token in enumerate(meta.kv_global_ids)}
    offsets = torch.full(indices.shape, len(storage), dtype=torch.long)
    valid = torch.zeros(indices.shape, dtype=torch.bool)
    for row, (query_id, selected) in enumerate(zip(meta.q_global_ids, indices.tolist())):
        query_sequence, query_position = meta.sequence_position(query_id)
        seen = set()
        for slot, token in enumerate(selected):
            if token == -1:
                continue
            sequence, position = meta.sequence_position(token)
            if token in seen:
                raise ValueError("Top-K must not contain duplicate valid token IDs")
            seen.add(token)
            if sequence != query_sequence or position > query_position:
                continue
            if token not in storage:
                raise ValueError("selected causal KV is missing; fetch or replicate it before evaluation")
            offsets[row, slot] = storage[token]
            valid[row, slot] = True
    return offsets, valid


def _gather_keys(keys: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    # Padding has its own zero row so -1 never aliases a real token or its gradient.
    padded = torch.cat((keys, keys.new_zeros((1, keys.shape[-1]))), dim=0)
    return padded[offsets]


def _masked_distribution(logits: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nonempty = valid.any(dim=-1, keepdim=True)
    masked = logits.masked_fill(~valid, -torch.inf)
    safe = torch.where(nonempty, masked, torch.zeros_like(masked))
    log_probability = safe.log_softmax(dim=-1).masked_fill(~valid, 0)
    probability = log_probability.exp().masked_fill(~valid, 0)
    return probability, log_probability


def _attention_logits(
    inputs: tuple[torch.Tensor, ...], offsets: torch.Tensor, attention_scale: float,
) -> torch.Tensor:
    query, compressed_kv, query_rope, key_rope = inputs
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("attention_scale must be the finite positive model scale")
    query_tokens, heads, compressed_dim = query.shape
    if compressed_kv.ndim != 2 or compressed_kv.shape[1] != compressed_dim:
        raise ValueError("compressed_kv must be [Tk, C] matching query [Tq, H, C]")
    if query_rope.ndim != 3 or query_rope.shape[:2] != (query_tokens, heads):
        raise ValueError("query_rope must be [Tq, H, Dr]")
    if key_rope.shape != (compressed_kv.shape[0], query_rope.shape[-1]):
        raise ValueError("key_rope must be [Tk, Dr]")
    scores = torch.einsum("thc,tkc->thk", query, _gather_keys(compressed_kv, offsets))
    scores = scores + torch.einsum("thd,tkd->thk", query_rope, _gather_keys(key_rope, offsets))
    return scores * attention_scale


def _main_inputs(
    inputs: tuple[torch.Tensor, ...], meta: DsaBatchMeta,
) -> tuple[torch.Tensor, ...]:
    tensors = _working_tensors(*inputs)
    if tensors[0].ndim != 3 or tensors[0].shape[0] != len(meta.q_global_ids) or tensors[0].shape[1] < 1:
        raise ValueError("query must be [local queries, positive H, C]")
    if tensors[1].ndim != 2 or tensors[1].shape[0] != len(meta.kv_global_ids):
        raise ValueError("compressed_kv must match the declared KV storage")
    return tensors


def sparse_attention_reference(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    batch_meta: DsaBatchMeta,
    *,
    attention_scale: float,
) -> tuple[torch.Tensor, DsaReferenceStats]:
    """Evaluate selected absorbed MQA with ordinary Torch autograd.

    Inputs follow [Tq,H,C], [Tk,C], [Tq,H,Dr], [Tk,Dr], [Tq,K]. Indices
    are global packed IDs; causal/packed filtering uses logical positions.
    Empty candidate rows return zero output, max=-inf and sum=0. The returned
    compressed output has no backend RoPE padding. FP16/BF16 are promoted to
    FP32, and FP64 is preserved for gradcheck. This materializes selected KV
    and is a small validation oracle, never a production backend.
    """
    inputs = _main_inputs((query, compressed_kv, query_rope, key_rope), batch_meta)
    offsets, valid = _selection(topk_indices, batch_meta)
    logits = _attention_logits(inputs, offsets, attention_scale)
    probability, _ = _masked_distribution(logits, valid[:, None, :])
    output = torch.einsum("thk,tkc->thc", probability, _gather_keys(inputs[1], offsets))
    maximum = logits.masked_fill(~valid[:, None, :], -torch.inf).amax(dim=-1)
    safe_maximum = torch.where(valid.any(dim=-1)[:, None], maximum, torch.zeros_like(maximum))
    exponent = (logits - safe_maximum[..., None]).masked_fill(~valid[:, None, :], -torch.inf).exp()
    return output, DsaReferenceStats(maximum, exponent.sum(dim=-1))


def _index_inputs(
    index_query: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor, meta: DsaBatchMeta,
) -> tuple[torch.Tensor, ...]:
    query, key, weight = _working_tensors(index_query, index_key, merge_weight)
    if query.ndim != 3 or query.shape[0] != len(meta.q_global_ids) or query.shape[1] < 1:
        raise ValueError("index_query must be [local queries, positive index heads, Di]")
    if key.shape != (len(meta.kv_global_ids), query.shape[-1]) or weight.shape != query.shape[:2]:
        raise ValueError("index_key [Tk, Di] and merge_weight [Tq, Hi] must match index_query")
    return query, key, weight


def indexer_reference(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    merge_weight: torch.Tensor,
    batch_meta: DsaBatchMeta,
    *,
    sparse_count: int,
) -> torch.Tensor:
    """Compute exact global Top-K, with ascending global ID as the oracle tie policy.

    Requires the full global key set, in any storage order. merge_weight is
    already scaled by the model and may be negative. No additional scale or
    weight activation is applied. Selection has no autograd path; this oracle
    tie policy still requires comparison with the installed CANN primitive.
    """
    if not isinstance(sparse_count, int) or isinstance(sparse_count, bool) or sparse_count <= 0:
        raise ValueError("sparse_count must be a positive integer")
    if set(batch_meta.kv_global_ids) != set(range(batch_meta.global_valid_queries)):
        raise ValueError("exact global Top-K requires the full global key set")
    query, key, weight = _index_inputs(index_query, index_key, merge_weight, batch_meta)
    with torch.no_grad():
        scores = (torch.einsum("thd,sd->ths", query, key).relu() * weight[..., None]).sum(dim=1)
        ids = torch.tensor(batch_meta.kv_global_ids, dtype=torch.int32)
        order = ids.argsort()
        indices = torch.full((len(batch_meta.q_global_ids), sparse_count), -1, dtype=torch.int32)
        for row, token in enumerate(batch_meta.q_global_ids):
            sequence, position = batch_meta.sequence_position(token)
            candidates = [offset for offset in order.tolist()
                          if batch_meta.sequence_position(int(ids[offset]))[0] == sequence
                          and batch_meta.sequence_position(int(ids[offset]))[1] <= position]
            candidates = torch.tensor(candidates, dtype=torch.long)
            winners = scores[row, candidates].argsort(descending=True, stable=True)[:sparse_count]
            selected = ids[candidates[winners]]
            indices[row, :len(selected)] = selected
    return indices


def selected_kl_reference(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    merge_weight: torch.Tensor,
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    batch_meta: DsaBatchMeta,
    *,
    attention_scale: float,
    normalization: DsaLossNormalization,
    loss_coeff: float = 1.0,
) -> torch.Tensor:
    """Compute selected-set teacher KL; gradients reach only indexer inputs.

    Teacher probabilities are normalized per main head, summed over heads,
    L1 normalized and detached. Empty rows contribute zero to the local sum.
    Only one loss_coeff and one downstream auxiliary grad scale are applied.
    Projection inputs must be detached by the caller before index projection;
    detaching index_query here would incorrectly discard parameter gradients.
    This function provides selected-set KL, not dense warm-up.
    """
    if not math.isfinite(loss_coeff) or loss_coeff < 0:
        raise ValueError("loss_coeff must be finite and nonnegative")
    if normalization.global_valid_queries != batch_meta.global_valid_queries:
        raise ValueError("KL normalization must use the metadata's global valid query count")
    main = _main_inputs((query, compressed_kv, query_rope, key_rope), batch_meta)
    index = _index_inputs(index_query, index_key, merge_weight, batch_meta)
    if index[0].dtype != main[0].dtype:
        raise ValueError("main and indexer inputs must use the same oracle working dtype")
    offsets, valid = _selection(topk_indices, batch_meta)
    with torch.no_grad():
        logits = _attention_logits(main, offsets, attention_scale)
        probability, _ = _masked_distribution(logits, valid[:, None, :])
        teacher = probability.sum(dim=1)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1)
    index_logits = (torch.einsum("thd,tkd->thk", index[0], _gather_keys(index[1], offsets)).relu()
                    * index[2][..., None]).sum(dim=1)
    _, log_probability = _masked_distribution(index_logits, valid)
    kl = functional.kl_div(log_probability, teacher, reduction="sum")
    return kl * (loss_coeff * normalization.local_sum_scale)
