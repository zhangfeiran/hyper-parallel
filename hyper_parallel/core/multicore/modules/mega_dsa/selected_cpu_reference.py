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
"""CPU selected-set attention and KL with bounded scratch and recomputed first-order VJPs."""

from __future__ import annotations

import math
from typing import Any

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import DsaReferenceStats


def _validate_selection(indices: torch.Tensor, meta: DsaBatchMeta) -> None:
    if (indices.device.type != 'cpu' or indices.dtype != torch.int32 or indices.ndim != 2
            or indices.shape[0] != len(meta.q_global_ids) or indices.shape[1] < 1):
        raise ValueError('bounded selected oracle requires CPU int32 [local queries,positive K] indices')


def _selected_row(indices: torch.Tensor, row: int, meta: DsaBatchMeta,
                  storage: dict[int, int]) -> torch.Tensor:
    query_sequence, query_position = meta.sequence_position(meta.q_global_ids[row])
    offsets, seen = [], set()
    for token in indices[row].tolist():
        if token == -1:
            continue
        sequence, position = meta.sequence_position(token)
        if token in seen:
            raise ValueError('Top-K must not contain duplicate valid token IDs')
        seen.add(token)
        if sequence != query_sequence or position > query_position:
            continue
        if token not in storage:
            raise ValueError('selected causal KV is missing; fetch or replicate it before evaluation')
        offsets.append(storage[token])
    return torch.tensor(offsets, dtype=torch.long)


def _validate_tensors(tensors: tuple[torch.Tensor, ...]) -> None:
    if (any(value.device.type != 'cpu' or value.dtype not in (torch.float32, torch.float64)
            or value.layout != torch.strided for value in tensors)
            or len({value.dtype for value in tensors}) != 1):
        raise ValueError('bounded selected oracle requires same-dtype dense CPU FP32/FP64 tensors')


def _validate_attention(inputs: tuple[torch.Tensor, ...], meta: DsaBatchMeta, scale: float) -> None:
    _validate_tensors(inputs)
    query, key, value, query_rope, key_rope = inputs
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('bounded selected oracle requires finite positive model scale')
    if query.ndim != 3 or query.shape[0] != len(meta.q_global_ids) or min(query.shape[1:]) <= 0:
        raise ValueError('bounded selected query requires [local queries,positive heads,positive dimension]')
    rows, heads, width = query.shape
    tokens = len(meta.kv_global_ids)
    if (key.shape not in ((tokens,width),(tokens,heads,width)) or value.ndim not in (2,3)
            or value.shape[0] != tokens or value.shape[-1] <= 0
            or (value.ndim == 3 and value.shape[1] != heads)
            or query_rope.ndim != 3 or query_rope.shape[:2] != (rows,heads) or query_rope.shape[-1] <= 0
            or key_rope.shape != (tokens,query_rope.shape[-1])):
        raise ValueError('bounded selected key/value/RoPE geometry does not match query or metadata')


def _head_rows(value: torch.Tensor, offsets: torch.Tensor, heads: int) -> torch.Tensor:
    selected = value.index_select(0, offsets)
    return selected.unsqueeze(0).expand(heads,-1,-1) if value.ndim == 2 else selected.transpose(0,1).contiguous()


def _probability(query: torch.Tensor, key: torch.Tensor, query_rope: torch.Tensor,
                 key_rope: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    dot = query@key[0].T if key.stride(0) == 0 else torch.einsum('hd,hkd->hk',query,key)
    logits = (dot+query_rope@key_rope.T)*scale
    return logits.softmax(-1), logits


class _SelectedAttention(torch.autograd.Function):
    """Save only source tensors and selection; recompute one selected row in backward."""

    @staticmethod
    @torch.autocast(device_type='cpu', enabled=False)
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                query_rope: torch.Tensor, key_rope: torch.Tensor, indices: torch.Tensor,
                meta: DsaBatchMeta, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate exact selected attention and natural-exponent statistics."""
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(query,key,value,query_rope,key_rope,indices)
        ctx.meta, ctx.scale = meta, scale
        rows, heads = query.shape[:2]
        output = query.new_zeros((rows,heads,value.shape[-1]))
        maximum = query.new_full((rows,heads),-torch.inf)
        denominator = query.new_zeros((rows,heads))
        storage = {token:offset for offset,token in enumerate(meta.kv_global_ids)}
        for row in range(rows):
            offsets = _selected_row(indices,row,meta,storage)
            if not offsets.numel():
                continue
            selected_key = _head_rows(key,offsets,heads)
            probability, logits = _probability(query[row],selected_key,query_rope[row],
                                                key_rope.index_select(0,offsets),scale)
            selected_value = _head_rows(value,offsets,heads)
            output[row] = (probability@selected_value[0] if selected_value.stride(0) == 0 else
                           torch.einsum('hk,hkv->hv',probability,selected_value))
            maximum[row] = logits.amax(-1)
            denominator[row] = (logits-maximum[row,:,None]).exp().sum(-1)
        ctx.mark_non_differentiable(maximum,denominator)
        return output, maximum, denominator

    @staticmethod
    @torch.autocast(device_type='cpu', enabled=False)
    def backward(ctx: Any, gradient: torch.Tensor | None, _maximum: None, _denominator: None) -> tuple:
        """Return independent dK/dV edges; aliasing the source C adds them exactly once."""
        if torch.is_grad_enabled():
            raise ValueError('bounded selected attention supports first-order backward only')
        query, key, value, query_rope, key_rope, indices = ctx.saved_tensors
        if gradient is None:
            return (None,)*8
        dq, dk, dv, dqr, dkr = (torch.zeros_like(value) for value in (query,key,value,query_rope,key_rope))
        heads = query.shape[1]
        storage = {token:offset for offset,token in enumerate(ctx.meta.kv_global_ids)}
        for row in range(query.shape[0]):
            offsets = _selected_row(indices,row,ctx.meta,storage)
            if not offsets.numel():
                continue
            selected_key = _head_rows(key,offsets,heads)
            probability, _logits = _probability(query[row],selected_key,query_rope[row],
                                                 key_rope.index_select(0,offsets),ctx.scale)
            selected_value = _head_rows(value,offsets,heads)
            grad_probability = (gradient[row]@selected_value[0].T if selected_value.stride(0) == 0 else
                                torch.einsum('hv,hkv->hk',gradient[row],selected_value))
            grad_logits = probability*(grad_probability-(probability*grad_probability).sum(-1,keepdim=True))
            grad_logits = grad_logits*ctx.scale
            dq[row] = (grad_logits@selected_key[0] if selected_key.stride(0) == 0 else
                       torch.einsum('hk,hkd->hd',grad_logits,selected_key))
            dqr[row] = grad_logits@key_rope.index_select(0,offsets)
            local_key = torch.einsum('hk,hd->khd',grad_logits,query[row])
            local_value = torch.einsum('hk,hv->khv',probability,gradient[row])
            dk.index_add_(0,offsets,local_key.sum(1) if key.ndim == 2 else local_key.contiguous())
            dv.index_add_(0,offsets,local_value.sum(1) if value.ndim == 2 else local_value.contiguous())
            dkr.index_add_(0,offsets,grad_logits.T@query_rope[row])
        return dq,dk,dv,dqr,dkr,None,None,None


def selected_attention_cpu(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           query_rope: torch.Tensor, key_rope: torch.Tensor,
                           indices: torch.Tensor, meta: DsaBatchMeta, *,
                           attention_scale: float) -> tuple[torch.Tensor, DsaReferenceStats]:
    """Evaluate arbitrary selected histories without retaining per-query selected KV graphs.

    Args:
        query: FP32/FP64 CPU [Tq,H,D] queries.
        key: Shared [Tk,D] or explicit per-head [Tk,H,D] keys.
        value: Shared [Tk,Dv] or explicit per-head [Tk,H,Dv] values.
        query_rope: [Tq,H,Dr] rotary queries in the same precision.
        key_rope: [Tk,Dr] shared rotary keys.
        indices: CPU int32 [Tq,K] global packed IDs, including -1 padding.
        meta: Logical query/KV storage and packed causal boundaries.
        attention_scale: Original positive model scale.

    Returns:
        Output [Tq,H,Dv] and nondifferentiable statistics. Empty candidate
        rows produce zero output, max=-inf and denominator=0. Duplicate IDs
        and missing causal keys are rejected; cross-sequence/future IDs are
        filtered as in the small oracle. C as both key and value receives
        dK+dV through its two original autograd edges. Working selection scratch
        is O(K*H*(D+Dv+Dr)), independent of Tq; source/output/gradient storage
        still scales with the actual model. Backward recomputes row probabilities.
        This CPU validation oracle supports only first-order VJPs, not production dispatch.
    """
    _validate_attention((query,key,value,query_rope,key_rope),meta,attention_scale)
    _validate_selection(indices,meta)
    output, maximum, denominator = _SelectedAttention.apply(query,key,value,query_rope,key_rope,
                                                          indices,meta,attention_scale)
    return output, DsaReferenceStats(maximum,denominator)


def _kl_row(index: tuple[torch.Tensor, ...], main: tuple[torch.Tensor, ...], offsets: torch.Tensor,
            row: int, scale: float) -> tuple[torch.Tensor, ...]:
    query, compressed, query_rope, key_rope = main
    probability, _logits = _probability(query[row],_head_rows(compressed,offsets,query.shape[1]),
                                        query_rope[row],key_rope.index_select(0,offsets),scale)
    teacher = probability.sum(0)
    teacher = teacher/teacher.sum().clamp_min(1)
    index_query, index_key, weight = index
    selected_key = index_key.index_select(0,offsets)
    raw = index_query[row]@selected_key.T
    logits = (raw.relu()*weight[row,:,None]).sum(0)
    log_probability = logits.log_softmax(-1)
    loss = (torch.xlogy(teacher,teacher)-teacher*log_probability).sum()
    return loss, teacher, log_probability, raw, selected_key


class _SelectedKl(torch.autograd.Function):
    """Recompute detached selected teacher and predictor for each first-order index VJP."""

    @staticmethod
    @torch.autocast(device_type='cpu', enabled=False)
    def forward(ctx: Any, index_query: torch.Tensor, index_key: torch.Tensor, weight: torch.Tensor,
                query: torch.Tensor, compressed: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
                indices: torch.Tensor, meta: DsaBatchMeta, scale: float, coefficient: float) -> torch.Tensor:
        """Accumulate selected KL with its declared coefficient and normalization once."""
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(index_query,index_key,weight,query,compressed,query_rope,key_rope,indices)
        ctx.meta, ctx.scale, ctx.coefficient = meta, scale, coefficient
        storage = {token:offset for offset,token in enumerate(meta.kv_global_ids)}
        loss = query.new_zeros(())
        for row in range(query.shape[0]):
            offsets = _selected_row(indices,row,meta,storage)
            if offsets.numel():
                loss = loss+_kl_row((index_query,index_key,weight),(query,compressed,query_rope,key_rope),
                                   offsets,row,scale)[0]
        return loss*coefficient

    @staticmethod
    @torch.autocast(device_type='cpu', enabled=False)
    def backward(ctx: Any, gradient: torch.Tensor | None) -> tuple:
        """Return index-only derivatives with a zero main teacher gradient."""
        if torch.is_grad_enabled():
            raise ValueError('bounded selected KL supports first-order backward only')
        tensors = ctx.saved_tensors
        if gradient is None:
            return (None,)*11
        index, main, indices = tensors[:3], tensors[3:7], tensors[7]
        dq, dk, dw = (torch.zeros_like(value) for value in index)
        storage = {token:offset for offset,token in enumerate(ctx.meta.kv_global_ids)}
        factor = gradient*ctx.coefficient
        for row in range(index[0].shape[0]):
            offsets = _selected_row(indices,row,ctx.meta,storage)
            if not offsets.numel():
                continue
            _loss, teacher, log_probability, raw, selected_key = _kl_row(index,main,offsets,row,ctx.scale)
            grad_logits = (log_probability.exp()*teacher.sum()-teacher)*factor
            grad_dot = (raw > 0)*index[2][row,:,None]*grad_logits[None]
            dq[row] = grad_dot@selected_key
            dk.index_add_(0,offsets,grad_dot.T@index[0][row])
            dw[row] = raw.relu()@grad_logits
        return dq,dk,dw,*(None,)*8


def selected_kl_cpu(index: tuple[torch.Tensor, ...], main: tuple[torch.Tensor, ...],
                    indices: torch.Tensor, meta: DsaBatchMeta, *, attention_scale: float,
                    normalization: DsaLossNormalization, loss_coeff: float = 1.) -> torch.Tensor:
    """Compute selected KL with bounded row scratch and gradients only to index inputs.

    Args:
        index: Same-dtype CPU FP32/FP64 IQ [Tq,Hi,Di], IK [Tk,Di], W [Tq,Hi].
        main: Absorbed Q [Tq,H,C], C [Tk,C], QR [Tq,H,Dr], KR [Tk,Dr].
        indices: Actual CPU int32 global packed selection, including -1 padding.
        meta: Logical query/key layout and packed causal domains.
        attention_scale: Original model scale for the detached main teacher.
        normalization: Complete global objective and declared downstream reduction.
        loss_coeff: Finite nonnegative coefficient, applied once. Downstream
            auxiliary cotangents are applied once by the scalar backward.

    Returns:
        Scalar selected KL; its teacher sums per-head probabilities then L1
        normalizes over the actual selection. Main tensors receive no gradient.
        Source tensor versions are retained, including detached main tensors.
        Backward recomputes per-row softmax, ReLU masks and matrix derivatives.
        No tensor of shape [Tq,H,K] or selected [Tq,K,C] is retained.
    """
    if len(index) != 3 or len(main) != 4:
        raise ValueError('bounded selected KL requires three index and four main tensors')
    _validate_tensors((*index,*main))
    _validate_attention((main[0],main[1],main[1],main[2],main[3]),meta,attention_scale)
    _validate_selection(indices,meta)
    query, key, weight = index
    if (query.ndim != 3 or query.shape[0] != len(meta.q_global_ids) or min(query.shape[1:]) <= 0
            or key.shape != (len(meta.kv_global_ids),query.shape[-1]) or weight.shape != query.shape[:2]
            or not isinstance(normalization,DsaLossNormalization)
            or normalization.global_valid_queries != meta.global_valid_queries
            or isinstance(loss_coeff,bool) or not isinstance(loss_coeff,(int,float))
            or not math.isfinite(loss_coeff) or loss_coeff < 0):
        raise ValueError('bounded selected KL requires matching index geometry and complete global loss normalization')
    return _SelectedKl.apply(*index,*(value.detach() for value in main),indices,meta,attention_scale,
                             loss_coeff*normalization.local_sum_scale)
