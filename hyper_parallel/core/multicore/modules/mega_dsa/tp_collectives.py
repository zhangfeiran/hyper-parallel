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
"""Explicit TP head reductions for tiled scores, selected teacher and shared gradients.

This Torch communication boundary is separate from the native fused CP worker.
It never gathers heads. Callers supply the same candidate rows and collective
schedule on every TP member, after any CP fetch/reordering has completed.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaLossNormalization


class _SumHeadScores(torch.autograd.Function):
    """TP partial scores become replicated scores; replicated cotangents stay local."""

    @staticmethod
    def forward(ctx: Any, boundary: DsaTpHeadCollectives, partial: torch.Tensor) -> torch.Tensor:
        """Publish the complete replicated score tile."""
        ctx.set_materialize_grads(False)
        return boundary.sum_detached(partial.detach())

    @staticmethod
    def backward(_ctx: Any, gradient: torch.Tensor | None) -> tuple:
        """Return one replicated cotangent to this head shard."""
        return None, gradient


def _working(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_floating_point():
        raise ValueError("TP fields must be floating-point tensors")
    return tensor.to(torch.float64 if tensor.dtype == torch.float64 else torch.float32)


def _distribution(logits: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if valid.dtype != torch.bool or valid.device != logits.device or valid.shape != logits.shape:
        raise ValueError("candidate validity must be boolean and match logits shape/device")
    nonempty = valid.any(dim=-1, keepdim=True)
    safe = torch.where(nonempty, logits.masked_fill(~valid, -torch.inf), torch.zeros_like(logits))
    log_probability = safe.log_softmax(-1).masked_fill(~valid, 0)
    return log_probability.exp().masked_fill(~valid, 0), log_probability


class DsaTpHeadCollectives:
    """Prepare an explicit TP group and require identical ordered candidate metadata.

    Args:
        group: Explicit process group, including a singleton group for TP=1.
        row_signature: Immutable host declaration of query IDs, candidate IDs,
            packed lengths and invocation identity. CP group ranks are omitted:
            TP peers can belong to different CP groups but must use identical rows.

    Note:
        Construction is collective. Forward score SUM has an identity backward,
        as in row-parallel TP: each TP rank backpropagates the same replicated
        KL objective once. Do not divide that objective by TP size or SUM its
        cotangent again. CP normalization remains the caller's declared contract.
        Shared compressed KV, K RoPE and index K gradient partials require the
        separate ``sum_shared_gradients`` call before reconnecting projections.
        All ranks must use the same gradient-presence mask and backward schedule.
    """

    def __init__(self, group: dist.ProcessGroup, *, row_signature: tuple) -> None:
        """Validate group membership and the setup-only row declaration collectively."""
        if group is None or not dist.is_initialized():
            raise ValueError("TP requires an initialized explicit process group")
        self.group = group
        self.members = tuple(dist.get_process_group_ranks(group))
        if not self.members or dist.get_rank(group) < 0:
            raise ValueError("caller must be a member of the TP group")
        if not isinstance(row_signature, tuple):
            raise TypeError("TP row signature must be a host tuple")
        declarations = [None] * len(self.members)
        dist.all_gather_object(declarations, row_signature, group=group)
        if any(signature != row_signature for signature in declarations):
            raise ValueError("TP members disagree on candidate rows or invocation")

    def sum_detached(self, partial: torch.Tensor) -> torch.Tensor:
        """SUM a detached tile in FP32 (FP64 for validation), retaining no input alias."""
        if partial.requires_grad:
            raise ValueError("detached TP SUM must not discard an active autograd edge")
        result = _working(partial).clone(memory_format=torch.contiguous_format)
        if len(self.members) > 1 and result.numel():
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.group)
        return result

    def index_scores(self, query: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """SUM local index heads before any Top-K, on one bounded candidate tile.

        Query is [Tq,local Hi,Di], keys [Tk,Di], weights [Tq,local Hi].
        Signed weights already contain the original model scaling; this method
        applies no head count, TP divisor or additional activation to weights.
        The output [Tq,Tk] can feed a streaming exact Top-K or selected KL.
        """
        if (query.ndim != 3 or query.shape[1] < 1 or keys.ndim != 2
                or keys.shape[1] != query.shape[2] or weights.shape != query.shape[:2]):
            raise ValueError("index tile requires [Tq,Hi,Di], [Tk,Di], [Tq,Hi]")
        if len({tensor.device for tensor in (query, keys, weights)}) != 1:
            raise ValueError("index fields must share a device")
        query, keys, weights = tuple(_working(tensor) for tensor in (query, keys, weights))
        partial = (torch.einsum("thd,kd->thk", query, keys).relu() * weights[..., None]).sum(1)
        return _SumHeadScores.apply(self, partial)

    def sum_selected_scores(self, partial: torch.Tensor) -> torch.Tensor:
        """SUM precomputed [Tq,K] local-head scores in identical selected-slot order.

        Supports per-query selected key sets, unlike the rectangular candidate
        tile in ``index_scores``. Partial values must already contain the local
        ReLU and signed model merge weights. The replicated-score VJP is identity.
        """
        if partial.ndim != 2:
            raise ValueError("selected index scores require [Tq,K]")
        return _SumHeadScores.apply(self, _working(partial))

    def teacher(self, local_logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Detach per-head softmax, SUM main heads across TP, then L1 normalize.

        Logits [Tq,local H,K] already include the original attention scale.
        Validity [Tq,K] includes padding, packed sequence and causal filtering.
        Empty rows contribute exactly zero and carry no teacher gradient.
        """
        if local_logits.ndim != 3 or local_logits.shape[1] < 1:
            raise ValueError("teacher logits require [Tq,positive local H,K]")
        if valid.shape != (local_logits.shape[0], local_logits.shape[2]):
            raise ValueError("teacher validity requires [Tq,K]")
        with torch.no_grad():
            logits = _working(local_logits)
            probability, _ = _distribution(logits, valid[:, None, :].expand_as(logits))
            summed = self.sum_detached(probability.sum(1))
            return summed / summed.sum(-1, keepdim=True).clamp_min(1)

    @staticmethod
    def selected_kl(scores: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor, *,
                    normalization: DsaLossNormalization, loss_coeff: float = 1.0) -> torch.Tensor:
        """Return the replicated KL objective with exactly one declared loss scale.

        Scores must already be the TP SUM from ``index_scores``. Teacher must
        be the detached global teacher from ``teacher``. Invalid slots and empty
        rows contribute zero. A caller using global-Q replication must account
        for its CP replication once, outside this TP boundary.
        """
        if teacher.shape != scores.shape or teacher.device != scores.device or teacher.requires_grad:
            raise ValueError("KL teacher must be detached and match the global scores")
        if not math.isfinite(loss_coeff) or loss_coeff < 0:
            raise ValueError("loss coefficient must be finite and nonnegative")
        _, log_probability = _distribution(scores, valid)
        return functional.kl_div(log_probability, teacher, reduction="sum") * (
            loss_coeff * normalization.local_sum_scale)

    def sum_shared_gradients(self, gradients: tuple[torch.Tensor | None, ...]) -> tuple:
        """SUM detached shared-field gradients once in one ordered TP collective.

        Input fields are complete CP owner-local partials in identical storage
        order on TP peers. Keep them FP32 through CP return and TP SUM, then cast
        once at the projection boundary. None denotes an absent branch, with
        identical presence on every TP member. No CP/TP division is applied.
        """
        active = tuple(tensor for tensor in gradients if tensor is not None)
        if not active:
            return gradients
        if any(tensor.requires_grad or tensor.dtype not in (torch.float32, torch.float64) for tensor in active):
            raise ValueError("shared gradient partials must be detached FP32/FP64")
        if len({(tensor.device, tensor.dtype) for tensor in active}) != 1:
            raise ValueError("shared gradient fields must share device and accumulation dtype")
        reduced = self.sum_detached(torch.cat(tuple(tensor.reshape(-1) for tensor in active)))
        fields = iter(reduced.split(tuple(tensor.numel() for tensor in active)))
        return tuple(None if tensor is None else next(fields).reshape(tensor.shape) for tensor in gradients)
