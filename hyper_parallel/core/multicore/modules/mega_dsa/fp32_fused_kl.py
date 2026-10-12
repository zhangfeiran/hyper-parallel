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
"""FP32 selected KL around fused BF16 LI/main with native CP/TP owner gradient returns."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.fp32_index_kl import (
    Fp32IndexKlProbe,
    Fp32IndexVjp,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import FusedCpForwardResult
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta


class Fp32OwnerKeyGather:
    """Cold-declared CP storage permutation and a detached FP32 owner-key gather.

    This gather has no autograd SUM. Its consumer returns local-query dK using
    the native CP owner return and TP SUM once, before projection backward.
    """

    def __init__(self, meta: DsaBatchMeta, group: dist.ProcessGroup, device: torch.device | str) -> None:
        """Verify complete owner storage and precompute the global packed permutation."""
        if group is None or group == dist.GroupMember.NON_GROUP_MEMBER:
            raise ValueError('FP32 owner key gather requires an explicit participating CP group')
        members = tuple(dist.get_process_group_ranks(group))
        if members != meta.cp_ranks or dist.get_rank(group) != meta.cp_rank:
            raise ValueError('FP32 owner key gather requires the prepared ordered CP group')
        self.meta, self.group, self.device = meta, group, torch.device(device)
        signature = (meta.global_cu_seqlens, meta.token_owners, meta.token_local_offsets,
                     meta.cp_ranks, meta.root_pes, meta.layout_id, meta.heap_generation)
        declaration = (signature, meta.kv_global_ids)
        peers = [None] * len(members)
        dist.all_gather_object(peers, declaration, group=group)
        rows = []
        for owner, (identity, ids) in enumerate(peers):
            expected = tuple(token for token in range(meta.global_valid_queries) if meta.token_owners[token] == owner)
            if identity != signature or set(ids) != set(expected) or len(ids) != len(expected):
                raise ValueError('FP32 CP key gather requires every owner key once and identical global ownership')
            rows.append(ids)
        self.max_rows = max(map(len, rows))
        addresses = {token: owner * self.max_rows + offset
                     for owner, ids in enumerate(rows) for offset, token in enumerate(ids)}
        self.global_order = torch.tensor([addresses[token] for token in range(meta.global_valid_queries)],
                                         dtype=torch.long, device=self.device)
        self._order = self.global_order
        self._version = self.global_order._version
        self._identity = (meta, group, self.device, self.max_rows)

    def gather(self, owner_key: torch.Tensor) -> torch.Tensor:
        """Queue one bounded-shape gather and restore complete global packed FP32 keys."""
        if (self._identity != (self.meta, self.group, self.device, self.max_rows)
                or self.global_order is not self._order or self.global_order._version != self._version):
            raise ValueError('FP32 CP key gather declaration changed')
        if (owner_key.shape != (len(self.meta.kv_global_ids), 128) or owner_key.dtype != torch.float32
                or owner_key.device != self.device or owner_key.requires_grad):
            raise ValueError('FP32 CP key gather requires detached owner-ordered FP32 [local K,128]')
        local = torch.zeros((self.max_rows, 128), dtype=torch.float32, device=self.device)
        local[:owner_key.shape[0]].copy_(owner_key)
        if len(self.meta.cp_ranks) == 1:
            gathered = local
        else:
            gathered = torch.empty((len(self.meta.cp_ranks) * self.max_rows, 128),
                                   dtype=torch.float32, device=self.device)
            handle = dist.all_gather_into_tensor(gathered, local, group=self.group, async_op=True)
            handle.wait()
        return gathered.index_select(0, self.global_order)


def selected_teacher_mass(main: tuple[torch.Tensor, ...], indices: torch.Tensor, *, scale: float) -> torch.Tensor:
    """Compute detached FP32 per-head selected probabilities in bounded per-query scratch.

    Main keys cover the complete packed batch; queries and selection follow
    caller-local row order. Returned mass sums only this TP rank's main heads.
    The caller TP SUMs mass before normalizing the teacher, without a divisor.
    """
    query, compressed, query_rope, key_rope = main
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('FP32 teacher scale must be finite and positive')
    rows, heads = query.shape[:2]
    if (query.shape != (rows, heads, 512) or compressed.ndim != 2 or compressed.shape[1] != 512
            or query_rope.shape != (rows, heads, 64) or key_rope.shape != (compressed.shape[0], 64)
            or indices.shape != (rows, 2048) or indices.dtype != torch.int32
            or any(value.device != query.device or value.requires_grad for value in (*main, indices))):
        raise ValueError('FP32 teacher requires detached local main Q/RoPE and complete main K/RoPE selection')
    mass = torch.zeros((rows, 2048), dtype=torch.float32, device=query.device)
    with torch.autocast(device_type=query.device.type, enabled=False):
        for row in range(rows):
            ids = indices[row].clamp_min(0).long()
            keys = compressed.index_select(0, ids).float()
            rope = key_rope.index_select(0, ids).float()
            logits = (query[row].float() @ keys.T + query_rope[row].float() @ rope.T) * scale
            probabilities = logits.masked_fill(indices[row][None] < 0, -torch.inf).softmax(dim=-1)
            mass[row].copy_(probabilities.sum(dim=0))
    return mass


@dataclass(frozen=True)
class Fp32FusedKlResult:
    """Own raw FP32 derivatives and receipts without retaining reusable transport scratch."""

    vjp: Fp32IndexVjp
    logits: torch.Tensor
    teacher: torch.Tensor
    score_trace: torch.Tensor


class Fp32FusedSelectedKlProbe:
    """Use authentic fused LI selection, FP32 KL, and existing native gradient returns.

    BF16 LI remains the selection producer. Teacher dot products and index KL
    use FP32, distinct from stock BF16 KL. The loss is TP replicated. No host
    selection read, loss normalization, or autograd CP gather is performed.
    """

    def __init__(self, meta: DsaBatchMeta, cp_group: dist.ProcessGroup,
                 tp_group: dist.ProcessGroup, device: torch.device | str, *, scale: float) -> None:
        """Prepare owner key replication before invoking any forward or backward graph."""
        if tp_group is None or tp_group == dist.GroupMember.NON_GROUP_MEMBER:
            raise ValueError('FP32 selected KL requires an explicit participating TP group')
        self.keys = Fp32OwnerKeyGather(meta, cp_group, device)
        self.tp_group, self.scale = tp_group, scale
        self.tp_size = dist.get_world_size(tp_group)
        self._declaration = (tp_group, scale, self.tp_size)
        self.last_result: Fp32FusedKlResult | None = None

    def gradients(self, result: FusedCpForwardResult, main: tuple[torch.Tensor, ...],
                  index: tuple[torch.Tensor, ...]) -> Fp32FusedKlResult:
        """Compute invocation-owned KL VJPs before training applies coefficient/normalization."""
        if self._declaration != (self.tp_group, self.scale, self.tp_size):
            raise ValueError('FP32 selected KL declaration changed')
        probe = Fp32IndexKlProbe.from_saved_attention(result.saved)
        key = self.keys.gather(index[1])
        teacher = selected_teacher_mass((main[0], result.global_keys[1][:, 0], main[2],
                                        result.global_keys[2][:, 0]), probe.indices, scale=self.scale)
        if teacher.numel() and self.tp_size > 1:
            dist.all_reduce(teacher, group=self.tp_group)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True)
        scores = probe.scores(index[0], key, index[2])
        logits = scores.logits.clone()
        if logits.numel() and self.tp_size > 1:
            dist.all_reduce(logits, group=self.tp_group)
        vjp = probe.vjp(scores, teacher, logits)
        output = Fp32FusedKlResult(vjp, logits, teacher, scores.trace)
        self.last_result = output
        return output
