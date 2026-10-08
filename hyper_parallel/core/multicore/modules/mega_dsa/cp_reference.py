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
"""Explicit replicated-Q/K CP baseline, independent of the future fused worker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
    CannDsaSelection,
    CannDsaStats,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)


class _OwnerGather(torch.autograd.Function):
    """Gather fields together, preserving absent teacher gradients in KL-only backward."""

    @staticmethod
    def forward(ctx: Any, layout: DsaCpLayout, shapes: tuple, *tensors: torch.Tensor) -> tuple:
        """Gather owner storage once and expose independent global fields."""
        ctx.layout = layout
        ctx.shapes = shapes
        ctx.input_dtype = tensors[0].dtype
        ctx.set_materialize_grads(False)
        packed = [tensor.index_select(0, layout.field_order(field)).flatten(1)
                  for field, tensor in enumerate(tensors)]
        ctx.widths = tuple(tensor.shape[1] for tensor in packed)
        local = torch.cat(packed, dim=1)
        padded = local.new_zeros((layout.padded_tokens, local.shape[1]))
        padded[:layout.local_tokens] = local
        gathered = local.new_empty((layout.cp_size * layout.padded_tokens, local.shape[1]))
        if layout.cp_size == 1:
            gathered.copy_(padded)
        else:
            dist.all_gather_into_tensor(gathered, padded.contiguous(), group=layout.group)
        global_packed = gathered.index_select(0, layout.global_order)
        return tuple(field.reshape(layout.batch_meta.global_valid_queries, *shape)
                     for field, shape in zip(global_packed.split(ctx.widths, dim=1), shapes))

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor | None) -> tuple:
        """Reduce active field gradients to owners while preserving absent fields."""
        layout = ctx.layout
        # BF16 sums lose remote contributions before the owner receives them.
        dtype = torch.float64 if ctx.input_dtype == torch.float64 else torch.float32
        total = layout.batch_meta.global_valid_queries
        fields = [torch.zeros((total, width), dtype=dtype, device=layout.device) if gradient is None
                  else gradient.to(dtype).reshape(total, width)
                  for gradient, width in zip(gradients, ctx.widths)]
        padded = torch.zeros((layout.cp_size * layout.padded_tokens, sum(ctx.widths)),
                             dtype=dtype, device=layout.device)
        padded.index_copy_(0, layout.global_order, torch.cat(fields, dim=1))
        owned = padded.new_empty((layout.padded_tokens, padded.shape[1]))
        if layout.cp_size == 1:
            owned.copy_(padded)
        else:
            dist.reduce_scatter_tensor(owned, padded, op=dist.ReduceOp.SUM, group=layout.group)
        returned = []
        for field, (value, gradient, shape) in enumerate(zip(owned[:layout.local_tokens].split(ctx.widths, dim=1),
                                                           gradients, ctx.shapes)):
            if gradient is None:
                returned.append(None)
            else:
                local = torch.empty_like(value)
                local.index_copy_(0, layout.field_order(field), value)
                returned.append(local.reshape(layout.local_tokens, *shape).to(ctx.input_dtype))
        return (None, None, *returned)


class DsaCpLayout:
    """Prepare complete owner shards with arbitrary local Q/K storage order.

    Construction is collective for CP>1. Metadata must agree across the ordered
    CP group. Q and KV must each cover the complete local owner set. Unequal or
    empty shards are padded for collectives, then reordered into global packed
    order on device. No tensor contents are read by the host during execution.
    All members must execute the same invocation and backward in the same order.
    """

    def __init__(
        self, batch_meta: DsaBatchMeta, device: torch.device | str, *, group: dist.ProcessGroup | None = None,
    ) -> None:
        """Collectively validate static ownership and prepare device permutations."""
        self.batch_meta = batch_meta
        self.device = torch.device(device)
        self.group = group
        self.cp_size = len(batch_meta.cp_ranks)
        owners = batch_meta.token_owners
        self.counts = tuple(owners.count(rank) for rank in range(self.cp_size))
        self.local_tokens = self.counts[batch_meta.cp_rank]
        self.padded_tokens = max(self.counts)
        self._validate_group()
        self._validate_shards()
        offsets = batch_meta.token_local_offsets
        self.global_order = self._ids(tuple(owner * self.padded_tokens + offset
                                            for owner, offset in zip(owners, offsets)))
        self.query_order = self._ids(tuple(sorted(range(self.local_tokens),
                                                 key=lambda row: offsets[batch_meta.q_global_ids[row]])))
        self.key_order = self._ids(tuple(sorted(range(self.local_tokens),
                                               key=lambda row: offsets[batch_meta.kv_global_ids[row]])))
        self.local_query_ids = self._ids(batch_meta.q_global_ids)

    def _ids(self, values: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def _validate_group(self) -> None:
        if not dist.is_initialized():
            if self.cp_size != 1 or self.batch_meta.cp_ranks != (0,) or self.group is not None:
                raise ValueError("CP>1 requires an initialized process group")
            return
        group = self.group if self.group is not None else dist.group.WORLD
        members = tuple(dist.get_process_group_ranks(group))
        error = None
        if members != self.batch_meta.cp_ranks or dist.get_rank(group) != self.batch_meta.cp_rank:
            error = "process group membership/order/rank must match CP metadata"
        errors = [error]
        if len(members) > 1:
            errors = [None] * len(members)
            dist.all_gather_object(errors, error, group=group)
        if any(item is not None for item in errors):
            raise ValueError("; ".join(item for item in errors if item is not None))

    def _validate_shards(self) -> None:
        meta = self.batch_meta
        owned = {token for token, owner in enumerate(meta.token_owners) if owner == meta.cp_rank}
        error = None
        if set(meta.q_global_ids) != owned or set(meta.kv_global_ids) != owned:
            error = "CP reference requires complete owner-local Q and KV shards"
        signature = (meta.global_cu_seqlens, meta.token_owners, meta.token_local_offsets, meta.cp_ranks,
                     meta.root_pes, meta.layout_id, meta.layer, meta.microbatch, meta.invocation,
                     meta.heap_generation, meta.index_namespace)
        declarations = [(signature, error)]
        if self.cp_size > 1:
            declarations = [None] * self.cp_size
            dist.all_gather_object(declarations, (signature, error), group=self.group)
        if any(item[1] is not None for item in declarations):
            raise ValueError("; ".join(item[1] for item in declarations if item[1] is not None))
        if any(item[0] != signature for item in declarations):
            raise ValueError("CP members disagree on global metadata/invocation")

    def gather_fields(self, tensors: tuple[torch.Tensor, ...], shapes: tuple[tuple[int, ...], ...]) -> tuple:
        """Gather one packed differentiable buffer for main and indexer fields.

        Fields alternate query/key for the four main states and the first two
        indexer states; the last indexer field is a query weight. Each field is
        returned in complete global token order. One backward collective avoids
        branch-dependent ordering between attention and KL gradient paths.
        """
        if len(tensors) != len(shapes) or len(tensors) not in (4, 7):
            raise ValueError("gather_fields expects four main fields or seven main/indexer fields")
        for tensor, shape in zip(tensors, shapes):
            if tensor.device != self.device or tensor.dtype != tensors[0].dtype:
                raise ValueError("CP fields must share the prepared device and floating dtype")
            if not tensor.is_floating_point() or tensor.shape != (self.local_tokens, *shape):
                raise ValueError("CP field shape does not match the prepared complete owner shard")
        return _OwnerGather.apply(self, shapes, *tensors)

    def field_order(self, field: int) -> torch.Tensor:
        """Return the prepared owner-storage permutation for a declared field."""
        return self.key_order if field in (1, 3, 5) else self.query_order



@dataclass(frozen=True)
class CannDsaCpResult:
    """Local attention output/stats/selections and this replica's KL contribution."""

    output: torch.Tensor
    stats: CannDsaStats
    global_indices: torch.Tensor
    kl_loss: torch.Tensor | None


class CannDsaCpReference:
    """Explicit full-Q/K replication baseline using the admitted CANN reference.

    This gathers Q as well as KV because the installed SFA right-down mask
    cannot express a shard's global query offset. Each rank runs complete SFA,
    then selects its local queries. It is a correctness baseline, not SHMEM,
    sparse communication, a fused launch, or a production training adapter.
    Main/indexer dimensions and optional KL are fixed at construction and must
    agree across ranks. Every rank must backpropagate its own returned output
    and KL contribution; do not all-reduce the scalar inside the loss graph.
    """

    def __init__(
        self, layout: DsaCpLayout, *, attention_scale: float, heads: int = 32, index_heads: int = 8,
        with_indexer: bool = True,
    ) -> None:
        """Prepare replicated native metadata and declare the collective field schema."""
        if not isinstance(with_indexer, bool):
            raise TypeError("with_indexer must be a boolean")
        if heads not in (32, 64, 128) or index_heads not in (8, 16, 32, 64):
            raise ValueError("unsupported CANN main/indexer head count")
        self.layout = layout
        lengths = tuple(b - a for a, b in zip(layout.batch_meta.global_cu_seqlens,
                                             layout.batch_meta.global_cu_seqlens[1:]))
        self.backend = CannDsaReference(CannDsaLayout(DsaBatchMeta.packed(lengths), layout.device),
                                        attention_scale=attention_scale)
        self.shapes = ((heads, 512), (512,), (heads, 64), (64,))
        if with_indexer:
            self.shapes += ((index_heads, 128), (128,), (index_heads,))
        self.with_indexer = with_indexer
        self._validate_schema()

    def _validate_schema(self) -> None:
        schema = (self.shapes, self.backend.attention_scale, self.with_indexer)
        schemas = [schema]
        if self.layout.cp_size > 1:
            schemas = [None] * self.layout.cp_size
            dist.all_gather_object(schemas, schema, group=self.layout.group)
        if any(item != schema for item in schemas):
            raise ValueError("CP members disagree on the prepared CANN field schema")

    def prepare_selection(self, global_indices: torch.Tensor) -> CannDsaSelection:
        """Admit the same complete CPU [global T,2048] selection snapshot on all ranks.

        This setup-only path retains native cardinality/uniqueness validation.
        Caller supplies identical snapshots on all members; no implicit gather
        or device read is performed. Device-generated selection uses the indexer.
        """
        return self.backend.prepare_selection(global_indices)

    def forward(
        self, main_inputs: tuple[torch.Tensor, ...], *, index_inputs: tuple[torch.Tensor, ...] = (),
        selection: CannDsaSelection | None = None, normalization: DsaLossNormalization | None = None,
        loss_coeff: float = 1.0,
    ) -> CannDsaCpResult:
        """Evaluate main attention and optional selected KL on complete replicated states.

        Args:
            main_inputs: Owner-local Q, compressed KV, Q RoPE, K RoPE in metadata storage order.
            index_inputs: Owner-local index Q/K and already scaled signed merge weights.
            selection: Prepared complete external selection; otherwise run native TopK.
            normalization: Global valid-query count and declared downstream reducer divisor.
            loss_coeff: Original nonnegative KL coefficient, applied exactly once.

        Returns:
            Output/stats/global selection rows in local query storage order.
            KL is the replicated normalized global scalar divided by CP size.
            Summing detached KL contributions recovers the global loss; an
            external parameter average requires the explicit reducer divisor.

        Note:
            Rank-dependent requires-grad masks or invocation/backward schedules
            are unsupported. Main teacher inputs stay detached inside native KL.
        """
        self._validate_inputs(main_inputs, index_inputs, selection, normalization, loss_coeff)
        inputs = self.layout.gather_fields((*main_inputs, *index_inputs), self.shapes)
        main, index = inputs[:4], inputs[4:]
        if selection is None:
            selection = self.backend.indexer(*index)
        output, stats = self.backend.attention(*main, selection)
        loss = None
        if self.with_indexer:
            loss = self.backend.kl_loss(*index, main, selection, stats,
                                        normalization=normalization, loss_coeff=loss_coeff) / self.layout.cp_size
        rows = self.layout.local_query_ids
        return CannDsaCpResult(output.index_select(0, rows),
                               CannDsaStats(stats.maximum.index_select(1, rows),
                                            stats.denominator.index_select(1, rows)),
                               selection.to_global_indices().index_select(0, rows), loss)

    def _validate_inputs(self, main_inputs, index_inputs, selection, normalization, loss_coeff) -> None:
        if not math.isfinite(loss_coeff) or loss_coeff < 0:
            raise ValueError("loss_coeff must be finite and nonnegative")
        if len(main_inputs) != 4 or len(index_inputs) != (3 if self.with_indexer else 0):
            raise ValueError("input fields must match the prepared main/indexer schema")
        if not self.with_indexer and selection is None:
            raise ValueError("attention-only CP execution requires a prepared external selection")
        if self.with_indexer and normalization is None:
            raise ValueError("CP KL requires explicit global loss normalization")
        if (normalization is not None
                and normalization.global_valid_queries != self.layout.batch_meta.global_valid_queries):
            raise ValueError("CP KL normalization must use the complete global query count")
        if any(tensor.dtype != torch.bfloat16 for tensor in (*main_inputs, *index_inputs)):
            raise ValueError("CANN CP reference inputs must be BF16")
