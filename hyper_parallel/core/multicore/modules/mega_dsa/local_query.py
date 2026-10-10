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
"""Prepared compact local-query positions for native exact indexer execution."""

from __future__ import annotations

from itertools import pairwise

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_indexer import MIXED_INDEXER_WORKSPACE_BYTES
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.torch.ops import _load_native


class DsaLocalQueryLayout:
    """Pack caller-local rows by sequence and global position without replicating Q.

    Genuine local cumulative counts may repeat when this rank owns no query in
    a sequence. Positions remain sequence-relative logical token positions.
    Construction transfers only static metadata; execution never reads it back.
    """

    def __init__(self, batch_meta: DsaBatchMeta, device: torch.device | str) -> None:
        """Prepare immutable permutations and causal metadata from admitted batch ownership."""
        if not isinstance(batch_meta, DsaBatchMeta):
            raise TypeError("local queries require DsaBatchMeta")
        self.batch_meta = batch_meta
        self.device = torch.device(device)
        self.global_ids = tuple(sorted(batch_meta.q_global_ids))
        storage = {token: row for row, token in enumerate(batch_meta.q_global_ids)}
        pack = tuple(storage[token] for token in self.global_ids)
        inverse = {row: packed for packed, row in enumerate(pack)}
        self.pack_order = torch.tensor(pack, dtype=torch.long, device=self.device)
        self.restore_order = torch.tensor(tuple(inverse[row] for row in range(len(pack))),
                                          dtype=torch.long, device=self.device)
        cumulative = tuple(sum(token < end for token in self.global_ids)
                           for end in batch_meta.global_cu_seqlens[1:])
        positions = tuple(batch_meta.sequence_position(token)[1] for token in self.global_ids)
        self.cumulative_queries = cumulative
        self.positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        self.query_lengths = torch.tensor(cumulative, dtype=torch.int32, device=self.device)
        self.key_lengths = torch.tensor(batch_meta.global_cu_seqlens[1:], dtype=torch.int32, device=self.device)
        self._prepared_tensors = self._tensors()
        self._versions = tuple(tensor._version for tensor in self._prepared_tensors)
        self._identity = (batch_meta, self.device, self.global_ids, self.cumulative_queries)

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return self.pack_order, self.restore_order, self.positions, self.query_lengths, self.key_lengths

    def validate_versions(self) -> None:
        """Reject mutated addresses or layout declarations before native loading or submission."""
        if (self._identity != (self.batch_meta, self.device, self.global_ids, self.cumulative_queries)
                or any(current is not prepared for current, prepared in zip(self._tensors(), self._prepared_tensors))
                or self._versions != tuple(tensor._version for tensor in self._tensors())):
            raise ValueError("local query metadata was modified; prepare a new layout")

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        """Restore canonical sequence/position order from the caller's local storage order."""
        self.validate_versions()
        if tensor.device != self.device or tensor.ndim < 1 or tensor.shape[0] != len(self.global_ids):
            raise ValueError("local query input must match the prepared device and caller row count")
        return tensor.index_select(0, self.pack_order)

    def restore(self, tensor: torch.Tensor) -> torch.Tensor:
        """Restore caller-local row order from a compact native output."""
        self.validate_versions()
        if tensor.device != self.device or tensor.ndim < 1 or tensor.shape[0] != len(self.global_ids):
            raise ValueError("local query output must match the prepared device and packed row count")
        return tensor.index_select(0, self.restore_order)

    def causal_bounds(self) -> tuple[tuple[int, int], ...]:
        """Return each compact row's exact and conservative right-down causal counts for offline checks."""
        self.validate_versions()
        bounds = []
        previous = 0
        for sequence, (start, end) in enumerate(pairwise(self.batch_meta.global_cu_seqlens)):
            count = self.cumulative_queries[sequence] - previous
            ids = self.global_ids[previous:previous + count]
            bounds.extend((token - start + 1, end - start - count + row + 1)
                          for row, token in enumerate(ids))
            previous += count
        return tuple(bounds)


def local_indexer_forward_probe(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
                                 layout: DsaLocalQueryLayout, schedule: MixedSfaSchedule,
                                 retained: torch.Tensor | None = None) -> tuple:
    """Execute exact LI on actual local Q and canonical full K, with two device phase closures.

    Args:
        query: Detached BF16 [local Q,64,128] in the prepared caller storage order.
        key: Detached BF16 [global K,1,128] in canonical packed order.
        weights: Already scaled BF16 or FP32 [local Q,64] in caller order.
        layout: Cold prepared genuine local counts, explicit positions and permutations.
        schedule: Mixed groups with one reserved coordinator and rounds=1.
        retained: Optional independent uint8 scratch, reusable only on the same stream
            or after an explicit event dependency. Empty Q needs no arithmetic scratch.

    Returns:
        Caller-ordered sequence-relative indices, values, two phase traces and scratch.
        This preparatory probe has no autograd or SHMEM transport; it is not yet a
        local-query CP training backend. No Q or weight collective is performed.
    """
    if not isinstance(layout, DsaLocalQueryLayout) or schedule.rounds != 1:
        raise ValueError("local LI requires a prepared layout and rounds=1")
    layout.validate_versions()
    tokens = len(layout.global_ids)
    expected = ((tokens, 64, 128), (layout.batch_meta.global_valid_queries, 1, 128), (tokens, 64))
    for tensor, shape, dtypes in zip((query, key, weights), expected,
                                    ((torch.bfloat16,), (torch.bfloat16,), (torch.bfloat16, torch.float32))):
        if (tensor.device != layout.device or tuple(tensor.shape) != shape or tensor.dtype not in dtypes
                or tensor.requires_grad or not tensor.is_contiguous()):
            raise ValueError("local LI requires detached contiguous states with prepared shapes/device/dtypes")
    required = MIXED_INDEXER_WORKSPACE_BYTES if tokens else 0
    if retained is not None and (retained.device != layout.device or retained.dtype != torch.uint8
                                or retained.ndim != 1 or retained.numel() < required
                                or retained.requires_grad or not retained.is_contiguous()):
        raise ValueError("local LI scratch must match the prepared device and bounded workspace")
    _load_native()
    if torch.ops.hyper_parallel.dsa_local_indexer_version() != 1:
        raise RuntimeError("local LI requires adapter ABI 1; rebuild this checkout's payload")
    if retained is None:
        retained = torch.empty(required, dtype=torch.uint8, device=layout.device)
    trace = torch.zeros((2, 20, 64), dtype=torch.int64, device=layout.device)
    indices = torch.full((tokens, 1, 2048), -2, dtype=torch.int32, device=layout.device)
    values = torch.full(indices.shape, float("nan"), dtype=torch.bfloat16, device=layout.device)
    config = schedule.runtime_config(layout.device)
    torch.ops.hyper_parallel.dsa_local_indexer_out(
        layout.pack(query), key, layout.pack(weights), layout.query_lengths, layout.key_lengths,
        config, trace, retained, layout.positions, indices, values)
    return layout.restore(indices), layout.restore(values), tuple(trace.unbind()), retained
