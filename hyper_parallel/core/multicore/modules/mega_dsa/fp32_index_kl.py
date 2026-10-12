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
"""Explicit FP32 index logits/VJP primitive with a caller-provided detached teacher."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedCpSavedAttention,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.torch.ops import _load_native


@dataclass(frozen=True)
class Fp32IndexScores:
    """Own local-head logits and keep original input/selection version guards."""

    probe: Fp32IndexKlProbe
    inputs: tuple[torch.Tensor, ...]
    versions: tuple[int, ...]
    logits: torch.Tensor
    logits_version: int
    trace: torch.Tensor
    selection_version: int


@dataclass(frozen=True)
class Fp32IndexVjp:
    """Own raw FP32 Q/K/weight derivatives, per-query loss and device receipts.

    Key gradients cover complete keys and include only this query shard's
    contribution. The caller performs CP ownership return and any TP SUM,
    then applies its declared projection/parameter precision boundaries.
    """

    gradients: tuple[torch.Tensor, ...]
    loss: torch.Tensor
    trace: torch.Tensor


class Fp32IndexKlProbe:
    """Prepare fixed global-packed selection for a native FP32 KL primitive.

    Scores sum only the supplied index heads. A TP caller must SUM those
    logits before VJP and supply the globally normalized detached teacher.
    This prototype does not install a model/autograd backend or a fallback.
    Native key accumulation uses FP32 atomics and rejects deterministic mode.
    """

    def __init__(self, batch_meta: DsaBatchMeta, indices: torch.Tensor, device: torch.device | str) -> None:
        """Validate an explicit CPU selection before any device submission."""
        total = batch_meta.global_valid_queries
        if batch_meta.kv_global_ids != tuple(range(total)):
            raise ValueError('FP32 index KL requires complete ordered global keys')
        if (indices.device.type != 'cpu' or indices.dtype not in (torch.int32, torch.int64)
                or indices.shape != (len(batch_meta.q_global_ids), 2048)):
            raise ValueError('FP32 index KL selection must be an explicit CPU integer [Tq,2048] snapshot')
        for query, row in zip(batch_meta.q_global_ids, indices.tolist()):
            start = batch_meta.global_cu_seqlens[batch_meta.sequence_position(query)[0]]
            padding = False
            selected = set()
            for key in row:
                if key == -1:
                    padding = True
                elif padding or key < start or key > query or key in selected:
                    raise ValueError('FP32 index KL requires unique legal causal IDs followed by -1 padding')
                else:
                    selected.add(key)
            if not selected:
                raise ValueError('FP32 index KL requires at least one selected key per query')
        self.batch_meta = batch_meta
        self.device = torch.device(device)
        self.indices = indices.to(device=self.device, dtype=torch.int32).clone().contiguous()
        self._prepared_indices = self.indices
        self._selection_version = self.indices._version
        self._saved_attention = None

    @classmethod
    def from_saved_attention(cls, saved: FusedCpSavedAttention) -> Fp32IndexKlProbe:
        """Reuse an authentic local fused LI selection without copying IDs to the host.

        The fused producer guarantees legal unique sequence-relative selected
        IDs. Its saved-state receipt and version guards remain authoritative;
        this entry does not admit an arbitrary device selection tensor.
        """
        if not isinstance(saved, FusedCpSavedAttention) or saved.query_scope != 'local':
            raise ValueError('FP32 index KL requires native local saved attention')
        saved.validate_versions()
        layout = saved.local_query_layout
        rows = len(layout.global_ids)
        if saved.indices.dtype != torch.int32 or saved.indices.shape != (rows, 1, 2048):
            raise ValueError('FP32 index KL requires native int32 [local queries,1,2048] selection')
        instance = object.__new__(cls)
        instance.batch_meta = replace(saved.batch_meta,
                                      kv_global_ids=tuple(range(saved.batch_meta.global_valid_queries)))
        instance.device = layout.device
        starts = (layout.global_ids_tensor - layout.positions).to(torch.int32)[:, None]
        packed = saved.indices[:, 0]
        instance.indices = layout.restore(torch.where(packed >= 0, packed + starts, -1)).contiguous()
        instance._prepared_indices = instance.indices
        instance._selection_version = instance.indices._version
        instance._saved_attention = saved
        return instance

    def _check_native(self) -> None:
        if torch.are_deterministic_algorithms_enabled():
            raise ValueError('FP32 index KL uses nondeterministic key atomics')
        if self.indices is not self._prepared_indices or self.indices._version != self._selection_version:
            raise ValueError('FP32 index KL selection changed after preparation')
        if self._saved_attention is not None:
            self._saved_attention.validate_versions()
        _load_native()
        version = getattr(torch.ops.hyper_parallel, 'dsa_fp32_index_kl_version', None)
        if version is None or version() != 1:
            raise RuntimeError('FP32 index KL requires native ABI 1; rebuild this checkout')

    def _validate_states(self, inputs: tuple[torch.Tensor, ...]) -> None:
        query, key, weight = inputs
        rows = len(self.batch_meta.q_global_ids)
        if (query.ndim != 3 or query.shape[0] != rows or not 1 <= query.shape[1] <= 128
                or query.shape[2] != 128 or key.shape != (self.batch_meta.global_valid_queries, 128)
                or weight.shape != query.shape[:2]):
            raise ValueError('FP32 index KL requires Q[Tq,H<=128,128], complete K[T,128] and weights[Tq,H]')
        if any(value.dtype != torch.float32 or value.device != self.device for value in inputs):
            raise ValueError('FP32 index KL preserves FP32 index states on the prepared device')

    def _submit(self, inputs: tuple[torch.Tensor, ...], teacher: torch.Tensor,
                logits: torch.Tensor, phase: int) -> tuple:
        native = tuple(value.detach().contiguous().clone() for value in inputs)
        target = teacher.detach().contiguous().clone()
        global_logits = logits.detach().contiguous().clone()
        local = torch.empty_like(global_logits)
        gradients = tuple(torch.empty_like(value) for value in native)
        loss = torch.empty((native[0].shape[0],), dtype=torch.float32, device=self.device)
        trace = torch.zeros((40, 4), dtype=torch.int64, device=self.device)
        if native[0].shape[0]:
            torch.ops.hyper_parallel.dsa_fp32_index_kl_out(
                *native, target, global_logits, self.indices, phase, local, *gradients, loss, trace)
        else:
            for value in gradients:
                value.zero_()
        return local, gradients, loss, trace

    def scores(self, query: torch.Tensor, key: torch.Tensor, weight: torch.Tensor) -> Fp32IndexScores:
        """Compute local-head logits without truncating FP32 index Q/K/weights.

        Returns:
            Version-guarded states and local-head logits. Selection padding
            contributes zero; native VJP applies the selection mask after TP SUM.
        """
        inputs = (query, key, weight)
        self._validate_states(inputs)
        self._check_native()
        empty = torch.zeros((query.shape[0], 2048), dtype=torch.float32, device=self.device)
        logits, _, _, trace = self._submit(inputs, empty, empty, 0)
        return Fp32IndexScores(self, inputs, tuple(value._version for value in inputs), logits,
                               logits._version, trace, self.indices._version)

    def vjp(self, saved: Fp32IndexScores, teacher: torch.Tensor, global_logits: torch.Tensor) -> Fp32IndexVjp:
        """Evaluate raw FP32 KL VJP after the caller's teacher/logits TP reductions.

        Args:
            saved: This probe's completed score result with unchanged inputs.
            teacher: Detached nonnegative finite target mass [Tq,2048], zero
                at every -1 selection. Caller prepares/validates its distribution.
            global_logits: Caller-SUM logits over all participating index heads.

        Returns:
            Unscaled raw derivatives and per-query KL loss. Coefficient,
            normalization and upstream scalar apply outside exactly once.
        """
        if (saved.probe is not self or saved.versions != tuple(value._version for value in saved.inputs)
                or saved.logits._version != saved.logits_version or self.indices._version != saved.selection_version):
            raise ValueError('FP32 index KL saved inputs, logits or selection changed')
        expected = (len(self.batch_meta.q_global_ids), 2048)
        for value in (teacher, global_logits):
            if value.shape != expected or value.dtype != torch.float32 or value.device != self.device:
                raise ValueError('FP32 index KL teacher/global logits must be FP32 [Tq,2048] on the prepared device')
        self._check_native()
        _, gradients, loss, trace = self._submit(saved.inputs, teacher, global_logits, 1)
        return Fp32IndexVjp(gradients, loss, trace)


def validate_fp32_index_kl_trace(trace: torch.Tensor, *, query_tokens: int, phase: int) -> dict:
    """Verify an explicit CPU phase receipt without implicit synchronization."""
    if (trace.device.type != 'cpu' or trace.dtype != torch.int64 or trace.shape != (40, 4)
            or isinstance(query_tokens, bool) or not isinstance(query_tokens, int) or query_tokens < 0
            or isinstance(phase, bool) or phase not in (0, 1)):
        raise ValueError('FP32 index KL requires a CPU int64 [40,4] receipt and valid query count/phase')
    if query_tokens == 0:
        if bool(trace.any()):
            raise ValueError('empty FP32 index KL must not write native receipts')
        return {'phase': phase, 'queries': 0, 'native_launch': False}
    for block, record in enumerate(trace.tolist()):
        rows = tuple(range(block, query_tokens, 40))
        expected = [phase + 1, len(rows), sum(row + 1 for row in rows), rows[-1] if rows else -1]
        if record != expected:
            raise ValueError(f'FP32 index KL block {block} has incomplete or invalid query evidence')
    return {'phase': phase, 'queries': query_tokens, 'native_launch': True}
