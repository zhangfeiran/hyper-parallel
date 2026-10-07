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
"""Explicit P0 replacement of an existing HP DSA module's operator boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaReference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaLossNormalization
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class CannDsaReferenceAttention(DeepseekV32DSAAttention):
    """Preserve model parameters/projections and use an explicit prepared reference.

    This replacement targets an already constructed ``DeepseekV32DSAAttention``
    through the existing ``replace_module`` mechanism. Its forward call requires
    ``dsa_reference=backend``; it has no automatic fallback or hidden backend
    construction. Prepared metadata defines the packed causal boundaries.
    The caller owns preparation, device activation and auxiliary upstream scale.
    """

    def __init__(
        self, *, module: nn.Module, module_fqn: str = "", context: Mapping[str, Any] | None = None,
    ) -> None:
        """Retain source state by identity without running its weight-fusing constructor.

        Args:
            module: Exact existing HP DSA attention module, before sharding.
            module_fqn: Replacement target name, for diagnostic messages.
            context: Existing replacement context; TP/CP must be disabled.
        """
        if type(module) is not DeepseekV32DSAAttention:  # pylint: disable=unidiomatic-typecheck
            raise TypeError(f"{module_fqn}: reference boundary requires an existing DeepseekV32DSAAttention")
        if any((context or {}).get(axis) for axis in ("tp", "cp")):
            raise ValueError("the P0 model reference boundary requires TP=CP=1")
        if module.index_topk != 2048 or module.kv_lora_rank != 512 or module.qk_rope_head_dim != 64:
            raise ValueError("the P0 model reference boundary requires K=2048, C=512 and Dr=64")
        # The base constructor creates a fused Parameter. This stage must retain
        # the already transformed Parameters and independent module registries.
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        for name, value in vars(module).items():
            self.__dict__[name] = value.copy() if isinstance(value, (dict, set)) else value

    def make_transforms(self) -> list[WeightConverter]:
        """Return no additional conversion: the existing HP checkpoint format is retained."""
        return []

    def _validate_reference(
        self, hidden_states: torch.Tensor, actual_seq_len: torch.Tensor | Sequence[int] | None,
        kwargs: dict[str, Any],
    ) -> CannDsaReference:
        backend = kwargs.get("dsa_reference")
        if not isinstance(backend, CannDsaReference):
            raise TypeError("the reference model requires dsa_reference=prepared CannDsaReference")
        layout = backend.layout
        meta = layout.batch_meta
        if hidden_states.ndim != 3 or hidden_states.shape[0] * hidden_states.shape[1] != meta.global_valid_queries:
            raise ValueError("model hidden states must match the prepared complete packed query count")
        if hidden_states.device != layout.device:
            raise ValueError("model hidden states and prepared reference must be on the same device")
        if backend.attention_scale != self.scaling:
            raise ValueError("reference attention_scale must equal the original model scaling")
        if self.layer_idx is not None and meta.layer != self.layer_idx:
            raise ValueError("prepared metadata layer does not match the attention module")
        if any(value is not None and (name.startswith(("actual_", "cu_seq")) or name == "packed_seq_params")
               for name, value in kwargs.items()):
            raise ValueError("use prepared metadata and actual_seq_len; packed length aliases are not accepted")
        self._validate_lengths(actual_seq_len, backend)
        return backend

    @staticmethod
    def _validate_lengths(
        actual_seq_len: torch.Tensor | Sequence[int] | None, backend: CannDsaReference,
    ) -> None:
        """Check optional lengths without reading the prepared device tensor."""
        layout = backend.layout
        if actual_seq_len is None or actual_seq_len is layout.length_tensor:
            return
        if isinstance(actual_seq_len, torch.Tensor):
            if actual_seq_len.device.type != "cpu" or actual_seq_len.dtype not in (torch.int32, torch.int64):
                raise ValueError("lengths must be a CPU integer snapshot or the prepared layout length_tensor")
            actual_seq_len = actual_seq_len.tolist()
        if tuple(actual_seq_len) != layout.cumulative_lengths:
            raise ValueError("actual_seq_len conflicts with prepared packed metadata")

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None, past_key_values: Any | None = None,
        position_ids: torch.Tensor | None = None, actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run inherited projections/output restoration with an explicit reference.

        Args:
            hidden_states: Original model BSN hidden states.
            position_embeddings: Original model cosine/sine RoPE tensors.
            attention_mask: Standard causal mask accepted by the source module.
            past_key_values: Unsupported, as in the source training path.
            position_ids: Position IDs accepted by the source module.
            actual_seq_len: Optional ends without zero, consistent with prepared
                metadata; a device tensor must be the prepared length_tensor.
            **kwargs: Must contain dsa_reference. Prepared metadata is required
                for packed boundaries; alternate length aliases are rejected.

        Returns:
            Projected attention output and None, preserving the source contract.
        """
        self._validate_reference(hidden_states, actual_seq_len, kwargs)
        return super().forward(hidden_states, position_embeddings, attention_mask, past_key_values,
                               position_ids, actual_seq_len, **kwargs)

    def _compute_sparse_attention(
        self, hidden_states: torch.Tensor, q_resid: torch.Tensor,
        attention_states: tuple[torch.Tensor, ...], position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        actual_seq_len: torch.Tensor | Sequence[int] | None, kwargs: dict[str, Any],
    ) -> torch.Tensor:
        del actual_seq_len
        backend = kwargs["dsa_reference"]
        index_query, index_key, weight = self._project_index_states(hidden_states, q_resid, position_embeddings)
        index_inputs = (index_query.flatten(0, 1), index_key.flatten(0, 1)[:, 0], weight.flatten(0, 1))
        selection = backend.indexer(*index_inputs)
        output, stats = backend.attention_bsnd(*attention_states, selection)
        if not (self.training and not self.freeze_dsa and self.dsa_loss_coeff):
            return output
        query, compressed, query_rope, key_rope = attention_states
        main_inputs = (query.flatten(0, 1), compressed.flatten(0, 1)[:, 0],
                       query_rope.flatten(0, 1), key_rope.flatten(0, 1)[:, 0])
        loss = backend.kl_loss(*index_inputs, main_inputs, selection, stats,
                               normalization=DsaLossNormalization(backend.layout.batch_meta.global_valid_queries),
                               loss_coeff=self.dsa_loss_coeff)
        return aux_loss_auto_scale(output, loss)
