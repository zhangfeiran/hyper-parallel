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
"""Parameter-preserving real attention boundary for prepared local CP fused training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention
from hyper_parallel.core.multicore.modules.mega_dsa.local_query import DsaLocalQueryLayout
from hyper_parallel.core.multicore.modules.mega_dsa.local_training import FusedDsaLocalCpTrainingProbe
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import DsaWorkspaceInvocation
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class FusedDsaCpAttention(DeepseekV32DSAAttention):
    """Retain real model projections and use caller-prepared owner-local CP execution.

    Hidden rows contain the complete owner shard in the common Q/KV caller
    order. Supplied RoPE embeddings already correspond to these global token
    positions. Preparation, parameter reduction, root lifetime and auxiliary
    upstream scale belong to the caller. TP is not yet supported. This opt-in
    boundary has no implicit backend construction or automatic fallback.
    """

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Preserve an existing HP attention module and its original checkpoint format.

        Args:
            module: Existing unsharded DeepseekV32DSAAttention.
            module_fqn: Diagnostic replacement path.
            context: Existing replacement context; TP must be disabled.
        """
        if type(module) is not DeepseekV32DSAAttention:  # pylint: disable=unidiomatic-typecheck
            raise TypeError(f"{module_fqn}: fused CP boundary requires an existing DeepseekV32DSAAttention")
        if (context or {}).get("tp"):
            raise ValueError("fused CP model boundary currently requires TP=1")
        if (module.num_heads not in (32, 64) or module.num_index_heads != 64
                or (module.index_topk, module.kv_lora_rank, module.qk_rope_head_dim, module.index_head_dim)
                != (2048, 512, 64, 128)):
            raise ValueError("fused CP model boundary requires H32/H64, Hi64, K2048, C512, Dr64 and Di128")
        # Re-running the base constructor would replace the source fused Parameter.
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        for name, value in vars(module).items():
            self.__dict__[name] = value.copy() if isinstance(value, (dict, set)) else value

    def make_transforms(self) -> list[WeightConverter]:
        """Retain the already transformed HP checkpoint without creating new Parameters."""
        return []

    def _validate_call(self, hidden_states, actual_seq_len, kwargs):
        probe, invocation, queries = (kwargs.get(name) for name in
                                      ("dsa_training", "dsa_invocation", "dsa_query_layout"))
        if (not isinstance(probe, FusedDsaLocalCpTrainingProbe)
                or not isinstance(invocation, DsaWorkspaceInvocation) or not isinstance(queries, DsaLocalQueryLayout)):
            raise TypeError("fused model requires prepared dsa_training, dsa_invocation and dsa_query_layout")
        backend = probe.forward_backend
        workspace = backend.workspace
        meta, base = invocation.batch_meta, backend.layout.batch_meta
        if (invocation.workspace is not workspace or workspace.closed or not workspace.participates
                or meta.heap_generation != workspace.root.generation):
            raise ValueError("fused model invocation requires its live participating workspace and heap generation")
        if replace(meta, layer=base.layer, microbatch=base.microbatch, invocation=base.invocation) != base:
            raise ValueError("fused model invocation must preserve the prepared ownership and storage layout")
        queries.validate_versions()
        if (queries.batch_meta != replace(meta, layer=queries.batch_meta.layer,
                                          microbatch=queries.batch_meta.microbatch,
                                          invocation=queries.batch_meta.invocation)
                or meta.q_global_ids != meta.kv_global_ids):
            raise ValueError("fused model requires complete owner Q/KV rows in the same caller order")
        if (hidden_states.ndim != 3 or hidden_states.shape[0]*hidden_states.shape[1] != len(meta.kv_global_ids)
                or hidden_states.device != backend.layout.device or hidden_states.device != queries.device
                or hidden_states.dtype != self.linear_qkv.weight.dtype):
            raise ValueError("fused model hidden states must match prepared local owner rows, device "
                             "and parameter dtype")
        if backend.heads != self.num_heads or backend.scale != self.scaling:
            raise ValueError("fused model backend must preserve original head count and attention scale")
        if self.layer_idx is not None and meta.layer != self.layer_idx:
            raise ValueError("prepared invocation layer must match the attention module")
        auxiliary = self.training and not self.freeze_dsa and bool(self.dsa_loss_coeff)
        if (probe.mode != ("lm_plus_kl" if auxiliary else "lm_only") or probe.loss_coeff != self.dsa_loss_coeff
                or not probe.fused_kl):
            raise ValueError("fused model requires a prepared probe matching training/freeze/coefficient gates "
                             "and fused KL")
        if any(value is not None and (name.startswith(("actual_", "cu_seq")) or name == "packed_seq_params"
                                     or name == "dsa_reference") for name, value in kwargs.items()):
            raise ValueError("use prepared fused metadata and actual_seq_len; length/backend aliases are not accepted")
        self._validate_lengths(actual_seq_len, queries)

    @staticmethod
    def _validate_lengths(actual_seq_len, queries):
        if actual_seq_len is None or actual_seq_len is queries.key_lengths:
            return
        if isinstance(actual_seq_len, torch.Tensor):
            if actual_seq_len.device.type != "cpu" or actual_seq_len.dtype not in (torch.int32, torch.int64):
                raise ValueError("lengths must be a CPU integer snapshot or the prepared query-layout key_lengths")
            actual_seq_len = actual_seq_len.tolist()
        if tuple(actual_seq_len) != queries.batch_meta.global_cu_seqlens[1:]:
            raise ValueError("actual_seq_len conflicts with the prepared global packed boundaries")

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                attention_mask: torch.Tensor | None = None, past_key_values: Any | None = None,
                position_ids: torch.Tensor | None = None,
                actual_seq_len: torch.Tensor | Sequence[int] | None = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        """Execute the original projections and output layer around an explicit CP training probe.

        Args:
            hidden_states: BSN complete owner-local hidden rows in prepared Q/KV order.
            position_embeddings: Original model RoPE tensors for those global positions.
            attention_mask: Standard source causal mask; packed causality comes from metadata.
            past_key_values: Unsupported as in the source training path.
            position_ids: Source-compatible local position IDs.
            actual_seq_len: Optional global cumulative ends agreeing with prepared metadata.
            **kwargs: Explicit dsa_training, dsa_invocation and dsa_query_layout.

        Returns:
            Caller-local projected attention output and None.
        """
        self._validate_call(hidden_states, actual_seq_len, kwargs)
        return super().forward(hidden_states, position_embeddings, attention_mask, past_key_values,
                               position_ids, actual_seq_len, **kwargs)

    def _compute_attention_output(self, hidden_states, position_embeddings, actual_seq_len, kwargs):
        batch, tokens = hidden_states.shape[:2]
        # Empty owners still enter native CP protocols, but have no rotary rows.
        embeddings = position_embeddings if batch*tokens else None
        q_resid, states, kv_weight = self._prepare_attention_states(hidden_states, embeddings)
        output = self._compute_sparse_attention(hidden_states, q_resid, states, embeddings, actual_seq_len, kwargs)
        output = output[..., :self.kv_lora_rank].permute(2, 0, 1, 3).reshape(
            self.num_heads, batch*tokens, self.kv_lora_rank)
        restored = torch.bmm(output, kv_weight[:, self.qk_nope_head_dim:].transpose(1, 2))
        return restored.view(self.num_heads, batch, tokens, self.v_head_dim).permute(1, 2, 0, 3).reshape(
            batch, tokens, self.num_heads*self.v_head_dim)

    def _compute_sparse_attention(self, hidden_states, q_resid, attention_states,
                                  position_embeddings, actual_seq_len, kwargs):
        del actual_seq_len
        probe = kwargs["dsa_training"]
        query, compressed, query_rope, key_rope = attention_states
        index_query, index_key, weight = self._project_index_states(hidden_states, q_resid, position_embeddings)
        main = (query.flatten(0, 1).contiguous(), compressed.flatten(0, 1)[:, 0].contiguous(),
                query_rope.flatten(0, 1).contiguous(), key_rope.flatten(0, 1)[:, 0].contiguous())
        index = (index_query.flatten(0, 1).contiguous(), index_key.flatten(0, 1)[:, 0].contiguous(),
                 weight.flatten(0, 1).contiguous())
        output, loss = probe.forward(kwargs["dsa_invocation"], main, index, query_layout=kwargs["dsa_query_layout"])
        if probe.mode == "lm_plus_kl":
            output = aux_loss_auto_scale(output, loss)
        return output.reshape(*hidden_states.shape[:2], self.num_heads, self.kv_lora_rank)
