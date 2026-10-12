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
"""Actual head-sharded attention projections around the explicit fused CP/TP training bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional

from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.modules.dsa_attention import (
    DeepseekV32DSAAttention,
    _absorb_query_projection,
    _apply_attention_rope,
    _apply_index_rope,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_tp import (
    FusedCpTpInvocation,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_model_boundary import (
    FusedDsaCpAttention,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training import (
    FusedDsaLocalCpTpTrainingProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.parameter_vjp import (
    ParameterVjpCapture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.query_projection import (
    project_query_with_fp32_vjp,
)
from hyper_parallel.core.multicore.modules.mega_dsa.source_bmm import (
    source_bmm_with_full_precision_weight_vjp,
)
from hyper_parallel.core.multicore.modules.mega_dsa.source_linear import (
    source_linear_with_full_precision_weight_vjp,
)
from hyper_parallel.core.multicore.modules.mega_dsa.source_rms_norm import (
    source_rms_norm_with_fp32_vjp,
)
from hyper_parallel.core.multicore.modules.mega_dsa.source_rope import (
    source_attention_rope_with_fp32_vjp,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_projections import (
    DsaTpProjectionCollectives,
    shard_dsa_tp_projections,
)
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class FusedDsaCpTpAttention(DeepseekV32DSAAttention):
    """Consume an unsharded HP attention module and physically shard its head projections.

    Construct before optimizer/FSDP wrapping, with explicit dsa_tp_group in the
    replacement context. Original Parameter objects and HP checkpoint keys
    remain; five projection weights and corresponding row biases have local
    storage. The shared QKV projection/norms and index-K projection remain
    replicated. Caller owns CP parameter reduction, root lifetime and compatible
    backward schedules. Global checkpoints use the explicit TP load/gather APIs.
    """

    def __init__(self, *, module: nn.Module, module_fqn: str = '',
                 context: Mapping[str, Any] | None = None) -> None:
        """Shard source storage and prepare TP projection VJPs without constructing a native backend."""
        if type(module) is not DeepseekV32DSAAttention:  # pylint: disable=unidiomatic-typecheck
            raise TypeError(f'{module_fqn}: fused TP boundary requires an unsharded HP DeepseekV32DSAAttention')
        if ((module.index_topk, module.kv_lora_rank, module.qk_rope_head_dim, module.index_head_dim)
                != (2048, 512, 64, 128)):
            raise ValueError('fused CP TP attention requires K2048, C512, Dr64 and Di128')
        group = (context or {}).get('dsa_tp_group')
        if group is None:
            raise ValueError('fused CP TP attention requires an explicit dsa_tp_group')
        capture = (context or {}).get('dsa_capture_output_parameter_vjps', False)
        capture_qkv = (context or {}).get('dsa_capture_qkv_parameter_vjps', False)
        capture_kv = (context or {}).get('dsa_capture_kv_parameter_vjps', False)
        capture_query = (context or {}).get('dsa_capture_query_parameter_vjps', False)
        fp32_index_kl = (context or {}).get('dsa_fp32_index_kl', False)
        fp32_query_vjp = (context or {}).get('dsa_fp32_query_vjp', False)
        fp32_compressed_vjp = (context or {}).get('dsa_fp32_compressed_vjp', False)
        fp32_native_query_vjp = (context or {}).get('dsa_fp32_native_query_vjp', False)
        fp32_rope_vjp = (context or {}).get('dsa_fp32_rope_vjp', False)
        projection_order = (context or {}).get('dsa_projection_reduction_order', 'collective')
        if projection_order not in ('collective', 'group_rank'):
            raise ValueError('TP projection reduction order must be collective or group_rank')
        if any(not isinstance(flag, bool) for flag in (capture, capture_qkv, capture_kv, capture_query,
                                                     fp32_index_kl, fp32_query_vjp, fp32_compressed_vjp,
                                                     fp32_native_query_vjp, fp32_rope_vjp)):
            raise TypeError('parameter VJP capture must be explicitly enabled by a boolean')
        if fp32_query_vjp and (not all((capture_qkv, capture_kv, capture_query))
                               or type(module.q_a_layernorm) is not nn.RMSNorm  # pylint: disable=unidiomatic-typecheck
                               or len(module.q_a_layernorm.normalized_shape) != 1
                               or module.q_a_layernorm.weight is None or module.q_a_layernorm.eps is None):
            raise ValueError('FP32 query VJP requires QKV/KV/query capture and affine one-dimensional nn.RMSNorm')
        if fp32_compressed_vjp and (not fp32_query_vjp
                or type(module.kv_a_layernorm) is not nn.RMSNorm  # pylint: disable=unidiomatic-typecheck
                or len(module.kv_a_layernorm.normalized_shape) != 1
                or module.kv_a_layernorm.weight is None or module.kv_a_layernorm.eps is None):
            raise ValueError('FP32 compressed VJP requires FP32 query VJP and affine one-dimensional KV nn.RMSNorm')
        if fp32_native_query_vjp and not fp32_compressed_vjp:
            raise ValueError('FP32 native query VJP requires FP32 compressed VJP')
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        for name, value in vars(module).items():
            self.__dict__[name] = value.copy() if isinstance(value, (dict, set)) else value
        self.fp32_index_kl = fp32_index_kl
        self.fp32_query_vjp = fp32_query_vjp
        if fp32_rope_vjp and not fp32_native_query_vjp:
            raise ValueError('FP32 RoPE VJP requires FP32 native query VJP')
        self.fp32_rope_vjp = fp32_rope_vjp
        self.fp32_compressed_vjp = fp32_compressed_vjp
        self.fp32_native_query_vjp = fp32_native_query_vjp
        layout = shard_dsa_tp_projections(self, group)
        self.local_heads = layout.local_heads
        self.local_index_heads = layout.local_index_heads
        self.tp_collectives = DsaTpProjectionCollectives(group, row_signature=(layout.global_heads,
                                                                             layout.global_index_heads,
                                                                             layout.shards),
                                                           reduction_order=projection_order)
        output_parameters = {'o_proj.weight': self.o_proj.weight}
        if self.o_proj.bias is not None:
            output_parameters['o_proj.bias'] = self.o_proj.bias
        if capture_qkv:
            output_parameters['linear_qkv.weight'] = self.linear_qkv.weight
            if self.linear_qkv.bias is not None:
                output_parameters['linear_qkv.bias'] = self.linear_qkv.bias
        self.capture_qkv_parameter_vjps = capture_qkv
        if capture_kv:
            output_parameters['kv_b_proj.weight'] = self.kv_b_proj.weight
        self.capture_kv_parameter_vjps = capture_kv
        if capture_query:
            output_parameters['q_b_proj.weight'] = self.q_b_proj.weight
            if self.q_b_proj.bias is not None:
                output_parameters['q_b_proj.bias'] = self.q_b_proj.bias
        if fp32_query_vjp:
            output_parameters['q_a_layernorm.weight'] = self.q_a_layernorm.weight
        if fp32_compressed_vjp:
            output_parameters['kv_a_layernorm.weight'] = self.kv_a_layernorm.weight
        self.capture_query_parameter_vjps = capture_query
        self.parameter_vjp_capture = (ParameterVjpCapture(output_parameters)
                                      if capture or capture_qkv or capture_kv or capture_query else None)

    def make_transforms(self) -> list[WeightConverter]:
        """Retain converted HP checkpoint keys; TP loading/gathering is explicit."""
        return []

    def _validate_call(self, hidden, actual_seq_len, kwargs):
        probe, invocation = kwargs.get('dsa_training'), kwargs.get('dsa_invocation')
        if not isinstance(probe, FusedDsaLocalCpTpTrainingProbe) or not isinstance(invocation, FusedCpTpInvocation):
            raise TypeError('fused TP attention requires prepared dsa_training and dsa_invocation')
        backend = probe.forward_backend
        workspace = backend.workspace
        meta, base = invocation.cp.batch_meta, backend.layout.batch_meta
        layout = self.dsa_tp_projection_layout
        if (invocation.backend is not probe.tp_forward or invocation.cp.workspace is not workspace
                or workspace.closed or not workspace.participates or meta.heap_generation != workspace.root.generation
                or layout.group is not probe.tp_forward.li.group
                or layout.members != probe.tp_forward.li.members):
            raise ValueError('fused TP model requires matching live CP workspace, heap and TP group')
        if replace(meta, layer=base.layer, microbatch=base.microbatch, invocation=base.invocation) != base:
            raise ValueError('fused TP model invocation must preserve prepared ownership and storage')
        queries = invocation.local
        queries.validate_versions()
        if (meta.q_global_ids != meta.kv_global_ids or queries.batch_meta != replace(meta,
                layer=queries.batch_meta.layer, microbatch=queries.batch_meta.microbatch,
                invocation=queries.batch_meta.invocation)):
            raise ValueError('fused TP model requires complete owner Q/KV rows in the same caller order')
        if (hidden.ndim != 3 or hidden.shape[0]*hidden.shape[1] != len(meta.kv_global_ids)
                or hidden.device != backend.layout.device or hidden.device != queries.device
                or hidden.dtype != self.linear_qkv.weight.dtype or backend.heads != self.local_heads
                or backend.scale != self.scaling or probe.tp_forward.index_heads != self.local_index_heads):
            raise ValueError('fused TP model states must match prepared rows, dtype, device and local head geometry')
        if self.layer_idx is not None and meta.layer != self.layer_idx:
            raise ValueError('fused TP model invocation layer must match the attention module')
        auxiliary = self.training and not self.freeze_dsa and bool(self.dsa_loss_coeff)
        if (probe.mode != ('lm_plus_kl' if auxiliary else 'lm_only') or probe.loss_coeff != self.dsa_loss_coeff
                or probe.fp32_index_kl != self.fp32_index_kl
                or probe.fp32_compressed_vjp != self.fp32_compressed_vjp
                or probe.fp32_rope_vjp != self.fp32_rope_vjp
                or probe.fp32_native_query_vjp != self.fp32_native_query_vjp
                or invocation.selected_kl != (auxiliary and not self.fp32_index_kl)):
            raise ValueError('fused TP model probe must match training/freeze/coefficient gates')
        if any(value is not None and (name.startswith(('actual_', 'cu_seq')) or name in
                                     ('packed_seq_params', 'dsa_reference', 'dsa_query_layout'))
               for name, value in kwargs.items()):
            raise ValueError('use prepared TP invocation and actual_seq_len; metadata/backend aliases are rejected')
        FusedDsaCpAttention._validate_lengths(actual_seq_len, queries)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                attention_mask: torch.Tensor | None = None, past_key_values: Any | None = None,
                position_ids: torch.Tensor | None = None,
                actual_seq_len: torch.Tensor | Sequence[int] | None = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        """Execute local head projections and fused CP/TP attention, returning a replicated O output.

        Main Q latent uses a SUM backward at its replicated input. O output SUM
        has an identity backward. Shared KV/index-K VJPs have already been SUMmed
        by the native training bridge; no projection-level TP reduction repeats
        them. The O bias is added after output SUM exactly once on every replica.
        """
        self._validate_call(hidden_states, actual_seq_len, kwargs)
        self._validate_forward_inputs(hidden_states, attention_mask, past_key_values, position_ids, kwargs)
        batch, tokens = hidden_states.shape[:2]
        embeddings = position_embeddings if batch*tokens else None
        q_resid, states, kv_weight = self._prepare_attention_states(hidden_states, embeddings)
        output = self._compute_sparse_attention(hidden_states, q_resid, states, embeddings, actual_seq_len, kwargs)
        value_up = kv_weight[:, self.qk_nope_head_dim:].transpose(1, 2)
        compressed = output.permute(2, 0, 1, 3).reshape(self.local_heads, batch*tokens, self.kv_lora_rank)
        restored = (source_bmm_with_full_precision_weight_vjp(compressed, value_up, self.kv_b_proj.weight)
                    if self.capture_kv_parameter_vjps else torch.bmm(compressed, value_up))
        # An explicit last dimension also admits zero owner rows.
        restored = restored.view(self.local_heads, batch, tokens, self.v_head_dim).permute(1, 2, 0, 3).reshape(
            batch, tokens, self.local_heads*self.v_head_dim)
        return self._project_output(restored).to(hidden_states.dtype), None

    def _project_output(self, restored):
        """Keep local O products, TP SUM and the replicated bias before one activation cast."""
        dtype = torch.float32 if restored.dtype in (torch.float16, torch.bfloat16) else restored.dtype
        # Rounding a local product before SUM loses small cross-rank residuals.
        with torch.autocast(device_type=restored.device.type, enabled=False):
            capture = getattr(self, 'parameter_vjp_capture', None)
            weight = (self.o_proj.weight.to(dtype) if capture is None else
                      capture.view('o_proj.weight', dtype=dtype))
            partial = functional.linear(  # pylint: disable=not-callable
                restored.to(dtype), weight)
            result = self.tp_collectives.sum_output(partial)
            if self.o_proj.bias is not None:
                bias = (self.o_proj.bias.to(dtype) if capture is None else
                        capture.view('o_proj.bias', dtype=dtype))
                result = result+bias
        return result

    def _project_query(self, q_resid):
        """SUM local Q-B input VJPs before converting back to the normalized latent dtype."""
        dtype = torch.float32 if q_resid.dtype in (torch.float16, torch.bfloat16) else q_resid.dtype
        with torch.autocast(device_type=q_resid.device.type, enabled=False):
            copied = self.tp_collectives.copy_query_latent(q_resid.to(dtype))
            if self.capture_query_parameter_vjps:
                weight = self.parameter_vjp_capture.view('q_b_proj.weight', dtype=dtype)
                bias = (None if self.q_b_proj.bias is None else
                        self.parameter_vjp_capture.view('q_b_proj.bias', dtype=dtype))
                projected = source_linear_with_full_precision_weight_vjp(
                    copied, weight, self.q_b_proj.weight, bias, self.q_b_proj.bias, full_precision_inputs=True,
                    full_precision_output=self.fp32_query_vjp)
            else:
                projected = project_query_with_fp32_vjp(copied, self.q_b_proj.weight, self.q_b_proj.bias)
        return projected.to(q_resid.dtype)

    def _project_attention_states(self, hidden_states):
        batch, tokens = hidden_states.shape[:2]
        if self.capture_qkv_parameter_vjps:
            dtype = torch.float64 if self.linear_qkv.weight.dtype == torch.float64 else torch.float32
            weight = self.parameter_vjp_capture.view('linear_qkv.weight', dtype=dtype)
            bias = (None if self.linear_qkv.bias is None else
                    self.parameter_vjp_capture.view('linear_qkv.bias', dtype=dtype))
            latent = source_linear_with_full_precision_weight_vjp(
                hidden_states, weight, self.linear_qkv.weight, bias, self.linear_qkv.bias,
                full_precision_output=self.fp32_query_vjp)
        else:
            latent = self.linear_qkv(hidden_states)
        q_latent, compressed, rope = latent.split((self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim), -1)
        if self.fp32_query_vjp:
            scale = self.parameter_vjp_capture.view('q_a_layernorm.weight', dtype=latent.dtype)
            q_resid = source_rms_norm_with_fp32_vjp(q_latent, scale, self.q_a_layernorm)
        else:
            q_resid = self.q_a_layernorm(q_latent)
        projected = self._project_query(q_resid)
        projected = projected.view(batch, tokens, self.local_heads, self.qk_head_dim)
        query, query_rope = projected.split((self.qk_nope_head_dim, self.qk_rope_head_dim), -1)
        weight = self.kv_b_proj.weight
        if self.capture_kv_parameter_vjps:
            dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
            weight = self.parameter_vjp_capture.view('kv_b_proj.weight', dtype=dtype)
        kv_weight = weight.view(self.local_heads, self.qk_nope_head_dim+self.v_head_dim, self.kv_lora_rank)
        if self.capture_kv_parameter_vjps:
            source = query.permute(2, 0, 1, 3).reshape(self.local_heads, batch*tokens, self.qk_nope_head_dim)
            absorbed = source_bmm_with_full_precision_weight_vjp(
                source, kv_weight[:, :self.qk_nope_head_dim], self.kv_b_proj.weight,
                full_precision_inputs=self.fp32_query_vjp, full_precision_output=self.fp32_native_query_vjp)
            absorbed = absorbed.view(self.local_heads, batch, tokens, self.kv_lora_rank).permute(1, 2, 0, 3)
        else:
            absorbed = _absorb_query_projection(query, kv_weight[:, :self.qk_nope_head_dim], num_heads=self.local_heads,
                                                batch_size=batch, seq_length=tokens, input_dim=self.qk_nope_head_dim,
                                                output_dim=self.kv_lora_rank)
        if self.fp32_compressed_vjp:
            scale = self.parameter_vjp_capture.view('kv_a_layernorm.weight', dtype=latent.dtype)
            compressed = source_rms_norm_with_fp32_vjp(compressed, scale, self.kv_a_layernorm)
        else:
            compressed = self.kv_a_layernorm(compressed.to(self.linear_qkv.weight.dtype))
        compressed = compressed.view(batch, tokens, 1, self.kv_lora_rank)
        rope_dtype = torch.float32 if self.fp32_rope_vjp else self.linear_qkv.weight.dtype
        rope = rope.to(rope_dtype).view(batch, tokens, 1, self.qk_rope_head_dim)
        query_rope = query_rope.to(rope_dtype)
        return q_resid, absorbed, compressed, query_rope, rope, kv_weight

    def _project_index_states(self, hidden_states, q_resid, position_embeddings):
        batch, tokens = hidden_states.shape[:2]
        query = self.indexer.wq_b(q_resid.detach().to(self.indexer.wq_b.weight.dtype)).view(
            batch, tokens, self.local_index_heads, self.index_head_dim)
        key = self.indexer.k_norm(self.indexer.wk(hidden_states.detach())).unsqueeze(2)
        weights = self.indexer.weights_proj(hidden_states.detach())
        weights = weights*self.num_index_heads**-.5*self.index_head_dim**-.5
        query, key = _apply_index_rope(query, key, position_embeddings, rope_head_dim=self.qk_rope_head_dim,
                                       index_head_dim=self.index_head_dim, interleaved=self.index_rotary_interleaved)
        return query, key, weights

    # Torch's built-in linear has no Python function body for static inference.
    # pylint: disable=not-callable
    def _project_fp32_index_states(self, hidden_states, position_embeddings):
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return self._project_fp32_index_states_impl(hidden_states, position_embeddings)

    def _project_fp32_index_states_impl(self, hidden_states, position_embeddings):
        hidden = hidden_states.detach().float()
        weight = self.linear_qkv.weight[:self.q_lora_rank].detach().float()
        bias = None if self.linear_qkv.bias is None else self.linear_qkv.bias[:self.q_lora_rank].detach().float()
        latent = functional.linear(hidden, weight, bias)
        norm = {name: parameter.detach().float() for name, parameter in self.q_a_layernorm.named_parameters()}
        q_resid = functional_call(self.q_a_layernorm, norm, (latent,)).detach()
        query = functional.linear(q_resid, self.indexer.wq_b.weight.float(),
                                  None if self.indexer.wq_b.bias is None else self.indexer.wq_b.bias.float())
        query = query.view(*hidden.shape[:2], self.local_index_heads, self.index_head_dim)
        key = functional.linear(hidden, self.indexer.wk.weight.float(),
                                None if self.indexer.wk.bias is None else self.indexer.wk.bias.float())
        norm = {name: parameter.float() for name, parameter in self.indexer.k_norm.named_parameters()}
        key = functional_call(self.indexer.k_norm, norm, (key,)).unsqueeze(2)
        weights = functional.linear(hidden, self.indexer.weights_proj.weight.float(),
                                    None if self.indexer.weights_proj.bias is None else
                                    self.indexer.weights_proj.bias.float())
        weights = weights*self.num_index_heads**-.5*self.index_head_dim**-.5
        embeddings = None if position_embeddings is None else tuple(value.float() for value in position_embeddings)
        query, key = _apply_index_rope(query, key, embeddings, rope_head_dim=self.qk_rope_head_dim,
                                       index_head_dim=self.index_head_dim, interleaved=self.index_rotary_interleaved)
        return query, key, weights
    # pylint: enable=not-callable

    def _prepare_attention_states(self, hidden_states, position_embeddings):
        q_resid, query, compressed, query_rope, key_rope, weights = self._project_attention_states(hidden_states)
        rotary = source_attention_rope_with_fp32_vjp if self.fp32_rope_vjp else _apply_attention_rope
        query_rope, key_rope = rotary(query_rope, key_rope, position_embeddings,
                                     interleaved=self.rotary_interleaved)
        return q_resid, (query, compressed, query_rope, key_rope), weights

    def _compute_sparse_attention(self, hidden_states, q_resid, attention_states,
                                  position_embeddings, actual_seq_len, kwargs):
        del actual_seq_len
        query, compressed, query_rope, key_rope = attention_states
        iq, ik, weight = self._project_index_states(hidden_states, q_resid, position_embeddings)
        main = (query.flatten(0, 1).contiguous(), compressed.flatten(0, 1)[:, 0].contiguous(),
                query_rope.flatten(0, 1).contiguous(), key_rope.flatten(0, 1)[:, 0].contiguous())
        index = (iq.flatten(0, 1).contiguous(), ik.flatten(0, 1)[:, 0].contiguous(), weight.flatten(0, 1).contiguous())
        probe = kwargs['dsa_training']
        li_index = None
        if self.fp32_index_kl and probe.index_active:
            li_index = index
            iq, ik, weight = self._project_fp32_index_states(hidden_states, position_embeddings)
            index = (iq.flatten(0, 1).contiguous(), ik.flatten(0, 1)[:, 0].contiguous(),
                     weight.flatten(0, 1).contiguous())
        if li_index is None:
            output, loss = probe.forward(kwargs['dsa_invocation'], main, index)
        else:
            output, loss = probe.forward(kwargs['dsa_invocation'], main, index, li_index=li_index)
        if probe.mode == 'lm_plus_kl':
            output = aux_loss_auto_scale(output, loss)
        return output.reshape(*hidden_states.shape[:2], self.local_heads, self.kv_lora_rank)
