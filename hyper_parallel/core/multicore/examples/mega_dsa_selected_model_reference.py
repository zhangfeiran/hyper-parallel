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
"""Unabsorbed CPU model reference for arbitrary actual global packed selected histories."""

from __future__ import annotations

import torch

from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.selected_cpu_reference import (
    selected_attention_cpu,
    selected_kl_cpu,
)


@torch.autocast(device_type='cpu', enabled=False)
def selected_unabsorbed_model_reference(
        model: DeepseekV32DSAAttention, hidden: torch.Tensor,
        embeddings: tuple[torch.Tensor, torch.Tensor], meta: DsaBatchMeta,
        indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate unabsorbed MLA and selected KL on the caller's actual selected set.

    Args:
        model: Actual HP attention layer with CPU FP32/FP64 parameters.
        hidden: Complete packed [1,T,hidden] CPU input in the same precision.
        embeddings: Explicit source cosine/sine values in the working precision.
            The optional NPU RoPE primitive must be explicitly mocked on CPU.
        meta: Complete ordered global queries and keys, with packed boundaries.
        indices: Actual global packed CPU int32 [T,K] selection. Histories may
            exceed K; no full-history substitution or new Top-K is performed.

    Returns:
        Unabsorbed projected output and optional detached-teacher selected KL.
        Main K and V use separate per-head up-projection edges into the original
        KV Parameter. Per-row attention/KL scratch is bounded; full source
        projection activations and full parameter gradients remain necessary.

    Raises:
        ValueError: Inputs do not describe complete CPU FP32/FP64 model states.
    """
    total, heads = meta.global_valid_queries, model.num_heads
    if (meta.q_global_ids != tuple(range(total)) or meta.kv_global_ids != tuple(range(total))
            or hidden.device.type != 'cpu' or hidden.dtype not in (torch.float32,torch.float64)
            or hidden.shape != (1,total,model.o_proj.out_features)
            or any(parameter.device.type != 'cpu' or parameter.dtype != hidden.dtype
                   for parameter in model.parameters())):
        raise ValueError('selected model oracle requires complete ordered CPU FP32/FP64 inputs and parameters')
    q_resid, main, _weights = model._prepare_attention_states(hidden,embeddings)
    query, compressed, qr, kr = main
    q_pass = model.q_b_proj(q_resid).reshape(total,heads,model.qk_head_dim)[...,:model.qk_nope_head_dim]
    expanded = compressed.reshape(total,model.kv_lora_rank)@model.kv_b_proj.weight.T
    key,value = expanded.reshape(total,heads,-1).split((model.qk_nope_head_dim,model.v_head_dim),-1)
    output,_stats = selected_attention_cpu(q_pass,key,value,qr.reshape(total,heads,model.qk_rope_head_dim),
        kr.reshape(total,model.qk_rope_head_dim),indices,meta,attention_scale=model.scaling)
    output = model.o_proj(output.reshape(1,total,heads*model.v_head_dim))
    if not (model.training and not model.freeze_dsa and model.dsa_loss_coeff):
        return output,None
    iq,ik,weight = model._project_index_states(hidden,q_resid,embeddings)
    index = (iq.reshape(total,model.num_index_heads,model.index_head_dim),
             ik.reshape(total,model.index_head_dim),weight.reshape(total,model.num_index_heads))
    main = (query.reshape(total,heads,model.kv_lora_rank),compressed.reshape(total,model.kv_lora_rank),
            qr.reshape(total,heads,model.qk_rope_head_dim),kr.reshape(total,model.qk_rope_head_dim))
    loss = selected_kl_cpu(index,main,indices,meta,attention_scale=model.scaling,
                           normalization=DsaLossNormalization(total),loss_coeff=model.dsa_loss_coeff)
    return output,loss
