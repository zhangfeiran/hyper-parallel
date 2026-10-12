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
"""Random standard-dimensional BF16 DSA fixtures with explicit source configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta

_STANDARD_DIMENSIONS = {'hidden_size':7168, 'q_lora_rank':1536, 'kv_lora_rank':512,
                        'num_attention_heads':128, 'qk_nope_head_dim':128, 'qk_rope_head_dim':64,
                        'v_head_dim':128, 'index_n_heads':64, 'index_head_dim':128, 'index_topk':2048}


def standard_model_fixture(config: Mapping[str, Any], *, seed: int = 20261258,
                           dtype: torch.dtype = torch.bfloat16,
                           device: torch.device | str = 'cpu') -> DeepseekV32DSAAttention:
    """Construct actual HP projections at the supplied standard checkpoint dimensions.

    Args:
        config: Explicit DeepSeek-V3.2 config with the standard dimensions and
            source RMSNorm/YaRN settings. Quantized checkpoint weights are not loaded.
        seed: Independent initialization seed; does not change global RNG state.
        dtype: Floating parameter dtype. BF16 is the native training target.
        device: CPU for initialization or meta for shape-only verification.

    Returns:
        Randomly initialized HP attention with all standard parameter shapes,
        interleaved main RoPE, half-layout index RoPE and source YaRN attention scale.
        This is an attention-layer fixture, not pretrained or full-LM acceptance.

    Raises:
        ValueError: Dimensions, seed, precision or source options are unsupported.
    """
    if (not isinstance(config, Mapping) or any(not isinstance(config.get(name), int) or config[name] != value
                                             for name, value in _STANDARD_DIMENSIONS.items())
            or config.get('attention_bias', False) or config.get('attention_dropout', 0.) != 0.
            or isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63-1
            or dtype not in (torch.bfloat16, torch.float32, torch.float64)):
        raise ValueError('standard DSA fixture requires exact standard dimensions, no dropout/bias and a valid seed')
    if torch.device(device).type not in ('cpu', 'meta'):
        raise ValueError('standard fixture initialization requires CPU or meta storage')
    epsilon = config.get('rms_norm_eps')
    rope = config.get('rope_scaling')
    if (not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not math.isfinite(epsilon)
            or epsilon <= 0 or not isinstance(rope, Mapping) or rope.get('type', rope.get('rope_type')) != 'yarn'):
        raise ValueError('standard DSA fixture requires positive RMSNorm epsilon and explicit YaRN settings')
    factor, magnitude = rope.get('factor'), rope.get('mscale_all_dim')
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (factor, magnitude)) or factor < 1 or magnitude < 0):
        raise ValueError('standard DSA fixture requires finite YaRN factor and magnitude')
    scale = 192**-.5*(1+.1*magnitude*math.log(factor))**2
    source_config = SimpleNamespace(**dict(config))
    source_config.dsa_loss_coeff = .3
    source_config.freeze_dsa = False
    source_config.rope_interleaved = True
    source_config.index_rope_interleaved = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        source = nn.Module()
        source.config, source.layer_idx, source.scaling = source_config, 0, scale
        source.q_a_proj = nn.Linear(7168, 1536, bias=False, device=device, dtype=dtype)
        source.kv_a_proj_with_mqa = nn.Linear(7168, 576, bias=False, device=device, dtype=dtype)
        source.q_a_layernorm = nn.RMSNorm(1536, eps=epsilon, device=device, dtype=dtype)
        source.kv_a_layernorm = nn.RMSNorm(512, eps=epsilon, device=device, dtype=dtype)
        source.q_b_proj = nn.Linear(1536, 128*192, bias=False, device=device, dtype=dtype)
        source.kv_b_proj = nn.Linear(512, 128*256, bias=False, device=device, dtype=dtype)
        source.o_proj = nn.Linear(128*128, 7168, bias=False, device=device, dtype=dtype)
        source.indexer = nn.Module()
        source.indexer.wq_b = nn.Linear(1536, 64*128, bias=False, device=device, dtype=dtype)
        source.indexer.wk = nn.Linear(7168, 128, bias=False, device=device, dtype=dtype)
        source.indexer.k_norm = nn.LayerNorm(128, device=device, dtype=dtype)
        source.indexer.weights_proj = nn.Linear(7168, 64, bias=False, device=device, dtype=dtype)
        model = DeepseekV32DSAAttention(module=source)
        with torch.no_grad():
            model.linear_qkv.weight.copy_(torch.cat((source.q_a_proj.weight, source.kv_a_proj_with_mqa.weight)))
        return model


def standard_model_inputs(meta: DsaBatchMeta, *, config: Mapping[str, Any], dtype: torch.dtype,
                          seed: int = 20261259) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Build standard-width hidden states and source YaRN packed-reset frequencies.

    Args:
        meta: Complete packed batch metadata with sequence-relative positions.
        config: Source configuration, including original context, YaRN correction
            bands, extension factor, theta and maximum context.
        dtype: Hidden and cosine/sine representation dtype.
        seed: Local CPU generator seed, independent of global RNG.

    Returns:
        Hidden [1,T,7168] and paired [1,T,64] cosine/sine source frequencies.
        The correction formula follows the pinned source's FP32 frequency table.

    Raises:
        ValueError: The declared source frequency schedule is incomplete or invalid.
    """
    if (meta.q_global_ids != tuple(range(meta.global_valid_queries))
            or config.get('hidden_size') != 7168 or config.get('qk_rope_head_dim') != 64
            or isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
            or dtype not in (torch.bfloat16, torch.float32, torch.float64)):
        raise ValueError('standard inputs require complete queries, standard dimensions and valid precision/seed')
    rope = config.get('rope_scaling', {})
    theta, maximum = config.get('rope_theta'), config.get('max_position_embeddings')
    original, factor = rope.get('original_max_position_embeddings'), rope.get('factor')
    fast, slow = rope.get('beta_fast'), rope.get('beta_slow')
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in (theta, maximum, original, factor, fast, slow))
            or theta <= 1 or factor < 1 or fast < slow
            or not isinstance(original, int) or not isinstance(maximum, int)):
        raise ValueError('standard inputs require complete positive source YaRN frequency settings')
    positions = torch.tensor([meta.sequence_position(token)[1] for token in meta.q_global_ids], dtype=torch.int64)
    if positions.numel() and int(positions.max()) >= maximum:
        raise ValueError('standard input positions exceed source maximum context')
    inverse = 1/(theta**(torch.arange(0, 64, 2, dtype=torch.float32)/64))
    if maximum > original:
        low = max(math.floor(64*math.log(original/(fast*2*math.pi))/(2*math.log(theta))), 0)
        high = min(math.ceil(64*math.log(original/(slow*2*math.pi))/(2*math.log(theta))), 63)
        ramp = ((torch.arange(32, dtype=torch.float32)-low)/(high-low if high != low else .001)).clamp(0, 1)
        smooth = 1-ramp
        inverse = inverse/factor*(1-smooth)+inverse*smooth
    phases = torch.outer(positions, inverse)
    phases = torch.cat((phases, phases), -1)[None]
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, meta.global_valid_queries, 7168, generator=generator).to(dtype)
    return hidden, (phases.cos().to(dtype), phases.sin().to(dtype))
