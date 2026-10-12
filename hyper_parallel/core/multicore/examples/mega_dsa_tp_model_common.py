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
"""Deterministic real HP attention projections with Native-supported global and local head geometry."""

from __future__ import annotations

import torch
from torch import nn

from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    build_model_fixture,
)


def tp_model_fixture(*, heads: int = 128, dtype: torch.dtype = torch.bfloat16,
                     projection_bias: bool = False, seed: int | None = None) -> DeepseekV32DSAAttention:
    """Build a random HP model with H64/H128, Hi64 and small hidden/LoRA/value dimensions.

    Hidden32/Q-LoRA64/V16 is a real projection fixture, not a pretrained model
    or the full standard checkpoint dimensions. Bias-enabled fixtures exercise
    TP row-bias slicing and the replicated O bias added after output SUM.
    """
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63-1):
        raise ValueError('model fixture seed must be a nonnegative integer with room for its projection seed')
    model = build_model_fixture(dtype=torch.float32, seed=20261007 if seed is None else seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(733 if seed is None else seed+1)
        model.q_b_proj = nn.Linear(64, heads*192, bias=projection_bias)
        model.kv_b_proj = nn.Linear(512, heads*144, bias=False)
        model.o_proj = nn.Linear(heads*16, 32, bias=projection_bias)
        model.indexer.wq_b = nn.Linear(64, 64*128, bias=projection_bias)
        model.indexer.weights_proj = nn.Linear(32, 64, bias=projection_bias)
    model.num_heads = heads
    model.config.num_attention_heads = heads
    model.num_index_heads = 64
    model.config.index_n_heads = 64
    return model.to(dtype=dtype)
