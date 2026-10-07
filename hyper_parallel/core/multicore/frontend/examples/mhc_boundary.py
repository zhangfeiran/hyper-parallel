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
"""Typed shifted MHC boundary; mapping predicts next mixes while InputMix uses previous pre."""

from __future__ import annotations

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml

RESIDUAL_SHAPE = ("T", 4, "H")
OUTPUT_SHAPE = ("T", "H")
MIX_SHAPE = ("T", 4)
MATRIX_SHAPE = ("T", 4, 4)
PHI_SHAPE = (24, "ExpandedH")
WEIGHT_SHAPE = ("H",)


@mc.program(target="ascend", schedule=mc.TaskDAG(policy="shifted_mhc_v1"))
def mhc_boundary(
    residual: ml.Tensor[ml.bf16, RESIDUAL_SHAPE],
    previous_output: ml.Tensor[ml.bf16, OUTPUT_SHAPE],
    previous_pre: ml.Tensor[ml.fp32, MIX_SHAPE],
    previous_post: ml.Tensor[ml.fp32, MIX_SHAPE],
    previous_residual: ml.Tensor[ml.fp32, MATRIX_SHAPE],
    phi: ml.Tensor[ml.fp32, PHI_SHAPE],
    alpha: ml.Tensor[ml.fp32, (3,)],
    bias: ml.Tensor[ml.fp32, (24,)],
    norm_weight: ml.Tensor[ml.bf16, WEIGHT_SHAPE],
    hc_eps: ml.Constexpr[float] = 1e-6,
    norm_eps: ml.Constexpr[float] = 1e-6,
    num_iters: ml.Constexpr[int] = 20,
) -> (ml.Tensor[ml.bf16, RESIDUAL_SHAPE], ml.Tensor[ml.fp32, MIX_SHAPE],
           ml.Tensor[ml.fp32, MIX_SHAPE], ml.Tensor[ml.fp32, MATRIX_SHAPE], ml.Tensor[ml.bf16, OUTPUT_SHAPE]):
    """Advance a four-stream shifted boundary using the native numerical contract.

    Args:
        residual: Four BF16 residual streams.
        previous_output: Previous block's BF16 output.
        previous_pre: Previous layer's input-mix coefficients.
        previous_post: Previous layer's output-mix coefficients.
        previous_residual: Previous layer's residual mixing matrix.
        phi: FP32 projection of 4H residual features into 24 coefficients.
        alpha: Three FP32 mapping scales.
        bias: Twenty-four FP32 mapping biases.
        norm_weight: BF16 input RMSNorm weights.
        hc_eps: Sinkhorn stability epsilon.
        norm_eps: NormCast and input RMSNorm epsilon.
        num_iters: Native Sinkhorn iteration count.
    """
    updated = ml.mhc_post(residual, previous_output, previous_post, previous_residual)
    next_pre, next_post, next_residual = ml.mhc_mapping(
        updated, phi, alpha, bias, hc_eps=hc_eps, norm_eps=norm_eps, num_iters=num_iters
    )
    mixed = ml.mhc_input_mix(updated, previous_pre)
    block_input = ml.rms_norm(mixed, norm_weight, eps=norm_eps)
    return updated, next_pre, next_post, next_residual, block_input
