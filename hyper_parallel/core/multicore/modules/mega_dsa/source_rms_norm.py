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
"""Source-rounded RMSNorm values with FP32 activation and pre-leaf scale VJPs."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional


class _SourceRmsNorm(torch.autograd.Function):
    """Retain source operand versions and compute the analytic first-order VJP."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight: torch.Tensor,
                original_weight: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Represent the exact source forward in the compute dtype."""
        ctx.set_materialize_grads(False)
        source = inputs.to(original_weight.dtype)
        ctx.save_for_backward(source, weight, original_weight)
        ctx.epsilon = epsilon
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return functional.rms_norm(source, original_weight.shape,
                                       weight.to(original_weight.dtype), epsilon).to(weight.dtype)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor | None) -> tuple:
        """Evaluate direct-scale RMSNorm derivatives without an intermediate activation cast."""
        if torch.is_grad_enabled():
            raise ValueError('source RMSNorm supports first-order backward only')
        if gradient is None:
            return None, None, None, None
        source, weight, _original_weight = ctx.saved_tensors
        with torch.autocast(device_type=source.device.type, enabled=False):
            values = source.to(weight.dtype)
            inverse = torch.rsqrt(values.square().mean(-1, keepdim=True)+ctx.epsilon)
            product = gradient.to(weight.dtype)*weight
            grad_inputs = (inverse*product-values*inverse.pow(3)*(product*values).mean(-1, keepdim=True)
                           if ctx.needs_input_grad[0] else None)
            grad_weight = ((gradient.to(weight.dtype)*values*inverse).reshape(-1, weight.numel()).sum(0)
                           if ctx.needs_input_grad[1] else None)
        return grad_inputs, grad_weight, None, None


def source_rms_norm_with_fp32_vjp(inputs: torch.Tensor, weight: torch.Tensor,
                                  original_norm: nn.RMSNorm) -> torch.Tensor:
    """Preserve a direct-scale source RMSNorm forward while carrying compute-dtype VJPs.

    Args:
        inputs: FP32/FP64 activations holding source-rounded values, with the
            last dimension equal to the original normalization width.
        weight: Differentiable compute-dtype view of the original scale Parameter.
        original_norm: Exact one-dimensional nn.RMSNorm with affine scale and
            explicit positive epsilon. Its Parameter identity/version is retained.

    Returns:
        Exact source-rounded values represented in the compute dtype. Input and
        weight derivatives use the independent direct-scale RMSNorm formula in
        that dtype. This is a declared first-order mixed-precision derivative;
        it does not claim equivalence to the source low-precision backward.

    Raises:
        ValueError: The original norm, precision, geometry or epsilon is unsupported.
    """
    # A subclass may implement a different primal; the analytic derivative is specific to nn.RMSNorm.
    if (type(original_norm) is not nn.RMSNorm  # pylint: disable=unidiomatic-typecheck
            or len(original_norm.normalized_shape) != 1
            or original_norm.weight is None or original_norm.eps is None):
        raise ValueError('source RMSNorm requires exact affine one-dimensional nn.RMSNorm with explicit epsilon')
    original = original_norm.weight
    dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
    if (original.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or inputs.dtype != dtype or weight.dtype != dtype or inputs.ndim < 1
            or inputs.shape[-1] != original.numel() or original.numel() <= 0
            or weight.shape != original.shape or inputs.device != original.device or weight.device != original.device
            or any(value.layout != torch.strided for value in (inputs, weight, original))
            or isinstance(original_norm.eps, bool) or not math.isfinite(original_norm.eps) or original_norm.eps <= 0):
        raise ValueError('source RMSNorm requires dense compute-dtype views, matching geometry and positive epsilon')
    return _SourceRmsNorm.apply(inputs, weight, original, original_norm.eps)
