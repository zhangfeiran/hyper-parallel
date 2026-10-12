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
"""Source-dtype linear values with weight VJPs returned before a low-precision leaf cast."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional


class _SourceLinear(torch.autograd.Function):
    """Keep source operands and saved Parameter versions while returning a full-precision weight VJP."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight: torch.Tensor, original_weight: torch.Tensor,
                bias: torch.Tensor | None, original_bias: torch.Tensor | None,
                full_precision_output: bool) -> torch.Tensor:
        """Evaluate the original activation/Parameter dtype linear, outside autocast."""
        ctx.set_materialize_grads(False)
        forward_inputs = inputs.to(original_weight.dtype)
        ctx.save_for_backward(inputs, forward_inputs, weight, original_weight, original_bias)
        ctx.has_bias = bias is not None
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            output = functional.linear(  # pylint: disable=not-callable
                forward_inputs, weight.to(original_weight.dtype),
                None if bias is None else bias.to(original_weight.dtype))
            return output.to(weight.dtype) if full_precision_output else output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor | None) -> tuple:
        """Compute VJPs in full precision and defer the weight/bias cast to their external views."""
        if torch.is_grad_enabled():
            raise ValueError('source linear supports first-order backward only')
        if gradient is None:
            return None, None, None, None, None, None
        inputs, forward_inputs, weight, _original_weight, _original_bias = ctx.saved_tensors
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            upstream = gradient.to(weight.dtype).reshape(-1, weight.shape[0])
            flattened = forward_inputs.to(weight.dtype).reshape(-1, weight.shape[1])
            grad_inputs = ((upstream@weight).reshape(inputs.shape).to(inputs.dtype)
                           if ctx.needs_input_grad[0] else None)
            grad_weight = (upstream.T@flattened if ctx.needs_input_grad[1] else None)
            grad_bias = (upstream.sum(0) if ctx.has_bias and ctx.needs_input_grad[3] else None)
        return grad_inputs, grad_weight, None, grad_bias, None, None


def source_linear_with_full_precision_weight_vjp(inputs: torch.Tensor, weight: torch.Tensor,
                                                 original_weight: torch.Tensor, bias: torch.Tensor | None = None,
                                                 original_bias: torch.Tensor | None = None, *,
                                                 full_precision_inputs: bool = False,
                                                 full_precision_output: bool = False) -> torch.Tensor:
    """Preserve source linear values and return weight/bias VJPs to full-precision capture views.

    Args:
        inputs: Original source-dtype activations, including empty token rows.
        weight: Differentiable full-precision view of the original weight.
        original_weight: Original Parameter retained for autograd version checks.
        bias: Optional full-precision bias view.
        original_bias: Original bias Parameter retained for version checks.
        full_precision_inputs: Accept compute-dtype inputs and return their VJP
            before a caller's TP SUM. Forward still consumes source-dtype inputs.
        full_precision_output: Represent the original rounded forward values in
            the compute dtype, allowing an unrounded downstream cotangent.

    Returns:
        Source-rounded projection values, optionally represented in compute dtype.
        Input VJP returns in the input dtype;
        weight/bias VJPs reach their view hooks before the original leaf cast.

    Raises:
        ValueError: Shapes, dtype or device do not match the declared source
            projection. Second-order backward is unsupported.
    """
    dtype = torch.float64 if original_weight.dtype == torch.float64 else torch.float32
    if any(not isinstance(flag, bool) for flag in (full_precision_inputs, full_precision_output)):
        raise TypeError('full_precision_inputs and full_precision_output must be booleans')
    input_dtype = dtype if full_precision_inputs else original_weight.dtype
    if (original_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or original_weight.ndim != 2 or 0 in original_weight.shape or weight.shape != original_weight.shape
            or weight.dtype != dtype or inputs.dtype != input_dtype or inputs.ndim < 1
            or inputs.shape[-1] != weight.shape[1] or inputs.device != weight.device
            or original_weight.device != weight.device
            or any(value.layout != torch.strided for value in (inputs, weight, original_weight))):
        raise ValueError('source linear requires matching dense source inputs and full-precision parameter views')
    if (bias is None) != (original_bias is None):
        raise ValueError('source linear requires both the original bias and its view')
    if bias is not None and (bias.shape != (weight.shape[0],) or original_bias.shape != bias.shape
                             or bias.dtype != dtype or original_bias.dtype != original_weight.dtype
                             or bias.device != weight.device or original_bias.device != weight.device
                             or bias.layout != torch.strided or original_bias.layout != torch.strided):
        raise ValueError('source linear bias view must match its original shape, dtype and device')
    return _SourceLinear.apply(inputs, weight, original_weight, bias, original_bias, full_precision_output)
