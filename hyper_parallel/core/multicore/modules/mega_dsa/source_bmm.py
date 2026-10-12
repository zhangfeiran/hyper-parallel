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
"""Source-dtype batched products with full-precision shared weight VJPs."""

from __future__ import annotations

from typing import Any

import torch


class _SourceBmm(torch.autograd.Function):
    """Preserve the batched source forward and defer parameter-view VJP rounding."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight: torch.Tensor,
                original_parameter: torch.Tensor, full_precision_output: bool) -> torch.Tensor:
        """Retain actual operands and the original shared Parameter version."""
        ctx.set_materialize_grads(False)
        forward_inputs = inputs.to(original_parameter.dtype)
        ctx.save_for_backward(inputs, forward_inputs, weight, original_parameter)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            output = torch.bmm(forward_inputs, weight.to(original_parameter.dtype))
            return output.to(weight.dtype) if full_precision_output else output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor | None) -> tuple:
        """Return a source-dtype activation VJP and an unrounded parameter-view VJP."""
        if torch.is_grad_enabled():
            raise ValueError('source bmm supports first-order backward only')
        if gradient is None:
            return None, None, None, None
        inputs, forward_inputs, weight, _original_parameter = ctx.saved_tensors
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            upstream = gradient.to(weight.dtype)
            grad_inputs = (torch.bmm(upstream, weight.transpose(1, 2)).to(inputs.dtype)
                           if ctx.needs_input_grad[0] else None)
            grad_weight = (torch.bmm(forward_inputs.to(weight.dtype).transpose(1, 2), upstream)
                           if ctx.needs_input_grad[1] else None)
        return grad_inputs, grad_weight, None, None


def source_bmm_with_full_precision_weight_vjp(inputs: torch.Tensor, weight: torch.Tensor,
                                              original_parameter: torch.Tensor, *,
                                              full_precision_inputs: bool = False,
                                              full_precision_output: bool = False) -> torch.Tensor:
    """Evaluate source-dtype batched products while combining weight-view VJPs before casting.

    Args:
        inputs: Dense source-dtype [heads, tokens, inner] activations.
        weight: Full-precision [heads, inner, output] view. Shared key/value
            views must derive from the same captured Parameter view so their
            contributions meet in full precision before its hook and leaf cast.
        original_parameter: Original source Parameter retained for version checks.
        full_precision_inputs: Accept compute-dtype inputs holding source-rounded
            values and return their VJP before a caller's downstream cast.
        full_precision_output: Represent source-rounded products in compute dtype
            so downstream VJPs reach this product before activation rounding.

    Returns:
        Source-rounded output, optionally in compute dtype, with first-order VJPs.

    Raises:
        ValueError: Dense geometry, compute precision or devices do not match.
            Second-order backward is unsupported.
    """
    dtype = torch.float64 if original_parameter.dtype == torch.float64 else torch.float32
    if any(not isinstance(flag, bool) for flag in (full_precision_inputs, full_precision_output)):
        raise TypeError('full_precision_inputs and full_precision_output must be booleans')
    input_dtype = dtype if full_precision_inputs else original_parameter.dtype
    if (original_parameter.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or inputs.ndim != 3 or weight.ndim != 3 or inputs.shape[0] != weight.shape[0]
            or inputs.shape[2] != weight.shape[1] or weight.shape[1] < 1 or weight.shape[2] < 1
            or inputs.dtype != input_dtype or weight.dtype != dtype
            or inputs.device != weight.device or weight.device != original_parameter.device
            or any(value.layout != torch.strided for value in (inputs, weight, original_parameter))):
        raise ValueError('source bmm requires dense matching source activations and full-precision weight views')
    return _SourceBmm.apply(inputs, weight, original_parameter, full_precision_output)
