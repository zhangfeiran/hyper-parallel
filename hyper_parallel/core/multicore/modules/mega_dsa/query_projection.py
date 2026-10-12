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
"""Preserve the source Q-B forward while retaining FP32 local input VJPs for TP SUM."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional


class _QueryProjection(torch.autograd.Function):
    """Original activation-dtype linear forward with first-order full-precision local VJPs."""

    @staticmethod
    def forward(ctx: Any, latent: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None) -> torch.Tensor:
        """Save original weight versions and keep the source forward operands and output dtype."""
        ctx.set_materialize_grads(False)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = None if bias is None else bias.dtype
        with torch.autocast(device_type=latent.device.type, enabled=False):
            inputs = latent.to(weight.dtype)
            ctx.save_for_backward(latent, weight, inputs)
            return functional.linear(inputs, weight, bias)  # pylint: disable=not-callable

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor | None) -> tuple:
        """Return the input VJP before its TP SUM; weight/bias VJPs keep their leaf dtypes."""
        if torch.is_grad_enabled():
            raise ValueError('query projection supports first-order backward only')
        if gradient is None:
            return None, None, None
        latent, weight, forward_inputs = ctx.saved_tensors
        dtype = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
        with torch.autocast(device_type=latent.device.type, enabled=False):
            upstream = gradient.to(dtype).reshape(-1, weight.shape[0])
            inputs = forward_inputs.to(dtype).reshape(-1, weight.shape[1])
            grad_latent, grad_weight, grad_bias = None, None, None
            if ctx.needs_input_grad[0]:
                grad_latent = (torch.zeros_like(latent) if not inputs.shape[0]
                               else (upstream@weight.to(dtype)).reshape(latent.shape).to(latent.dtype))
            if ctx.needs_input_grad[1]:
                grad_weight = (torch.zeros_like(weight) if not inputs.shape[0]
                               else (upstream.transpose(0, 1)@inputs).to(weight.dtype))
            if ctx.has_bias and ctx.needs_input_grad[2]:
                grad_bias = upstream.sum(0).to(ctx.bias_dtype)
        return grad_latent, grad_weight, grad_bias


def project_query_with_fp32_vjp(latent: torch.Tensor, weight: torch.Tensor,
                                bias: torch.Tensor | None = None) -> torch.Tensor:
    """Compute source-dtype Q-B values and an unrounded local input VJP.

    Args:
        latent: Output of the TP input-copy boundary. Low-precision weights
            require an FP32 latent so backward reaches TP SUM before casting.
        weight: Original local-head Parameter, with its version tracking intact.
        bias: Optional original local-head bias in the weight dtype.

    Returns:
        Source-weight-dtype projection values. The caller retains a cast from
        the normalized latent to FP32 before its TP input-copy boundary.

    Raises:
        ValueError: Operand shapes, devices or dtypes do not describe a dense
            real projection. Second-order backward is unsupported.
    """
    expected = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
    if (weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or weight.layout != torch.strided or latent.layout != torch.strided
            or weight.ndim != 2 or 0 in weight.shape or latent.ndim < 1 or latent.shape[-1] != weight.shape[1]
            or not weight.is_floating_point() or latent.dtype != expected or latent.device != weight.device):
        raise ValueError('query projection requires a matching dense real weight and full-precision latent')
    if bias is not None and (bias.layout != torch.strided or bias.shape != (weight.shape[0],)
                             or bias.dtype != weight.dtype
                             or bias.device != weight.device):
        raise ValueError('query projection bias must match the local weight output, dtype and device')
    return _QueryProjection.apply(latent, weight, bias)
