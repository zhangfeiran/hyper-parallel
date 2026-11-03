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
"""Retain padded HF parameter gradients without clearing active expert matrices."""

from typing import Any

# Native tensor APIs are the model adapter contract.
# pylint: disable=forbidden-backend-import
import torch


class _RealGatePrefix(torch.autograd.Function):
    """Expose real Gate/Up matrices and clear only the unused gradient tail."""

    @staticmethod
    def forward(ctx: Any, weight: torch.Tensor, rows: int) -> torch.Tensor:
        """Return an alias of the real matrices while retaining the complete parameter shape.

        Args:
            ctx: Autograd context retaining the original shape.
            weight: Full packed HF Gate/Up parameter.
            rows: Number of local parameterized experts.
        """
        ctx.shape, ctx.rows = weight.shape, rows
        return weight[:rows]

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Copy active gradients once and initialize only the unused tail.

        Args:
            ctx: Retained original parameter shape and active row count.
            gradient: Complete real-expert gradient matrices.
        """
        output = gradient.new_empty(ctx.shape)
        output[:ctx.rows].copy_(gradient)
        output[ctx.rows:].zero_()
        return output, None


def real_gate_prefix(weight: torch.Tensor, rows: int) -> torch.Tensor:
    """Expose native GMM weights with an optimized, full-shape padded gradient.

    Args:
        weight: Full packed Gate/Up parameter.
        rows: Number of parameterized local experts.

    Returns:
        A storage-sharing real-expert prefix with the same derivative as slicing.
    """
    return weight if weight.shape[0] == rows else _RealGatePrefix.apply(weight, rows)
