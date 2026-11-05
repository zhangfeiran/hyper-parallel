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
"""Manifold hyper-connection post-processing functions."""

from functools import lru_cache
from typing import Any, Tuple

import torch  # pylint: disable=forbidden-backend-import

try:
    import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import
except ImportError:
    omni_training_custom_ops = None


@lru_cache(maxsize=None)
def _uses_native_mhc(device: torch.device) -> bool:
    """Select A5 built-ins without querying or initializing accelerators on CPU paths."""
    if device.type != "npu":
        return False
    name = torch.npu.get_device_name(device)
    return "Ascend950" in name or "Ascend910_95" in name


class _MhcPost(torch.autograd.Function):
    """Autograd bridge for the NPU MHC post custom operator."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        """Run the NPU MHC post forward operator.

        Args:
            ctx: Autograd context retaining operands and the selected provider.
            x: Residual streams with shape [batch, sequence, streams, hidden].
            h_res: Source-to-destination stream mixing coefficients.
            h_out: Output of the wrapped attention or expert block.
            h_post: Per-stream coefficients applied to the block output.

        Returns:
            Mixed residual streams with the same shape as x.
        """
        ctx.use_native_mhc = _uses_native_mhc(x.device)
        if not ctx.use_native_mhc and omni_training_custom_ops is None:
            raise ImportError("MHC AscendC requires omni_training_custom_ops")
        ctx.save_for_backward(x, h_res, h_out, h_post)
        if ctx.use_native_mhc:
            return torch.ops.npu.npu_mhc_post(x, h_res, h_out, h_post)
        return torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post(
            x,
            h_res,
            h_out,
            h_post,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the NPU MHC post backward operator.

        Args:
            ctx: Forward context containing saved operands and provider identity.
            grad_output: Upstream gradient for the mixed residual streams.

        Returns:
            Gradients for x, h_res, h_out and h_post in forward argument order.
        """
        x, h_res, h_out, h_post = ctx.saved_tensors
        operation = (torch.ops.npu.npu_mhc_post_backward if ctx.use_native_mhc
                     else torch.ops.custom.npu_ai_infra_mhc_post_grad)
        grad_x, grad_h_res, grad_h_out, grad_h_post = operation(
            grad_output.contiguous(),
            x,
            h_res,
            h_out,
            h_post,
        )
        return grad_x, grad_h_res, grad_h_out, grad_h_post


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    num_stream: int,
) -> torch.Tensor:
    """Combine transformed and residual streams after an MHC-wrapped block.

    Args:
        x: Output of the wrapped block.
        residual: Flattened residual streams.
        h_post: Per-stream output mixing coefficients.
        h_res: Residual stream mixing matrix.
        num_stream: Number of residual streams.

    Returns:
        Flattened mixed residual streams.
    """
    x_shape = x.size()
    residual = residual.reshape(x_shape[0], x_shape[1], num_stream, -1)
    if x.device.type == "npu" and (_uses_native_mhc(x.device) or omni_training_custom_ops is not None):
        return _MhcPost.apply(residual, h_res, x, h_post).flatten(2)
    output = h_post.unsqueeze(-1) * x.unsqueeze(-2)
    output = output + torch.sum(
        h_res.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return output.flatten(2).to(x.dtype)
