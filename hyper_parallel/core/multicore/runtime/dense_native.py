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
"""Schema-checked dense activation bindings with an explicit native VJP."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=1)
def verify_dense_dispatch() -> tuple[str, str]:
    """Verify exact forward/backward operator contracts before NPU invocation."""
    schemas = (
        ("npu_swiglu", "npu::npu_swiglu(Tensor input, int dim=-1) -> Tensor"),
        ("npu_swiglu_backward", "npu::npu_swiglu_backward(Tensor grad_output, Tensor input, int dim=-1) -> Tensor"),
    )
    actual = []
    for name, expected in schemas:
        packet = getattr(torch.ops.npu, name, None)
        if packet is None or str(packet.default._schema) != expected:
            raise RuntimeError(f"Dense native dispatcher contract mismatch: {name}")
        actual.append(expected)
    return tuple(actual)


class NativeSwiGLU(torch.autograd.Function):
    """Use the native activation backward while retaining each call's packed input."""

    @staticmethod
    def forward(ctx: Any, packed: torch.Tensor) -> torch.Tensor:
        """Execute the packed native activation on the caller's current stream.

        Args:
            ctx: Invocation-owned autograd state.
            packed: Contiguous BF16/FP16/FP32 packed gate/up matrix on NPU.
        """
        verify_dense_dispatch()
        ctx.save_for_backward(packed)
        return torch.ops.npu.npu_swiglu.default(packed, dim=-1)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        """Return the native packed-input VJP for this exact forward.

        Args:
            ctx: State saved by this invocation.
            gradient: Gradient of the unpacked activation.
        """
        packed, = ctx.saved_tensors
        return (torch.ops.npu.npu_swiglu_backward.default(gradient.contiguous(), packed, dim=-1),)
