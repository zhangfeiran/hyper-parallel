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
"""常规 BF16 FFN 的独立 SiLU 和乘法舍入边界；设备绑定由后续阶段提供。"""

from __future__ import annotations

import inspect

import torch
from torch.nn import functional

from hyper_parallel.core.multicore.language.types import DType, TensorType
from hyper_parallel.core.multicore.primitives.registry import REGISTRY, OpSchema


def _matrix_type(value):
    if not isinstance(value, TensorType) or len(value.shape) != 2 or value.dtype != DType.BF16:
        raise ValueError("Common dense activation requires a two-dimensional BF16 tensor")
    return value


def _silu_types(arguments):
    return (_matrix_type(arguments["value"]),)


def _multiply_types(arguments):
    left, right = (_matrix_type(arguments[name]) for name in ("left", "right"))
    if left != right:
        raise ValueError("Common dense multiplication requires identical shape, dtype and layout without broadcasting")
    return (left,)


def _silu_reference(value: torch.Tensor) -> torch.Tensor:
    return functional.silu(value)


def _multiply_reference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left * right


silu_dense = REGISTRY.register(OpSchema("dense.silu", 1, inspect.signature(_silu_reference),
                                       _silu_types, _silu_reference))
multiply_dense = REGISTRY.register(OpSchema("dense.multiply", 1, inspect.signature(_multiply_reference),
                                           _multiply_types, _multiply_reference))
