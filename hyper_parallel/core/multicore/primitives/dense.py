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
"""Dense matrix primitives with explicit storage and activation rounding contracts."""

from __future__ import annotations

import inspect

import torch
from torch.nn import functional

from hyper_parallel.core.multicore.language.types import DType, TensorType
from hyper_parallel.core.multicore.primitives.registry import REGISTRY, OpSchema


def _matrix(value: object) -> TensorType:
    if not isinstance(value, TensorType) or len(value.shape) != 2:
        raise ValueError("Dense matrix primitives require ordinary rank-two Tensor inputs")
    if value.dtype not in (DType.BF16, DType.FP16, DType.FP32):
        raise ValueError("Dense matrix primitives require floating-point storage")
    return value


def _matmul_types(arguments: dict[str, object]) -> tuple[TensorType, ...]:
    left, right = _matrix(arguments["left"]), _matrix(arguments["right"])
    transpose_left, transpose_right = arguments["transpose_left"], arguments["transpose_right"]
    if type(transpose_left) not in (bool,) or type(transpose_right) not in (bool,):
        raise TypeError("Matmul transpose attributes must be static booleans")
    if left.dtype != right.dtype:
        raise ValueError("Dense matmul input storage dtypes must match")
    rows, contracted = left.shape[::-1] if transpose_left else left.shape
    other, columns = right.shape[::-1] if transpose_right else right.shape
    if contracted != other and type(contracted) is type(other):
        raise ValueError("Dense matmul contracted dimensions must match, including symbolic identities")
    return (TensorType(left.dtype, (rows, columns)),)


def _swiglu_types(arguments: dict[str, object]) -> tuple[TensorType, ...]:
    packed = _matrix(arguments["packed"])
    width = arguments["intermediate_size"]
    if type(width) not in (int,) or width <= 0:
        raise ValueError("Dense SwiGLU requires a positive constexpr intermediate_size")
    if arguments["layout"] != "gate_up":
        raise ValueError("Dense SwiGLU supports the packed gate_up layout")
    if isinstance(packed.shape[1], int) and packed.shape[1] != 2 * width:
        raise ValueError("Dense SwiGLU packed width must be exactly twice intermediate_size")
    return (TensorType(packed.dtype, (packed.shape[0], width)),)


def _matmul_reference(left: torch.Tensor, right: torch.Tensor, transpose_left: bool = False,
                      transpose_right: bool = False) -> torch.Tensor:
    return torch.mm(left.t() if transpose_left else left, right.t() if transpose_right else right)


def _swiglu_reference(packed: torch.Tensor, intermediate_size: int, layout: str = "gate_up") -> torch.Tensor:
    if layout != "gate_up" or packed.shape[-1] != 2 * intermediate_size:
        raise ValueError("Dense SwiGLU input does not match its declared packed layout")
    gate, up = packed.float().chunk(2, dim=-1)
    return (functional.silu(gate) * up).to(packed.dtype)


matmul = REGISTRY.register(OpSchema(
    "dense.matmul", 1, inspect.signature(_matmul_reference), _matmul_types, _matmul_reference,
))
swiglu_dense = REGISTRY.register(OpSchema(
    "dense.swiglu", 1, inspect.signature(_swiglu_reference), _swiglu_types, _swiglu_reference,
))
