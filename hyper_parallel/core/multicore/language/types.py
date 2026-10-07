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
"""Logical tensor types and specialization annotations, independent of native ABI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import get_args


class DType(Enum):
    """Storage dtype; equal byte widths do not imply equal semantics."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"


@dataclass(frozen=True)
class TensorType:
    """Dense logical tensor with static or symbolic dimensions."""

    dtype: DType
    shape: tuple[int | str, ...]
    layout: str = "contiguous"

    def __post_init__(self):
        if not isinstance(self.dtype, DType):
            raise TypeError("Tensor dtype must be a DSL DType")
        if not isinstance(self.shape, tuple) or any(
            type(dim) not in (int, str)
            or (isinstance(dim, int) and dim < 0)
            or (isinstance(dim, str) and not dim.isidentifier())
            for dim in self.shape
        ):
            raise ValueError("Tensor shape must be a tuple of nonnegative integers or symbolic identifiers")
        if self.layout != "contiguous":
            raise ValueError("V0 currently supports only contiguous logical layouts")


@dataclass(frozen=True)
class ConstexprType:
    """A host scalar whose value participates in specialization."""

    scalar_type: object

    def __post_init__(self):
        candidates = get_args(self.scalar_type) if isinstance(self.scalar_type, UnionType) else (self.scalar_type,)
        if not candidates or any(
            not any(item is scalar for scalar in (int, float, bool, str, type(None))) for item in candidates
        ):
            raise ValueError("Constexpr requires scalar types or a union of scalar types")

    def accepts(self, value: object) -> bool:
        """Check exact scalar types, excluding bool from integer parameters.

        Args:
            value: Logical value, scalar or captured symbol to validate.
        """
        candidates = get_args(self.scalar_type) if isinstance(self.scalar_type, UnionType) else (self.scalar_type,)
        return type(value) in candidates


class Tensor:
    """Annotation factory: Tensor[dtype, (dimension, ...)]."""

    def __class_getitem__(cls, arguments: tuple) -> TensorType:
        return TensorType(*arguments)


class Constexpr:
    """Annotation factory: Constexpr[int] or Constexpr[float | None]."""

    def __class_getitem__(cls, scalar_type: object) -> ConstexprType:
        return ConstexprType(scalar_type)
