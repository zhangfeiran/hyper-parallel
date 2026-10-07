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
"""Typed DSL annotations and registered semantic primitives."""

from __future__ import annotations

from hyper_parallel.core.multicore.language.types import (
    Constexpr,
    DType,
    Tensor,
    TensorType,
)
from hyper_parallel.core.multicore.primitives.gate import (
    add,
    cast,
    divide,
    gather,
    multiply,
    reduce_sum,
    softplus,
    sqrt,
    stop_gradient,
    topk_indices,
)

bf16 = DType.BF16
fp16 = DType.FP16
fp32 = DType.FP32
int32 = DType.INT32
int64 = DType.INT64
boolean = DType.BOOL

__all__ = [
    "Constexpr",
    "DType",
    "Tensor",
    "TensorType",
    "add",
    "bf16",
    "boolean",
    "cast",
    "divide",
    "fp16",
    "fp32",
    "gather",
    "int32",
    "int64",
    "multiply",
    "reduce_sum",
    "softplus",
    "sqrt",
    "stop_gradient",
    "topk_indices",
]
