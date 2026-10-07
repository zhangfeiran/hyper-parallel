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
"""Gate semantic primitives with Torch CPU references, without native claims."""

from __future__ import annotations

import inspect
from numbers import Real

import torch
from torch.nn import functional

from hyper_parallel.core.multicore.language.types import DType, TensorType
from hyper_parallel.core.multicore.primitives.registry import REGISTRY, OpSchema

TORCH_DTYPES = {
    DType.BF16: torch.bfloat16,
    DType.FP16: torch.float16,
    DType.FP32: torch.float32,
    DType.INT32: torch.int32,
    DType.INT64: torch.int64,
    DType.BOOL: torch.bool,
}


def _tensor(value):
    if not isinstance(value, TensorType):
        raise TypeError("Expected a tensor argument")
    return value


def _floating(value):
    tensor = _tensor(value)
    if tensor.dtype not in (DType.BF16, DType.FP16, DType.FP32):
        raise ValueError("Expected a floating tensor")
    return tensor


def _unary_types(arguments):
    return (_floating(arguments["value"]),)


def _binary_types(arguments):
    left = _floating(arguments["left"])
    right = arguments["right"]
    if not isinstance(right, TensorType):
        if not isinstance(right, Real) or isinstance(right, bool):
            raise TypeError("Expected a tensor or numeric scalar")
        return (left,)
    if left.dtype != right.dtype:
        raise ValueError("Binary tensor dtypes must match")
    shape = []
    for index in range(1, max(len(left.shape), len(right.shape)) + 1):
        lhs = left.shape[-index] if index <= len(left.shape) else 1
        rhs = right.shape[-index] if index <= len(right.shape) else 1
        if lhs != rhs and lhs != 1 and rhs != 1:
            raise ValueError(f"Cannot prove broadcast compatibility of {lhs!r} and {rhs!r}")
        shape.append(rhs if lhs == 1 else lhs)
    return (TensorType(left.dtype, tuple(reversed(shape))),)


def _axis(arguments, tensor):
    axis = arguments["axis"]
    if type(axis) not in (int,) or not -len(tensor.shape) <= axis < len(tensor.shape):
        raise ValueError("axis must be a static integer within tensor rank")
    return axis % len(tensor.shape)


def _topk_types(arguments):
    tensor = _floating(arguments["value"])
    axis = _axis(arguments, tensor)
    count = arguments["k"]
    if type(count) not in (int,) or count <= 0 or type(arguments["sorted"]) not in (bool,):
        raise ValueError("topk requires positive static k and boolean sorted")
    if isinstance(tensor.shape[axis], int) and count > tensor.shape[axis]:
        raise ValueError("topk k exceeds selection dimension")
    shape = list(tensor.shape)
    shape[axis] = count
    return (TensorType(DType.INT64, tuple(shape)),)


def _gather_types(arguments):
    tensor = _tensor(arguments["value"])
    indices = _tensor(arguments["indices"])
    axis = _axis(arguments, tensor)
    if indices.dtype != DType.INT64 or len(indices.shape) != len(tensor.shape):
        raise ValueError("gather requires int64 indices with matching rank")
    for index, (source, selected) in enumerate(zip(tensor.shape, indices.shape)):
        if index != axis and source != selected:
            raise ValueError("V0 gather requires equal non-selection dimensions")
    return (TensorType(tensor.dtype, indices.shape),)


def _sum_types(arguments):
    tensor = _floating(arguments["value"])
    axis = _axis(arguments, tensor)
    if type(arguments["keepdim"]) not in (bool,):
        raise ValueError("keepdim must be a static boolean")
    shape = list(tensor.shape)
    if arguments["keepdim"]:
        shape[axis] = 1
    else:
        shape.pop(axis)
    return (TensorType(tensor.dtype, tuple(shape)),)


def _cast_types(arguments):
    tensor = _tensor(arguments["value"])
    dtype = arguments["dtype"]
    if not isinstance(dtype, DType):
        raise TypeError("cast requires a DSL dtype")
    return (TensorType(dtype, tensor.shape),)


def _topk(value, k, axis=-1, sorted=False):
    return torch.topk(value, k, dim=axis, sorted=sorted).indices


def _gather(value, indices, axis=-1):
    return torch.gather(value, axis, indices)


def _sum(value, axis=-1, keepdim=False):
    return value.sum(dim=axis, keepdim=keepdim)


def _cast(value, dtype):
    return value.to(TORCH_DTYPES[dtype])


def _register(name, reference, inference, parameters):
    signature = inspect.Signature(
        [
            inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
            for key, default in parameters
        ]
    )
    return REGISTRY.register(OpSchema(f"gate.{name}", 1, signature, inference, reference))


_REQUIRED = inspect.Parameter.empty
softplus = _register("softplus", functional.softplus, _unary_types, [("value", _REQUIRED)])
sqrt = _register("sqrt", torch.sqrt, _unary_types, [("value", _REQUIRED)])
stop_gradient = _register(
    "stop_gradient",
    torch.Tensor.detach,
    lambda arguments: (_tensor(arguments["value"]),),
    [("value", _REQUIRED)],
)
add = _register("add", torch.add, _binary_types, [("left", _REQUIRED), ("right", _REQUIRED)])
divide = _register("divide", torch.divide, _binary_types, [("left", _REQUIRED), ("right", _REQUIRED)])
multiply = _register(
    "multiply",
    torch.multiply,
    _binary_types,
    [("left", _REQUIRED), ("right", _REQUIRED)],
)
topk_indices = _register(
    "topk_indices",
    _topk,
    _topk_types,
    [("value", _REQUIRED), ("k", _REQUIRED), ("axis", -1), ("sorted", False)],
)
gather = _register(
    "gather",
    _gather,
    _gather_types,
    [("value", _REQUIRED), ("indices", _REQUIRED), ("axis", -1)],
)
reduce_sum = _register(
    "reduce_sum",
    _sum,
    _sum_types,
    [("value", _REQUIRED), ("axis", -1), ("keepdim", False)],
)
cast = _register("cast", _cast, _cast_types, [("value", _REQUIRED), ("dtype", _REQUIRED)])
