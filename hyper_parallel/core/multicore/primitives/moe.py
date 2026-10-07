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
"""MoE task-region schemas and local expert-major CPU references."""

from __future__ import annotations

import inspect

import torch

from hyper_parallel.core.multicore.language.types import (
    DType,
    RaggedTensorType,
    RouteMetadataType,
    TensorListType,
    TensorType,
)
from hyper_parallel.core.multicore.language.values import RaggedTensor, RouteMetadata
from hyper_parallel.core.multicore.primitives.registry import REGISTRY, OpSchema
from hyper_parallel.core.multicore.tasks.swiglu import _encode_clamp_limit


def _matrix(value):
    if not isinstance(value, RaggedTensorType) or value.dtype != DType.BF16:
        raise ValueError("MoE compatibility primitives require BF16 ragged matrices")
    return value


def _dispatch_types(arguments):
    value = _matrix(arguments["routed_x"])
    metadata = arguments["route_meta"]
    if not isinstance(metadata, RouteMetadataType):
        raise TypeError("dispatch requires RouteMetadata")
    return RaggedTensorType(value.dtype, ("ReceiveRows", value.shape[1])), TensorType(
        DType.INT64, (metadata.num_experts,),
    )


def _gmm_types(arguments):
    value = _matrix(arguments["value"])
    weights, groups = arguments["weights"], arguments["groups"]
    if not isinstance(weights, TensorListType) or weights.dtype != value.dtype:
        raise ValueError("grouped_matmul requires matching BF16 expert matrices")
    if not isinstance(groups, TensorType) or groups.dtype != DType.INT64 or len(groups.shape) != 1:
        raise ValueError("grouped_matmul groups must be an INT64 cumulative group-list vector")
    columns = None
    if weights.shape is not None:
        contracted, columns = weights.shape
        if isinstance(contracted, int) and isinstance(value.shape[1], int) and contracted != value.shape[1]:
            raise ValueError("grouped_matmul contracted dimensions disagree")
    return (RaggedTensorType(value.dtype, (value.shape[0], columns)),)


def _swiglu_types(arguments):
    value = _matrix(arguments["packed"])
    if arguments["layout"] != "gate_up":
        raise ValueError("MoE native SwiGLU supports only the packed gate_up layout")
    _encode_clamp_limit(arguments["limit"])
    columns = value.shape[1]
    if isinstance(columns, int):
        if columns <= 0 or columns % 2:
            raise ValueError("Packed SwiGLU width must be positive and even")
        columns //= 2
    else:
        columns = None
    return (RaggedTensorType(value.dtype, (value.shape[0], columns)),)


def _combine_types(arguments):
    value = _matrix(arguments["projected"])
    if not isinstance(arguments["route_meta"], RouteMetadataType):
        raise TypeError("combine requires the dispatch RouteMetadata")
    return (RaggedTensorType(value.dtype, ("Rows", value.shape[1])),)


def _groups(metadata, rows):
    if not isinstance(metadata, RouteMetadata) or metadata.group_list.device.type != "cpu":
        raise ValueError("CPU MoE interpretation requires local CPU RouteMetadata")
    groups = metadata.group_list.tolist()
    if any(end < begin for begin, end in zip([0, *groups], groups)) or groups[-1] != rows:
        raise ValueError("Cumulative expert groups must be monotonic and cover exactly the valid rows")
    return groups


def _dispatch_reference(routed_x, route_meta):
    _groups(route_meta, routed_x.valid_rows)
    return routed_x, route_meta.group_list


def _gmm_reference(value, weights, groups):
    matrices = tuple(weights.unbind(0)) if isinstance(weights, torch.Tensor) else tuple(weights)
    ends = _groups(RouteMetadata(groups), value.valid_rows)
    if len(matrices) != len(ends):
        raise ValueError("Expert matrix count must match the runtime group list")
    outputs = [value.storage[begin:end] @ matrix for begin, end, matrix in zip([0, *ends], ends, matrices)]
    active = torch.cat(outputs, dim=0)
    padding = active.new_zeros((value.storage.shape[0] - value.valid_rows, active.shape[1]))
    return RaggedTensor(torch.cat((active, padding)), value.valid_rows)


def _swiglu_reference(packed, layout="gate_up", limit=None):
    if layout != "gate_up":
        raise ValueError("SwiGLU reference requires gate_up")
    gate, up = packed.storage[:packed.valid_rows].float().chunk(2, dim=-1)
    if limit is not None:
        # Native clamp derivatives are zero at the boundaries, as well as outside.
        gate = torch.where(gate < limit, gate, torch.full_like(gate, limit))
        up = torch.where((up > -limit) & (up < limit), up, up.clamp(-limit, limit).detach())
    active = (gate * torch.sigmoid(gate) * up).to(packed.storage.dtype)
    padding = active.new_zeros((packed.storage.shape[0] - packed.valid_rows, active.shape[1]))
    return RaggedTensor(torch.cat((active, padding)), packed.valid_rows)


def _combine_reference(projected, route_meta):
    _groups(route_meta, projected.valid_rows)
    return projected


def _register(name, reference, inference):
    return REGISTRY.register(OpSchema("moe." + name, 1, inspect.signature(reference), inference, reference))


dispatch = _register("dispatch", _dispatch_reference, _dispatch_types)
grouped_matmul = _register("grouped_matmul", _gmm_reference, _gmm_types)
swiglu_packed = _register("swiglu_packed", _swiglu_reference, _swiglu_types)
combine = _register("combine", _combine_reference, _combine_types)
