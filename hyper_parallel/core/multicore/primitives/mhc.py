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
"""Shifted four-stream MHC schemas and differentiable CPU references."""

from __future__ import annotations

import inspect
import math
import struct

import torch

from hyper_parallel.core.multicore.language.types import DType, TensorType
from hyper_parallel.core.multicore.primitives.registry import REGISTRY, OpSchema


def _tensor(value, dtype, rank):
    if not isinstance(value, TensorType) or value.dtype != dtype or len(value.shape) != rank:
        raise ValueError(f"MHC requires a contiguous rank-{rank} {dtype.value} tensor")
    return value


def _same(value, dtype, shape):
    if _tensor(value, dtype, len(shape)).shape != shape:
        raise ValueError(f"MHC shape mismatch: expected {shape}, got {value.shape}")


def _epsilon(value):
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("MHC epsilon must be a positive finite float")
    try:
        encoded = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as error:
        raise ValueError("MHC epsilon must fit FP32") from error
    if not math.isfinite(encoded) or encoded <= 0:
        raise ValueError("MHC epsilon must remain positive and finite in FP32")


def _post_types(arguments):
    residual = _tensor(arguments["residual"], DType.BF16, 3)
    rows, streams, hidden = residual.shape
    if streams != 4:
        raise ValueError("MHC requires four residual streams")
    _same(arguments["previous_output"], DType.BF16, (rows, hidden))
    _same(arguments["previous_post"], DType.FP32, (rows, 4))
    _same(arguments["previous_residual"], DType.FP32, (rows, 4, 4))
    return (residual,)


def _mapping_types(arguments):
    value = _tensor(arguments["updated"], DType.BF16, 3)
    rows, streams, hidden = value.shape
    if streams != 4:
        raise ValueError("MHC mapping requires four residual streams")
    phi = _tensor(arguments["phi"], DType.FP32, 2)
    if phi.shape[0] != 24 or isinstance(hidden, int) and phi.shape[1] != 4 * hidden:
        raise ValueError("MHC phi must have shape [24, 4H]")
    _same(arguments["alpha"], DType.FP32, (3,))
    _same(arguments["bias"], DType.FP32, (24,))
    _epsilon(arguments["hc_eps"])
    _epsilon(arguments["norm_eps"])
    iterations = arguments["num_iters"]
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations != 20:
        raise ValueError("Native MHC requires exactly 20 Sinkhorn iterations")
    return (TensorType(DType.FP32, (rows, 4)), TensorType(DType.FP32, (rows, 4)),
            TensorType(DType.FP32, (rows, 4, 4)))


def _input_mix_types(arguments):
    value = _tensor(arguments["updated"], DType.BF16, 3)
    rows, streams, hidden = value.shape
    if streams != 4:
        raise ValueError("MHC input mix requires four residual streams")
    _same(arguments["previous_pre"], DType.FP32, (rows, 4))
    return (TensorType(DType.BF16, (rows, hidden)),)


def _rms_types(arguments):
    value = _tensor(arguments["value"], DType.BF16, 2)
    _same(arguments["weight"], DType.BF16, (value.shape[-1],))
    _epsilon(arguments["eps"])
    return (value,)


def _post_reference(residual, previous_output, previous_post, previous_residual):
    return (torch.einsum("tji,tjh->tih", previous_residual.float(), residual.float())
            + previous_post.float().unsqueeze(-1) * previous_output.float().unsqueeze(-2)).to(residual.dtype)


def _mapping_reference(updated, phi, alpha, bias, hc_eps=1e-6, norm_eps=1e-6, num_iters=20):
    values = updated.float()
    inv_rms = torch.rsqrt(values.square().mean(dim=(-2, -1), keepdim=False) + norm_eps)
    logits = (values.flatten(1) @ phi.float().T) * inv_rms.unsqueeze(-1)
    pre = torch.sigmoid(logits[:, :4] * alpha[0] + bias[:4]) + hc_eps
    post = 2.0 * torch.sigmoid(logits[:, 4:8] * alpha[1] + bias[4:8])
    matrix = (logits[:, 8:] * alpha[2] + bias[8:]).reshape(-1, 4, 4)
    matrix = torch.softmax(matrix, dim=-1) + hc_eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(1, num_iters):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + hc_eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + hc_eps)
    return pre, post, matrix


def _input_mix_reference(updated, previous_pre):
    return (updated.float() * previous_pre.float().unsqueeze(-1)).sum(-2).to(updated.dtype)


def _rms_reference(value, weight, eps=1e-6):
    data = value.float()
    return (data * torch.rsqrt(data.square().mean(-1, keepdim=True) + eps) * weight.float()).to(value.dtype)


def _register(name, reference, inference):
    return REGISTRY.register(OpSchema("mhc." + name, 1, inspect.signature(reference), inference, reference))


mhc_post = _register("post", _post_reference, _post_types)
mhc_mapping = _register("mapping", _mapping_reference, _mapping_types)
mhc_input_mix = _register("input_mix", _input_mix_reference, _input_mix_types)
rms_norm = _register("rms_norm", _rms_reference, _rms_types)
