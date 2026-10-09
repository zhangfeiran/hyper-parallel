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
"""Dense SwiGLU FFN with ordinary tensors and no expert/route metadata."""

from __future__ import annotations

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml

TOKEN_SHAPE = ("T", "H")
PACKED_WEIGHT = ("H", "PackedI")
DOWN_WEIGHT = ("I", "H")


@mc.program(target="ascend", schedule=mc.TaskDAG(policy="dense_v1"))
def dense_ffn(
    x: ml.Tensor[ml.bf16, TOKEN_SHAPE],
    gate_up: ml.Tensor[ml.bf16, PACKED_WEIGHT],
    down: ml.Tensor[ml.bf16, DOWN_WEIGHT],
    intermediate_size: ml.Constexpr[int],
) -> ml.Tensor[ml.bf16, TOKEN_SHAPE]:
    """Apply a bias-free packed FFN while retaining explicit activation rounding.

    Args:
        x: Flattened tokens after the caller's attention/normalization.
        gate_up: Gate/up columns packed in one caller-owned matrix.
        down: Caller-owned output projection matrix.
        intermediate_size: Packed activation's logical intermediate width.
    """
    packed = ml.matmul(x, gate_up)
    activation = ml.swiglu_dense(packed, intermediate_size=intermediate_size)
    return ml.matmul(activation, down)
