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
"""三投影常规 FFN 的语义程序，独立保留 BF16 SiLU 与乘法结果。"""

from __future__ import annotations

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.primitives.dense_common import multiply_dense, silu_dense

TOKEN_SHAPE = ("T", "H")
INPUT_WEIGHT = ("I", "H")
OUTPUT_WEIGHT = ("H", "I")


@mc.program(target="ascend", schedule=mc.TaskDAG(policy="dense_v1"))
def common_ffn(
    x: ml.Tensor[ml.bf16, TOKEN_SHAPE],
    gate: ml.Tensor[ml.bf16, INPUT_WEIGHT],
    up: ml.Tensor[ml.bf16, INPUT_WEIGHT],
    down: ml.Tensor[ml.bf16, OUTPUT_WEIGHT],
) -> ml.Tensor[ml.bf16, TOKEN_SHAPE]:
    """保留三个原始 NK 投影及分步激活，供原始 common 基线验收。

    Args:
        x: 二维 BF16 词元输入。
        gate: 原始门控投影权重。
        up: 原始上投影权重。
        down: 原始下投影权重。
    """
    projected_gate = ml.linear(x, gate)
    projected_up = ml.linear(x, up)
    activated_gate = silu_dense(projected_gate)
    activation = multiply_dense(activated_gate, projected_up)
    return ml.linear(activation, down)
