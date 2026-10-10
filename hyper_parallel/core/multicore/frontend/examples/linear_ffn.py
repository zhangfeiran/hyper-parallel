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
"""用显式 Linear 数值契约描述打包 FFN，保留原始 NK 权重布局。"""

from __future__ import annotations

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml

TOKEN_SHAPE = ("T", "H")
PACKED_WEIGHT = ("PackedI", "H")
DOWN_WEIGHT = ("H", "I")


@mc.program(target="ascend", schedule=mc.TaskDAG(policy="dense_v1"))
def linear_ffn(
    x: ml.Tensor[ml.bf16, TOKEN_SHAPE],
    gate_up: ml.Tensor[ml.bf16, PACKED_WEIGHT],
    down: ml.Tensor[ml.bf16, DOWN_WEIGHT],
    intermediate_size: ml.Constexpr[int],
) -> ml.Tensor[ml.bf16, TOKEN_SHAPE]:
    """执行打包 Linear/SwiGLU/Linear，将数值来源和物理权重布局分开。

    Args:
        x: 展平后的 token 输入。
        gate_up: 沿输出行拼接、采用 NK 布局的 gate/up 权重。
        down: 按输出×输入维度存储的 NK 输出投影权重。
        intermediate_size: 激活的逻辑通道数。
    """
    packed = ml.linear(x, gate_up)
    activation = ml.swiglu_dense(packed, intermediate_size=intermediate_size)
    return ml.linear(activation, down)
