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
"""Native MoE region; router, permutation and replica decisions remain in the module bridge."""

from __future__ import annotations

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml

ROUTED_SHAPE = ("Rows", "H")
UP_SHAPE = ("H", "PackedI")
DOWN_SHAPE = ("I", "H")


@mc.program(target="ascend", schedule=mc.TaskDAG(policy="moe_ratr_v1"))
def moe_region(
    routed_x: ml.RaggedTensor[ml.bf16, ROUTED_SHAPE],
    w13: ml.TensorList[ml.bf16, UP_SHAPE],
    w2: ml.TensorList[ml.bf16, DOWN_SHAPE],
    route_meta: ml.RouteMetadata,
    limit: ml.Constexpr[float | None] = None,
) -> ml.RaggedTensor[ml.bf16, ROUTED_SHAPE]:
    """Preserve the original numerical order and transport boundary.

    Args:
        routed_x: Router-expanded BF16 rows; effective counts stay dynamic.
        w13: Expert matrices with packed gate/up columns.
        w2: Expert down-projection matrices.
        route_meta: Runtime dispatch/combine metadata owned by the module.
        limit: Optional native packed SwiGLU clamp specialization.

    Returns:
        Combined router-expanded rows for the existing weighted unpermutation.
    """
    received, groups = ml.dispatch(routed_x, route_meta)
    packed = ml.grouped_matmul(received, w13, groups)
    activated = ml.swiglu_packed(packed, layout="gate_up", limit=limit)
    projected = ml.grouped_matmul(activated, w2, groups)
    return ml.combine(projected, route_meta)
