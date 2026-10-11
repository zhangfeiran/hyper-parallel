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
"""Conservative, shape-matched admission of optional expert replicas."""

from functools import lru_cache
import math
from typing import Optional

import torch

_BASIS_POINTS = 10000


def reduction_basis_points(value: float) -> int:
    """Round an admission threshold upwards for identical integer CPU/device decisions.

    Args:
        value: Required reduction of the maximum received rows, between zero and one.

    Returns:
        Integer basis points; zero preserves the original placement policy.
    """
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value <= 1 or not math.isfinite(value)):
        raise ValueError("replica_minimum_load_reduction must be finite and between zero and one")
    return math.ceil(value * _BASIS_POINTS)


@lru_cache(maxsize=64)
def default_admission_profile(backend: str, transport: str, tokens: int, hidden: int,
                                   intermediate: int, experts: int, ranks: int, top_k: int,
                                   budget: int, device: torch.device) -> tuple[float, int]:
    """Select only an admission range supported by complete-step measurements.

    Args:
        backend: Native, push or pull execution backend.
        transport: Expert weight and gradient transport.
        tokens: Local token count.
        hidden: Expert hidden dimension.
        intermediate: SwiGLU intermediate dimension.
        experts: Logical expert count.
        ranks: Expert parallel ranks.
        top_k: Distinct experts selected per token.
        budget: Requested guest slots per rank.
        device: Execution device, used only for the architecture match.

    Returns:
        Minimum fractional and absolute peak-load reductions. A fraction of one
        retains only capacity-required replicas in unmatched workloads.

    Note:
        This is a conservative admission heuristic, not a latency guarantee.
        The transport-specific boundaries cover BF16 home versus FP32 replica
        execution; they do not extrapolate to other shapes or architectures.
    """
    if (backend != "push" or device.type != "npu"
            or (hidden, intermediate, experts, ranks, top_k, budget) != (5120, 1792, 24, 4, 8, 1)
            or tokens not in (2048, 4096)):
        return 1.0, 0
    if not torch.npu.get_device_name(device).startswith("Ascend910B"):
        return 1.0, 0
    if transport == "shmem_signal_kernel_gradient":
        return 0.4, tokens * top_k * (3 if tokens == 4096 else 4) // 4
    return (0.4, tokens * top_k) if transport == "p2p" and tokens == 4096 else (1.0, 0)


def resolve_replica_admission(value: Optional[float], **execution: object) -> tuple[float, int]:
    """Use an explicit threshold or the conservative matching default.

    Args:
        value: Explicit threshold, or None for the measured admission ranges.
        **execution: Execution identity accepted by default_admission_profile.

    Returns:
        Required fractional and absolute reductions of the peak receive load.
        An explicit fractional threshold replaces the automatic profile entirely.
    """
    if value is not None:
        reduction_basis_points(value)
        return value, 0
    return default_admission_profile(**execution)
