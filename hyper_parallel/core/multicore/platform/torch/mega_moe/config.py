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
"""Converged private configuration for the Torch MegaMoe backend."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from ._framework import get_torch, get_torch_distributed

_COMMUNICATION_SPLIT = 128
_FUSION_MODES = frozenset({"legacy", "permute", "unpermute", "both"})
_FUSION_MODE_ENV = "HYPER_PARALLEL_MEGA_MOE_FUSION_MODE"
FusionMode = Literal["legacy", "permute", "unpermute", "both"]


@dataclass(frozen=True)
class MegaMoeConfig:
    """Validated static shape and topology used only inside the backend."""

    local_num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    receive_capacity: int
    ep_size: int
    ep_group: Any | None
    rank_id: int
    num_cube_cores: int
    fusion_mode: FusionMode = "legacy"
    dispatch_split: int = _COMMUNICATION_SPLIT
    combine_split: int = _COMMUNICATION_SPLIT
    swiglu_split: int = _COMMUNICATION_SPLIT

    @property
    def local_experts(self) -> int:
        """Return experts owned by this expert-parallel rank."""
        return self.num_experts // self.ep_size

    @property
    def routed_slots(self) -> int:
        """Return routed rows produced by local Top-K expansion."""
        return self.local_num_tokens * self.top_k


def _resolve_receive_capacity(expert_capacity_factor: float, routed_slots: int) -> int:
    """Resolve an aligned receive limit from the public capacity multiplier."""
    requested = math.ceil(routed_slots * expert_capacity_factor)
    return (
        (requested + _COMMUNICATION_SPLIT - 1)
        // _COMMUNICATION_SPLIT
        * _COMMUNICATION_SPLIT
    )


def _resolve_fusion_mode(top_k: int | None = None) -> FusionMode:
    """Resolve and validate the private route-fusion experiment selector."""
    fusion_mode = os.getenv(_FUSION_MODE_ENV, "legacy")
    if fusion_mode not in _FUSION_MODES:
        expected = ", ".join(sorted(_FUSION_MODES))
        raise ValueError(
            f"{_FUSION_MODE_ENV} must be one of {expected}, got {fusion_mode!r}."
        )
    if top_k is not None and fusion_mode != "legacy" and top_k > 16:
        raise ValueError("route-fused MegaMoe currently supports top_k <= 16.")
    return cast(FusionMode, fusion_mode)


def build_config(specification: Mapping[str, Any], tensor: Any) -> MegaMoeConfig:
    """Resolve hardware and process-group values on the first NPU input.

    Args:
        specification: Static resource specification from the public module.
        tensor: First input tensor used to resolve device-specific limits.

    Returns:
        Validated backend configuration bound to the input device.
    """
    torch = get_torch()
    dist = get_torch_distributed()

    device = torch.device(tensor.device)
    if device.type != "npu":
        raise ValueError(f"MegaMoeExperts requires an NPU input, got {device}.")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(
            f"MegaMoeExperts supports BF16 activations, got {tensor.dtype}."
        )

    ep_group = specification["ep_group"]
    ep_size = specification["ep_size"]
    if dist.is_initialized():
        group_size = dist.get_world_size(ep_group)
        rank_id = dist.get_rank(ep_group)
    else:
        if ep_group is not None:
            raise RuntimeError(
                "ep_group cannot be used before torch.distributed initialization."
            )
        group_size = 1
        rank_id = 0
    if group_size != ep_size:
        raise ValueError(
            f"ep_group world size ({group_size}) does not match ep_size ({ep_size})."
        )

    device_index = (
        device.index if device.index is not None else torch.npu.current_device()
    )
    device_limits = torch.npu.get_device_limit(device_index)
    num_cube_cores = device_limits.get("cube_core_num")
    if not isinstance(num_cube_cores, int) or num_cube_cores <= 0:
        raise RuntimeError(
            "torch.npu.get_device_limit() returned an invalid cube_core_num "
            f"for NPU {device_index}: {num_cube_cores!r}."
        )

    local_num_tokens = specification["local_num_tokens"]
    top_k = specification["top_k"]
    receive_capacity = _resolve_receive_capacity(
        specification["expert_capacity_factor"],
        local_num_tokens * top_k,
    )
    fusion_mode = _resolve_fusion_mode(top_k)
    config = MegaMoeConfig(
        local_num_tokens=local_num_tokens,
        hidden_size=specification["hidden_size"],
        intermediate_size=specification["intermediate_size"],
        num_experts=specification["num_experts"],
        top_k=top_k,
        receive_capacity=receive_capacity,
        ep_size=ep_size,
        ep_group=ep_group,
        rank_id=rank_id,
        num_cube_cores=num_cube_cores,
        fusion_mode=fusion_mode,
    )
    if config.local_num_tokens % _COMMUNICATION_SPLIT:
        raise ValueError(
            "local_num_tokens must be divisible by the fixed "
            f"communication split {_COMMUNICATION_SPLIT}, got {config.local_num_tokens}."
        )
    if config.receive_capacity < config.routed_slots:
        raise ValueError(
            "receive_capacity cannot be smaller than the local routed row count "
            f"{config.routed_slots}, got {config.receive_capacity}."
        )
    return config
