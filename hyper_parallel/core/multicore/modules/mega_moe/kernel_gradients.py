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
"""Invocation metadata for W2 return inside the fused backward kernel."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

import torch

from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute

from .plan import MegaMoePlan


@dataclass(frozen=True)
class KernelGradientReturn:
    """Retain ordinary metadata and completion words through kernel submission."""

    metadata: torch.Tensor
    completion: torch.Tensor


def prepare_kernel_gradient_return(plan: MegaMoePlan, route: ReplicaRoute | None, provider: Any,
                                   gradient: torch.Tensor, guest: torch.Tensor | None) -> KernelGradientReturn | None:
    """Encode only proven W2 completion events and preserve peer accumulation order."""
    if (route is None or not route.plan.transfers or not plan.replica_w2_events
            or not getattr(provider, "kernel_gradients", False)):
        return None
    if gradient.dtype != torch.float32 or gradient.requires_grad or not gradient.is_contiguous():
        raise ValueError("Kernel gradient return requires detached contiguous FP32 owner gradients")
    if (guest is None or guest.dtype != torch.float32 or guest.requires_grad or not guest.is_contiguous()
            or guest.device != gradient.device or guest.shape[1:] != gradient.shape[1:]):
        raise ValueError("Kernel gradient return requires matching detached contiguous FP32 guest gradients")
    home = route.plan.config.home_experts
    incoming = [item for item in route.plan.transfers if item.target_rank == route.rank]
    owned = sorted((item for item in route.plan.transfers if item.owner_rank == route.rank),
                   key=lambda item: item.target_rank)
    epoch, ready, ack = provider.kernel_gradient_signals()
    completion = torch.zeros((plan.spec.num_cube_cores, 16), dtype=torch.int32, device=gradient.device)
    values = [1, route.rank, route.plan.config.replica_slots_per_rank, epoch, gradient[0].numel(),
              gradient.data_ptr(), guest.data_ptr(), ready, ack, completion.data_ptr(), len(incoming), len(owned)]
    for item in incoming:
        values.extend((item.owner_rank, item.target_slot - home, plan.replica_w2_events[item.target_slot]))
    for item in owned:
        values.extend((item.target_rank, item.target_slot - home, item.owner_slot,
                       plan.replica_w2_events[item.owner_slot]))
    data = struct.pack("<" + "Q" * len(values), *values)
    metadata = torch.tensor(list(data), dtype=torch.uint8, device=gradient.device)
    stream = torch.npu.current_stream(gradient.device)
    metadata.record_stream(stream)
    completion.record_stream(stream)
    return KernelGradientReturn(metadata, completion)
