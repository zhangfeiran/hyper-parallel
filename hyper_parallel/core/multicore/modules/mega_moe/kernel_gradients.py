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
"""Invocation metadata for ordered projection returns inside fused backward."""

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
    matrices: tuple[int, ...] = (1,)


def _validate_gradient_pair(gradient: torch.Tensor, guest: torch.Tensor | None) -> None:
    """Reject borrowed, cast-requiring or incompatible gradient storage."""
    if gradient.dtype != torch.float32 or gradient.requires_grad or not gradient.is_contiguous():
        raise ValueError("Kernel gradient return requires detached contiguous FP32 owner gradients")
    if (guest is None or guest.dtype != torch.float32 or guest.requires_grad or not guest.is_contiguous()
            or guest.device != gradient.device or guest.shape[1:] != gradient.shape[1:]):
        raise ValueError("Kernel gradient return requires matching detached contiguous FP32 guest gradients")


def _matrix_record(route: ReplicaRoute, provider: Any, events: tuple[int, ...],
                   gradient: torch.Tensor, guest: torch.Tensor, completion: torch.Tensor) -> list[int]:
    """Encode one epoch with complete local producers before ordered peer reads."""
    home = route.plan.config.home_experts
    incoming = [item for item in route.plan.transfers if item.target_rank == route.rank]
    owned = sorted((item for item in route.plan.transfers if item.owner_rank == route.rank),
                   key=lambda item: item.target_rank)
    epoch, ready, ack = provider.kernel_gradient_signals()
    values = [1, route.rank, route.plan.config.replica_slots_per_rank, epoch, gradient[0].numel(),
              gradient.data_ptr(), guest.data_ptr(), ready, ack, completion.data_ptr(), len(incoming), len(owned)]
    for item in incoming:
        values.extend((item.owner_rank, item.target_slot - home, events[item.target_slot]))
    for item in owned:
        values.extend((item.target_rank, item.target_slot - home, item.owner_slot, events[item.owner_slot]))
    return values


def prepare_kernel_gradient_return(plan: MegaMoePlan, route: ReplicaRoute | None, provider: Any,
                                   gradient: torch.Tensor, guest: torch.Tensor | None, *,
                                   gradient_w13: torch.Tensor | None = None,
                                   guest_w13: torch.Tensor | None = None) -> KernelGradientReturn | None:
    """Encode proven projection fences and retain FP32 peer order and ACK leases."""
    if (route is None or not route.plan.transfers or not plan.replica_w2_events
            or not getattr(provider, "kernel_gradients", False)):
        return None
    pairs = [(plan.replica_w2_events, gradient, guest)]
    w13_events = getattr(plan, "replica_w13_events", ())
    if w13_events and gradient_w13 is not None:
        pairs.append((w13_events, gradient_w13, guest_w13))
    for _, owner, replica in pairs:
        _validate_gradient_pair(owner, replica)
    completion = torch.zeros((len(pairs), plan.spec.num_cube_cores, 16),
                             dtype=torch.int32, device=gradient.device)
    records = [_matrix_record(route, provider, events, owner, replica, completion[index])
               for index, (events, owner, replica) in enumerate(pairs)]
    # The two-matrix envelope retains the original single-matrix records. Epochs
    # are distinct because ready/ACK words are shared across ordered matrices.
    values = records[0] if len(records) == 1 else [2, 3, 3 + len(records[0])] + records[0] + records[1]
    data = struct.pack("<" + "Q" * len(values), *values)
    metadata = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(gradient.device)
    if len(records) == 1:
        completion = completion[0]
    stream = torch.npu.current_stream(gradient.device)
    metadata.record_stream(stream)
    completion.record_stream(stream)
    return KernelGradientReturn(metadata, completion, (1,) if len(records) == 1 else (1, 0))
