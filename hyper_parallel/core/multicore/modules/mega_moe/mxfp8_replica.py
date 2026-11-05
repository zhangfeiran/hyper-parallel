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
"""Exact byte transport for MX weight axes and FP32 owner-gradient reduction."""

from __future__ import annotations

from dataclasses import replace

import torch

from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute
from hyper_parallel.core.expert_parallel.hot_replica.transport import prefetch_weights, return_gradients

from .mxfp8_weights import MXFP8PhysicalWeight


def materialize_replica_weights(
    weights: tuple[MXFP8PhysicalWeight, ...], route: ReplicaRoute | None, *, backward: bool
) -> tuple[MXFP8PhysicalWeight, ...]:
    """Append B physical slots without dequantizing or requantizing any axis.

    Only current-phase axes are exchanged. Forward consumes column weights;
    backward consumes freshly gathered row weights. Unused axes are dropped
    from the materialized view while the original FSDP root remains unchanged.
    Empty slots are zeroed and have no routed rows. The eager transport keeps
    the existing projected B2 placement policy independent of BF16 kernel ABIs.

    Args:
        weights: Current physical owner weight axes, never cached across FSDP phases.
        route: Immutable replica plan and exact physical dispatch metadata.
    """
    if route is None or route.home_only:
        return weights
    fields = ("row_data", "row_scale") if backward else ("col_data", "col_scale")
    originals = []
    for weight in weights:
        for field in fields:
            tensor = getattr(weight, field)
            if tensor is None:
                raise ValueError(f"MXFP8 replica phase is missing {field}.")
            originals.append(tensor)
    aliases = tuple(tensor.view(torch.uint8).reshape(tensor.shape[0], -1) for tensor in originals)
    config = route.plan.config
    with prefetch_weights(aliases, route) as pool:
        active = {
            item.target_slot - config.home_experts for item in route.plan.transfers if item.target_rank == route.rank
        }
        for guest in pool.weights:
            for slot in range(config.replica_slots_per_rank):
                if slot not in active:
                    guest[slot].zero_()
        assembled = []
        for tensor, alias, guest in zip(originals, aliases, pool.weights):
            data = torch.cat((alias, guest), dim=0).view(tensor.dtype)
            assembled.append(data.reshape((config.home_experts + config.replica_slots_per_rank), *tensor.shape[1:]))
    result = []
    for index, weight in enumerate(weights):
        values = dict.fromkeys(("col_data", "col_scale", "row_data", "row_scale"))
        values.update(zip(fields, assembled[index * len(fields) : (index + 1) * len(fields)]))
        result.append(
            replace(
                weight,
                logical_shape=((config.home_experts + config.replica_slots_per_rank), *weight.logical_shape[1:]),
                **values,
            )
        )
    return tuple(result)


def reduce_replica_gradients(
    gradients: tuple[torch.Tensor, ...], route: ReplicaRoute | None
) -> tuple[torch.Tensor, ...]:
    """Merge guest partials in FP32 and return only the logical owner matrices.

    Args:
        gradients: Fresh FP32 physical partials, including active guests.
        route: Immutable replica plan and exact physical dispatch metadata.
    """
    if route is None or route.home_only:
        return gradients
    if any(gradient.dtype != torch.float32 for gradient in gradients):
        raise TypeError("MXFP8 replica owner reduction requires FP32 gradient partials.")
    return return_gradients(gradients, route, consume=True)
