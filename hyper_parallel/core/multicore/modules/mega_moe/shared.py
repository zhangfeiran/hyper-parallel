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
"""Owned local shared tensors and static tiling for the integrated kernel branch."""

import struct
from typing import Any

import numpy as np
import torch

from .backward.tiling_tables import (
    get_act_grad_tiling_bytes, get_gate_grad_tiling_bytes, get_w1_grad_tiling_bytes,
    get_w2_grad_tiling_bytes, get_swiglu_grad_tiling_bytes,
)
from .forward.tiling_tables import get_up_proj_tiling_bytes, get_down_proj_tiling_bytes, get_swiglu_tiling_bytes
from .shared_schedule import GMM_TILING_BYTES


def _static_gmm(data: bytes, tokens: int, *, weight_gradient: bool) -> bytes:
    """Set the static M/K extent in every core's independent tiling record."""
    words = np.frombuffer(data, dtype="<u4").copy().reshape(-1, GMM_TILING_BYTES // 4)
    anchors = np.flatnonzero(words[0] == 393216)
    if len(anchors) != 1:
        raise ValueError("Shared GMM tiling has no unique matmul layout anchor")
    base = int(anchors[0]) - 19
    positions = (base + 3, base + 4, base + 7) if weight_gradient else (14, base + 1, base + 5)
    words[:, positions] = tokens
    return words.tobytes()


def shared_tiling_bytes(spec: Any, *, backward: bool) -> bytes:
    """Serialize one group list and static shared GMM/SwiGLU tiling records.

    Args:
        spec: Bound MegaMoe shape and hardware specification.
        backward: Select gradient tilings instead of forward tilings.

    Returns:
        The shared runtime buffer consumed through the native ABI.
    """
    options = dict(hidden_size=spec.hidden_size, intermediate_size=spec.shared_intermediate_size,
                   num_groups=1, num_cube_cores=spec.num_cube_cores)
    builders = ((get_act_grad_tiling_bytes, get_gate_grad_tiling_bytes,
                 get_w1_grad_tiling_bytes, get_w2_grad_tiling_bytes) if backward
                else (get_up_proj_tiling_bytes, get_down_proj_tiling_bytes))
    records = [struct.pack("<q", spec.local_num_tokens)]
    for index, builder in enumerate(builders):
        data = builder(4096, **options)
        if len(data) != GMM_TILING_BYTES * spec.num_cube_cores:
            raise ValueError("Shared GMM tiling stride differs from the device ABI")
        records.append(_static_gmm(data, spec.local_num_tokens, weight_gradient=backward and index >= 2))
    swiglu = get_swiglu_grad_tiling_bytes if backward else get_swiglu_tiling_bytes
    records.append(swiglu(128, intermediate_size=spec.shared_intermediate_size)[:80])
    return b"".join(records)


def pointer_table(tensors: tuple[torch.Tensor, ...], *, idle_vectors: bool = False) -> torch.Tensor:
    """Upload scheduling flags followed by addresses; callers retain all targets."""
    addresses = torch.tensor([int(idle_vectors)] + [tensor.data_ptr() for tensor in tensors],
                             dtype=torch.int64, pin_memory=True)
    return addresses.to(device=tensors[0].device, non_blocking=True).view(torch.uint8)


def shared_forward_buffers(x: torch.Tensor, intermediate: int) -> tuple[torch.Tensor, ...]:
    """Allocate owned shared intermediates whose full extent is overwritten."""
    return (x.new_empty((x.shape[0], 2 * intermediate)),
            x.new_empty((x.shape[0], intermediate)), torch.empty_like(x))


def shared_backward_buffers(
    dy: torch.Tensor, saved: tuple[torch.Tensor, ...], *, idle_vectors: bool = False,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Allocate gradients and keep indirect targets alive through kernel completion."""
    x, w1, w2, up, activation = saved
    intermediate = w2.shape[1]
    dw2 = w2.new_empty((intermediate, w2.shape[0]))
    dactivation, dup, dx = torch.empty_like(activation), torch.empty_like(up), torch.empty_like(x)
    dw1 = w1.new_empty((w1.shape[1], w1.shape[0]))
    tensors = (dy.contiguous(), activation, dw2, w2, dactivation, up, dup, w1, dx, x, dw1)
    return tensors, pointer_table(tensors, idle_vectors=idle_vectors)
