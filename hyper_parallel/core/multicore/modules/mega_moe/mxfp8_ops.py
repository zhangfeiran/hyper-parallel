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
"""Byte-alias launch adapter for the separately versioned native MX task graph."""

from __future__ import annotations

from functools import lru_cache
from collections.abc import Sequence

import torch

from hyper_parallel.core.multicore.scheduler.mxfp8 import MxLaunchDescriptor, serialize_mx_launch_metadata
from hyper_parallel.core.multicore.torch.ops import _load_native

from .spec import MegaMoeSpec


@lru_cache(maxsize=1)
def _load_mxfp8_native() -> None:
    _load_native()
    if torch.ops.hyper_parallel.mxfp8_task_graph_version() != 5:
        raise RuntimeError("MXFP8 task-graph adapter ABI mismatch; rebuild the local multicore component.")


def _byte_aliases(buffers: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    aliases = []
    for tensor in buffers:
        if not tensor.is_contiguous():
            raise ValueError("MXFP8 task buffers must be contiguous; implicit physical weight copies are forbidden.")
        aliases.append(tensor.view(torch.uint8).reshape(-1))
    return aliases


def serialize_mxfp8_launch(
    buffers: Sequence[torch.Tensor], runtime: torch.Tensor, events: torch.Tensor, spec: MegaMoeSpec,
    *, kernel_gradients: bool = False,
) -> bytes:
    """Serialize exact current byte bounds without reading device tensor contents.

    Args:
        buffers: Current physical buffers in the native launch order.
        runtime: Device runtime image with current task and event capacities.
        events: Symmetric event and persistent ready-generation storage.
        spec: Validated physical expert topology and current capacity.
    """
    content = serialize_mx_launch_metadata(
        [tensor.numel() * tensor.element_size() for tensor in buffers],
        rank=spec.rank_id,
        ep=spec.ep_size,
        experts=spec.num_experts,
        hidden=spec.hidden_size,
        local_tokens=spec.local_num_tokens,
        cube_count=spec.num_cube_cores,
        runtime_bytes=runtime.numel(),
        event_bytes=events.numel(),
        vector_policy=spec.mx_vector_policy,
    )
    if not kernel_gradients:
        return content
    if spec.mx_vector_policy != "window2" or spec.replica_transport != "shmem_signal_kernel_gradient":
        raise ValueError("MX gradient return requires its dedicated transport and window2 queue")
    record = MxLaunchDescriptor.from_buffer_copy(content)
    record.version, record.reserved = 11, 10
    return bytes(record)


def prepare_mxfp8_launch(
    buffers: Sequence[torch.Tensor], runtime: torch.Tensor, events: torch.Tensor, spec: MegaMoeSpec,
    *, kernel_gradients: bool = False,
) -> torch.Tensor:
    """Transfer the byte-bounds launch descriptor to the active device.

    Args:
        buffers: Current physical buffers in the native launch order.
        runtime: Device runtime image with current task and event capacities.
        events: Symmetric event and persistent ready-generation storage.
        spec: Validated physical expert topology and current capacity.
    """
    metadata = serialize_mxfp8_launch(buffers, runtime, events, spec, kernel_gradients=kernel_gradients)
    return torch.frombuffer(bytearray(metadata), dtype=torch.uint8).to(device=runtime.device)


def launch_mxfp8_task_graph(
    buffers: Sequence[torch.Tensor],
    runtime: torch.Tensor,
    events: torch.Tensor,
    diagnostic: torch.Tensor,
    launch: torch.Tensor,
    *,
    vector_policy: str = "legacy_odd",
    swiglu_row_only: bool = False,
) -> None:
    """Run the native fused task graph with alias-preserving physical arguments.

    Args:
        buffers: Current physical buffers in the native launch order.
        runtime: Device runtime image with current task and event capacities.
        events: Symmetric event and persistent ready-generation storage.
        diagnostic: Explicit diagnostic storage; ordinary launches keep capture disabled.
        launch: Native task-graph launch callable.
    """
    aliases = _byte_aliases(buffers)
    _load_mxfp8_native()
    if vector_policy not in ("legacy_odd", "window2") or swiglu_row_only:
        raise ValueError("The integrated MX payload supports legacy_odd/window2 dual-axis graphs only.")
    torch.ops.hyper_parallel.mxfp8_task_graph(aliases, runtime, events, diagnostic, launch)
