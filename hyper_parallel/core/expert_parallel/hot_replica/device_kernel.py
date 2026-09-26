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
"""Independent Ascend AIV launch boundary for shared expert replica planning."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from .capacity import ExpertReplicaConfig


@lru_cache(maxsize=1)
def _planner_library() -> ctypes.CDLL:
    """Load only the planner library, with no multicore or SHMEM initialization."""
    path = Path(__file__).parent / "_device_kernel" / "libhyper_parallel_replica_planner.so"
    if not path.is_file():
        raise RuntimeError("Build the shared device planner using the CMake instructions in hot_replica/README.md")
    library = ctypes.CDLL(str(path))
    try:
        version = library.planner_abi_version
    except AttributeError as error:
        raise RuntimeError("Rebuild the shared device planner: missing ABI version") from error
    version.argtypes = []
    version.restype = ctypes.c_int
    if version() != 2:
        raise RuntimeError("Rebuild the shared device planner: incompatible ABI version")
    library.launch_planner.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_int64] * 3)
    library.launch_planner.restype = ctypes.c_int
    return library


def _validate_device(counts: torch.Tensor, config: ExpertReplicaConfig) -> None:
    """Keep topology arithmetic and the single-AIV scratch within their bounds."""
    if counts.device.type != "npu":
        raise ValueError("The fused device replica planner requires an Ascend NPU")
    scratch = 5 * config.ep_size * config.num_experts + 3 * config.num_experts + 4 * config.ep_size
    if scratch > 23040 or config.ep_size * config.physical_experts > torch.iinfo(torch.int32).max:
        raise ValueError("Replica topology exceeds the fused device planner scratch or index capacity")


def launch_device_planner(counts: torch.Tensor, config: ExpertReplicaConfig,
                          capacity: int | None, target: int | None,
                          minimum_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one task with invocation-owned buffers on the current NPU stream.

    Callers supply producer dependencies for counts. Stream recording keeps all
    launch arguments alive even if counts originated on another stream. The
    returned buffers have no dependency on a mutable graph replay workspace.
    """
    _validate_device(counts, config)
    library = _planner_library()
    with torch.npu.device(counts.device):
        counts = counts.to(torch.int64).contiguous()
        ranks, width = config.ep_size, config.slots_per_rank
        control = torch.empty(1 + 2 * ranks * width + ranks * ranks + (ranks + 1) * ranks * width,
                              dtype=torch.int64, device=counts.device)
        dispatch = torch.empty((ranks, ranks * width), dtype=torch.int32, device=counts.device)
        stream = torch.npu.current_stream(counts.device)
        for tensor in (counts, control, dispatch):
            tensor.record_stream(stream)
        status = library.launch_planner(stream.npu_stream, counts.data_ptr(), control.data_ptr(), dispatch.data_ptr(),
                                        ranks, config.num_experts, config.replica_slots_per_rank,
                                        -1 if capacity is None else capacity, target or 0, minimum_rows)
        if status:
            raise RuntimeError(f"Replica planner launch failed with ACL error {status}")
    return control, dispatch
