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
"""Materialize dense plans into CPU references or compiled native adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from hyper_parallel.core.multicore._build.build_dense import build_dense_payload
from hyper_parallel.core.multicore.runtime.dense import DenseExecutable, DenseKernelPlan
from hyper_parallel.core.multicore.runtime.dense_compiled import CompiledDenseExecutable
from hyper_parallel.core.multicore.runtime.dense_tile import ResidentDenseExecutable
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy


@dataclass(frozen=True)
class DenseExecutionConfig:
    """Explicit backend selection through module and existing replacement context."""

    backend: str = "host_stream"
    cann_root: Path | None = None
    cache_root: Path | None = None
    soc: str | None = None
    tile_policy: DenseTilePolicy | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("host_stream", "resident_tiles"):
            raise ValueError("Dense execution backend must be host_stream or resident_tiles")
        if self.tile_policy is not None and not isinstance(self.tile_policy, DenseTilePolicy):
            raise ValueError("Dense tile_policy requires a DenseTilePolicy")


def materialize_dense(plan: DenseKernelPlan, device: torch.device,
                      cache_root: Path | None = None, execution: DenseExecutionConfig | None = None) -> DenseExecutable:
    """Compile native provider calls for NPU while keeping CPU references explicit.

    Args:
        plan: Canonical dense task graph.
        device: Device for all invocation inputs.
        cache_root: Optional caller-owned native build cache.
        execution: Explicit resident candidate or host-stream provider configuration.
    """
    execution = DenseExecutionConfig() if execution is None else execution
    if not isinstance(execution, DenseExecutionConfig):
        raise ValueError("Dense execution requires a DenseExecutionConfig")
    if device.type == "cpu":
        return plan.materialize(device)
    cache_root = execution.cache_root if execution.cache_root is not None else cache_root
    if cache_root is None:
        cache_root = Path(__file__).resolve().parents[4] / "build/native/dense"
    if execution.backend == "resident_tiles":
        cann_root = execution.cann_root or Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
        device_api = torch.get_device_module(device)
        with device_api.device(device):
            soc = execution.soc or device_api.get_device_name(device)
        return ResidentDenseExecutable(plan, device, cann_root=cann_root, cache_root=cache_root, soc=soc,
                                       policy=execution.tile_policy)
    manifest = build_dense_payload(plan, cache_root)
    return CompiledDenseExecutable(plan, device, manifest)
