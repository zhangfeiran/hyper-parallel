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
"""Immutable host plans with legacy images, row ownership and explicit native boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hyper_parallel.core.multicore.ir.schedule import PipelineScheduleIR, PipelineStage
from hyper_parallel.core.multicore.runtime.abi import FamilyABI, NativeManifest
from hyper_parallel.core.multicore.runtime.gate import GateExecutable


@dataclass(frozen=True)
class RuntimeImage:
    """Normal/profiled wire bytes for a single family-native entry point."""

    kernel_name: str
    stages: tuple[PipelineStage, ...]
    normal: bytes
    profiled: bytes


@dataclass(frozen=True)
class Binding:
    """Logical value or saved-state binding in native argument order."""

    name: str
    position: int
    value_id: int | None


@dataclass(frozen=True)
class KernelPlan:
    """Gate compilation result with artifact-verified native materialization."""

    abi: FamilyABI
    schedule: PipelineScheduleIR
    forward: RuntimeImage
    backward: RuntimeImage
    bindings: tuple[Binding, ...]
    backward_bindings: tuple[Binding, ...]
    saved_state: tuple[str, ...]
    external_backward_calls: tuple[str, ...]
    top_k: int
    scale: float
    token_count: int
    expert_count: int

    def materialize(self, device: str = "npu:0", *, payload_root: Path | None = None) -> GateExecutable:
        """Bind the plan to a verified Gate payload and device.

        Args:
            device: Ascend device where all inputs and resources will reside.
            payload_root: Build-produced payload; defaults to HP_GATE_PAYLOAD_ROOT.

        Returns:
            Callable with native forward/backward and optional per-call profiling.
        """
        return GateExecutable(self, device, payload_root)

    def export_manifest(self) -> dict[str, object]:
        """Export host plan contracts without runtime pointers or a native claim."""
        return {
            **self.abi.export_manifest(),
            "execution_mode": self.schedule.execution_mode,
            "forward_kernel": self.forward.kernel_name,
            "backward_kernel": self.backward.kernel_name,
            "native_status": "unbound",
            "top_k": self.top_k,
            "scale": self.scale,
            "use_vision_bias": False,
            "token_count": self.token_count,
            "expert_count": self.expert_count,
            "available_row_workers": len(self.schedule.partitions),
            "device_tiling": "required_from_legacy_host",
            "saved_state": self.saved_state,
            "external_backward_calls": self.external_backward_calls,
        }

    def verify_native(self, manifest: NativeManifest) -> None:
        """Check matching native build metadata before future materialization.

        Args:
            manifest: Native artifact's build-provided ABI/source identity.
        """
        self.abi.verify_native(manifest)

    def explain(self) -> str:
        """Describe native stages, bindings and source provenance as deterministic JSON."""
        data = {
            "manifest": self.export_manifest(),
            "stages": [
                {
                    "stage": stage.logical_name,
                    "native_id": self.abi.task_id(stage.logical_name),
                    "source": [str(span) for span in stage.source_spans],
                    "reason": stage.reason,
                }
                for stage in self.schedule.stages
            ],
            "partitions": [
                {"worker": part.worker_id, "first_row": part.first_row, "rows": part.row_count}
                for part in self.schedule.partitions
            ],
            "bindings": [
                {"name": binding.name, "position": binding.position, "value": binding.value_id}
                for binding in self.bindings
            ],
            "backward_bindings": [
                {"name": binding.name, "position": binding.position, "value": binding.value_id}
                for binding in self.backward_bindings
            ],
        }
        return json.dumps(data, indent=2, sort_keys=True)
