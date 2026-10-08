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
"""Artifact-verified Gate binding, current-stream resources and native autograd."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
import struct
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.core.multicore.runtime.abi import NativeManifest, family_abi
from hyper_parallel.core.multicore.runtime.generated import native_calls

if TYPE_CHECKING:
    from hyper_parallel.core.multicore.runtime.plan import KernelPlan, RuntimeImage

_VENDOR = "hyper_parallel_multicore_gate_v1"
_PROFILE_BYTES = 24 * 64 + 48 * (64 + 16 * 32)
_HEADER = struct.Struct("<QIIIIII32x")
_RECORD = struct.Struct("<QQIIII")
_LOADED_PAYLOAD: dict[Path, str] = {}


def verify_gate_payload(root: Path) -> dict[str, object]:
    """Verify family identity, build fingerprint and every packaged artifact.

    Args:
        root: Payload produced by the isolated Gate build.

    Returns:
        Validated build manifest; no shared library is loaded by this check.
    """
    root = Path(root).resolve()
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    family_abi("gate").verify_native(NativeManifest(**{
        field.name: data[field.name] for field in fields(NativeManifest)
    }))
    if data["vendor"] != _VENDOR:
        raise ValueError("Gate vendor identity mismatch")
    expected = hashlib.sha256(json.dumps(
        {"build": data["build"], "artifacts": data["artifacts"]}, sort_keys=True,
    ).encode()).hexdigest()
    if expected != data["build_fingerprint"]:
        raise ValueError("Gate build fingerprint mismatch")
    artifacts = data["artifacts"]
    required = {
        f"vendors/{_VENDOR}/op_api/lib/libcust_opapi.so",
        "framework/torch/libhyper_parallel_mega_gate_torch.so", "set_env.bash",
    }
    if not required.issubset(artifacts):
        raise ValueError("Gate payload manifest lacks required artifacts")
    for relative, digest in artifacts.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or escaping Gate artifact: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Gate artifact hash mismatch: {relative}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != set(artifacts) | {"manifest.json"}:
        raise ValueError("Gate payload contains unrecorded artifacts")
    return data


def _environment_paths(name):
    return tuple(Path(value).resolve() for value in os.environ.get(name, "").split(os.pathsep) if value)


def _load_payload(root, manifest):
    vendor = root / "vendors" / _VENDOR
    configured = _environment_paths("ASCEND_CUSTOM_OPP_PATH")
    if not configured or configured[0] != vendor or vendor / "op_api/lib" not in _environment_paths("LD_LIBRARY_PATH"):
        raise ValueError("Source the Gate payload set_env.bash before starting Python")
    # The framework caches ACLNN handles and op metadata process-wide. An old
    # family vendor in the same process cannot be safely switched by editing env.
    if len(configured) != 1:
        raise ValueError("Gate compatibility execution requires an isolated custom OPP process")
    build = manifest["build"]
    if build["torch"] != torch.__version__ or build["torch_cxx11_abi"] != torch.compiled_with_cxx11_abi():
        raise ValueError("Gate payload Torch version/C++ ABI mismatch")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "")).resolve()
    if str(cann) != build["cann_root"] or (cann / "opp/version.info").read_text() != build["cann_version"]:
        raise ValueError("Gate payload CANN identity mismatch")
    torch_npu = importlib.import_module("torch_npu")
    if torch_npu.__version__ != build["torch_npu"]:
        raise ValueError("Gate payload torch_npu version mismatch")
    if _LOADED_PAYLOAD:
        if _LOADED_PAYLOAD != {root: manifest["build_fingerprint"]}:
            raise ValueError("A different Gate payload is already loaded in this process")
        return
    ctypes.CDLL(str(vendor / "op_api/lib/libcust_opapi.so"), mode=ctypes.RTLD_LOCAL)
    torch.ops.load_library(str(root / "framework/torch/libhyper_parallel_mega_gate_torch.so"))
    _LOADED_PAYLOAD[root] = manifest["build_fingerprint"]


class GateProfile:
    """One invocation's private profile buffer and completion event."""

    def __init__(self, direction: str, image: RuntimeImage, buffer: torch.Tensor, event: Any) -> None:
        """Own the completed invocation's buffer, native image and NPU event."""
        self.direction = direction
        self.image = image
        self.buffer = buffer
        self.event = event

    def records(self) -> tuple[dict[str, object], ...]:
        """Wait for this invocation and return source-mapped native stage intervals."""
        self.event.synchronize()
        wire = bytes(self.buffer.cpu().tolist())
        records = []
        for worker in range(48):
            offset = 24 * 64 + worker * (64 + 16 * 32)
            _, count, dropped, core, block, capacity, _ = _HEADER.unpack_from(wire, offset)
            if not count:
                continue
            if dropped or count != len(self.image.stages) or capacity != 16 or core != 2 or block != worker:
                raise ValueError(f"Invalid or dropped Gate profile records for worker {worker}")
            for index in range(count):
                start, end, desc, task, stage_index, owner = _RECORD.unpack_from(wire, offset + 64 + index * 32)
                if task != index or desc != 0x30000 + index or stage_index != worker or owner != worker or end < start:
                    raise ValueError("Gate profile descriptor/worker identity mismatch")
                stage = self.image.stages[index]
                records.append({
                    "direction": self.direction, "worker": worker, "stage": stage.logical_name,
                    "start_cycle": start, "end_cycle": end,
                    "source": tuple(str(span) for span in stage.source_spans),
                })
        return tuple(records)


class GateExecutable:
    """A Gate plan bound to a verified payload and one NPU device.

    Descriptors are immutable across calls. Forward caches belong to each
    autograd invocation; profiled calls allocate separate record buffers.
    """

    def __init__(self, plan: KernelPlan, device: str | torch.device, payload_root: Path | None = None) -> None:
        """Verify native artifacts and materialize the plan's normal/profiled images."""
        root = payload_root or os.environ.get("HP_GATE_PAYLOAD_ROOT")
        if root is None:
            raise ValueError("Activate the isolated Gate payload before materializing a plan")
        root = Path(root).resolve()
        self.manifest = verify_gate_payload(root)
        plan.verify_native(NativeManifest(**{
            field.name: self.manifest[field.name] for field in fields(NativeManifest)
        }))
        _load_payload(root, self.manifest)
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("Gate native materialization requires an NPU device")
        if self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        name = torch.npu.get_device_name(self.device).lower()
        soc = self.manifest["build"]["soc"]
        if (soc == "ascend910b" and not name.startswith("ascend910b")) or (
            soc == "ascend910_93" and not name.startswith(("ascend910c", "ascend910_93"))
        ):
            raise ValueError(f"Gate payload SoC mismatch: {soc}, {name}")
        available = torch.npu.get_device_properties(self.device).vector_core_num
        if available <= 0 or len(plan.schedule.partitions) != min(plan.token_count, available, 48):
            raise ValueError(f"Gate plan topology does not match native host tiling: device has {available} AIV cores")
        self.plan = plan
        with torch.npu.device(self.device):
            self.configs = tuple(torch.tensor(list(wire), dtype=torch.uint8, device=self.device) for wire in (
                plan.forward.normal, plan.forward.profiled, plan.backward.normal, plan.backward.profiled,
            ))
            self.image_mask = torch.zeros(1, dtype=torch.bool, device=self.device)
            self.disabled_profile = torch.zeros(1, dtype=torch.uint8, device=self.device)
            self.ready = torch.npu.Event()
            self.ready.record()
        self.profiles = []

    def _resources(self, backward, profile):
        stream = torch.npu.current_stream(self.device)
        stream.wait_event(self.ready)
        config = self.configs[2 * backward + profile]
        buffer = (
            torch.zeros(_PROFILE_BYTES, dtype=torch.uint8, device=self.device) if profile else self.disabled_profile
        )
        for tensor in (config, buffer, self.image_mask):
            tensor.record_stream(stream)
        return config, buffer

    def _complete(self, backward, profile, buffer):
        if profile:
            event = torch.npu.Event()
            event.record(torch.npu.current_stream(self.device))
            image = self.plan.backward if backward else self.plan.forward
            self.profiles.append(GateProfile("backward" if backward else "forward", image, buffer, event))

    def _forward(self, logits, bias, profile):
        config, buffer = self._resources(False, profile)
        stream = torch.npu.current_stream(self.device)
        for tensor in (logits, bias):
            tensor.record_stream(stream)
        outputs = native_calls.mega_gate_route(
            logits, bias, bias, self.image_mask, config, buffer, self.plan.top_k, self.plan.scale, False,
        )
        self._complete(False, profile, buffer)
        return outputs

    def backward(self, saved: tuple[torch.Tensor, ...], grad_weights: torch.Tensor, profile: bool) -> torch.Tensor:
        """Run the pinned RouteGrad kernel followed by its original CANN postprocessing.

        Args:
            saved: Invocation-owned logits, indices, scores, selected scores and denominator.
            grad_weights: Incoming gradient of routed weights.
            profile: Record this backward invocation's native stages.
        """
        logits, indices, scores, selected, denominator = saved
        config, buffer = self._resources(True, profile)
        stream = torch.npu.current_stream(self.device)
        for tensor in (*saved, grad_weights):
            tensor.record_stream(stream)
        result = native_calls.mega_gate_route_grad(
            logits, scores, selected, denominator, indices, grad_weights.contiguous(), logits,
            config, buffer, self.plan.top_k, self.plan.scale, False,
        )
        self._complete(True, profile, buffer)
        return result

    def __call__(
        self, logits: torch.Tensor, bias: torch.Tensor, *, profile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the compiled Route, preserving selection-only bias and native autograd.

        Args:
            logits: Contiguous FP32 [tokens, experts] on the bound device.
            bias: Contiguous FP32 [experts] selection bias on the same device.
            profile: Allocate private forward/backward stage record buffers.
        """
        self._validate_inputs(logits, bias)
        if not isinstance(profile, bool):
            raise TypeError("profile must be bool")
        with torch.npu.device(self.device):
            if torch.is_grad_enabled() and logits.requires_grad:
                return _GateAutograd.apply(logits, bias.detach(), self, profile)
            return self._forward(logits, bias.detach(), profile)[:2]

    def _validate_inputs(self, logits, bias):
        for tensor, shape in ((logits, (self.plan.token_count, self.plan.expert_count)),
                              (bias, (self.plan.expert_count,))):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"Gate input must match compiled shape {shape}")
            if tensor.device != self.device or tensor.dtype != torch.float32 or not tensor.is_contiguous():
                raise ValueError("Gate inputs must be contiguous FP32 on the bound device")

    def take_profiles(self) -> tuple[GateProfile, ...]:
        """Transfer ownership of collected per-call profiles without synchronizing."""
        profiles, self.profiles = tuple(self.profiles), []
        return profiles


class _GateAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, logits: torch.Tensor, bias: torch.Tensor, executable: GateExecutable, profile: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Save invocation-owned caches and detach integer selection outputs.

        Args:
            ctx: Autograd context.
            logits: Differentiable projection logits.
            bias: Detached correction bias.
            executable: Verified device binding and descriptor resources.
            profile: Enable private stage-record buffers.
        """
        ctx.set_materialize_grads(False)
        weights, indices, scores, selected, denominator = executable._forward(logits, bias, profile)
        ctx.save_for_backward(logits, indices, scores, selected, denominator)
        ctx.executable = executable
        ctx.profile = profile
        ctx.mark_non_differentiable(indices)
        return weights, indices

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, grad_weights: torch.Tensor | None, _grad_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None, None, None]:
        """Return logits gradients through the native recipe; bias remains selection-only.

        Args:
            ctx: Autograd context owning this invocation's saved tensors.
            grad_weights: Optional incoming routing-weight gradient.
            _grad_indices: Always absent for nondifferentiable expert indices.
        """
        if grad_weights is None or not ctx.needs_input_grad[0]:
            return None, None, None, None
        with torch.npu.device(ctx.executable.device):
            grad_logits = ctx.executable.backward(ctx.saved_tensors, grad_weights, ctx.profile)
        return grad_logits, None, None, None
