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
"""Isolated MHC payload binding and invocation-owned native caches/counters/autograd."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
import struct
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.core.multicore.runtime.abi import NativeManifest, family_abi
from hyper_parallel.core.multicore.runtime.generated import native_calls

if TYPE_CHECKING:
    from hyper_parallel.core.multicore.runtime.mhc import MhcKernelPlan, MhcRuntimeImage

_VENDOR = "hyper_parallel_multicore_mhc_v1"
_LOADED: dict[Path, str] = {}
_HEADER = struct.Struct("<QIIIIII32x")
_RECORD = struct.Struct("<QQIIII")


def verify_mhc_payload(root: Path) -> dict[str, object]:
    """Check fixed family/source/schema, build closure and every packaged artifact.

    Args:
        root: Payload produced by the isolated MHC builder.
    """
    root = Path(root).resolve()
    data = json.loads((root / "manifest.json").read_text())
    family_abi("mhc").verify_native(NativeManifest(**{
        field.name: data[field.name] for field in fields(NativeManifest)}))
    expected = hashlib.sha256(json.dumps({"build": data["build"], "artifacts": data["artifacts"]},
                                         sort_keys=True).encode()).hexdigest()
    if data["vendor"] != _VENDOR or data["build_fingerprint"] != expected:
        raise ValueError("MHC vendor/build fingerprint mismatch")
    required = {f"vendors/{_VENDOR}/op_api/lib/libcust_opapi.so", "set_env.bash",
                "framework/torch/libhyper_parallel_mega_mhc_torch.so"}
    if not required.issubset(data["artifacts"]):
        raise ValueError("MHC manifest lacks required native artifacts")
    for relative, digest in data["artifacts"].items():
        path = (root / relative).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError(f"MHC artifact missing, escaping or corrupted: {relative}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != set(data["artifacts"]) | {"manifest.json"}:
        raise ValueError("MHC payload contains unrecorded artifacts")
    return data


def _load(root, manifest):
    vendor = root / "vendors" / _VENDOR
    paths = tuple(Path(value).resolve() for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
                  if value)
    if paths != (vendor,):
        raise ValueError("MHC requires an isolated custom OPP process activated before Torch import")
    build = manifest["build"]
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "")).resolve()
    if str(cann) != build["cann_root"] or (cann / "opp/version.info").read_text() != build["cann_version"]:
        raise ValueError("MHC CANN identity mismatch")
    backend = importlib.import_module("torch_npu")
    if (torch.__version__ != build["torch"] or backend.__version__ != build["torch_npu"]
            or torch.compiled_with_cxx11_abi() != build["torch_cxx11_abi"]):
        raise ValueError("MHC framework/C++ ABI mismatch")
    if _LOADED:
        if _LOADED != {root: manifest["build_fingerprint"]}:
            raise ValueError("A different MHC payload is already loaded")
        return
    ctypes.CDLL(str(vendor / "op_api/lib/libcust_opapi.so"), mode=ctypes.RTLD_LOCAL)
    torch.ops.load_library(str(root / "framework/torch/libhyper_parallel_mega_mhc_torch.so"))
    _LOADED[root] = manifest["build_fingerprint"]


class MhcProfile:
    """One invocation's cycle buffer, completion event and stage/source identities."""

    def __init__(self, image: MhcRuntimeImage, direction: str, buffer: torch.Tensor, event: Any,
                 sources: dict[str, object]) -> None:
        """Keep invocation-owned device records alive until explicit inspection."""
        self.image, self.direction, self.buffer, self.event = image, direction, buffer, event
        self.sources = sources

    def records(self) -> tuple[dict[str, object], ...]:
        """Wait for this invocation and decode actual AIC/AIV cycle records."""
        self.event.synchronize()
        wire = self.buffer.cpu().numpy().tobytes()
        cube, vector = struct.unpack_from("<II", self.image.normal, 24)
        result = []
        offset = 0
        for core, slots, capacity in ((1, 24, cube), (2, 48, vector)):
            for worker in range(slots):
                _, count, dropped, kind, block, recorded_capacity, _ = _HEADER.unpack_from(wire, offset)
                if dropped or count > capacity:
                    raise ValueError("MHC profiling dropped or exceeded allocated records")
                if count and (kind != core or block != worker or recorded_capacity != capacity):
                    raise ValueError("MHC profile core identity mismatch")
                for index in range(count):
                    start, end, desc, task, stage, owner = _RECORD.unpack_from(wire, offset + 64 + index * 32)
                    if end < start or not start or task >= len(self.image.tasks):
                        raise ValueError("MHC profile cycle/task identity mismatch")
                    name = next(name for name, first, count in self.image.stages if first <= task < first + count)
                    result.append({"direction": self.direction, "stage": name, "core": core, "worker": worker,
                                   "task": task, "desc": desc, "stage_index": stage, "owner": owner,
                                   "start_cycle": start, "end_cycle": end, "source": str(self.sources[name])})
                offset += 64 + capacity * 32
        return tuple(result)


@dataclass(frozen=True)
class _DescriptorResources:
    configs: dict[str, tuple[torch.Tensor, ...]]
    initialized: Any


class MhcExecutable:
    """Bind immutable descriptors; each call owns counters, caches and optional profile buffers."""

    def __init__(self, plan: MhcKernelPlan, device: str, payload_root: Path | None) -> None:
        """Validate payload/topology and initialize stream-safe descriptor tensors."""
        self.plan = plan
        self.root = Path(payload_root or os.environ.get(
            "HP_MHC_PAYLOAD_ROOT", Path(__file__).resolve().parents[4] / "build/native/mhc/payload")).resolve()
        self.manifest = verify_mhc_payload(self.root)
        _load(self.root, self.manifest)
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("MHC executable requires an NPU device")
        if self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        limits = torch.npu.get_device_limit(self.device.index)
        if (self.device.type != "npu" or limits["cube_core_num"] != plan.spec.num_cube_cores
                or limits["vector_core_num"] != plan.spec.num_vector_cores):
            raise ValueError("MHC planned cores do not match the actual mixed-kernel topology")
        self._closed = False
        self._profiles = []
        configs = {direction: tuple(_bytes_tensor(data, self.device) for data in (image.normal, image.profiled))
                         for direction, image in (("forward", plan.forward), ("backward", plan.backward))
                         if image is not None}
        initialized = torch.npu.Event()
        initialized.record(torch.npu.current_stream(self.device))
        self._bindings = _DescriptorResources(configs, initialized)

    def close(self) -> None:
        """Reject future forwards while outstanding autograd invocations retain their resources."""
        self._closed = True
        self._bindings = None

    def take_profiles(self) -> tuple[MhcProfile, ...]:
        """Transfer completed-or-pending invocation profiles to the caller."""
        profiles, self._profiles = tuple(self._profiles), []
        return profiles

    def __call__(self, *inputs: torch.Tensor, profile: bool = False) -> tuple[torch.Tensor, ...]:
        """Run the five-output shifted boundary in semantic argument order.

        Args:
            *inputs: Residual, previous output/pre/post/residual mix, phi, alpha, bias and norm weight.
            profile: Record actual forward/backward cycle intervals with private buffers.
        """
        if not isinstance(profile, bool):
            raise TypeError("MHC profile must be a boolean")
        if self._closed:
            raise RuntimeError("MHC executable is closed")
        _validate_inputs(self.plan, self.device, inputs)
        needs_grad = torch.is_grad_enabled() and any(value.requires_grad for value in inputs)
        if needs_grad and self.plan.backward is None:
            raise ValueError("MHC plan has no backward recipe")
        if needs_grad:
            return _MhcFunction.apply(self, profile, *inputs)
        outputs, _ = self._forward(inputs, profile, False)
        return outputs

    def _resources(self, direction, profile, resources=None):
        image = self.plan.backward if direction == "backward" else self.plan.forward
        stream = torch.npu.current_stream(self.device)
        resources = resources or self._bindings
        stream.wait_event(resources.initialized)
        config = resources.configs[direction][int(profile)]
        config.record_stream(stream)
        event_capacity = struct.unpack_from("<I", image.normal, 12)[0]
        counters = torch.zeros(event_capacity * 4, dtype=torch.uint8, device=self.device)
        cube, vector = struct.unpack_from("<II", image.normal, 24)
        size = 24 * (64 + cube * 32) + 48 * (64 + vector * 32) if profile else 0
        buffer = torch.zeros(size, dtype=torch.uint8, device=self.device)
        return image, config, counters, buffer

    def _complete(self, image, direction, buffer, profile):
        if profile:
            event = torch.npu.Event()
            event.record(torch.npu.current_stream(self.device))
            sources = (self.plan.recipe.stage_sources() if direction == "forward"
                       else self.plan.recipe.backward_sources())
            self._profiles.append(MhcProfile(image, direction, buffer, event, sources))

    def _forward(self, inputs, profile, need_cache):
        residual, previous_output, pre, post, matrix, phi, alpha, bias, weight = inputs
        shape, rows, hidden = residual.shape[:-2], self.plan.spec.token_count, self.plan.spec.hidden_size
        image, config, counters, buffer = self._resources("forward", profile)
        values = (previous_output.reshape(1, rows, hidden).contiguous(),
                  residual.reshape(1, rows, 4, hidden).contiguous(),
                  pre.reshape(1, rows, 4).contiguous(), post.reshape(1, rows, 4).contiguous(),
                  matrix.reshape(1, rows, 4, 4).contiguous(), phi, alpha, bias, weight)
        outputs = _allocate_forward(residual, rows, hidden)
        _record(values)
        native_calls.mega_mhc(*values, config, counters, buffer, *outputs,
                                          self.plan.recipe.hc_eps, self.plan.recipe.norm_eps,
                                          self.plan.recipe.num_iters, need_cache)
        self._complete(image, "forward", buffer, profile)
        public = (outputs[0].reshape(*shape, 4, hidden), outputs[1].reshape(*shape, 4),
                  outputs[2].reshape(*shape, 4), outputs[3].reshape(*shape, 4, 4),
                  outputs[4].reshape(*shape, hidden))
        return public, outputs[5:]

    def _backward(self, inputs, public, cache, grads, profile, resources):
        residual, previous_output, pre, post, matrix, phi, alpha, bias, weight = inputs
        rows, hidden, shape = self.plan.spec.token_count, self.plan.spec.hidden_size, residual.shape[:-2]
        grads = tuple(torch.zeros_like(value) if grad is None else grad.contiguous()
                      for value, grad in zip(public, grads))
        updated = public[0].reshape(1, rows, 4, hidden).contiguous()
        image, config, counters, buffer = self._resources("backward", profile, resources)
        values = (grads[4].reshape(1, rows, hidden), grads[2].reshape(1, rows, 4),
                  grads[3].reshape(1, rows, 4, 4), updated, phi, alpha, bias,
                  pre.reshape(1, rows, 4).contiguous(), *cache[:4], grads[1].reshape(1, rows, 4),
                  cache[4], cache[5], weight, grads[0].reshape(1, rows, 4, hidden),
                  residual.reshape(1, rows, 4, hidden).contiguous(),
                  previous_output.reshape(1, rows, hidden).contiguous(), post.reshape(1, rows, 4).contiguous(),
                  matrix.reshape(1, rows, 4, 4).contiguous())
        outputs = tuple(torch.empty_like(value) for value in (
            values[17], phi, alpha, bias, values[18], values[7], values[19], values[20]))
        outputs += (torch.empty_like(weight, dtype=torch.float32),)
        _record(values)
        native_calls.mega_mhc_grad(*values, config, counters, buffer, *outputs, self.plan.recipe.hc_eps)
        self._complete(image, "backward", buffer, profile)
        return (outputs[0].reshape(*shape, 4, hidden), outputs[4].reshape(*shape, hidden),
                outputs[5].reshape(pre.shape), outputs[6].reshape(post.shape), outputs[7].reshape(matrix.shape),
                outputs[1], outputs[2], outputs[3], outputs[8])


def _bytes_tensor(data, device):
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone().to(device)


def _record(values):
    stream = torch.npu.current_stream(values[0].device)
    for value in values:
        value.record_stream(stream)


def _allocate_forward(residual, rows, hidden):
    def _empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=residual.device)
    return (torch.empty((1, rows, 4, hidden), dtype=torch.bfloat16, device=residual.device),
            _empty((1, rows, 4), torch.float32), _empty((1, rows, 4), torch.float32),
            _empty((1, rows, 16), torch.float32), _empty((1, rows, hidden), torch.bfloat16),
            _empty((1, rows, 24), torch.float32), _empty((1, rows, 1), torch.float32),
            _empty((40, 1, rows, 4), torch.float32), _empty((40, 1, rows, 4, 4), torch.float32),
            _empty((1, rows, hidden), torch.bfloat16), _empty((1, rows, 1), torch.float32))


def _validate_inputs(plan, device, inputs):
    if len(inputs) != 9:
        raise ValueError("MHC execution requires exactly nine tensors")
    residual = inputs[0]
    if residual.ndim < 3:
        raise ValueError("MHC residual must end in [4,H]")
    shape, hidden = tuple(residual.shape[:-2]), plan.spec.hidden_size
    expected = ((*shape, 4, hidden), (*shape, hidden), (*shape, 4), (*shape, 4), (*shape, 4, 4),
                (24, 4 * hidden), (3,), (24,), (hidden,))
    dtypes = (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32, torch.float32,
              torch.float32, torch.float32, torch.float32, torch.bfloat16)
    if residual.numel() != plan.spec.token_count * 4 * hidden:
        raise ValueError("MHC token shape does not match the compiled plan")
    for value, dimensions, dtype in zip(inputs, expected, dtypes):
        if (value.device != device or tuple(value.shape) != dimensions or value.dtype != dtype
                or not value.is_contiguous()):
            raise ValueError("MHC input device/dtype/shape/layout does not match the native contract")


class _MhcFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, executable: MhcExecutable, profile: bool,
                *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Retain the exact invocation's native caches and descriptor ownership.

        Args:
            ctx: Autograd invocation context.
            executable: Validated family-specific binding.
            profile: Whether to retain native cycle records.
            *inputs: Nine tensors in semantic argument order.
        """
        ctx.resources = executable._bindings
        public, cache = executable._forward(inputs, profile, True)
        ctx.save_for_backward(*inputs, *public, *cache)
        ctx.executable, ctx.profile = executable, profile
        ctx.set_materialize_grads(False)
        return public

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *grads: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        """Use the original backward recipe and compose with direct autograd gradients.

        Args:
            ctx: Context retaining this invocation's resources and caches.
            *grads: Five optional upstream gradients in public output order.
        """
        saved = ctx.saved_tensors
        gradients = ctx.executable._backward(saved[:9], saved[9:14], saved[14:], grads, ctx.profile, ctx.resources)
        return (None, None, *(gradient if needed else None for gradient, needed in zip(
            gradients, ctx.needs_input_grad[2:])))
