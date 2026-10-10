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
"""Resident forward candidate with invocation-owned state and explicit native VJP."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.core.multicore._build.build_dense import build_dense_payload
from hyper_parallel.core.multicore._build.build_dense_tile import (
    build_dense_tile_payload,
    dense_tile_build_identity,
    verify_dense_tile_payload,
)
from hyper_parallel.core.multicore.backends.dense_codegen import saved_value_ids
from hyper_parallel.core.multicore.backends.dense_tile import DenseSdkTiler, bind_dense_tiles
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy, compile_dense_tiles, simulate_dense_tiles
from hyper_parallel.core.multicore.runtime.dense import DenseExecutable, DenseKernelPlan
from hyper_parallel.core.multicore.runtime.dense_compiled import CompiledDenseExecutable
from hyper_parallel.core.multicore.runtime.dense_scratch import shared_dense_scratch

_LOADED: dict[str, str] = {}


def _default_policy(plan, hardware_workers):
    row_inputs = [value for value in plan.ir.inputs if value.name == "x"]
    if len(row_inputs) != 1:
        raise ValueError("Dense row input must identify exactly one semantic tensor parameter")
    rows = dict(plan.value_types)[row_inputs[0].id].shape[0]
    return DenseTilePolicy(cube_workers=min(24, hardware_workers, max(1, (rows + 63) // 64)))


def _load(manifest, identity):
    data = verify_dense_tile_payload(manifest, identity)
    library = (manifest.parent / data["library"]).resolve()
    fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    namespace = data["namespace"]
    if namespace in _LOADED and _LOADED[namespace] != fingerprint:
        raise ValueError("A different resident dense payload owns this operator namespace")
    if (namespace not in _LOADED and getattr(getattr(torch.ops, namespace), "launch", None) is not None
            and str(library) not in torch.ops.loaded_libraries):
        raise ValueError("An unowned resident dense operator namespace is already registered")
    torch.ops.load_library(str(library))
    _LOADED[namespace] = fingerprint
    return data, getattr(torch.ops, namespace)


class ResidentDenseExecutable(DenseExecutable):
    """Experimental mixed AIC/AIV forward; native provider calls implement backward.

    The forward executes all admitted row-local primitives in one device launch.
    Backward still uses the independent, generated current-stream VJP. Each call
    owns its intermediates; stream-isolated scratch is reused and parameters are borrowed.
    No device correctness or speedup is implied by successful native compilation.
    """

    def __init__(self, plan: DenseKernelPlan, device: torch.device, *, cann_root: Path, cache_root: Path,
                 soc: str, policy: DenseTilePolicy | None = None) -> None:
        """Bind sealed kernels, exact SDK tilings and separately owned reverse mode.

        Args:
            plan: Canonical BF16 dense SSA graph.
            device: Actual NPU device; CPU references use DenseExecutable directly.
            cann_root: Selected standalone CANN SDK.
            cache_root: Writable caller-owned cache, outside native vendor installs.
            soc: Explicit SoC, checked again at first native registration.
            policy: Optional row/prefetch policy with worker count within SDK limits.
        """
        if device.type != "npu":
            raise ValueError("Resident dense tile execution requires an NPU device")
        super().__init__(plan, device)
        cache_root = Path(cache_root).resolve()
        manifest = build_dense_tile_payload(cann_root, cache_root / "resident", soc)
        identity = dense_tile_build_identity(cann_root, soc)
        data, self.native_ops = _load(manifest, identity)
        tiler = DenseSdkTiler(manifest.parent / data["tiling_library"], soc)
        if policy is None:
            policy = _default_policy(plan, tiler.cube_workers)
        self.tile_plan = compile_dense_tiles(plan, policy)
        simulate_dense_tiles(self.tile_plan)
        self.binding = bind_dense_tiles(self.tile_plan)
        bank = tiler.bank(self.tile_plan, self.binding)
        self.workspace_bytes = tiler.workspace_bytes
        self.scratch = shared_dense_scratch(self.device, len(self.binding.value_ids),
                                           self.tile_plan.event_count * 32, self.workspace_bytes, self)
        self.config = self._bytes(self.binding.config)
        self.tilings = self._bytes(bank)
        self.ones = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device=self.device)
        self.ready = torch.get_device_module(self.device).Event()
        self.ready.record(torch.get_device_module(self.device).current_stream(self.device))
        self.saved_ids = saved_value_ids(plan)
        self.vjp = CompiledDenseExecutable(plan, self.device, build_dense_payload(plan, cache_root / "vjp"))
        self.native_identity = {**data, "tile_plan": self.tile_plan.export_manifest(),
                                "tiling": tiler.export_manifest(self.binding, bank),
                                "scratch": {"reuse": "device_shared_per_stream", "max_cached_streams": 2,
                                            "saved_values": "invocation_owned"},
                                "backward_execution_mode": "native_host_stream_adapter",
                                "backward_payload": self.vjp.native_identity}

    def _bytes(self, value):
        return torch.frombuffer(bytearray(value), dtype=torch.uint8).clone().to(self.device)

    def close(self) -> None:
        """Release cached kernel scratch while preserving each pending invocation's backward values."""
        super().close()
        self.scratch.release_owner(self)

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Launch one private invocation without caching current weight addresses.

        Args:
            *inputs: Dense program inputs in semantic parameter order.
        """
        self._bind_inputs(inputs)
        return _ResidentDenseFunction.apply(self, *inputs)

    def forward_values(self, inputs: tuple[torch.Tensor, ...]) -> dict[int, torch.Tensor]:
        """Allocate private values and enqueue the resident kernel using stream-owned scratch.

        Args:
            inputs: Exact contiguous caller-owned tokens and weights for this call.
        """
        values = self._bind_inputs(inputs)
        types = dict(self.plan.value_types)
        for buffer in self.plan.buffers:
            values[buffer.value_id] = torch.empty(types[buffer.value_id].shape, dtype=torch.bfloat16,
                                                  device=self.device)
        if self.tile_plan.rows:
            self._launch(values)
        return values

    def _launch(self, values):
        stream = torch.get_device_module(self.device).current_stream(self.device)
        stream.wait_event(self.ready)
        tensors = [values[value_id] for value_id in self.binding.value_ids]
        inputs = [values[value.id] for value in self.plan.ir.inputs]
        outputs = [values[buffer.value_id] for buffer in self.plan.buffers]
        with self.scratch.lease(stream, pointers=len(tensors),
                                event_elements=self.tile_plan.event_count * 32) as scratch:
            scratch.prepare(tuple(tensor.data_ptr() for tensor in tensors))
            self.native_ops.launch.default(inputs, outputs, scratch.pointers, self.config, self.tilings,
                                           scratch.events, self.ones, scratch.workspace, scratch.overflow,
                                           self.tile_plan.policy.cube_workers)


class _ResidentDenseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, executable: ResidentDenseExecutable,
                *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Save the actual resident intermediates used by this invocation's VJP.

        Args:
            ctx: Private autograd invocation state.
            executable: Sealed resident candidate plus generated native VJP.
            *inputs: Exact caller-owned input and parameter tensors.
        """
        ctx.executable, ctx.input_count = executable, len(inputs)
        ctx.set_materialize_grads(False)
        values = executable.forward_values(inputs)
        ctx.save_for_backward(*inputs, *(values[value_id] for value_id in executable.saved_ids))
        outputs = tuple(values[value.id] for value in executable.plan.ir.outputs)
        return outputs if executable.plan.ir.returns_tuple else outputs[0]

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *output_grads: torch.Tensor | None) -> tuple[object, ...]:
        """Apply independent provider VJPs to the exact resident forward cache.

        Args:
            ctx: Retained forward invocation, valid even after module close.
            *output_grads: Public output cotangents, including unused tuple outputs.
        """
        tensors = ctx.saved_tensors
        gradients = ctx.executable.vjp.native_ops.backward.default(
            list(tensors[:ctx.input_count]), list(tensors[ctx.input_count:]), list(output_grads))
        if len(gradients) != ctx.input_count:
            raise RuntimeError("Resident dense VJP returned an invalid input-gradient contract")
        return (None, *(gradient if needed else None for gradient, needed in
                        zip(gradients, ctx.needs_input_grad[1:])))
