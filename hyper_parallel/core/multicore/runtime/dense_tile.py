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
"""常驻前向、真实保存状态及按完整梯度图选择的 autograd 反向。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.core.multicore._build.build_dense import build_dense_payload
from hyper_parallel.core.multicore._build.build_dense_tile import (
    build_dense_tile_payload,
    dense_tile_build_identity,
)
from hyper_parallel.core.multicore.backends.dense_codegen import saved_value_ids
from hyper_parallel.core.multicore.backends.dense_backward import HEADER as MATRIX_HEADER
from hyper_parallel.core.multicore.backends.dense_linear import DenseLinearSdkTiler, bind_dense_linear_tiles
from hyper_parallel.core.multicore.backends.dense_reference import DenseReferenceTiler
from hyper_parallel.core.multicore.backends.dense_tile import DenseSdkTiler, bind_dense_tiles
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy, compile_dense_tiles, simulate_dense_tiles
from hyper_parallel.core.multicore.runtime.dense import DenseExecutable, DenseKernelPlan
from hyper_parallel.core.multicore.runtime.dense_backward_router import DenseBackwardRouter
from hyper_parallel.core.multicore.runtime.dense_compiled import CompiledDenseExecutable
from hyper_parallel.core.multicore.runtime.dense_payload import load_dense_payload
from hyper_parallel.core.multicore.runtime.dense_saved import DenseSavedUse, shared_dense_saved
from hyper_parallel.core.multicore.runtime.dense_scratch import shared_dense_scratch

class ResidentDenseExecutable(DenseExecutable):
    """常驻前向及完整梯度图选择；未绑定能力的调用沿用原有完整主机 VJP。"""

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
        data, self.native_ops = load_dense_payload(manifest, identity)
        tiler = DenseSdkTiler(manifest.parent / data["tiling_library"], soc)
        tiler.reference = DenseReferenceTiler(cann_root, manifest.parent / data["reference_library"], soc,
                                             tiler.cube_workers, torch.are_deterministic_algorithms_enabled())
        if policy is None:
            policy = tiler.recommended_policy(plan)
        self.tile_plan = compile_dense_tiles(plan, policy)
        simulate_dense_tiles(self.tile_plan)
        self.linear_forward = any(operation.logical_name == "dense.linear" for operation in plan.ir.operations)
        if self.linear_forward:
            shared_tiler = DenseLinearSdkTiler(manifest.parent / data["tiling_library"], soc)
            shared_tiler.reference = tiler.reference
            tiler = shared_tiler
            self.binding = bind_dense_linear_tiles(self.tile_plan)
        else:
            self.binding = bind_dense_tiles(self.tile_plan)
        counts = MATRIX_HEADER.unpack_from(self.binding.config) if self.linear_forward else None
        self.launch_counts = (counts[5], counts[8], counts[7]) if counts is not None else ()
        bank = tiler.bank(self.tile_plan, self.binding)
        self.workspace_bytes = tiler.sdk.workspace_bytes if self.linear_forward else tiler.workspace_bytes
        self.scratch = shared_dense_scratch(self.device, len(self.binding.value_ids),
                                           self.tile_plan.event_count * 32, self.workspace_bytes, self)
        self.config = self._bytes(self.binding.config)
        self.tilings = self._bytes(bank)
        self.ones = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device=self.device)
        self.ready = torch.get_device_module(self.device).Event()
        self.ready.record(torch.get_device_module(self.device).current_stream(self.device))
        self.saved_ids = saved_value_ids(plan)
        self.vjp = CompiledDenseExecutable(plan, self.device, build_dense_payload(plan, cache_root / "vjp"))
        self.backward_router = DenseBackwardRouter(plan, self.device, self.vjp.native_ops.backward.default,
                                                   cann_root=cann_root, cache_root=cache_root, soc=soc, policy=policy)
        public_ids = {value.id for value in plan.ir.outputs}
        self.saved_shapes = {value_id: dict(plan.value_types)[value_id].shape for value_id in self.saved_ids
                             if value_id not in public_ids}
        self.saved_pool = shared_dense_saved(self.device, self.vjp.native_ops.storage_exclusive.default, self,
                                            view_factory=self.vjp.native_ops.saved_views.default)
        self.native_identity = {**data, "tile_plan": self.tile_plan.export_manifest(),
                                "tiling": tiler.export_manifest(self.binding, bank),
                                "scratch": {"reuse": "device_shared_per_stream", "max_cached_streams": 2,
                                            "saved_values": "storage_alias_guarded"},
                                "saved_buffers": {"reuse": "after_last_storage_alias", "max_cached_invocations": 2},
                                "backward_execution_mode": "not_executed",
                                "backward_runtime": self.backward_router.statistics(),
                                "backward_payload": self.vjp.native_identity}

    def _bytes(self, value):
        return torch.frombuffer(bytearray(value), dtype=torch.uint8).clone().to(self.device)

    def close(self) -> None:
        """释放前向与反向缓存所有权，保持已保存调用的待执行反向有效。"""
        super().close()
        self.backward_router.close()
        self.scratch.release_owner(self)
        self.saved_pool.release_owner(self)

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Launch one private invocation without caching current weight addresses.

        Args:
            *inputs: Dense program inputs in semantic parameter order.
        """
        self._bind_inputs(inputs)
        return _ResidentDenseFunction.apply(self, *inputs)

    def forward_values(self, inputs: tuple[torch.Tensor, ...]
                       ) -> tuple[dict[int, torch.Tensor], DenseSavedUse | None]:
        """借用已无存储别名的内部保存值，其余输出归本次调用所有。

        Args:
            inputs: Exact contiguous caller-owned tokens and weights for this call.
        """
        values = self._bind_inputs(inputs)
        types = dict(self.plan.value_types)
        use = None
        if self.saved_shapes and self.tile_plan.rows:
            stream = None if self.device.type == "cpu" else torch.get_device_module(
                self.device).current_stream(self.device)
            saved, use = self.saved_pool.acquire(self.saved_shapes, stream)
            values.update(saved)
        for buffer in self.plan.buffers:
            if buffer.value_id not in values:
                values[buffer.value_id] = torch.empty(types[buffer.value_id].shape, dtype=torch.bfloat16,
                                                      device=self.device)
        if self.tile_plan.rows:
            self._launch(values)
        return values, use

    def _launch(self, values):
        stream = torch.get_device_module(self.device).current_stream(self.device)
        stream.wait_event(self.ready)
        tensors = [values[value_id] for value_id in self.binding.value_ids]
        inputs = [values[value.id] for value in self.plan.ir.inputs]
        outputs = [values[buffer.value_id] for buffer in self.plan.buffers]
        with self.scratch.lease(stream, pointers=len(tensors),
                                event_elements=self.tile_plan.event_count * 32) as scratch:
            scratch.prepare(tuple(tensor.data_ptr() for tensor in tensors))
            launch = self.native_ops.launch_backward if self.linear_forward else self.native_ops.launch
            launch.default(inputs, outputs, scratch.pointers, self.config, self.tilings,
                           scratch.events, self.ones, scratch.workspace, scratch.overflow,
                           self.tile_plan.policy.cube_workers, *self.launch_counts)


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
        values, ctx.saved_use = executable.forward_values(inputs)
        ctx.save_for_backward(*inputs, *(values[value_id] for value_id in executable.saved_ids))
        outputs = tuple(values[value.id] for value in executable.plan.ir.outputs)
        return outputs if executable.plan.ir.returns_tuple else outputs[0]

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *output_grads: torch.Tensor | None) -> tuple[object, ...]:
        """根据真实需求选择完整设备反向或原主机 VJP，读取实际保存状态。

        Args:
            ctx: 即使模块已关闭，也仍然有效的前向调用状态。
            *output_grads: 实际输出余切，包括未使用公共输出的 None。
        """
        tensors = ctx.saved_tensors
        if ctx.saved_use is not None:
            device = ctx.executable.device
            stream = None if device.type == "cpu" else torch.get_device_module(device).current_stream(device)
            ctx.saved_use.consume(tensors[ctx.input_count:], stream)
        gradients = ctx.executable.backward_router(tuple(tensors[:ctx.input_count]),
                                                   tuple(tensors[ctx.input_count:]), tuple(output_grads),
                                                   tuple(ctx.needs_input_grad[1:]))
        runtime = ctx.executable.backward_router.statistics()
        ctx.executable.native_identity["backward_execution_mode"] = runtime["execution_mode"]
        ctx.executable.native_identity["backward_runtime"] = runtime
        if len(gradients) != ctx.input_count:
            raise RuntimeError("Resident dense VJP returned an invalid input-gradient contract")
        return (None, *(gradient if needed else None for gradient, needed in
                        zip(gradients, ctx.needs_input_grad[1:])))
