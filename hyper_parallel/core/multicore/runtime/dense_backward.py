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
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""显式反向 SSA 的独立设备执行入口；完整 FFN 的未绑定能力严格拒绝。"""

from __future__ import annotations

from pathlib import Path

import torch

from hyper_parallel.core.multicore._build.build_dense_tile import build_dense_tile_payload, dense_tile_build_identity
from hyper_parallel.core.multicore.backends.dense_backward import (
    HEADER,
    DenseBackwardSdkTiler,
    bind_dense_backward_tiles,
)
from hyper_parallel.core.multicore.backends.dense_codegen import saved_value_ids
from hyper_parallel.core.multicore.compiler.dense_backward import compile_dense_backward
from hyper_parallel.core.multicore.compiler.dense_backward_tiles import compile_dense_backward_tiles
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy
from hyper_parallel.core.multicore.runtime.dense import DenseKernelPlan
from hyper_parallel.core.multicore.runtime.dense_scratch import shared_dense_scratch
from hyper_parallel.core.multicore.runtime.dense_tile import _load


class ResidentDenseBackward:
    """一次设备启动执行已准入的完整梯度子图，不拼接主机 VJP 绕过能力检查。"""

    def __init__(self, plan: DenseKernelPlan, device: torch.device, *, cann_root: Path, cache_root: Path,
                 soc: str, policy: DenseTilePolicy | None = None,
                 input_mask: tuple[bool, ...] | None = None, output_mask: tuple[bool, ...] | None = None) -> None:
        """封存本次输入需求及余切存在性，所有能力通过之后才构建设备状态。

        Args:
            plan: 已准入的 BF16 前向 SSA。
            device: 本次调用的真实 NPU 设备。
            cann_root: 明确选定的 CANN SDK。
            cache_root: 调用方的可写原生缓存目录。
            soc: 明确的 Ascend910B 硬件型号。
            policy: 反向矩形域的行高和列分区；缺省 worker 数来自实际 SDK。
            input_mask: 调用方确实需要的输入梯度，缺省全部需要。
            output_mask: 本次余切的主机存在性，缺省全部存在。
        """
        if device.type != "npu":
            raise ValueError("Resident dense backward requires an NPU device")
        self.device, self.plan = device, plan
        self.backward = compile_dense_backward(plan)
        provisional = DenseTilePolicy(cube_workers=1) if policy is None else policy
        self.tile_plan = compile_dense_backward_tiles(self.backward, provisional, output_mask, input_mask)
        self.binding = bind_dense_backward_tiles(self.tile_plan)
        self.binding.admit_native()
        manifest = build_dense_tile_payload(cann_root, Path(cache_root) / "resident", soc)
        data, self.native_ops = _load(manifest, dense_tile_build_identity(cann_root, soc))
        tiler = DenseBackwardSdkTiler(manifest.parent / data["tiling_library"], soc)
        if policy is None:
            policy = DenseTilePolicy(cube_workers=min(24, tiler.sdk.cube_workers))
            self.tile_plan = compile_dense_backward_tiles(self.backward, policy, output_mask, input_mask)
            self.binding = bind_dense_backward_tiles(self.tile_plan)
        bank = tiler.bank(self.binding)
        self.config = self._bytes(self.binding.config)
        self.tilings = self._bytes(bank)
        self.ones = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device=device)
        self.ready = torch.get_device_module(device).Event()
        self.ready.record(torch.get_device_module(device).current_stream(device))
        self.scratch = shared_dense_scratch(device, max(1, len(self.binding.value_ids)),
                                           self.tile_plan.event_count * 32, tiler.sdk.workspace_bytes, self)
        self.saved_ids = saved_value_ids(plan)
        self.native_identity = {"payload": data, "execution_mode": "resident_backward_candidate",
                                "device_status": "unvalidated", "plan": self.tile_plan.export_manifest(),
                                "tiling": tiler.export_manifest(self.binding, bank)}

    def _bytes(self, value):
        if not value:
            return torch.empty(0, dtype=torch.uint8, device=self.device)
        return torch.frombuffer(bytearray(value), dtype=torch.uint8).clone().to(self.device)

    def _values(self, inputs, saved, output_grads):
        if (len(inputs) != len(self.plan.ir.inputs) or len(saved) != len(self.saved_ids)
                or tuple(value is not None for value in output_grads) != self.tile_plan.output_mask):
            raise ValueError("Resident backward invocation differs from its bound inputs/saved/cotangent contract")
        values = {value.id: tensor for value, tensor in zip(self.plan.ir.inputs, inputs)}
        values.update(zip(self.saved_ids, saved))
        values.update((seed.value.id, output_grads[seed.output_index]) for seed in self.backward.seeds
                      if output_grads[seed.output_index] is not None)
        types = dict(self.backward.value_types)
        seed_ids = {seed.value.id for seed in self.backward.seeds}
        for value_id in self.binding.value_ids:
            if value_id in self.binding.output_ids:
                values[value_id] = torch.empty(types[value_id].shape, dtype=torch.bfloat16, device=self.device)
            else:
                value = values[value_id]
                if (value.device != self.device or value.dtype != torch.bfloat16
                        or tuple(value.shape) != types[value_id].shape):
                    raise ValueError(f"Resident backward tensor {value_id} differs from its sealed type")
                if value_id in seed_ids:
                    values[value_id] = value.contiguous()
                elif not value.is_contiguous():
                    raise ValueError("Resident backward requires contiguous borrowed primal tensors")
        return values

    def __call__(self, inputs: tuple[torch.Tensor, ...], saved: tuple[torch.Tensor, ...],
                 output_grads: tuple[torch.Tensor | None, ...]) -> tuple[torch.Tensor | None, ...]:
        """借用实际保存张量及余切，所有中间梯度和返回梯度归本次调用所有。

        Args:
            inputs: 原始输入与实际权重，按前向参数顺序排列。
            saved: 此次前向真正保留的原始值，按生成适配器的顺序排列。
            output_grads: 各公共输出的实际余切，与封存的存在性一致。
        """
        values = self._values(inputs, saved, output_grads)
        if self.tile_plan.tasks:
            stream = torch.get_device_module(self.device).current_stream(self.device)
            stream.wait_event(self.ready)
            tensors = tuple(values[value_id] for value_id in self.binding.value_ids)
            produced = set(self.binding.output_ids)
            borrowed = [values[value_id] for value_id in self.binding.value_ids if value_id not in produced]
            outputs = [values[value_id] for value_id in self.binding.value_ids if value_id in produced]
            with self.scratch.lease(stream, pointers=len(tensors),
                                    event_elements=self.tile_plan.event_count * 32) as scratch:
                scratch.prepare(tuple(value.data_ptr() for value in tensors))
                header = HEADER.unpack_from(self.binding.config)
                self.native_ops.launch_backward.default(borrowed, outputs, scratch.pointers, self.config,
                                                        self.tilings, scratch.events, self.ones, scratch.workspace,
                                                        scratch.overflow, self.tile_plan.policy.cube_workers,
                                                        header[5], header[8], header[7])
        return tuple(values[value] if value is not None else None for value in self.binding.gradients)
