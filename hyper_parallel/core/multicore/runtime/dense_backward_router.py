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
"""按实际 autograd 需求选择完整设备梯度图，并明确记录未绑定能力。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable

import torch

from hyper_parallel.core.multicore.backends.dense_backward import bind_dense_backward_tiles
from hyper_parallel.core.multicore.compiler.dense_backward import compile_dense_backward
from hyper_parallel.core.multicore.compiler.dense_backward_tiles import compile_dense_backward_tiles
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy
from hyper_parallel.core.multicore.runtime.dense import DenseKernelPlan
from hyper_parallel.core.multicore.runtime.dense_backward import ResidentDenseBackward


@dataclass
class _Route:
    native: ResidentDenseBackward | None
    unbound: tuple[str, ...]
    uses: int = 0
    retired: bool = False


class DenseBackwardRouter:
    """每次调用只选择一种完整 VJP，拒绝拼接部分设备梯度和主机补算。"""

    def __init__(self, plan: DenseKernelPlan, device: torch.device, host_vjp: Callable, *,
                 cann_root: Path, cache_root: Path, soc: str, policy: DenseTilePolicy,
                 max_cached_variants: int = 16) -> None:
        """配置有界存在性缓存；缓存不保留输入、保存值、余切或梯度。

        Args:
            plan: 前向的规范 BF16 SSA。
            device: 已由前向归一化的实际执行设备。
            host_vjp: 原有完整主机 VJP，供未绑定能力的调用使用。
            cann_root: 前向已选择的 SDK。
            cache_root: 调用方原生缓存目录。
            soc: 前向实际硬件型号。
            policy: 前向已经准入的 worker 及行列分区。
            max_cached_variants: 可选梯度存在性组合的缓存上限。
        """
        if type(max_cached_variants) not in (int,) or max_cached_variants < 0:
            raise ValueError("Dense backward variant cache limit must be a nonnegative integer")
        self.plan, self.device, self.host_vjp = plan, device, host_vjp
        self.backward = compile_dense_backward(plan)
        self.options = {"cann_root": cann_root, "cache_root": cache_root, "soc": soc, "policy": policy}
        self.max_cached_variants = max_cached_variants
        self._routes = OrderedDict()
        self._lock = Lock()
        self._closed = False
        self._native_calls, self._host_calls = 0, 0

    @staticmethod
    def _release(route):
        if route.retired and not route.uses and route.native is not None:
            route.native.scratch.release_owner(route.native)

    def _acquire(self, input_mask, output_mask):
        if (len(input_mask) != len(self.backward.gradients)
                or any(type(value) not in (bool,) for value in input_mask)):
            raise ValueError("Dense gradient input mask must match the input positions")
        if (len(output_mask) != len(self.backward.seeds)
                or any(type(value) not in (bool,) for value in output_mask)):
            raise ValueError("Dense gradient output mask must match the return positions")
        key = input_mask, output_mask
        with self._lock:
            route = self._routes.get(key)
            if route is None:
                tiles = compile_dense_backward_tiles(self.backward, self.options["policy"], output_mask, input_mask)
                binding = bind_dense_backward_tiles(tiles)
                unbound = binding.unbound_capabilities()
                native = None if unbound else ResidentDenseBackward(
                    self.plan, self.device, **self.options, input_mask=input_mask, output_mask=output_mask)
                route = _Route(native, unbound)
                if self._closed or not self.max_cached_variants:
                    route.retired = True
                else:
                    self._routes[key] = route
                    if len(self._routes) > self.max_cached_variants:
                        _old_key, previous = self._routes.popitem(last=False)
                        previous.retired = True
                        self._release(previous)
            else:
                self._routes.move_to_end(key)
            route.uses += 1
            return route

    def close(self) -> None:
        """释放缓存所有权；活跃调用和未来的待执行反向仍使用实际保存张量。"""
        with self._lock:
            self._closed = True
            for route in self._routes.values():
                route.retired = True
                self._release(route)
            self._routes.clear()

    def __call__(self, inputs: tuple[torch.Tensor, ...], saved: tuple[torch.Tensor, ...],
                 output_grads: tuple[torch.Tensor | None, ...], input_mask: tuple[bool, ...]
                 ) -> tuple[torch.Tensor | None, ...]:
        """使用真实需求选择完整梯度图，不从设备读取控制数据。

        Args:
            inputs: 实际前向输入与借用权重。
            saved: 此次前向真正保存的张量。
            output_grads: autograd 传入的可选余切。
            input_mask: ctx.needs_input_grad 中各张量参数的真实需求。
        """
        route = self._acquire(input_mask, tuple(value is not None for value in output_grads))
        try:
            if route.native is None:
                result = self.host_vjp(list(inputs), list(saved), list(output_grads))
            else:
                result = route.native(inputs, saved, output_grads)
            if len(result) != len(self.plan.ir.inputs):
                raise RuntimeError("Dense backward returned an invalid input-gradient contract")
            with self._lock:
                if route.native is None:
                    self._host_calls += 1
                else:
                    self._native_calls += 1
            return tuple(gradient if needed else None for gradient, needed in zip(result, input_mask))
        finally:
            with self._lock:
                route.uses -= 1
                self._release(route)

    def statistics(self) -> dict[str, object]:
        """记录实际执行方式和原生产物；存在主机调用时不能视为完整设备反向。"""
        with self._lock:
            mode = ("mixed_invocations" if self._native_calls and self._host_calls else
                    "resident_dense_backward" if self._native_calls else
                    "native_host_stream_adapter" if self._host_calls else "not_executed")
            return {"execution_mode": mode, "native_calls": self._native_calls, "host_calls": self._host_calls,
                    "cached_variants": len(self._routes), "max_cached_variants": self.max_cached_variants,
                    "closed": self._closed,
                    "unbound_capabilities": sorted({capability for route in self._routes.values()
                                                    for capability in route.unbound}),
                    "native_payloads": [route.native.native_identity for route in self._routes.values()
                                        if route.native is not None]}
