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
"""通过共享 AST 原语 DAG 接入训练的稠密 FFN 模块。"""

from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Callable

import torch
from torch import nn

from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from hyper_parallel.core.multicore.frontend.program import Program
from hyper_parallel.core.multicore.language.types import ConstexprType
from hyper_parallel.core.multicore.runtime.dense import DenseExecutable, DenseKernelPlan, DenseSpec
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig, materialize_dense


class MegaFFN(nn.Module):
    """支持打包 gate/up 或三个独立 NK 投影的无偏置 BF16 FFN。

    CPU 调用执行显式参考；NPU 调用按配置使用生成的主机流调用或常驻 worker。
    外部权重只在本次调用中借用，不缓存调用张量或复制权重。
    """

    def __init__(self, hidden_size: int, intermediate_size: int, *, create_parameters: bool = True,
                 program: Program | None = None, device: torch.device | str | None = None,
                 execution: DenseExecutionConfig | None = None) -> None:
        """配置形状、语义程序及参数所有权。

        Args:
            hidden_size: 输入与输出的通道数。
            intermediate_size: 未打包的中间通道数。
            create_parameters: 是否持有参数；False 时每次调用借用外部权重。
            program: 声明 x/gate_up/down 或 x/gate/up/down 的稠密 AST 程序。
            device: 参数的初始设备；默认为 CPU，供普通模块迁移使用。
            execution: 默认主机流调用，或显式常驻分块配置。
        """
        super().__init__()
        if any(type(size) not in (int,) or size <= 0 for size in (hidden_size, intermediate_size)):
            raise ValueError("MegaFFN hidden and intermediate sizes must be positive integers")
        if type(create_parameters) not in (bool,):
            raise TypeError("MegaFFN create_parameters must be a boolean")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.program = dense_ffn if program is None else program
        self.execution = DenseExecutionConfig() if execution is None else execution
        if not isinstance(self.execution, DenseExecutionConfig):
            raise ValueError("MegaFFN execution requires a DenseExecutionConfig")
        if (not isinstance(self.program, Program) or self.program.schedule is None
                or getattr(self.program.schedule, "policy", None) != "dense_v1"):
            raise ValueError("MegaFFN requires a dense_v1 AST program")
        self._executables: dict[tuple[int, torch.device], DenseExecutable] = {}
        self._closed = False
        parameters = self.program.parameter_types()
        self._constants = {}
        if "intermediate_size" in parameters:
            if not isinstance(parameters["intermediate_size"], ConstexprType):
                raise ValueError("MegaFFN intermediate_size must be a declared constexpr")
            self._constants["intermediate_size"] = intermediate_size
        self.weight_shapes = self._weight_shapes()
        self.weight_names = ("gate_up", "down") if len(self.weight_shapes) == 2 else ("gate", "up", "down")
        packed_kn = ((hidden_size, 2 * intermediate_size), (intermediate_size, hidden_size))
        self.weight_layout = "kn" if self.weight_shapes == packed_kn else "nk"
        for name, shape in zip(self.weight_names, self.weight_shapes):
            value = nn.Parameter(torch.empty(shape, dtype=torch.bfloat16, device=device)) if create_parameters else None
            self.register_parameter(name, value)
        if create_parameters:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """按各投影的实际输入通道数初始化参数。"""
        if getattr(self, self.weight_names[0]) is None:
            return
        with torch.no_grad():
            for name in self.weight_names:
                fan_in = self.intermediate_size if name == "down" else self.hidden_size
                getattr(self, name).uniform_(-1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, x: torch.Tensor, *,
                weights: tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
        """沿末尾通道执行 FFN，保留所有前导维度。

        Args:
            x: 形状为 [..., hidden_size] 的连续 BF16 词元。
            weights: 外部权重，按 weight_names 顺序传入，形状必须符合程序输入注解。
        """
        if self._closed:
            raise RuntimeError("MegaFFN is closed")
        if (not isinstance(x, torch.Tensor) or x.ndim < 2 or x.shape[-1] != self.hidden_size
                or x.dtype != torch.bfloat16 or not x.is_contiguous()):
            raise ValueError("MegaFFN input must be contiguous BF16 tokens with the configured final dimension")
        if getattr(self, self.weight_names[0]) is not None:
            if weights is not None:
                raise ValueError("External MegaFFN weights require create_parameters=False")
            weights = tuple(getattr(self, name) for name in self.weight_names)
        if not isinstance(weights, tuple) or len(weights) != len(self.weight_names):
            kind = "gate_up/down weight pair" if len(self.weight_names) == 2 else "gate/up/down weight triple"
            raise ValueError(f"MegaFFN requires a {kind} in the declared {self.weight_names} order")
        rows = x.numel() // self.hidden_size
        key = (rows, x.device)
        if key not in self._executables:
            self._executables[key] = materialize_dense(self._plan(rows), x.device, execution=self.execution)
        output = self._executables[key](x.view(rows, self.hidden_size), *weights)
        return output.view(x.shape)

    def _plan(self, rows: int) -> DenseKernelPlan:
        spec = DenseSpec({"T": rows, "H": self.hidden_size, "PackedI": 2 * self.intermediate_size,
                          "I": self.intermediate_size})
        plan = self.program.plan(spec, **self._constants)
        if len(plan.ir.inputs) not in (3, 4) or len(plan.ir.outputs) != 1 or plan.ir.returns_tuple:
            raise ValueError("MegaFFN programs must have three or four tensor inputs and one tensor output")
        if dict(plan.value_types)[plan.ir.outputs[0].id].shape != (rows, self.hidden_size):
            raise ValueError("MegaFFN program output must preserve the token/hidden shape")
        return plan

    def _weight_shapes(self) -> tuple[tuple[int, int], ...]:
        """从已校验的程序输入注解取得权重布局，不根据函数体名称选择实现。"""
        plan = self._plan(0)
        types = dict(plan.value_types)
        shapes = tuple(types[value.id].shape for value in plan.ir.inputs)
        expected = ((self.hidden_size, 2 * self.intermediate_size), (self.intermediate_size, self.hidden_size))
        alternate = ((2 * self.intermediate_size, self.hidden_size), (self.hidden_size, self.intermediate_size))
        separate = ((self.intermediate_size, self.hidden_size),) * 2 + ((self.hidden_size, self.intermediate_size),)
        names = tuple(value.name for value in plan.ir.inputs)
        valid_names = names == (("x", "gate_up", "down") if len(shapes) == 3 else ("x", "gate", "up", "down"))
        if not valid_names or shapes[0] != (0, self.hidden_size) or shapes[1:] not in (expected, alternate, separate):
            raise ValueError("MegaFFN input annotations must declare matching KN or NK gate/up and down weights")
        return shapes[1:]

    def close(self) -> None:
        """释放缓存计划并拒绝新前向；已保存的待执行反向仍然有效。"""
        self._closed = True
        for executable in self._executables.values():
            executable.close()
        self._executables.clear()

    def execution_manifest(self) -> list[dict[str, object]]:
        """报告绑定计划、原生产物和保存缓存统计，不保留调用张量。"""
        return [{"tokens": rows, "device": str(device), "plan": executable.plan.export_manifest(),
                 "native": deepcopy(getattr(executable, "native_identity", None)),
                 "saved_buffers": executable.saved_pool.statistics() if hasattr(executable, "saved_pool") else None}
                for (rows, device), executable in self._executables.items()]

    def _apply(self, fn: Callable, recurse: bool = True) -> MegaFFN:
        for executable in self._executables.values():
            executable.close()
        self._executables.clear()
        return super()._apply(fn, recurse)
