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
"""Training-compatible dense FFN using the shared AST primitive DAG."""

from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Callable

import torch
from torch import nn

from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from hyper_parallel.core.multicore.frontend.program import Program
from hyper_parallel.core.multicore.runtime.dense import DenseExecutable, DenseKernelPlan, DenseSpec
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig, materialize_dense


class MegaFFN(nn.Module):
    """Bias-free BF16 SwiGLU with packed gate/up columns and ordinary dense inputs.

    CPU calls execute explicit references. NPU calls execute generated native
    host-stream adapters; resident-worker fusion is a separate backend.
    External weights are borrowed for the invocation without caching or copying.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, *, create_parameters: bool = True,
                 program: Program | None = None, device: torch.device | str | None = None,
                 execution: DenseExecutionConfig | None = None) -> None:
        """Configure shape, semantic program and parameter ownership.

        Args:
            hidden_size: Input/output channel count.
            intermediate_size: Unpacked SwiGLU channel count.
            create_parameters: Own packed parameters; False borrows weights per call.
            program: Dense AST program accepting tokens, packed gate/up, down and intermediate_size.
            device: Initial parameter device; defaults to CPU for standard module placement.
            execution: Host-stream default or explicit experimental resident tile configuration.
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
        if create_parameters:
            self.gate_up = nn.Parameter(torch.empty(hidden_size, 2 * intermediate_size,
                                                   dtype=torch.bfloat16, device=device))
            self.down = nn.Parameter(torch.empty(intermediate_size, hidden_size,
                                                dtype=torch.bfloat16, device=device))
            self.reset_parameters()
        else:
            self.register_parameter("gate_up", None)
            self.register_parameter("down", None)

    def reset_parameters(self) -> None:
        """Initialize each packed projection with its actual input fan-in."""
        if self.gate_up is None:
            return
        with torch.no_grad():
            self.gate_up.uniform_(-1 / math.sqrt(self.hidden_size), 1 / math.sqrt(self.hidden_size))
            self.down.uniform_(-1 / math.sqrt(self.intermediate_size), 1 / math.sqrt(self.intermediate_size))

    def forward(self, x: torch.Tensor, *,
                weights: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Apply FFN to the final channel axis, preserving all leading dimensions.

        Args:
            x: Contiguous BF16 tokens with shape [..., hidden_size].
            weights: Invocation-owned (gate_up [H, 2I], down [I, H]) when parameters are external.
        """
        if self._closed:
            raise RuntimeError("MegaFFN is closed")
        if (not isinstance(x, torch.Tensor) or x.ndim < 2 or x.shape[-1] != self.hidden_size
                or x.dtype != torch.bfloat16 or not x.is_contiguous()):
            raise ValueError("MegaFFN input must be contiguous BF16 tokens with the configured final dimension")
        if self.gate_up is not None:
            if weights is not None:
                raise ValueError("External MegaFFN weights require create_parameters=False")
            weights = (self.gate_up, self.down)
        if not isinstance(weights, tuple) or len(weights) != 2:
            raise ValueError("MegaFFN requires a gate_up/down weight pair for this invocation")
        rows = x.numel() // self.hidden_size
        key = (rows, x.device)
        if key not in self._executables:
            self._executables[key] = materialize_dense(self._plan(rows), x.device, execution=self.execution)
        output = self._executables[key](x.view(rows, self.hidden_size), *weights)
        return output.view(x.shape)

    def _plan(self, rows: int) -> DenseKernelPlan:
        spec = DenseSpec({"T": rows, "H": self.hidden_size, "PackedI": 2 * self.intermediate_size,
                          "I": self.intermediate_size})
        plan = self.program.plan(spec, intermediate_size=self.intermediate_size)
        if len(plan.ir.inputs) != 3 or len(plan.ir.outputs) != 1 or plan.ir.returns_tuple:
            raise ValueError("MegaFFN programs must have three tensor inputs and one tensor output")
        if dict(plan.value_types)[plan.ir.outputs[0].id].shape != (rows, self.hidden_size):
            raise ValueError("MegaFFN program output must preserve the token/hidden shape")
        return plan

    def close(self) -> None:
        """Release cached plans and reject new forwards; pending backward remains valid."""
        self._closed = True
        for executable in self._executables.values():
            executable.close()
        self._executables.clear()

    def execution_manifest(self) -> list[dict[str, object]]:
        """Report bound plans and native payload identities without invocation tensors."""
        return [{"tokens": rows, "device": str(device), "plan": executable.plan.export_manifest(),
                 "native": deepcopy(getattr(executable, "native_identity", None))}
                for (rows, device), executable in self._executables.items()]

    def _apply(self, fn: Callable, recurse: bool = True) -> MegaFFN:
        for executable in self._executables.values():
            executable.close()
        self._executables.clear()
        return super()._apply(fn, recurse)
