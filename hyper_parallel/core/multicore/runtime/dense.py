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
"""Dense task plans with invocation-owned values and caller-owned parameters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from hyper_parallel.core.multicore.backends.providers import NativeProvider
from hyper_parallel.core.multicore.ir.program import ProgramIR, Value
from hyper_parallel.core.multicore.language.types import TensorType
from hyper_parallel.core.multicore.primitives.gate import TORCH_DTYPES
from hyper_parallel.core.multicore.primitives.registry import REGISTRY
from hyper_parallel.core.multicore.runtime.dense_native import NativeSwiGLU


@dataclass(frozen=True)
class DenseSpec:
    """Static dimensions without expert topology, routes or device pointers."""

    dimensions: Mapping[str, int]

    def __post_init__(self) -> None:
        values = dict(self.dimensions)
        if any(not isinstance(key, str) or not key.isidentifier() or type(value) not in (int,) or value < 0
               for key, value in values.items()):
            raise ValueError("Dense dimensions must map symbolic identifiers to nonnegative integers")
        object.__setattr__(self, "dimensions", MappingProxyType(values))

    def bind(self, tensor_type: TensorType) -> TensorType:
        """Resolve a logical shape while retaining its dtype and layout.

        Args:
            tensor_type: Ordinary dense semantic tensor type.
        """
        try:
            shape = tuple(self.dimensions[dim] if isinstance(dim, str) else dim for dim in tensor_type.shape)
        except KeyError as error:
            raise ValueError(f"Unbound dense symbolic dimension: {error.args[0]}") from error
        return TensorType(tensor_type.dtype, shape, tensor_type.layout)


@dataclass(frozen=True)
class DenseTask:
    """One primitive invocation with SSA-derived producer dependencies."""

    index: int
    provider: NativeProvider
    dependencies: tuple[int, ...]
    reads: tuple[int, ...]
    writes: tuple[int, ...]

    def export_manifest(self) -> dict[str, object]:
        """Export logical scheduling metadata without device task IDs."""
        return {"index": self.index, "provider": self.provider.export_manifest(),
                "dependencies": self.dependencies, "reads": self.reads, "writes": self.writes}


@dataclass(frozen=True)
class DenseBuffer:
    """An intermediate's conservative live interval and backward retention."""

    value_id: int
    first_task: int
    last_task: int
    size_bytes: int
    saved_for_backward: bool


@dataclass(frozen=True)
class DenseKernelPlan:
    """Executable primitive DAG; fused implementations have separate admission rules."""

    ir: ProgramIR
    spec: DenseSpec
    tasks: tuple[DenseTask, ...]
    value_types: tuple[tuple[int, TensorType], ...]
    buffers: tuple[DenseBuffer, ...]

    def export_manifest(self) -> dict[str, object]:
        """Describe a host-stream plan without claiming resident-worker code generation."""
        return {"family": "dense", "format_version": 1, "native_status": "unbound",
                "execution_mode": "current_stream_primitives", "dimensions": dict(self.spec.dimensions),
                "numeric_policy": self.ir.numeric_policy, "tasks": [task.export_manifest() for task in self.tasks],
                "buffers": [{"value_id": buffer.value_id, "first_task": buffer.first_task,
                             "last_task": buffer.last_task, "size_bytes": buffer.size_bytes,
                             "saved_for_backward": buffer.saved_for_backward} for buffer in self.buffers]}

    def explain(self) -> str:
        """Describe bindings, live intervals and original source locations."""
        return json.dumps({"manifest": self.export_manifest(), "semantic_ir": json.loads(self.ir.dump())},
                          sort_keys=True, indent=2)

    def materialize(self, device: torch.device | str) -> DenseExecutable:
        """Bind a reusable plan without retaining invocation tensors or parameters.

        Args:
            device: CPU for explicit references, or an activated Ascend NPU.
        """
        return DenseExecutable(self, torch.device(device))


class DenseExecutable:
    """Execute trusted providers with each forward's private value table."""

    def __init__(self, plan: DenseKernelPlan, device: torch.device) -> None:
        """Select execution device; native operators remain stream ordered.

        Args:
            plan: Validated static primitive DAG.
            device: CPU reference or native NPU device.
        """
        if device.type not in ("cpu", "npu"):
            raise ValueError("Dense Ascend plans support CPU references or NPU execution")
        self.plan = plan
        if device.type == "cpu":
            device = torch.device("cpu")
        elif device.index is None:
            device = torch.device("npu", torch.get_device_module(device).current_device())
        self.device = device
        self._closed = False

    def close(self) -> None:
        """Reject future forwards without changing saved autograd state."""
        self._closed = True

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run one invocation; autograd retains this invocation's exact values.

        Args:
            *inputs: Input tensors in semantic program parameter order.
        """
        values = self._bind_inputs(inputs)
        for task, operation in zip(self.plan.tasks, self.plan.ir.operations):
            arguments = {key: values[value.id] if isinstance(value, Value) else value
                         for key, value in operation.arguments}
            result = self._invoke(task.provider, arguments)
            outputs = (result,) if len(operation.outputs) == 1 else result
            for output, tensor in zip(operation.outputs, outputs):
                values[output.id] = tensor
            for buffer in self.plan.buffers:
                if buffer.last_task == task.index and buffer.value_id not in task.writes:
                    values.pop(buffer.value_id, None)
        results = tuple(values[value.id] for value in self.plan.ir.outputs)
        return results if self.plan.ir.returns_tuple else results[0]

    def _bind_inputs(self, inputs: tuple[torch.Tensor, ...]) -> dict[int, torch.Tensor]:
        if self._closed:
            raise RuntimeError("Dense executable is closed")
        if len(inputs) != len(self.plan.ir.inputs):
            raise ValueError("Dense invocation input count does not match its program")
        types = dict(self.plan.value_types)
        values = {}
        for logical, tensor in zip(self.plan.ir.inputs, inputs):
            expected = types[logical.id]
            if (not isinstance(tensor, torch.Tensor) or tensor.device != self.device
                    or tensor.dtype != TORCH_DTYPES[expected.dtype] or tuple(tensor.shape) != expected.shape
                    or not tensor.is_contiguous()):
                raise ValueError(f"Dense input {logical.name} does not match its device/dtype/shape/layout contract")
            values[logical.id] = tensor
        return values

    def _invoke(self, provider: NativeProvider, arguments: dict[str, object]) -> torch.Tensor:
        schema = REGISTRY.schema(provider.logical_name, provider.version)
        bound = schema.signature.bind(**arguments)
        if self.device.type == "cpu":
            return schema.reference(*bound.args, **bound.kwargs)
        if provider.implementation == "aten.mm.default":
            left, right = arguments["left"], arguments["right"]
            return torch.mm(left.t() if arguments["transpose_left"] else left,
                            right.t() if arguments["transpose_right"] else right)
        if provider.implementation == "npu.npu_swiglu.default":
            return NativeSwiGLU.apply(arguments["packed"])
        raise ValueError(f"No trusted dense executor for {provider.implementation}")
