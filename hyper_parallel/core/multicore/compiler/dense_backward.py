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
"""将已准入原语的显式反向 recipe 转成有序、带类型的梯度 SSA。"""

from __future__ import annotations

from dataclasses import dataclass

from hyper_parallel.core.multicore.ir.program import Operation, SourceSpan, Value
from hyper_parallel.core.multicore.ir.dense_matrix import DenseMatrixReference
from hyper_parallel.core.multicore.language.types import TensorType
from hyper_parallel.core.multicore.primitives.dense import linear, matmul, swiglu_dense
from hyper_parallel.core.multicore.primitives.dense_common import multiply_dense, silu_dense
from hyper_parallel.core.multicore.primitives.registry import REGISTRY
from hyper_parallel.core.multicore.runtime.dense import DenseKernelPlan


@dataclass(frozen=True)
class DenseGradientSeed:
    """按返回值位置借用可为空的余切，重复返回保留逐次累加顺序。"""

    output_index: int
    target: int
    value: Value


@dataclass(frozen=True)
class DenseGradientNode:
    """一次数值 recipe 或一次已舍入梯度的有序相加。"""

    index: int
    kind: str
    inputs: tuple[int, ...]
    output: Value
    dependencies: tuple[int, ...]
    operation: int | None
    target: int
    gradient_source: int | None
    primal_inputs: tuple[int, ...] = ()
    matrix: tuple[int, int, int, bool, bool] | None = None
    operand_transposes: tuple[tuple[bool, ...], tuple[bool, ...]] = ((), ())
    transpose_output: bool = False
    source: SourceSpan | None = None
    numeric_reference: DenseMatrixReference | None = None


@dataclass(frozen=True)
class DenseGradientContribution:
    """数值贡献与累加槽位绑定，供保留原始调用顺序的主机生成器使用。"""

    node: int
    operation: int
    source_target: int
    target: int


@dataclass(frozen=True)
class DenseBackwardPlan:
    """显式反向 SSA；设备准入和 worker 调度仍需分别完成。"""

    forward: DenseKernelPlan
    seeds: tuple[DenseGradientSeed, ...]
    nodes: tuple[DenseGradientNode, ...]
    contributions: tuple[DenseGradientContribution, ...]
    gradients: tuple[int | None, ...]
    value_types: tuple[tuple[int, TensorType], ...]

    def defined_values(self, output_mask: tuple[bool, ...]) -> frozenset[int]:
        """从余切的主机存在性推导可用梯度值，不读取任何设备数据。

        Args:
            output_mask: 与程序返回位置一一对应的精确布尔值。
        """
        if len(output_mask) != len(self.seeds) or any(type(value) not in (bool,) for value in output_mask):
            raise ValueError("Dense gradient output mask must match the return positions")
        defined = {seed.value.id for seed in self.seeds if output_mask[seed.output_index]}
        for node in self.nodes:
            active = any(value in defined for value in node.inputs) if node.kind == "gradient_add" else (
                node.gradient_source in defined)
            if active:
                defined.add(node.output.id)
        return frozenset(defined)

    def export_manifest(self) -> dict[str, object]:
        """记录数值顺序、绑定和收缩几何，不宣称设备反向已放行。"""
        return {"format_version": 1, "native_status": "unbound", "numeric_policy": self.forward.ir.numeric_policy,
                "execution_mode": "explicit_dense_backward_recipes",
                "seeds": [{"output_index": seed.output_index, "target": seed.target, "value": seed.value.id}
                          for seed in self.seeds],
                "nodes": [{"index": node.index, "kind": node.kind, "inputs": node.inputs,
                           "output": node.output.id, "shape": node.output.type.shape,
                           "dtype": node.output.type.dtype.value, "dependencies": node.dependencies,
                           "operation": node.operation, "target": node.target,
                           "gradient_source": node.gradient_source, "primal_inputs": node.primal_inputs,
                           "matrix": node.matrix, "operand_transposes": node.operand_transposes,
                           "transpose_output": node.transpose_output,
                           "numeric_reference": node.numeric_reference.export_manifest()
                           if node.numeric_reference is not None else None,
                           "source": str(node.source) if node.source is not None else None}
                          for node in self.nodes],
                "input_gradients": self.gradients,
                "accumulation": "round_each_contribution_before_ordered_add"}


class _Builder:
    def __init__(self, forward: DenseKernelPlan) -> None:
        """建立原始张量绑定、梯度版本和数值贡献列表。

        Args:
            forward: 已校验的稠密前向计划。
        """
        self.forward = forward
        self.types = dict(forward.value_types)
        self.next_id = max(self.types, default=-1) + 1
        self.latest, self.producers = {}, {}
        self.seeds, self.nodes, self.contributions = [], [], []

    def _value(self, target):
        value = Value(self.next_id, f"gradient_{target}_{self.next_id}", self.types[target])
        self.types[value.id] = value.type
        self.next_id += 1
        return value

    def _add(self, target, value, operation=None, source=None):
        if target in self.latest:
            inputs = (self.latest[target], value.id)
            merged = self._value(target)
            dependencies = tuple(sorted({self.producers[item] for item in inputs if item in self.producers}))
            self.nodes.append(DenseGradientNode(len(self.nodes), "gradient_add", inputs, merged, dependencies,
                                                operation, target, None, source=source))
            self.producers[merged.id] = len(self.nodes) - 1
            value = merged
        self.latest[target] = value.id

    def seed(self, index: int, output: Value) -> None:
        """按原始返回位置建立余切和可能的有序累加。

        Args:
            index: 返回值位置。
            output: 原始返回的语义值。
        """
        value = self._value(output.id)
        self.seeds.append(DenseGradientSeed(index, output.id, value))
        self._add(output.id, value)

    def contribution(self, index: int, operation: Operation, kind: str) -> None:
        """记录一个受支持 recipe 的数值贡献及后续累加。

        Args:
            index: 原始操作位置。
            operation: 原始、规范化的操作及源码范围。
            kind: 已校验 provider 的显式数值 recipe。
        """
        args = dict(operation.arguments)
        output, = operation.outputs
        gradient = self.latest.get(output.id)
        if kind in ("silu_input_grad", "multiply_left_grad", "multiply_right_grad"):
            name = "value" if kind == "silu_input_grad" else "left" if kind == "multiply_left_grad" else "right"
            primal = "value" if kind == "silu_input_grad" else "right" if name == "left" else "left"
            target, primals = args[name].id, (args[primal].id,)
            matrix, transforms, transpose_output = None, ((), ()), False
        elif kind == "swiglu_input_grad":
            target, primals = args["packed"].id, (args["packed"].id,)
            matrix, transforms, transpose_output = None, ((), ()), False
        else:
            left, right = args["left"].id, args["right"].id
            rows, columns = self.types[output.id].shape
            contracted = self.types[left].shape[0 if args["transpose_left"] else 1]
            if kind == "matmul_left_grad":
                target, primals = left, (right,)
                matrix = (rows, contracted, columns, False, not args["transpose_right"])
                transforms, transpose_output = ((), (args["transpose_right"], True)), args["transpose_left"]
            elif kind == "linear_weight_grad":
                target, primals = right, (left,)
                matrix = (columns, contracted, rows, True, args["transpose_left"])
                transforms, transpose_output = ((True,), (args["transpose_left"],)), not args["transpose_right"]
            else:
                target, primals = right, (left,)
                matrix = (contracted, columns, rows, not args["transpose_left"], False)
                transforms, transpose_output = ((args["transpose_left"], True), ()), args["transpose_right"]
        value = self._value(target)
        inputs = (*primals, *((gradient,) if gradient is not None else ()))
        dependencies = (self.producers[gradient],) if gradient in self.producers else ()
        node = DenseGradientNode(len(self.nodes), kind, inputs, value, dependencies, index, target, gradient,
                                 primals, matrix, transforms, transpose_output, operation.source,
                                 _linear_gradient_reference(operation, kind, matrix))
        self.nodes.append(node)
        self.producers[value.id] = node.index
        self.contributions.append(DenseGradientContribution(node.index, index, output.id, target))
        self._add(target, value, index, operation.source)


def _linear_gradient_reference(operation, kind, matrix):
    if operation.logical_name != "dense.linear" or matrix is None:
        return None
    rows, columns, contracted, _left, _right = matrix
    if kind == "matmul_left_grad":
        return DenseMatrixReference("dense.linear.input_grad", (rows, columns, contracted, False, False))
    # TorchNPU 的 matmul_backward 先计算 X.T @ dY，再由原始权重转置恢复 NK 输出。
    return DenseMatrixReference("dense.linear.weight_grad", (columns, rows, contracted, True, False), True)


def _recipes(task, operation):
    schema = REGISTRY.schema(operation.logical_name, operation.version)
    if (task.provider.logical_name, task.provider.version) != (schema.logical_name, schema.version):
        raise ValueError("Dense backward provider does not match its canonical schema")
    expected = {"dense.matmul": (matmul.schema, "aten.mm.default", ("matmul_left_grad", "matmul_right_grad")),
                "dense.linear": (linear.schema, "aten.linear.default", ("matmul_left_grad", "linear_weight_grad")),
                "dense.swiglu": (swiglu_dense.schema, "npu.npu_swiglu.default", ("swiglu_input_grad",)),
                "dense.silu": (silu_dense.schema, "aten.silu.default", ("silu_input_grad",)),
                "dense.multiply": (multiply_dense.schema, "aten.mul.Tensor",
                                   ("multiply_left_grad", "multiply_right_grad"))}
    contract = expected.get(schema.logical_name)
    if (contract is None or schema is not contract[0]
            or contract[1:] != (task.provider.implementation, task.provider.backward)):
        raise ValueError("Dense backward requires a supported explicit provider recipe")
    return task.provider.backward


def compile_dense_backward(plan: DenseKernelPlan) -> DenseBackwardPlan:
    """保留原始 provider 数值 recipe、反向顺序及逐次梯度累加。

    Args:
        plan: 已校验的带类型稠密前向计划。
    """
    if plan.ir.numeric_policy != "preserve_numeric_order" or len(plan.tasks) != len(plan.ir.operations):
        raise ValueError("Dense backward requires a complete plan with preserved numeric order")
    builder = _Builder(plan)
    for index, output in enumerate(plan.ir.outputs):
        builder.seed(index, output)
    for index in reversed(range(len(plan.tasks))):
        operation = plan.ir.operations[index]
        for recipe in _recipes(plan.tasks[index], operation):
            builder.contribution(index, operation, recipe)
    available = {value.id for value in plan.ir.inputs}
    available.update(buffer.value_id for buffer in plan.buffers if buffer.saved_for_backward)
    if any(value not in available for node in builder.nodes for value in node.primal_inputs):
        raise ValueError("Dense backward recipe requires an unavailable forward saved value")
    return DenseBackwardPlan(plan, tuple(builder.seeds), tuple(builder.nodes), tuple(builder.contributions),
                             tuple(builder.latest.get(value.id) for value in plan.ir.inputs),
                             tuple(sorted(builder.types.items())))
