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
"""原始 NK Linear 前向的共享矩阵 ABI，不改变旧 Matmul 前向的绑定。"""

from __future__ import annotations

from dataclasses import dataclass

from hyper_parallel.core.multicore.backends.dense_backward import (
    DEPENDENCY,
    HEADER,
    MAGIC,
    OPERATION,
    VERSION,
    DenseBackwardSdkTiler,
)
from hyper_parallel.core.multicore.compiler.dense_tile import (
    DenseTilePlan,
    column_partition,
    compile_dense_tiles,
    simulate_dense_tiles,
)
from hyper_parallel.core.multicore.ir.dense_matrix import DenseMatrixReference, forward_matrix_reference


@dataclass(frozen=True)
class DenseLinearBinding:
    """真实前向 SSA 的槽位、操作主序事件及显式数值来源。"""

    value_ids: tuple[int, ...]
    config: bytes
    matmul_shapes: tuple[tuple[int, int, int, bool, bool] | None, ...]
    tile_shapes: tuple[tuple[int, int], ...]
    matrix_references: tuple[DenseMatrixReference | None, ...]
    required_capabilities: tuple[str, ...]
    event_map: tuple[int, ...]

    def admit_native(self) -> None:
        """只放行共享核心已经支持的行内矩阵、打包激活、SiLU 和逐元素乘法。"""
        unsupported = set(self.required_capabilities) - {"matmul_nn", "matmul_nt", "swiglu", "silu", "multiply"}
        if unsupported:
            raise ValueError(f"Dense Linear forward has unbound native capabilities: {sorted(unsupported)}")


def bind_dense_linear_tiles(plan: DenseTilePlan) -> DenseLinearBinding:
    """绑定规范前向队列，事件布局显式映射到共享矩阵核心。

    Args:
        plan: 由 typed SSA 生成的前向工作队列，不包含张量地址。
    """
    if plan != compile_dense_tiles(plan.dense, plan.policy):
        raise ValueError("Dense Linear binding differs from its canonical forward plan")
    if not any(operation.logical_name == "dense.linear" for operation in plan.dense.ir.operations):
        raise ValueError("Dense Linear binding requires an explicit Linear primitive")
    simulate_dense_tiles(plan)
    value_ids = tuple(value_id for value_id, _ in plan.dense.value_types)
    slots, types = {value: index for index, value in enumerate(value_ids)}, dict(plan.dense.value_types)
    count = len(plan.dense.tasks)
    tiles = (plan.rows + plan.policy.rows_per_tile - 1) // plan.policy.rows_per_tile
    partitions = tuple(column_partition(plan.dense, task.index, plan.policy) for task in plan.dense.tasks)
    signals = tuple(parts * (2 if task.provider.worker == "vector" else 1)
                    for task, (_, parts) in zip(plan.dense.tasks, partitions))
    descriptors, dependencies, shapes, tile_shapes, references, capabilities = [], [], [], [], [], set()
    for task, (width, parts) in zip(plan.dense.tasks, partitions):
        operation = plan.dense.ir.operations[task.index]
        arguments = dict(operation.arguments)
        output, = operation.outputs
        rows, columns = types[output.id].shape
        reference = forward_matrix_reference(operation, types)
        if reference is None:
            capability = operation.logical_name.split(".")[1]
            kind = {"swiglu": 1, "silu": 5, "multiply": 6}[capability]
            name = {"swiglu": "packed", "silu": "value", "multiply": "left"}[capability]
            left = arguments[name].id
            right = arguments["right"].id if capability == "multiply" else value_ids[0]
            contracted, transpose, shape = columns * 2 if kind == 1 else 0, False, None
        else:
            left, right = arguments["left"].id, arguments["right"].id
            contracted, transpose = reference.matrix[2], arguments["transpose_right"]
            kind, shape = 0, (rows, columns, contracted, False, transpose)
            capability = "matmul_nt" if transpose else "matmul_nn"
        if any(dimension > 2**31 - 1 for dimension in (rows, columns, contracted)):
            raise ValueError("Dense Linear dimensions exceed the signed SDK tiling boundary")
        waits = tuple((parent * tiles, tiles, signals[parent], 1) for parent in task.dependencies)
        descriptors.append(OPERATION.pack(kind, slots[left], slots[right], slots[output.id], rows, columns,
                                          contracted, 0, int(transpose), 0, 0, width, parts,
                                          task.index * tiles, tiles, signals[task.index],
                                          len(dependencies), len(waits)))
        dependencies.extend(waits)
        shapes.append(shape)
        tile_shapes.append((min(rows, plan.policy.rows_per_tile), min(columns, width)))
        references.append(reference)
        if rows:
            capabilities.add(capability)
    config = HEADER.pack(MAGIC, VERSION, plan.policy.rows_per_tile, plan.policy.cube_workers,
                         plan.policy.prefetch_tiles, count, len(value_ids), plan.event_count,
                         len(dependencies), tiles, 0)
    config += b"".join(descriptors) + b"".join(DEPENDENCY.pack(*wait) for wait in dependencies)
    event_map = tuple(operation * tiles + tile for tile in range(tiles) for operation in range(count))
    return DenseLinearBinding(value_ids, config, tuple(shapes), tuple(tile_shapes), tuple(references),
                              tuple(sorted(capabilities)), event_map)


class DenseLinearSdkTiler(DenseBackwardSdkTiler):
    """前向使用同一个真实 NN/NT SDK 矩阵核心，保留独立的规范计划检查。"""

    def bank(self, plan: DenseTilePlan, binding: DenseLinearBinding) -> bytes:
        """查询实际 Linear 布局的参考顺序，封存逐操作原始分块参数。

        Args:
            plan: 本次编译使用的规范前向队列。
            binding: 同一前向队列的共享 ABI，不接受反向计划替代前向证明。
        """
        if binding != bind_dense_linear_tiles(plan):
            raise ValueError("Dense Linear SDK binding differs from its canonical forward plan")
        return self._bank(binding)

    def export_manifest(self, binding: DenseLinearBinding, bank: bytes) -> dict[str, object]:
        """明确前向的真实事件布局与数值来源，不将它伪装为反向执行。"""
        return {**super().export_manifest(binding, bank), "direction": "forward",
                "event_layout": "operation_major", "logical_to_native_events": binding.event_map,
                "matrix_access": "sdk_linear_reference_candidate"}
