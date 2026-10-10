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
"""矩阵的物理寻址与原始算子数值参考分别记录，不复制设备权重以改变布局。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from hyper_parallel.core.multicore.ir.program import Operation
from hyper_parallel.core.multicore.language.types import TensorType


@dataclass(frozen=True)
class DenseMatrixReference:
    """数值参考几何与物理 GEMM 坐标的映射，全部来自编译期元数据。"""

    source: str
    matrix: tuple[int, int, int, bool, bool]
    coordinates_transposed: bool = False

    def __post_init__(self) -> None:
        if (type(self.source) not in (str,) or self.source not in
                ("dense.matmul", "dense.linear", "dense.linear.input_grad", "dense.linear.weight_grad")
                or type(self.matrix) not in (tuple,) or len(self.matrix) != 5
                or any(type(value) not in (int,) or value < 0 for value in self.matrix[:3])
                or any(type(value) not in (bool,) for value in self.matrix[3:])
                or type(self.coordinates_transposed) not in (bool,)):
            raise ValueError("Dense matrix reference requires a canonical source and exact geometry types")

    def export_manifest(self) -> dict[str, object]:
        """将原语来源、参考布局及输出坐标映射纳入绑定身份。"""
        return asdict(self)


def forward_matrix_reference(operation: Operation, types: Mapping[int, TensorType]) -> DenseMatrixReference | None:
    """从规范矩阵原语推导参考布局，保留真实的数学收缩几何。

    Args:
        operation: 已由 provider 注册表验证过身份的矩阵原语。
        types: 已绑定具体维度的值类型。
    """
    if operation.logical_name not in ("dense.matmul", "dense.linear"):
        return None
    arguments = dict(operation.arguments)
    output, = operation.outputs
    rows, columns = types[output.id].shape
    contracted = types[arguments["left"].id].shape[0 if arguments["transpose_left"] else 1]
    reference_right = True if operation.logical_name == "dense.linear" else arguments["transpose_right"]
    return DenseMatrixReference(operation.logical_name,
                                (rows, columns, contracted, arguments["transpose_left"], reference_right))
