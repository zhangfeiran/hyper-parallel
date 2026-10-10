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
"""反向矩形域、跨轴依赖及实际 SDK 左转置 tiling 的地址无关绑定。"""

from __future__ import annotations

import ctypes
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

from hyper_parallel.core.multicore.backends.dense_reference import ACCESS
from hyper_parallel.core.multicore.backends.dense_reference import reference_access
from hyper_parallel.core.multicore.backends.dense_tile import DenseSdkTiler
from hyper_parallel.core.multicore.compiler.dense_backward_tiles import (
    DenseBackwardTilePlan,
    compile_dense_backward_tiles,
    row_local_dependency,
    simulate_dense_backward_tiles,
)
from hyper_parallel.core.multicore.ir.dense_matrix import DenseMatrixReference

HEADER = struct.Struct("<11I")
OPERATION = struct.Struct("<18I")
DEPENDENCY = struct.Struct("<4I")
MAGIC = 0x444E4231
VERSION = 1
KINDS = {"matmul": 0, "swiglu_grad": 2, "gradient_add": 3, "zero": 4, "multiply": 6, "silu_grad": 7}


@dataclass(frozen=True)
class DenseBackwardBinding:
    """仅保留实际消费者、生产者及别名后的紧凑张量槽位。"""

    value_ids: tuple[int, ...]
    aliases: tuple[tuple[int, int], ...]
    output_ids: tuple[int, ...]
    gradients: tuple[int | None, ...]
    config: bytes
    matmul_shapes: tuple[tuple[int, int, int, bool, bool] | None, ...]
    tile_shapes: tuple[tuple[int, int], ...]
    required_capabilities: tuple[str, ...]
    matrix_references: tuple[DenseMatrixReference | None, ...] = ()

    def unbound_capabilities(self) -> tuple[str, ...]:
        """按稳定顺序返回尚未实现的设备能力，不吞掉调度或绑定错误。"""
        supported = {"matmul_nn", "matmul_tn", "matmul_nt", "swiglu_grad", "add", "zero", "multiply", "silu_grad"}
        return tuple(sorted(set(self.required_capabilities) - supported))

    def admit_native(self) -> None:
        """设备能力必须全部实现；双侧转置及非空转置输出尚未放行。"""
        unsupported = self.unbound_capabilities()
        if unsupported:
            raise ValueError(f"Dense resident backward has unbound native capabilities: {sorted(unsupported)}")


def _resolved(value, aliases):
    while value in aliases:
        value = aliases[value]
    return value


def _operands(node):
    if node.kind == "matmul_left_grad":
        return node.gradient_source, node.primal_inputs[0]
    if node.kind == "linear_weight_grad":
        return node.gradient_source, node.primal_inputs[0]
    if node.kind == "matmul_right_grad":
        return node.primal_inputs[0], node.gradient_source
    if node.kind in ("swiglu_input_grad", "silu_input_grad", "multiply_left_grad", "multiply_right_grad"):
        return node.primal_inputs[0], node.gradient_source
    return node.inputs


def _dependencies(plan, region, producers, aliases):
    node = plan.backward.nodes[region.node]
    result = set()
    for value in node.inputs:
        parent = producers.get(_resolved(value, aliases))
        if parent is None or not parent.events:
            continue
        result.add((parent.events[0], len(parent.events), parent.signals, int(row_local_dependency(region, parent))))
    return tuple(sorted(result))


def bind_dense_backward_tiles(plan: DenseBackwardTilePlan) -> DenseBackwardBinding:
    """序列化规范反向计划，包括未准入的能力，拒绝损坏的任务和事件。

    Args:
        plan: 从完整显式反向 SSA、输入需求和余切存在性导出的计划。
    """
    if plan != compile_dense_backward_tiles(plan.backward, plan.policy, plan.output_mask, plan.input_mask):
        raise ValueError("Dense backward binding differs from the canonical worker plan")
    simulate_dense_backward_tiles(plan)
    aliases = dict(plan.aliases)
    nodes = {region.node: plan.backward.nodes[region.node] for region in plan.regions}
    gradients = tuple(_resolved(value, aliases) if value is not None and requested else None
                      for value, requested in zip(plan.backward.gradients, plan.input_mask))
    defined = plan.backward.defined_values(plan.output_mask)
    gradients = tuple(value if value in defined else None for value in gradients)
    values = {value for value in gradients if value is not None}
    for node in nodes.values():
        values.add(node.output.id)
        values.update(_resolved(value, aliases) for value in _operands(node))
    value_ids = tuple(sorted(values))
    slots = {value: index for index, value in enumerate(value_ids)}
    producers = {node.output.id: region for region in plan.regions for node in (nodes[region.node],)}
    descriptors, dependencies, shapes, tile_shapes, capabilities = [], [], [], [], set()
    for region in plan.regions:
        node = nodes[region.node]
        left, right = (_resolved(value, aliases) for value in _operands(node))
        contracted, transpose_left, transpose_right = (region.matrix[2:] if region.matrix else (0, False, False))
        if region.rows and region.columns:
            if region.kind == "matmul":
                capabilities.add("matmul_" + ("t" if transpose_left else "n") + ("t" if transpose_right else "n"))
            else:
                capabilities.add("add" if region.kind == "gradient_add" else region.kind)
        if region.transpose_output and region.rows and region.columns and region.kind != "zero":
            capabilities.add("transpose_output")
        waits = _dependencies(plan, region, producers, aliases)
        descriptor = (KINDS[region.kind], slots[left], slots[right], slots[node.output.id],
                      region.rows, region.columns, contracted, int(transpose_left), int(transpose_right),
                      int(region.transpose_output), region.phase, region.width, region.parts,
                      region.events[0] if region.events else 0, len(region.events), region.signals,
                      len(dependencies), len(waits))
        descriptors.append(OPERATION.pack(*descriptor))
        dependencies.extend(waits)
        shapes.append(region.matrix if region.kind == "matmul" else None)
        tile_shapes.append((min(region.rows, plan.policy.rows_per_tile), min(region.columns, region.width)))
    if len(descriptors) > 64 or len(value_ids) > 128 or len(dependencies) > 2**31 - 1:
        raise ValueError("Dense backward plan exceeds the native descriptor boundary")
    phase_tiles = tuple(max((len(region.events) for region in plan.regions if region.phase == phase), default=0)
                        for phase in (0, 1))
    config = HEADER.pack(MAGIC, VERSION, plan.policy.rows_per_tile, plan.policy.cube_workers,
                         plan.policy.prefetch_tiles, len(descriptors), len(value_ids), plan.event_count,
                         len(dependencies), *phase_tiles)
    config += b"".join(descriptors) + b"".join(DEPENDENCY.pack(*wait) for wait in dependencies)
    return DenseBackwardBinding(value_ids, plan.aliases, tuple(node.output.id for node in nodes.values()),
                                gradients, config, tuple(shapes), tuple(tile_shapes), tuple(sorted(capabilities)),
                                tuple(nodes[region.node].numeric_reference for region in plan.regions))


class DenseBackwardSdkTiler:
    """每个反向矩形保留整矩阵跨度和完整 K，不使用 BF16 token 局部求和。"""

    def __init__(self, library: Path, soc: str) -> None:
        """读取已密封平台库，并绑定独立的左右转置参数 ABI。"""
        self.sdk = DenseSdkTiler(library, soc)
        self.generate = self.sdk.library.hyper_parallel_dense_backward_tiling
        self.generate.argtypes = [ctypes.c_char_p, *([ctypes.c_uint32] * 7), ctypes.c_void_p, ctypes.c_uint32]
        self.generate.restype = ctypes.c_int
        self.reference = None
        self.reference_record = None
        self.reference_shapes = ()

    def bank(self, plan: DenseBackwardTilePlan, binding: DenseBackwardBinding) -> bytes:
        """在设备地址建立前检查全部能力，生成逐操作原始 tiling。

        Args:
            plan: 真实调用所需的规范反向工作队列。
            binding: 规范反向 SSA 的描述符，不包含调用方张量地址。
        """
        if binding != bind_dense_backward_tiles(plan):
            raise ValueError("Dense backward SDK binding differs from its canonical worker plan")
        return self._bank(binding)

    def _bank(self, binding):
        binding.admit_native()
        workers = HEADER.unpack_from(binding.config)[3]
        if workers > self.sdk.cube_workers:
            raise ValueError("Dense backward cube worker count exceeds the selected SDK hardware target")
        references = binding.matrix_references or (None,) * len(binding.matmul_shapes)
        queries = tuple(reference.matrix if reference is not None else
                        (*shape[:3], shape[4]) if shape is not None and shape[4] and not shape[3] else None
                        for shape, reference in zip(binding.matmul_shapes, references))
        shapes = tuple(dict.fromkeys(query for query in queries if query is not None and all(query[:3])))
        if getattr(self, "reference", None) is not None and shapes != self.reference_shapes:
            self.reference_record = self.reference.query(shapes)
            self.reference_shapes = shapes
        record = getattr(self, "reference_record", None) or {}
        reference = {tuple(result["shape"]): result for result in record.get("results", [])}
        blocks, reasons = [], []
        for shape, tile, query, numeric in zip(binding.matmul_shapes, binding.tile_shapes, queries, references):
            block = ctypes.create_string_buffer(self.sdk.tiling_bytes)
            access = bytes(ACCESS.size)
            reason = "no_matrix"
            if shape is not None and all(shape[:3]):
                rows, columns, contracted, transpose_left, transpose_right = shape
                status = self.generate(self.sdk.soc.encode(), rows, tile[0], columns, tile[1], contracted,
                                       int(transpose_left), int(transpose_right), block, self.sdk.tiling_bytes)
                if status:
                    raise ValueError(f"Dense backward SDK matrix tiling failed for {shape}: status={status}")
                reason = "no_reference_query"
                if query in reference:
                    access, reason = reference_access(reference[query], query, self.sdk.cube_workers,
                                                      allow_right_transpose=True, allow_left_transpose=True,
                                                      allow_sequential_reference=numeric is not None)
                    if numeric is not None and numeric.coordinates_transposed:
                        m_span, n_span, m_extent, n_extent, chunk, close = ACCESS.unpack(access)
                        access = ACCESS.pack(n_span, m_span, n_extent, m_extent, chunk, close)
                if (numeric is not None and numeric.source.startswith("dense.linear") and reason not in
                        ("sdk_k_block_access", "native_matmul_v3_sequential", "sdk_single_k_block_sequential",
                         "sdk_single_partition_sequential", "sdk_full_l1_k_sequential")):
                    raise ValueError(f"Dense Linear numerical reference is not admitted: {query}: {reason}")
            blocks.append(block.raw + access)
            reasons.append(reason)
        self.access_reasons = tuple(reasons)
        return b"".join(blocks)

    def export_manifest(self, binding: DenseBackwardBinding, bank: bytes) -> dict[str, object]:
        """封存原始字节；顺序 SDK 执行的全模型数值仍需独立设备验收。"""
        stride = self.sdk.tiling_bytes + ACCESS.size
        if len(bank) != len(binding.matmul_shapes) * stride:
            raise ValueError("Dense backward tiling bank length differs from the native ABI")
        return {"soc": self.sdk.soc, "cube_workers": self.sdk.cube_workers,
                "workspace_bytes": self.sdk.workspace_bytes, "tiling_bytes": self.sdk.tiling_bytes,
                "bank_stride_bytes": stride, "matrix_access": "sdk_nt_reference_candidate"
                if getattr(self, "reference_record", None) is not None else "sequential_sdk_unproved",
                "reference": getattr(self, "reference_record", None),
                "matrix_references": [reference.export_manifest() if reference is not None else None
                                      for reference in binding.matrix_references],
                "access_reasons": getattr(self, "access_reasons", ()),
                "required_capabilities": binding.required_capabilities,
                "config_sha256": hashlib.sha256(binding.config).hexdigest(),
                "bank_sha256": hashlib.sha256(bank).hexdigest()}
