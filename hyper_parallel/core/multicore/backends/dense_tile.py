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
"""Exact descriptor serialization and SDK tiling for resident dense candidates."""

from __future__ import annotations

import ctypes
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

from hyper_parallel.core.multicore.backends.dense_reference import ACCESS, DenseReferenceTiler, reference_access
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePlan

HEADER = struct.Struct("<8I")
OPERATION = struct.Struct("<7IiI")
MAGIC = 0x444E5331
VERSION = 2
EVENT_STRIDE_BYTES = 128


@dataclass(frozen=True)
class DenseTileBinding:
    """Compact tensor slots and address-free device descriptors."""

    value_ids: tuple[int, ...]
    config: bytes
    matmul_shapes: tuple[tuple[int, int, int, bool] | None, ...]


def _operation_descriptor(plan, task, slots):
    operation = plan.dense.ir.operations[task.index]
    arguments = dict(operation.arguments)
    types = dict(plan.dense.value_types)
    output, = operation.outputs
    columns = types[output.id].shape[1]
    if operation.logical_name == "dense.matmul":
        left, right = arguments["left"], arguments["right"]
        contracted, transpose = types[left.id].shape[1], arguments["transpose_right"]
        descriptor = (0, slots[left.id], slots[right.id], slots[output.id], columns, contracted, int(transpose))
        shape = (plan.rows, columns, contracted, transpose)
    else:
        packed = arguments["packed"]
        descriptor = (1, slots[packed.id], 0, slots[output.id], columns, types[packed.id].shape[1], 0)
        shape = None
    if len(task.dependencies) > 1:
        raise ValueError("Resident dense forward descriptor admits one row-local producer per primitive")
    dependency = task.dependencies[0] if task.dependencies else -1
    count = 2 if dependency >= 0 and plan.dense.tasks[dependency].provider.worker == "vector" else 1
    return OPERATION.pack(*descriptor, dependency, count), shape


def bind_dense_tiles(plan: DenseTilePlan) -> DenseTileBinding:
    """Serialize compact SSA slots with the exact native header/operation ABI.

    Args:
        plan: Admitted, concrete BF16 forward tile candidate.
    """
    value_ids = tuple(value_id for value_id, _ in plan.dense.value_types)
    slots = {value_id: slot for slot, value_id in enumerate(value_ids)}
    sizes = (plan.rows, plan.policy.rows_per_tile, plan.policy.cube_workers, plan.policy.prefetch_tiles)
    if any(size < 0 or size > 2**31 - 1 for size in sizes):
        raise ValueError("Dense tile dimensions exceed the signed SDK tiling boundary")
    descriptors, shapes = [], []
    for task in plan.dense.tasks:
        descriptor, shape = _operation_descriptor(plan, task, slots)
        if shape is not None and any(dimension > 2**31 - 1 for dimension in shape[:3]):
            raise ValueError("Dense matrix dimensions exceed the signed SDK tiling boundary")
        descriptors.append(descriptor)
        shapes.append(shape)
    config = HEADER.pack(MAGIC, VERSION, *sizes, len(descriptors), len(value_ids)) + b"".join(descriptors)
    return DenseTileBinding(value_ids, config, tuple(shapes))


class DenseSdkTiler:
    """Generate the selected SDK's raw TCubeTiling through a sealed host library."""

    def __init__(self, library: Path, soc: str, reference: DenseReferenceTiler | None = None) -> None:
        """Read target limits and raw tiling size without allocating device tensors.

        Args:
            library: Caller-verified host tiling shared library.
            soc: Exact SoC selected by the native build.
            reference: 可选的隔离 SDK 参考查询，正式运行时由已验证产物创建。
        """
        self.soc = soc
        self.reference = reference
        self.reference_record = None
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        query = self.library.hyper_parallel_dense_platform
        query.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64),
                          ctypes.POINTER(ctypes.c_uint32)]
        query.restype = ctypes.c_int
        workers, workspace, size = ctypes.c_uint32(), ctypes.c_uint64(), ctypes.c_uint32()
        status = query(soc.encode(), ctypes.byref(workers), ctypes.byref(workspace), ctypes.byref(size))
        if status != 0 or workers.value == 0 or size.value == 0 or size.value % 4:
            raise ValueError(f"Dense SDK platform query failed for {soc}: status={status}")
        self.cube_workers, self.workspace_bytes, self.tiling_bytes = workers.value, workspace.value, size.value
        self.generate = self.library.hyper_parallel_dense_matmul_tiling
        self.generate.argtypes = [ctypes.c_char_p, *([ctypes.c_uint32] * 5), ctypes.c_void_p, ctypes.c_uint32]
        self.generate.restype = ctypes.c_int

    def bank(self, plan: DenseTilePlan, binding: DenseTileBinding) -> bytes:
        """Bind every matrix primitive, keeping activation slots at SDK-sized offsets.

        Args:
            plan: Concrete tile policy and row dimensions.
            binding: Address-free descriptors for that exact SSA plan.
        """
        if plan.policy.cube_workers > self.cube_workers:
            raise ValueError("Dense cube worker count exceeds the selected SDK hardware target")
        if binding != bind_dense_tiles(plan):
            raise ValueError("Dense SDK tiling binding differs from its SSA tile plan")
        shapes = tuple(shape for shape in binding.matmul_shapes if shape is not None and plan.rows)
        self.reference_record = self.reference.query(shapes) if self.reference is not None and shapes else None
        results = iter(self.reference_record["results"] if self.reference_record is not None else ())
        blocks, reasons = [], []
        for shape in binding.matmul_shapes:
            block = ctypes.create_string_buffer(self.tiling_bytes)
            access, reason = bytes(ACCESS.size), "no_matrix"
            if shape is not None and plan.rows:
                rows, columns, contracted, transpose = shape
                tile_rows = min(rows, plan.policy.rows_per_tile)
                status = self.generate(self.soc.encode(), rows, tile_rows, columns, contracted, int(transpose),
                                       block, self.tiling_bytes)
                if status != 0:
                    raise ValueError(f"Dense SDK matrix tiling failed for {shape}: status={status}")
                reason = "no_reference_query"
                if self.reference_record is not None:
                    access, reason = reference_access(next(results), shape, self.cube_workers)
            blocks.append(block.raw + access)
            reasons.append(reason)
        self.access_reasons = tuple(reasons)
        return b"".join(blocks)

    def export_manifest(self, binding: DenseTileBinding, bank: bytes) -> dict[str, object]:
        """Seal target metadata and exact serialized bytes, excluding Tensor addresses.

        Args:
            binding: Concrete device task descriptors.
            bank: Raw SDK tiling bank generated for those descriptors.
        """
        expected = len(binding.matmul_shapes) * (self.tiling_bytes + ACCESS.size)
        if len(bank) != expected:
            raise ValueError("稠密 SDK tiling bank 的长度不符合访问参数 ABI")
        return {"soc": self.soc, "cube_workers": self.cube_workers, "workspace_bytes": self.workspace_bytes,
                "tiling_bytes": self.tiling_bytes, "config_sha256": hashlib.sha256(binding.config).hexdigest(),
                "access_bytes": ACCESS.size, "bank_stride_bytes": self.tiling_bytes + ACCESS.size,
                "reference": self.reference_record, "access_reasons": self.access_reasons,
                "bank_sha256": hashlib.sha256(bank).hexdigest()}
