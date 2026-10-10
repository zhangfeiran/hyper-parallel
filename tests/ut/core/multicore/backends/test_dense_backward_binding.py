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
"""反向原生描述符、真实 worker 循环和未绑定能力的拒绝。"""

from __future__ import annotations

import struct
import unittest
from dataclasses import replace
from types import SimpleNamespace

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.backends.dense_backward import (
    DEPENDENCY,
    HEADER,
    OPERATION,
    DenseBackwardSdkTiler,
    bind_dense_backward_tiles,
)
from hyper_parallel.core.multicore.compiler.dense_backward import compile_dense_backward
from hyper_parallel.core.multicore.compiler.dense_backward_tiles import compile_dense_backward_tiles
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from tests.common.mark_utils import arg_mark


def _ffn(rows, **policy):
    forward = dense_ffn.plan(mc.DenseSpec({"T": rows, "H": 17, "I": 19, "PackedI": 38}), intermediate_size=19)
    return compile_dense_backward_tiles(compile_dense_backward(forward), DenseTilePolicy(**policy))


def _linear(rows, hidden=17, output=37, **masks):
    forward = mc.from_source("def linear(x, weight):\n    return ml.matmul(x, weight)\n",
                             signature={"x": ml.Tensor[ml.bf16, (rows, hidden)],
                                        "weight": ml.Tensor[ml.bf16, (hidden, output)]},
                             symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1")).plan(mc.DenseSpec({}))
    return compile_dense_backward_tiles(compile_dense_backward(forward),
                                        DenseTilePolicy(cube_workers=3, column_workers=4), **masks)


def _decode_queues(binding):
    header = HEADER.unpack_from(binding.config)
    height, workers, prefetch, count = header[2:6]
    operations = [OPERATION.unpack_from(binding.config, HEADER.size + index * OPERATION.size)
                  for index in range(count)]
    offset = HEADER.size + count * OPERATION.size
    dependencies = [DEPENDENCY.unpack_from(binding.config, offset + index * DEPENDENCY.size)
                    for index in range(header[8])]
    queues = {}
    for worker_kind, total in (("cube", workers), ("vector", workers * 2)):
        for worker in range(total):
            cube = worker if worker_kind == "cube" else worker // 2
            tasks = []
            for phase, tiles in enumerate(header[9:11]):
                for base in range(0, tiles, workers * prefetch):
                    for index, operation in enumerate(operations):
                        kind = operation[0]
                        if operation[10] != phase or (kind == 0) != (worker_kind == "cube"):
                            continue
                        parts = operation[12]
                        stop = min(operation[14], base + workers * prefetch)
                        for fragment in range(base * parts + cube, stop * parts, workers):
                            tile, part = divmod(fragment, parts)
                            first, end = tile * height, min((tile + 1) * height, operation[4])
                            if worker_kind == "vector":
                                middle = first + (end - first + 1) // 2
                                first, end = (first, middle) if worker % 2 == 0 else (middle, end)
                            waits = set()
                            for begin, length, required, local in dependencies[operation[16]:
                                                                               operation[16] + operation[17]]:
                                events = (begin + tile,) if local else range(begin, begin + length)
                                waits.update((event, required) for event in events)
                            tasks.append((index, tile, first, end, part * operation[11],
                                          min((part + 1) * operation[11], operation[5]), part,
                                          tuple(sorted(waits)), operation[13] + tile))
            if tasks:
                queues[(worker_kind, worker)] = tasks
    return queues


class TestDenseBackwardBinding(unittest.TestCase):
    """地址无关的 ABI 必须表达真实收缩域、可选梯度和全部生产者汇合。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_complete_ffn_domains_and_global_join_are_serialized(self):
        """Feature: 完整 FFN 反向 ABI。
        Description: 检查 token 行、两组完整 token 权重收缩以及 SDK 能力拒绝。
        Expectation: dWgu 等待全部 dP 行；完整 FFN 保留未绑定右转置状态。
        """
        plan = _ffn(129, rows_per_tile=64, cube_workers=3, column_workers=4)
        binding = bind_dense_backward_tiles(plan)
        header = HEADER.unpack_from(binding.config)
        self.assertEqual(header[2:9], (64, 3, 2, 5, 11, plan.event_count, 3))
        self.assertEqual(binding.matmul_shapes, ((129, 19, 17, False, True), (19, 17, 129, True, False),
                                                None, (129, 17, 38, False, True), (17, 38, 129, True, False)))
        operations = [OPERATION.unpack_from(binding.config, HEADER.size + index * OPERATION.size)
                      for index in range(5)]
        self.assertEqual([operation[10] for operation in operations], [0, 1, 0, 0, 1])
        offset = HEADER.size + 5 * OPERATION.size + operations[4][16] * DEPENDENCY.size
        self.assertEqual(DEPENDENCY.unpack_from(binding.config, offset),
                         (plan.regions[2].events[0], 3, plan.regions[2].signals, 0))
        with self.assertRaisesRegex(ValueError, "matmul_nt"):
            binding.admit_native()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_only_requested_input_gradients_keep_their_complete_ancestors(self):
        """Feature: 真实输入梯度需求。
        Description: 只请求线性层权重梯度，或在 FFN 中只请求 gate/up 权重梯度。
        Expectation: 无需求分支被删除，所需行生产者仍存在，K 轴始终完整。
        """
        linear = _linear(129, input_mask=(False, True))
        binding = bind_dense_backward_tiles(linear)
        binding.admit_native()
        self.assertEqual(binding.matmul_shapes, ((17, 37, 129, True, False),))
        self.assertEqual(binding.required_capabilities, ("matmul_tn",))
        self.assertIsNone(binding.gradients[0])
        self.assertEqual(set(binding.value_ids), {0, linear.backward.seeds[0].value.id,
                                                linear.backward.nodes[1].output.id})
        full = _ffn(129, cube_workers=3)
        gate_up = compile_dense_backward_tiles(full.backward, full.policy, input_mask=(False, True, False))
        self.assertEqual([region.node for region in gate_up.regions], [0, 2, 4])
        self.assertEqual(gate_up.regions[-1].matrix[2], 129)
        with self.assertRaisesRegex(ValueError, "input positions"):
            compile_dense_backward_tiles(full.backward, full.policy, input_mask=(False, 1, False))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_serialized_worker_loop_matches_every_planned_queue(self):
        """Feature: Python 与设备循环的调度一致性。
        Description: 独立解码真实二进制 ABI，回放设备的阶段、窗口和 worker 循环。
        Expectation: 所有片段、空通道和依赖与规范队列逐项一致。
        """
        for rows in (0, 1, 7, 65, 129):
            for workers in (1, 3, 20):
                for columns in (1, 4, 7):
                    for prefetch in (1, 2, 4):
                        with self.subTest(rows=rows, workers=workers, columns=columns, prefetch=prefetch):
                            plan = _ffn(rows, cube_workers=workers, column_workers=columns,
                                        prefetch_tiles=prefetch)
                            indices = {region.node: index for index, region in enumerate(plan.regions)}
                            expected = {worker: [(indices[task.operation], task.tile, task.row_begin, task.row_end,
                                                  task.column_begin, task.column_end, task.column_tile,
                                                  task.dependencies, task.trigger) for task in tasks]
                                        for worker, tasks in plan.queues().items()}
                            self.assertEqual(_decode_queues(bind_dense_backward_tiles(plan)), expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_corrupt_queues_and_nullable_gradients_cannot_enter_native(self):
        """Feature: 反向准入和空余切。
        Description: 删除生产任务或改变输入需求，并比较实际为空的返回梯度。
        Expectation: 损坏计划严格拒绝，空余切不创建梯度任务。
        """
        plan = _ffn(65, cube_workers=3)
        for corrupt in (replace(plan, tasks=plan.tasks[1:]), replace(plan, event_count=plan.event_count + 1),
                        replace(plan, input_mask=(False, True, True))):
            with self.subTest(corrupt=corrupt.input_mask), self.assertRaisesRegex(ValueError, "canonical"):
                bind_dense_backward_tiles(corrupt)
        empty = _linear(65, output_mask=(False,))
        binding = bind_dense_backward_tiles(empty)
        self.assertEqual(binding.gradients, (None, None))
        self.assertFalse(binding.output_ids)
        self.assertEqual(len(binding.config), HEADER.size)
        binding.admit_native()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_zero_tokens_use_zero_workers_without_transpose_capability(self):
        """Feature: 空 token 权重梯度。
        Description: 空 FFN 删除未请求输入梯度，仅保留两组权重梯度。
        Expectation: 非空权重梯度由零填充发布，不要求未执行的右转置矩阵。
        """
        full = _ffn(0, cube_workers=3)
        plan = compile_dense_backward_tiles(full.backward, full.policy, input_mask=(False, True, True))
        binding = bind_dense_backward_tiles(plan)
        binding.admit_native()
        self.assertEqual(binding.required_capabilities, ("zero",))
        self.assertEqual([region.kind for region in plan.regions if region.events], ["zero", "zero"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_sdk_tiling_keeps_full_k_and_actual_left_transpose(self):
        """Feature: 左转置 SDK tiling 参数。
        Description: 用原始结构大小的主机库替身记录反向矩形配置。
        Expectation: tile 只切输出 M/N，K 仍为完整 token，能力错误早于 SDK 调用。
        """
        calls = []
        tiler = DenseBackwardSdkTiler.__new__(DenseBackwardSdkTiler)
        tiler.sdk = SimpleNamespace(soc="test-target", cube_workers=20, workspace_bytes=128, tiling_bytes=200)

        def _generate(_soc, *args):
            *dimensions, block, capacity = args
            calls.append(tuple(dimensions))
            block.raw = struct.pack("<7I", *dimensions) + bytes(capacity - 28)
            return 0

        tiler.generate = _generate
        plan = _linear(1024, hidden=131, output=73, input_mask=(False, True))
        binding = bind_dense_backward_tiles(plan)
        bank = tiler.bank(plan, binding)
        self.assertEqual(calls, [(131, 64, 73, 32, 1024, 1, 0)])
        self.assertEqual(len(bank), 224)
        self.assertEqual(tiler.export_manifest(binding, bank)["matrix_access"], "sequential_sdk_unproved")
        with self.assertRaisesRegex(ValueError, "unbound"):
            unsupported = _ffn(65)
            tiler.bank(unsupported, bind_dense_backward_tiles(unsupported))
        with self.assertRaisesRegex(ValueError, "canonical"):
            tiler.bank(plan, replace(binding, matmul_shapes=((131, 73, 512, True, False),)))
        self.assertEqual(len(calls), 1)
