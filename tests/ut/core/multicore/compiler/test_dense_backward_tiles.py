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
"""反向矩形域、全 token 收缩以及真实队头阻塞关系的 CPU 验证。"""

import unittest
from collections import Counter
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.nn import functional

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.compiler.dense_backward import compile_dense_backward
from hyper_parallel.core.multicore.compiler.dense_backward_tiles import (
    compile_dense_backward_tiles,
    simulate_dense_backward_tiles,
)
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from tests.common.mark_utils import arg_mark


def _plan(rows, hidden=80, intermediate=131, **policy):
    forward = dense_ffn.plan(mc.DenseSpec({"T": rows, "H": hidden, "I": intermediate, "PackedI": 2 * intermediate}),
                            intermediate_size=intermediate)
    return compile_dense_backward_tiles(compile_dense_backward(forward), DenseTilePolicy(**policy))


def _shared_plan(rows, hidden=513, columns=37, branches=3, transpose=False, **policy):
    statements = [f"    result_{index} = ml.matmul(x, weight, transpose_right={transpose})"
                  for index in range(branches)]
    returned = ", ".join(f"result_{index}" for index in range(branches))
    source = "def shared(x, weight):\n" + "\n".join(statements) + f"\n    return {returned}\n"
    shape = (columns, hidden) if transpose else (hidden, columns)
    forward = mc.from_source(source, signature={"x": ml.Tensor[ml.bf16, (rows, hidden)],
                                               "weight": ml.Tensor[ml.bf16, shape]},
                             symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1")).plan(mc.DenseSpec({}))
    return compile_dense_backward_tiles(compile_dense_backward(forward), DenseTilePolicy(**policy),
                                        input_mask=(False, True))


def _forward_values(forward, inputs):
    values = {value.id: tensor for value, tensor in zip(forward.ir.inputs, inputs)}
    for operation in forward.ir.operations:
        args = dict(operation.arguments)
        if operation.logical_name == "dense.matmul":
            left, right = values[args["left"].id], values[args["right"].id]
            result = (left.t() if args["transpose_left"] else left) @ (
                right.t() if args["transpose_right"] else right)
        else:
            packed = values[args["packed"].id]
            gate, up = packed.float().chunk(2, dim=-1)
            result = (functional.silu(gate) * up).to(packed.dtype)
        values[operation.outputs[0].id] = result
    return values


def _replay(plan, inputs, cotangents):
    values = _forward_values(plan.backward.forward, inputs)
    for seed, cotangent in zip(plan.backward.seeds, cotangents):
        if cotangent is not None:
            values[seed.value.id] = cotangent
    matrices = {}
    for region in plan.regions:
        node = plan.backward.nodes[region.node]
        if region.matrix is not None and region.kind != "zero":
            matrices[node.index] = torch.empty(region.matrix[:2], dtype=torch.bfloat16)
            values[node.output.id] = matrices[node.index].t() if node.transpose_output else matrices[node.index]
        else:
            values[node.output.id] = torch.empty(node.output.type.shape, dtype=torch.bfloat16)
    for value, source in plan.aliases:
        values[value] = values[source]
    regions = {region.node: region for region in plan.regions}
    for task in simulate_dense_backward_tiles(plan):
        node, region = plan.backward.nodes[task.operation], regions[task.operation]
        rows, columns = slice(task.row_begin, task.row_end), slice(task.column_begin, task.column_end)
        if region.kind == "zero":
            values[node.output.id][rows, columns] = 0
        elif node.matrix is not None:
            primal = values[node.primal_inputs[0]]
            gradient = values[node.gradient_source]
            left, right = (gradient, primal) if node.kind == "matmul_left_grad" else (primal, gradient)
            left = left.t() if node.matrix[3] else left
            right = right.t() if node.matrix[4] else right
            matrices[node.index][rows, columns] = left[rows] @ right[:, columns]
        elif region.kind == "gradient_add":
            first, second = (values[value][rows, columns] for value in node.inputs)
            values[node.output.id][rows, columns] = first + second
        else:
            gate, up = values[node.primal_inputs[0]][rows].float().chunk(2, dim=-1)
            gate, up = gate[:, columns], up[:, columns]
            delta = values[node.gradient_source][rows, columns].float()
            sigmoid = gate.sigmoid()
            target = values[node.output.id]
            target[rows, columns] = (delta * up * sigmoid * (1 + gate * (1 - sigmoid))).to(target.dtype)
            target[rows, slice(task.column_begin + region.columns, task.column_end + region.columns)] = (
                delta * functional.silu(gate)).to(target.dtype)
    return tuple(values.get(value) for value in plan.backward.gradients)


class TestDenseBackwardTiles(unittest.TestCase):
    """每个输出片段收缩完整 K，使用实际 worker 队头回放检验调度。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_shared_weights_accumulate_without_waiting_for_future_windows(self):
        """Feature: 多个权重贡献的逐行累加。
        Description: 用多种预取窗口回放同一权重的连续梯度相加，包括空 token。
        Expectation: 同一权重行仅等待该行完整贡献，不阻塞未来的 Vector 生产者。
        """
        for tokens in (0, 7, 129):
            for branches in (2, 3, 4):
                for workers in (1, 3, 20):
                    for height in (8, 64, 224):
                        with self.subTest(tokens=tokens, branches=branches, workers=workers, height=height):
                            plan = _shared_plan(tokens, branches=branches, cube_workers=workers,
                                                rows_per_tile=height, column_workers=4, prefetch_tiles=1)
                            self.assertEqual(len(simulate_dense_backward_tiles(plan)), len(plan.tasks))
                            kinds = {region.node: region.kind for region in plan.regions}
                            for task in plan.tasks:
                                if kinds[task.operation] == "gradient_add":
                                    self.assertEqual(len(task.dependencies), 2)
                            self.assertTrue(all(region.matrix[2] == tokens for region in plan.regions
                                                if region.kind == "matmul"))
        with patch("hyper_parallel.core.multicore.compiler.dense_backward_tiles.row_local_dependency",
                   side_effect=lambda region, parent: region.phase == parent.phase == 0):
            old = _shared_plan(7, cube_workers=3, rows_per_tile=64, prefetch_tiles=1)
        with self.assertRaisesRegex(ValueError, "deadlock"):
            simulate_dense_backward_tiles(old)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_zero_weight_domains_follow_logical_output_coordinates(self):
        """Feature: 转置权重的空 token 输出域。
        Description: 零收缩 recipe 的原始矩阵域与实际权重逻辑行不同。
        Expectation: 零填充覆盖逻辑输出域，后续相加沿相同行依赖，所有元素只写一次。
        """
        plan = _shared_plan(0, hidden=37, columns=513, transpose=True,
                            cube_workers=3, rows_per_tile=64, prefetch_tiles=1, column_workers=4)
        self.assertEqual(len(simulate_dense_backward_tiles(plan)), len(plan.tasks))
        for region in plan.regions:
            self.assertEqual((region.rows, region.columns), (513, 37))
            if region.kind == "zero":
                self.assertEqual(region.matrix, (37, 513, 0, True, False))
            owners = Counter((row, column) for task in plan.tasks if task.operation == region.node
                             for row in range(task.row_begin, task.row_end)
                             for column in range(task.column_begin, task.column_end))
            self.assertEqual(owners, Counter((row, column) for row in range(513) for column in range(37)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_shared_weight_numeric_order_matches_independent_autograd(self):
        """Feature: 多贡献权重梯度的数值顺序。
        Description: 与独立完整图比较三次 Matmul 的权重梯度和原始贡献相加顺序。
        Expectation: 不产生提前舍入的 token 部分收缩，零填充、矩阵及相加回放正确。
        """
        generator = torch.Generator(device="cpu").manual_seed(823)
        for tokens in (0, 7, 129):
            with self.subTest(tokens=tokens):
                plan = _shared_plan(tokens, cube_workers=3, rows_per_tile=64,
                                    prefetch_tiles=1, column_workers=4)
                inputs = tuple(torch.randn(shape, dtype=torch.bfloat16, generator=generator) * 0.05
                               for shape in ((tokens, 513), (513, 37)))
                cotangents = tuple(torch.randn(tokens, 37, dtype=torch.bfloat16, generator=generator) * 0.05
                                   for _ in range(3))
                actual = _replay(plan, inputs, cotangents)
                weight = inputs[1].detach().clone().requires_grad_()
                outputs = tuple(inputs[0] @ weight for _ in range(3))
                expected, = torch.autograd.grad(outputs, (weight,), cotangents)
                contributions = tuple(inputs[0].t() @ delta for delta in cotangents)
                ordered = (contributions[2] + contributions[1]) + contributions[0]
                self.assertIsNone(actual[0])
                torch.testing.assert_close(actual[1], expected, rtol=0.02, atol=0.002)
                torch.testing.assert_close(actual[1], ordered, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_worker_queues_finish_for_token_tails_and_multiple_windows(self):
        """Feature: 两阶段反向工作队列。
        Description: 枚举行尾部、worker 数、预取窗口和输出列分区。
        Expectation: 各 worker 完成全部行窗口后进入完整收缩，不产生阻塞环。
        """
        for rows in (0, 1, 7, 65, 129, 257):
            for workers in (1, 3, 20):
                for prefetch in (1, 2, 4):
                    for columns in (1, 4):
                        with self.subTest(rows=rows, workers=workers, prefetch=prefetch, columns=columns):
                            plan = _plan(rows, cube_workers=workers, prefetch_tiles=prefetch, column_workers=columns)
                            self.assertEqual(len(simulate_dense_backward_tiles(plan)), len(plan.tasks))
                            phases = {region.node: region.phase for region in plan.regions}
                            for queue in plan.queues().values():
                                sequence = [phases[task.operation] for task in queue]
                                self.assertEqual(sequence, sorted(sequence))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_numerical_queue_replay_matches_independent_whole_autograd(self):
        """Feature: 反向矩形域的数值回放。
        Description: 执行真实队列片段，与完整 Torch 图的独立 autograd 比较。
        Expectation: 输入及两组权重梯度正确，空 token、非对齐通道和尾部均被覆盖。
        """
        generator = torch.Generator(device="cpu").manual_seed(73)
        for rows in (0, 1, 7, 65, 129):
            with self.subTest(rows=rows):
                plan = _plan(rows, hidden=17, intermediate=19, cube_workers=3, column_workers=4)
                inputs = tuple(torch.randn(shape, dtype=torch.bfloat16, generator=generator) * 0.05
                               for shape in ((rows, 17), (17, 38), (19, 17)))
                cotangent = torch.randn((rows, 17), dtype=torch.bfloat16, generator=generator)
                actual = _replay(plan, inputs, (cotangent,))
                reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
                forward = _forward_values(plan.backward.forward, reference_inputs)
                output = forward[plan.backward.forward.ir.outputs[0].id]
                expected = torch.autograd.grad(output, reference_inputs, cotangent)
                for gradient, target in zip(actual, expected):
                    torch.testing.assert_close(gradient, target, rtol=0.02, atol=0.002)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_weight_gradient_waits_for_every_token_producer_and_keeps_full_k(self):
        """Feature: 跨轴梯度收缩。
        Description: 检查 dWgu 的输入汇合及与原 token 数不同的输出矩形域。
        Expectation: 每片权重输出收缩完整 token 轴，并等待所有 dP 行片段。
        """
        plan = _plan(129, cube_workers=3, column_workers=4)
        packed = next(region for region in plan.regions if region.kind == "swiglu_grad")
        weight = plan.regions[-1]
        self.assertEqual(weight.matrix[:3], (80, 262, 129))
        self.assertEqual((packed.columns, packed.signals), (131, 6))
        expected = tuple((event, packed.signals) for event in packed.events)
        self.assertTrue(all(task.dependencies == expected for task in plan.tasks if task.operation == weight.node))
        for region in plan.regions:
            coverage = Counter((row, column) for task in plan.tasks if task.operation == region.node
                               for row in range(task.row_begin, task.row_end)
                               for column in range(task.column_begin, task.column_end))
            self.assertEqual(coverage, Counter((row, column) for row in range(region.rows)
                                              for column in range(region.columns)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_window_first_order_deadlocks_on_global_contraction(self):
        """Feature: 全局消费者的调度位置。
        Description: 把权重收缩错误地放到第一个行预取窗口之后。
        Expectation: 实际队头回放发现消费者阻塞后续生产者，正确阶段顺序可完成。
        """
        plan = _plan(129, hidden=8, intermediate=6, cube_workers=1, prefetch_tiles=1)
        phases = {region.node: region.phase for region in plan.regions}
        broken = {worker: tuple(sorted(tasks, key=lambda task: (
            task.tile, phases[task.operation], task.operation, task.column_tile)))
                  for worker, tasks in plan.queues().items()}
        with patch.object(type(plan), "queues", return_value=broken), self.assertRaisesRegex(ValueError, "deadlock"):
            simulate_dense_backward_tiles(plan)
        self.assertEqual(len(simulate_dense_backward_tiles(plan)), len(plan.tasks))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_empty_lane_signals_and_empty_tokens_produce_weight_zeros(self):
        """Feature: 空 Vector 通道及空 token。
        Description: 删除尾部空通道的完成发布，并检查空 token 的权重矩形。
        Expectation: 缺少发布被拒绝，空 token 权重梯度由零填充任务覆盖完整输出。
        """
        plan = _plan(1, hidden=8, intermediate=6, cube_workers=1)
        empty = next(task for task in plan.tasks if task.worker_kind == "vector" and task.row_begin == task.row_end)
        broken = replace(plan, tasks=tuple(task for task in plan.tasks if task is not empty))
        with self.assertRaisesRegex(ValueError, "deadlock"):
            simulate_dense_backward_tiles(broken)
        empty = _plan(0, hidden=8, intermediate=6, cube_workers=1)
        self.assertTrue(all(region.kind == "zero" for region in empty.regions if region.events))
        self.assertEqual(len(simulate_dense_backward_tiles(empty)), len(empty.tasks))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_partial_cotangents_alias_single_contribution_and_keep_transposed_output(self):
        """Feature: 分叉、可选余切和转置权重梯度。
        Description: 两个 Matmul 共享行输入，仅保留一个输出余切。
        Expectation: 不对单个贡献再次相加；转置输出保留原始梯度类型与物理矩阵域。
        """
        program = mc.from_source("def fork(x, a, b):\n    first = ml.matmul(x, a)\n"
                                 "    second = ml.matmul(x, b, transpose_right=True)\n    return first, second\n",
                                 signature={"x": ml.Tensor[ml.bf16, (65, 4)],
                                            "a": ml.Tensor[ml.bf16, (4, 3)], "b": ml.Tensor[ml.bf16, (5, 4)]},
                                 symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1"))
        backward = compile_dense_backward(program.plan(mc.DenseSpec({})))
        policy = DenseTilePolicy(cube_workers=1)
        for mask in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(mask=mask):
                plan = compile_dense_backward_tiles(backward, policy, mask)
                self.assertEqual(len(simulate_dense_backward_tiles(plan)), len(plan.tasks))
                if mask in ((True, False), (False, True)):
                    self.assertEqual(len(plan.aliases), 1)
                if mask[1]:
                    region = next(region for region in plan.regions if region.transpose_output)
                    self.assertEqual(region.matrix[:3], (4, 5, 65))
                    self.assertEqual(backward.nodes[region.node].output.type.shape, (5, 4))
                if mask == (False, False):
                    self.assertEqual(plan.tasks, ())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_fanout_and_duplicate_cotangents_match_whole_autograd(self):
        """Feature: 分叉和重复余切的数值回放。
        Description: 独立比较全部余切组合、转置权重梯度和重复返回的种子累加。
        Expectation: 只累加已存在且已舍入的贡献，未使用输入梯度保持 None。
        """
        sources = ("def branch(x, a, b):\n    first = ml.matmul(x, a)\n"
                   "    second = ml.matmul(x, b, transpose_right=True)\n    return first, second\n",
                   "def repeated(x, a, b):\n    first = ml.matmul(x, a)\n    return first, first\n")
        generator = torch.Generator(device="cpu").manual_seed(86)
        for source in sources:
            program = mc.from_source(source, signature={"x": ml.Tensor[ml.bf16, (65, 4)],
                                                       "a": ml.Tensor[ml.bf16, (4, 3)],
                                                       "b": ml.Tensor[ml.bf16, (5, 4)]},
                                     symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1"))
            backward = compile_dense_backward(program.plan(mc.DenseSpec({})))
            for mask in ((True, True), (True, False), (False, True), (False, False)):
                with self.subTest(source=source, mask=mask):
                    plan = compile_dense_backward_tiles(backward, DenseTilePolicy(cube_workers=3), mask)
                    inputs = tuple(torch.randn(shape, dtype=torch.bfloat16, generator=generator) * 0.05
                                   for shape in ((65, 4), (4, 3), (5, 4)))
                    cotangents = tuple(torch.randn(seed.value.type.shape, dtype=torch.bfloat16, generator=generator)
                                       if active else None for seed, active in zip(backward.seeds, mask))
                    actual = _replay(plan, inputs, cotangents)
                    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
                    forward = _forward_values(backward.forward, reference_inputs)
                    selected = [(forward[value.id], cotangent) for value, cotangent in
                                zip(backward.forward.ir.outputs, cotangents) if cotangent is not None]
                    if selected:
                        expected = torch.autograd.grad(tuple(item[0] for item in selected), reference_inputs,
                                                       tuple(item[1] for item in selected), allow_unused=True)
                    else:
                        expected = (None,) * len(inputs)
                    for gradient, target in zip(actual, expected):
                        if target is None:
                            self.assertIsNone(gradient)
                        else:
                            torch.testing.assert_close(gradient, target, rtol=0.02, atol=0.002)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_damaged_recipe_or_event_is_rejected_and_plan_is_unbound(self):
        """Feature: 反向计划准入。
        Description: 更改收缩几何和任务事件，检查清单的执行状态。
        Expectation: 损坏输入被拒绝，CPU 队列通过不代表设备反向已放行。
        """
        plan = _plan(7, cube_workers=1)
        node = replace(plan.backward.nodes[0], matrix=(7, 131, 1, False, True))
        backward = replace(plan.backward, nodes=(node, *plan.backward.nodes[1:]))
        with self.assertRaisesRegex(ValueError, "canonical"):
            compile_dense_backward_tiles(backward, plan.policy)
        broken = replace(plan, tasks=(replace(plan.tasks[0], dependencies=((plan.event_count, 1),)), *plan.tasks[1:]))
        with self.assertRaisesRegex(ValueError, "invalid event"):
            simulate_dense_backward_tiles(broken)
        self.assertEqual(plan.export_manifest()["native_status"], "unbound")
