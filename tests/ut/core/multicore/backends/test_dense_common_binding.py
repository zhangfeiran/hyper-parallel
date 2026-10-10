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
"""常规三投影 FFN 的参数、检查点和实际双向工作队列。"""

from __future__ import annotations

import unittest
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.core.multicore.backends.dense_backward import bind_dense_backward_tiles
from hyper_parallel.core.multicore.backends.dense_codegen import generate_dense_cpp
from hyper_parallel.core.multicore.backends.dense_linear import bind_dense_linear_tiles
from hyper_parallel.core.multicore.compiler.dense_backward import compile_dense_backward
from hyper_parallel.core.multicore.compiler.dense_backward_tiles import (
    compile_dense_backward_tiles,
    simulate_dense_backward_tiles,
)
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy, compile_dense_tiles
from hyper_parallel.core.multicore.frontend.examples.common_ffn import common_ffn
from hyper_parallel.core.multicore.frontend.examples.linear_ffn import linear_ffn
from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.core.multicore.modules.mega_ffn.module import MegaFFN
from tests.common.mark_utils import arg_mark
from tests.ut.core.multicore.backends.test_dense_backward_binding import _decode_queues


class _SourceFFN(nn.Module):
    def __init__(self, hidden: int = 17, intermediate: int = 19) -> None:
        """创建三个独立的无偏置 BF16 源投影。"""
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """通过独立源算子保留两个 BF16 激活结果。"""
        return self.down_proj(functional.silu(self.gate_proj(value)) * self.up_proj(value))


class TestDenseCommonBinding(unittest.TestCase):
    """独立原始算子、权重所有权和解码后的双向队列均须保持声明语义。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_module_and_borrowed_weights_match_independent_three_projection_ffn(self):
        """Feature: 三投影常规 FFN 模块。
        Description: 比较独立原始 FFN，并通过外部借用权重执行另一份调用。
        Expectation: 输出和四组梯度逐值一致，借用权重的地址与版本保持不变。
        """
        for rows in (0, 1, 7, 129):
            with self.subTest(rows=rows):
                torch.manual_seed(13)
                source = _SourceFFN()
                candidate = MegaFFNAdapter(module=source, context={"mega_ffn_program": common_ffn})
                self.assertEqual(candidate.weight_names, ("gate", "up", "down"))
                x = torch.randn(rows, 17, dtype=torch.bfloat16, requires_grad=True)
                baseline_x = x.detach().clone().requires_grad_()
                expected, actual = source(baseline_x), candidate(x)
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                actual.backward(gradient)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, baseline_x.grad, rtol=0, atol=0)
                for value, baseline in zip(candidate.parameters(), source.parameters()):
                    torch.testing.assert_close(value.grad, baseline.grad, rtol=0, atol=0)
                borrowed = tuple(value.detach().clone().requires_grad_() for value in source.parameters())
                identity = tuple((value.data_ptr(), value._version) for value in borrowed)
                module = MegaFFN(17, 19, program=common_ffn, create_parameters=False)
                borrowed_x = baseline_x.detach().clone().requires_grad_()
                output = module(borrowed_x, weights=borrowed)
                output.backward(gradient)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                torch.testing.assert_close(borrowed_x.grad, baseline_x.grad, rtol=0, atol=0)
                for value, baseline in zip(borrowed, source.parameters()):
                    torch.testing.assert_close(value.grad, baseline.grad, rtol=0, atol=0)
                self.assertEqual(identity, tuple((value.data_ptr(), value._version) for value in borrowed))
                module.close()
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    module(borrowed_x, weights=borrowed)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_independent_training_flags_and_checkpoint_restore_original_projections(self):
        """Feature: 三投影所有权与检查点。
        Description: 冻结 up 投影，按正式转换入口恢复原始三权重。
        Expectation: 保留各自的 requires_grad 与 NK 布局；打包路径仍拒绝混合训练策略。
        """
        source = _SourceFFN()
        source.up_proj.weight.requires_grad_(False)
        candidate = MegaFFNAdapter(module=source, context={"mega_ffn_program": common_ffn})
        self.assertEqual([value.requires_grad for value in candidate.parameters()], [True, False, True])
        candidate(torch.randn(7, 17, dtype=torch.bfloat16, requires_grad=True)).sum().backward()
        self.assertIsNone(candidate.up.grad)
        self.assertIsNotNone(candidate.gate.grad)
        candidate._weight_conversions = candidate.make_transforms()
        recovered = revert_weight_conversion(candidate, candidate.state_dict())
        self.assertEqual(set(recovered), set(source.state_dict()))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(recovered[name], value, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "training policies"):
            MegaFFNAdapter(module=source, context={"mega_ffn_program": linear_ffn})
        square = MegaFFNAdapter(module=_SourceFFN(8, 4), context={"mega_ffn_program": linear_ffn})
        self.assertEqual(square.weight_layout, "nk")
        self.assertEqual(tuple(square.down.shape), (8, 4))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_binary_forward_and_backward_queues_match_actual_branches_and_global_joins(self):
        """Feature: 三投影双向设备队列。
        Description: 独立解码 ABI，比较分叉、合并、可选梯度和权重的完整 K 收缩。
        Expectation: 与真实逐 worker 队列一致；事件编号映射和完整行汇合均无缺失。
        """
        for rows in (0, 1, 65, 129):
            for column_workers in (1, 4):
                with self.subTest(rows=rows, columns=column_workers):
                    module = MegaFFN(17, 19, program=common_ffn, create_parameters=False)
                    forward = compile_dense_tiles(module._plan(rows),
                                                  DenseTilePolicy(cube_workers=3, column_workers=column_workers))
                    binding = bind_dense_linear_tiles(forward)
                    mapping = binding.event_map
                    expected = {worker: [(task.operation, task.tile, task.row_begin, task.row_end,
                                          task.column_begin, task.column_end, task.column_tile,
                                          tuple(sorted((mapping[event], count) for event, count in task.dependencies)),
                                          mapping[task.trigger]) for task in tasks]
                                for worker, tasks in forward.queues().items()}
                    self.assertEqual(_decode_queues(binding), expected)
                    binding.admit_native()
                    derivative = compile_dense_backward(forward.dense)
                    for mask in ((True,) * 4, (False, True, False, False), (True, True, False, True)):
                        backward = compile_dense_backward_tiles(derivative, forward.policy, input_mask=mask)
                        backward_binding = bind_dense_backward_tiles(backward)
                        backward_binding.admit_native()
                        simulate_dense_backward_tiles(backward)
                        indices = {region.node: index for index, region in enumerate(backward.regions)}
                        expected = {worker: [(indices[task.operation], task.tile, task.row_begin, task.row_end,
                                              task.column_begin, task.column_end, task.column_tile,
                                              task.dependencies, task.trigger) for task in tasks]
                                    for worker, tasks in backward.queues().items()}
                        self.assertEqual(_decode_queues(backward_binding), expected)
                        for region in backward.regions:
                            node = derivative.nodes[region.node]
                            if node.kind == "linear_weight_grad":
                                self.assertEqual(region.matrix[2], rows)
                    with self.assertRaisesRegex(ValueError, "canonical"):
                        bind_dense_linear_tiles(replace(forward, event_count=forward.event_count + 1))
                    generated = generate_dense_cpp(forward.dense, "hp_dense_common_test")
                    self.assertIn("at::silu(", generated)
                    self.assertIn("at::silu_backward(", generated)
