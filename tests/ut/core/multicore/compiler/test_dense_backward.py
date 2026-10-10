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
"""显式反向 recipe 的类型、累加、余切与原始保存状态契约。"""

import json
import unittest
from dataclasses import replace

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.backends.dense import dense_artifacts
from hyper_parallel.core.multicore.compiler.dense_backward import compile_dense_backward
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from tests.common.mark_utils import arg_mark


def _ffn():
    return dense_ffn.plan(mc.DenseSpec({"T": 129, "H": 80, "I": 131, "PackedI": 262}), intermediate_size=131)


class TestDenseBackward(unittest.TestCase):
    """梯度 SSA 与固定数值 recipe 分别接受独立契约检查。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_ffn_has_full_token_contractions_and_ordered_recipes(self):
        """Feature: 完整 FFN 反向。
        Description: 检查五个既有数值调用的类型、依赖和收缩维度。
        Expectation: 权重梯度独立完成完整 token 收缩，行梯度按真实生产关系连接。
        """
        plan = compile_dense_backward(_ffn())
        self.assertEqual([node.kind for node in plan.nodes],
                         ["matmul_left_grad", "matmul_right_grad", "swiglu_input_grad",
                          "matmul_left_grad", "matmul_right_grad"])
        self.assertEqual([node.matrix for node in plan.nodes],
                         [(129, 131, 80, False, True), (131, 80, 129, True, False), None,
                          (129, 80, 262, False, True), (80, 262, 129, True, False)])
        self.assertEqual([node.dependencies for node in plan.nodes], [(), (), (0,), (2,), (2,)])
        self.assertEqual([node.output.type.shape for node in plan.nodes],
                         [(129, 131), (131, 80), (129, 262), (129, 80), (80, 262)])
        self.assertEqual(plan.gradients, (plan.nodes[3].output.id, plan.nodes[4].output.id, plan.nodes[1].output.id))
        self.assertEqual(plan.defined_values((False,)), frozenset())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_all_matmul_transposes_preserve_original_operand_history(self):
        """Feature: Matmul 显式反向的转置。
        Description: 枚举左右转置并检查物理矩阵域与最终梯度视图。
        Expectation: 不重写原始转置链；原始输入形状与梯度返回形状一致。
        """
        for transpose_left in (False, True):
            for transpose_right in (False, True):
                left = (3, 5) if transpose_left else (5, 3)
                right = (7, 3) if transpose_right else (3, 7)
                program = mc.from_source(
                    f"def projection(x, w):\n    return ml.matmul(x, w, transpose_left={transpose_left}, "
                    f"transpose_right={transpose_right})\n",
                    signature={"x": ml.Tensor[ml.fp32, left], "w": ml.Tensor[ml.fp32, right]}, symbols={"ml": ml},
                    schedule=mc.TaskDAG("dense_v1"))
                with self.subTest(left=transpose_left, right=transpose_right):
                    plan = compile_dense_backward(program.plan(mc.DenseSpec({})))
                    first, second = plan.nodes
                    self.assertEqual(first.matrix, (5, 3, 7, False, not transpose_right))
                    self.assertEqual(second.matrix, (3, 7, 5, not transpose_left, False))
                    self.assertEqual((first.output.type.shape, second.output.type.shape), (left, right))
                    self.assertEqual(first.operand_transposes, ((), (transpose_right, True)))
                    self.assertEqual(second.operand_transposes, ((transpose_left, True), ()))
                    self.assertEqual((first.transpose_output, second.transpose_output),
                                     (transpose_left, transpose_right))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_fanout_merges_rounded_contributions_in_original_reverse_order(self):
        """Feature: 分叉梯度累加。
        Description: 两个输出共享行输入，检查各贡献的先后顺序和缺失余切传播。
        Expectation: 先处理后定义的输出，再累加先定义的输出；缺失分支保持未定义。
        """
        program = mc.from_source("def fork(x, a, b):\n    first = ml.matmul(x, a)\n"
                                 "    second = ml.matmul(x, b, transpose_right=True)\n    return first, second\n",
                                 signature={"x": ml.Tensor[ml.bf16, (7, 4)],
                                            "a": ml.Tensor[ml.bf16, (4, 3)], "b": ml.Tensor[ml.bf16, (5, 4)]},
                                 symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1"))
        plan = compile_dense_backward(program.plan(mc.DenseSpec({})))
        self.assertEqual([item.operation for item in plan.contributions], [1, 1, 0, 0])
        merged, = [node for node in plan.nodes if node.kind == "gradient_add"]
        self.assertEqual(merged.inputs, (plan.nodes[0].output.id, plan.nodes[2].output.id))
        defined = plan.defined_values((True, False))
        self.assertNotIn(plan.nodes[0].output.id, defined)
        self.assertIn(merged.output.id, defined)
        self.assertNotIn(plan.gradients[2], defined)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_duplicate_returns_merge_seeds_before_provider_recipe(self):
        """Feature: 重复返回及可选余切。
        Description: 同一个结果返回两次，分别提供不同的余切存在性。
        Expectation: 余切先按返回位置累加，数值 recipe 只执行一次。
        """
        program = mc.from_source("def repeated(x, w):\n    result = ml.matmul(x, w)\n    return result, result\n",
                                 signature={"x": ml.Tensor[ml.bf16, (7, 4)], "w": ml.Tensor[ml.bf16, (4, 3)]},
                                 symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1"))
        plan = compile_dense_backward(program.plan(mc.DenseSpec({})))
        self.assertEqual(plan.nodes[0].kind, "gradient_add")
        self.assertEqual(plan.nodes[0].inputs, tuple(seed.value.id for seed in plan.seeds))
        self.assertEqual(len(plan.contributions), 2)
        for mask in ((True, False), (False, True), (True, True)):
            with self.subTest(mask=mask):
                self.assertIn(plan.nodes[-1].output.id, plan.defined_values(mask))
        self.assertEqual(plan.defined_values((False, False)), frozenset())
        for mask in ((True,), (1, False)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                plan.defined_values(mask)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_rejects_missing_saved_values_recipes_and_reassociation(self):
        """Feature: 显式反向准入。
        Description: 删除保存值、更改 provider recipe 顺序或请求求和重关联。
        Expectation: 损坏的数值及生命周期契约在生成原生调用之前被拒绝。
        """
        plan = _ffn()
        provider = replace(plan.tasks[0].provider, backward=tuple(reversed(plan.tasks[0].provider.backward)))
        broken = (replace(plan, buffers=tuple(replace(buffer, saved_for_backward=False) for buffer in plan.buffers)),
                  replace(plan, tasks=(replace(plan.tasks[0], provider=provider), *plan.tasks[1:])),
                  replace(plan, ir=replace(plan.ir, numeric_policy="reassociate")),
                  replace(plan, tasks=plan.tasks[:-1]))
        for candidate in broken:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                compile_dense_backward(candidate)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_source_artifacts_include_explicit_backward_bindings(self):
        """Feature: 反向计划封存。
        Description: 通过正式产物生成入口导出反向数据流。
        Expectation: 清单保留完整收缩及未绑定状态，源码闭包包含新反向编译器。
        """
        _, _, files = dense_artifacts(_ffn())
        backward = json.loads(files["bindings/backward.json"])
        self.assertEqual(backward["native_status"], "unbound")
        self.assertEqual(backward["nodes"][4]["matrix"][2], 129)
        self.assertIn("compiler/dense_backward.py", json.loads(files["definition.json"])["sources"])
