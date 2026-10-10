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
"""常规 FFN 的独立 BF16 舍入语义，与真实源投影自动求导对照。"""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional

from hyper_parallel.core.multicore.frontend.examples.common_ffn import common_ffn
from hyper_parallel.core.multicore.language.types import DType, TensorType
from hyper_parallel.core.multicore.primitives.dense_common import multiply_dense, silu_dense
from tests.common.mark_utils import arg_mark


class TestDenseCommon(unittest.TestCase):
    """三投影和两次存储舍入不能被打包 SwiGLU 的单次舍入替代。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_three_original_projections_and_all_gradients_match(self):
        """Feature: 原始 common 语义。
        Description: 对照三个独立 Linear、BF16 SiLU 和 BF16 乘法。
        Expectation: 空输入和非对齐尾部的输出与四组梯度逐值一致。
        """
        torch.manual_seed(17)
        for rows in (0, 1, 7, 129):
            with self.subTest(rows=rows):
                projections = (nn.Linear(17, 19, bias=False, dtype=torch.bfloat16),
                               nn.Linear(17, 19, bias=False, dtype=torch.bfloat16),
                               nn.Linear(19, 17, bias=False, dtype=torch.bfloat16))
                expected_x = torch.randn(rows, 17, dtype=torch.bfloat16, requires_grad=True)
                actual = tuple(value.detach().clone().requires_grad_() for value in
                               (expected_x, *(layer.weight for layer in projections)))
                reference = projections[2](functional.silu(projections[0](expected_x)) * projections[1](expected_x))
                output = common_ffn.interpret(*actual)
                cotangent = torch.randn_like(reference)
                targets = (expected_x, *(layer.weight for layer in projections))
                expected_gradients = torch.autograd.grad(reference, targets, cotangent)
                actual_gradients = torch.autograd.grad(output, actual, cotangent)
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                for value, expected in zip(actual_gradients, expected_gradients):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
        self.assertEqual([operation.logical_name for operation in common_ffn.lower().operations],
                         ["dense.linear", "dense.linear", "dense.silu", "dense.multiply", "dense.linear"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_bf16_storage_boundary_differs_from_packed_activation(self):
        """Feature: 分步 BF16 舍入。
        Description: 用同一实际 BF16 输入，比较分步与打包 FP32 激活。
        Expectation: 分步结果匹配原始 common，且至少一个元素区别于打包契约。
        """
        generator = torch.Generator(device="cpu").manual_seed(31)
        gate = torch.randn(129, 19, generator=generator, dtype=torch.bfloat16)
        up = torch.randn(129, 19, generator=generator, dtype=torch.bfloat16)
        stepped = multiply_dense.schema.reference(silu_dense.schema.reference(gate), up)
        packed = (functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
        torch.testing.assert_close(stepped, functional.silu(gate) * up, rtol=0, atol=0)
        self.assertTrue(bool(torch.any(stepped != packed)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_unsupported_dtype_and_broadcasting_are_rejected(self):
        """Feature: 激活存储契约准入。
        Description: 尝试 FP32 激活或可广播但不一致的乘法形状。
        Expectation: 规范不允许改变 BF16 舍入边界或隐式广播。
        """
        with self.assertRaisesRegex(ValueError, "BF16"):
            silu_dense.schema.infer_types_and_shapes({"value": TensorType(DType.FP32, (7, 19))})
        with self.assertRaisesRegex(ValueError, "without broadcasting"):
            multiply_dense.schema.infer_types_and_shapes({"left": TensorType(DType.BF16, (7, 19)),
                                                          "right": TensorType(DType.BF16, (1, 19))})
