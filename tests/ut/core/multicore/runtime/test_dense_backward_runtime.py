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
"""反向运行时拒绝隐式原始张量复制，并保留实际余切与梯度需求。"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.backends.dense_backward import bind_dense_backward_tiles
from hyper_parallel.core.multicore.backends.dense_codegen import saved_value_ids
from hyper_parallel.core.multicore.compiler.dense_backward import compile_dense_backward
from hyper_parallel.core.multicore.compiler.dense_backward_tiles import compile_dense_backward_tiles
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from hyper_parallel.core.multicore.runtime.dense_backward import ResidentDenseBackward
from tests.common.mark_utils import arg_mark


def _cpu_protocol():
    forward = mc.from_source("def linear(x, weight):\n    return ml.matmul(x, weight)\n",
                             signature={"x": ml.Tensor[ml.bf16, (7, 5)], "weight": ml.Tensor[ml.bf16, (5, 9)]},
                             symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1")).plan(mc.DenseSpec({}))
    executable = ResidentDenseBackward.__new__(ResidentDenseBackward)
    executable.plan, executable.device = forward, torch.device("cpu")
    executable.backward = compile_dense_backward(forward)
    executable.tile_plan = compile_dense_backward_tiles(executable.backward, DenseTilePolicy(cube_workers=1),
                                                       input_mask=(False, True))
    executable.binding = bind_dense_backward_tiles(executable.tile_plan)
    executable.saved_ids = saved_value_ids(forward)
    return executable


class TestDenseBackwardRuntime(unittest.TestCase):
    """CPU 只核对真实调用边界，设备数值由独立系统用例负责。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_borrowed_storage_is_preserved_and_strided_cotangents_are_normalized(self):
        """Feature: 实际输入与余切生命周期。
        Description: 建立只请求权重梯度的 CPU 协议，传入带步长余切。
        Expectation: 输入存储保持借用，只有余切归一化，返回梯度具有独立存储。
        """
        executable = _cpu_protocol()
        inputs = (torch.randn(7, 5, dtype=torch.bfloat16), torch.randn(5, 9, dtype=torch.bfloat16))
        gradient = torch.randn(7, 18, dtype=torch.bfloat16)[:, ::2]
        values = executable._values(inputs, (), (gradient,))
        self.assertIs(values[0], inputs[0])
        normalized = values[executable.backward.seeds[0].value.id]
        self.assertTrue(normalized.is_contiguous())
        torch.testing.assert_close(normalized, gradient, atol=0, rtol=0)
        output = values[executable.binding.gradients[1]]
        self.assertEqual(tuple(output.shape), (5, 9))
        self.assertNotEqual(output.untyped_storage().data_ptr(), inputs[1].untyped_storage().data_ptr())
        strided = torch.randn(7, 10, dtype=torch.bfloat16)[:, ::2]
        with self.assertRaisesRegex(ValueError, "contiguous borrowed"):
            executable._values((strided, inputs[1]), (), (gradient,))
        with self.assertRaisesRegex(ValueError, "cotangent contract"):
            executable._values(inputs, (), (None,))
        with self.assertRaisesRegex(ValueError, "sealed type"):
            executable._values(inputs, (), (gradient.float(),))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_output_transpose_rejection_happens_before_build_or_device_allocation(self):
        """Feature: 非空转置输出的能力边界。
        Description: 以设备元数据替身请求转置权重梯度，并监视构建入口。
        Expectation: 未绑定的输出转置先被拒绝，不创建隐式主机补算或替代权重路径。
        """
        plan = mc.from_source("def linear(x, weight):\n    return ml.matmul(x, weight, transpose_right=True)\n",
                              signature={"x": ml.Tensor[ml.bf16, (7, 5)],
                                         "weight": ml.Tensor[ml.bf16, (9, 5)]},
                              symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1")).plan(mc.DenseSpec({}))
        with patch("hyper_parallel.core.multicore.runtime.dense_backward.build_dense_tile_payload") as build:
            with self.assertRaisesRegex(ValueError, "transpose_output"):
                ResidentDenseBackward(plan, SimpleNamespace(type="npu"), cann_root=None, cache_root=None,
                                      soc="Ascend910B3")
        build.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_full_ffn_admits_all_gradients_before_native_build(self):
        """Feature: 完整 FFN 的反向设备绑定。
        Description: 请求所有真实梯度，在原生构建边界使用明确异常停止 CPU 协议。
        Expectation: NN/TN/NT 与激活梯度的完整图通过准入，进入原生构建。
        """
        plan = dense_ffn.plan(mc.DenseSpec({"T": 7, "H": 5, "I": 9, "PackedI": 18}), intermediate_size=9)
        with patch("hyper_parallel.core.multicore.runtime.dense_backward.build_dense_tile_payload",
                   side_effect=RuntimeError("native-build-boundary")) as build:
            with self.assertRaisesRegex(RuntimeError, "native-build-boundary"):
                ResidentDenseBackward(plan, SimpleNamespace(type="npu"), cann_root=None,
                                      cache_root=Path("metadata-only"), soc="Ascend910B3")
        build.assert_called_once()
