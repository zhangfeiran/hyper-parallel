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
"""完整梯度图选择、变体缓存和关闭后反向的 CPU 调用协议。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from hyper_parallel.core.multicore.runtime.dense import DenseKernelPlan
from hyper_parallel.core.multicore.runtime.dense_backward_router import DenseBackwardRouter
from tests.common.mark_utils import arg_mark


def _glu():
    return mc.from_source("def glu(x):\n    return ml.swiglu_dense(x, intermediate_size=9)\n",
                          signature={"x": ml.Tensor[ml.bf16, (7, 18)]}, symbols={"ml": ml},
                          schedule=mc.TaskDAG("dense_v1")).plan(mc.DenseSpec({}))


class _NativeDouble:
    def __init__(self, _plan: DenseKernelPlan, _device: torch.device, **options: object) -> None:
        """记录梯度需求，替代真实设备构建。"""
        self.input_mask = options["input_mask"]
        self.output_mask = options["output_mask"]
        self.scratch = SimpleNamespace(release_owner=Mock())
        self.native_identity = {"scope": "cpu_protocol_double"}
        self.calls = []

    def __call__(self, inputs: tuple[torch.Tensor, ...], saved: tuple[torch.Tensor, ...],
                 output_grads: tuple[torch.Tensor | None, ...]) -> tuple[torch.Tensor | None, ...]:
        """保存实际传入对象身份，返回与需求一致的协议替身。"""
        self.calls.append((inputs, saved, output_grads))
        return tuple(torch.zeros_like(value) if needed and any(self.output_mask) else None
                     for value, needed in zip(inputs, self.input_mask))


def _router(plan, host, **options):
    return DenseBackwardRouter(plan, torch.device("cpu"), host, cann_root=None, cache_root=None,
                               soc="metadata-double", policy=DenseTilePolicy(cube_workers=1), **options)


class TestDenseBackwardRouter(unittest.TestCase):
    """主机替身仅核对选择及生命周期，真实数值由 NPU 系统用例负责。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_unbound_output_transpose_uses_one_complete_host_vjp(self):
        """Feature: 非空转置输出的未绑定能力。
        Description: 请求转置权重的全部梯度，并监视设备构建入口。
        Expectation: 整次调用使用原主机 VJP，不能拼接部分设备梯度。
        """
        plan = mc.from_source("def linear(x, weight):\n    return ml.matmul(x, weight, transpose_right=True)\n",
                              signature={"x": ml.Tensor[ml.bf16, (7, 5)],
                                         "weight": ml.Tensor[ml.bf16, (9, 5)]},
                              symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1")).plan(mc.DenseSpec({}))
        values = tuple(torch.zeros(shape, dtype=torch.bfloat16) for shape in ((7, 5), (9, 5)))
        host = Mock(return_value=values)
        router = _router(plan, host)
        delta = torch.zeros(7, 9, dtype=torch.bfloat16)
        with patch("hyper_parallel.core.multicore.runtime.dense_backward_router.ResidentDenseBackward") as native:
            self.assertEqual(len(router(values, (), (delta,), (True, True))), 2)
            self.assertEqual(len(router(values, (), (delta,), (True, True))), 2)
        native.assert_not_called()
        self.assertEqual(host.call_count, 2)
        statistics = router.statistics()
        self.assertEqual(statistics["execution_mode"], "native_host_stream_adapter")
        self.assertEqual(statistics["cached_variants"], 1)
        self.assertEqual(statistics["unbound_capabilities"], ["transpose_output"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_full_ffn_requests_all_three_device_gradients(self):
        """Feature: 完整 FFN 的设备梯度图。
        Description: 使用真实三个输入需求及输出余切，核对完整设备反向的选择。
        Expectation: dX、gate/up 和 down 梯度属于同次设备调用，主机 VJP 不执行。
        """
        plan = dense_ffn.plan(mc.DenseSpec({"T": 7, "H": 5, "I": 9, "PackedI": 18}), intermediate_size=9)
        values = tuple(torch.zeros(shape, dtype=torch.bfloat16) for shape in ((7, 5), (5, 18), (9, 5)))
        host = Mock()
        router = _router(plan, host)
        delta = torch.zeros(7, 5, dtype=torch.bfloat16)
        with patch("hyper_parallel.core.multicore.runtime.dense_backward_router.ResidentDenseBackward",
                   side_effect=_NativeDouble) as native:
            actual = router(values, (), (delta,), (True, True, True))
        native.assert_called_once()
        self.assertEqual(len(actual), 3)
        self.assertTrue(all(value is not None for value in actual))
        host.assert_not_called()
        self.assertEqual(router.statistics()["execution_mode"], "resident_dense_backward")
        self.assertEqual(router.statistics()["unbound_capabilities"], [])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_bound_native_cache_retirement_and_pending_backward_after_close(self):
        """Feature: 原生变体的缓存和退休。
        Description: 触发 LRU 淘汰及关闭，在活跃调用退出之前检查 scratch 所有权。
        Expectation: 缓存有界，退休后只在最后活跃使用结束时释放，关闭后仍可完成反向。
        """
        host = Mock()
        router = _router(_glu(), host, max_cached_variants=1)
        inputs = (torch.zeros(7, 18, dtype=torch.bfloat16),)
        delta = torch.zeros(7, 9, dtype=torch.bfloat16)
        instances = []

        def _build(*args, **options):
            native = _NativeDouble(*args, **options)
            instances.append(native)
            return native

        with patch("hyper_parallel.core.multicore.runtime.dense_backward_router.ResidentDenseBackward",
                   side_effect=_build):
            router(inputs, (), (delta,), (True,))
            held = router._acquire((True,), (True,))
            router(inputs, (), (None,), (True,))
            self.assertEqual(router.statistics()["cached_variants"], 1)
            instances[0].scratch.release_owner.assert_not_called()
            held.uses -= 1
            router._release(held)
            instances[0].scratch.release_owner.assert_called_once_with(instances[0])
            router.close()
            instances[1].scratch.release_owner.assert_called_once_with(instances[1])
            result = router(inputs, (), (delta,), (True,))
            self.assertEqual(tuple(result[0].shape), (7, 18))
            instances[2].scratch.release_owner.assert_called_once_with(instances[2])
        host.assert_not_called()
        self.assertEqual(router.statistics()["native_calls"], 3)
        self.assertEqual(router.statistics()["cached_variants"], 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_cached_nonboolean_mask_cannot_alias_a_true_route(self):
        """Feature: 热缓存的精确布尔需求。
        Description: 先建立 True 变体，再传入与其缓存键相等的整数或浮点数。
        Expectation: 缓存命中之前拒绝错误类型，不执行设备或主机反向。
        """
        host = Mock()
        router = _router(_glu(), host)
        values = (torch.zeros(7, 18, dtype=torch.bfloat16),)
        delta = torch.zeros(7, 9, dtype=torch.bfloat16)
        with patch("hyper_parallel.core.multicore.runtime.dense_backward_router.ResidentDenseBackward",
                   side_effect=_NativeDouble) as native:
            router(values, (), (delta,), (True,))
            for invalid in (1, 1.0):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "input positions"):
                    router(values, (), (delta,), (invalid,))
            native.assert_called_once()
        host.assert_not_called()
        self.assertEqual(router.statistics()["native_calls"], 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_invalid_masks_cannot_create_native_or_fall_back_to_host(self):
        """Feature: 控制元数据准入。
        Description: 传入非布尔输入需求或错误余切数量。
        Expectation: 计划错误明确抛出，不创建新设备入口，也不隐式退回主机。
        """
        host = Mock()
        router = _router(_glu(), host)
        with patch("hyper_parallel.core.multicore.runtime.dense_backward_router.ResidentDenseBackward") as native:
            with self.assertRaisesRegex(ValueError, "input positions"):
                router((), (), (None,), (1,))
            with self.assertRaisesRegex(ValueError, "return positions"):
                router((), (), (), (True,))
        native.assert_not_called()
        host.assert_not_called()
