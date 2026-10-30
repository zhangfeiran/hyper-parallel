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
"""CPU interface regressions for native combine plan ownership and its literal adjoint."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import native_combine, torch_route


class _LiteralKernel:
    """Supply independent eager kernels while exercising the real autograd bridge on CPU."""

    def __init__(self, hidden: int, top_k: int, device_index: int) -> None:
        self.hidden = hidden
        self.top_k = top_k

    def __call__(self, expert: torch.Tensor, scores: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        output = expert.new_zeros((expert.shape[0] // self.top_k, self.hidden))
        return output.index_add_(0, order // self.top_k, expert * scores)

    def gradients(self, expert: torch.Tensor, scores: torch.Tensor, order: torch.Tensor,
                  dy: torch.Tensor, out: torch.Tensor, need_weight_grad: bool) -> torch.Tensor:
        gathered = dy.index_select(0, order // self.top_k)
        torch.mul(gathered, scores, out=out)
        return gathered * expert if need_weight_grad else expert[:0]


class TestNativeCombine(unittest.TestCase):
    """Keep native activation lazy and exercise state, masks, aliases and shape-plan reuse."""

    def tearDown(self) -> None:
        native_combine._kernel.cache_clear()
        native_combine._require_payload.cache_clear()
        native_combine._device_supported.cache_clear()

    def test_missing_or_wrong_payload_is_rejected_before_device_work(self) -> None:
        for version in (None, 1, 2):
            namespace = SimpleNamespace()
            if version is not None:
                namespace.mega_moe_native_combine_version = Mock(return_value=version)
            with (
                self.subTest(version=version),
                patch.object(native_combine.multicore_ops, "_load_native"),
                patch.object(torch, "ops", SimpleNamespace(hyper_parallel=namespace)),
            ):
                native_combine._require_payload.cache_clear()
                if version == 2:
                    native_combine._require_payload()
                else:
                    with self.assertRaisesRegex(RuntimeError, "payload ABI 2"):
                        native_combine._require_payload()

    def test_cpu_and_unsupported_shapes_never_query_native_capabilities(self) -> None:
        expert = torch.zeros(24, 16, dtype=torch.bfloat16)
        weights = torch.zeros(4, 6)
        order = torch.arange(24)
        with patch.object(native_combine, "_device_supported") as query:
            self.assertFalse(native_combine._is_supported(expert, order, weights))
            fake = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), dtype=torch.float32,
                                   shape=(24, 16), is_contiguous=lambda: True)
            self.assertFalse(native_combine._is_supported(fake, order, weights))
            fake.dtype = torch.bfloat16
            self.assertFalse(native_combine._is_supported(fake, order, torch.zeros(4, 9)))
        query.assert_not_called()

    def test_token_changes_reuse_one_plan_and_preserve_full_gradients(self) -> None:
        torch.manual_seed(39)
        create = Mock(side_effect=_LiteralKernel)
        with (
            patch.object(native_combine, "_NativeCombine", create),
            patch.object(torch_route, "_is_supported", return_value=True),
        ):
            for tokens in (4, 9):
                for need_router in (False, True):
                    with self.subTest(tokens=tokens, need_router=need_router):
                        source = torch.randn(tokens * 6, 16, dtype=torch.bfloat16)
                        scores = torch.randn(tokens, 6)
                        order = torch.randperm(tokens * 6)
                        dy = torch.randn(16, tokens, dtype=torch.bfloat16).T
                        a = source.clone().requires_grad_()
                        b = scores.clone().requires_grad_(need_router)
                        c = source.clone().requires_grad_()
                        d = scores.clone().requires_grad_(need_router)
                        weights = b.reshape(-1).bfloat16()[order].unsqueeze(-1)
                        expected = a.new_zeros((tokens, 16)).index_add_(0, order // 6, a * weights)
                        actual = torch_route.restore_torch_output(c, order, d)
                        native_inputs = (a, b) if need_router else (a,)
                        fused_inputs = (c, d) if need_router else (c,)
                        expected_grads = torch.autograd.grad(expected, native_inputs, dy)
                        actual_grads = torch.autograd.grad(actual, fused_inputs, dy)
                        for reference, candidate in ((expected, actual), *zip(expected_grads, actual_grads)):
                            bits = torch.int16 if reference.dtype == torch.bfloat16 else torch.int32
                            self.assertTrue(torch.equal(reference.view(bits), candidate.view(bits)))
        create.assert_called_once_with(16, 6, None)

    def test_gradient_output_cannot_overwrite_a_source(self) -> None:
        backend = object.__new__(native_combine._NativeCombine)
        expert = torch.randn(4, 16, dtype=torch.bfloat16)
        dy = torch.randn_like(expert)
        scores = torch.ones(4, 1, dtype=torch.bfloat16)
        order = torch.arange(4)
        for output in (expert, dy):
            with self.subTest(alias=output is expert), self.assertRaisesRegex(ValueError, "must not alias"):
                backend.gradients(expert, scores, order, dy, output, True)
