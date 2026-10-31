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

    def __call__(self, expert: torch.Tensor, scores: torch.Tensor, order: torch.Tensor,
                 *, inverse_order: torch.Tensor | None = None) -> torch.Tensor:
        if inverse_order is not None and not torch.equal(order[inverse_order], torch.arange(order.numel())):
            raise ValueError('The combine inverse does not recover the original rows.')
        output = expert.new_zeros((expert.shape[0] // self.top_k, self.hidden))
        return output.index_add_(0, order // self.top_k, expert * scores)

    def gradients(self, expert: torch.Tensor, scores: torch.Tensor, order: torch.Tensor,
                  dy: torch.Tensor, out: torch.Tensor, need_weight_grad: bool,
                  *, source_ordered: bool = False) -> torch.Tensor:
        gathered = dy.index_select(0, order // self.top_k)
        torch.mul(gathered, scores, out=out)
        if not need_weight_grad:
            return expert[:0]
        row_scores = (gathered * expert).sum(dim=-1)
        if source_ordered:
            result = torch.zeros_like(row_scores)
            result.index_put_((order,), row_scores, accumulate=True)
            return result.float()
        return row_scores


class TestNativeCombine(unittest.TestCase):
    """Keep native activation lazy and exercise state, masks, aliases and shape-plan reuse."""

    def tearDown(self) -> None:
        native_combine._kernel.cache_clear()
        native_combine._require_payload.cache_clear()
        native_combine._device_supported.cache_clear()

    def test_fused_scores_require_the_validated_backward_geometry(self) -> None:
        image = native_combine._runtime_image(5120, 6, backward=True, reduce_scores=True)
        events = int.from_bytes(image[12:16], "little")
        task = native_combine.TaskDescC.from_buffer_copy(image, 64 + events * 4)
        self.assertEqual(task.extra_value_2, 1)
        for hidden, top_k, backward in [(16, 6, True), (5120, 2, True), (5120, 6, False)]:
            with self.subTest(hidden=hidden, top_k=top_k, backward=backward):
                with self.assertRaisesRegex(ValueError, "backward H5120/K6"):
                    native_combine._runtime_image(hidden, top_k, backward=backward, reduce_scores=True)

    def test_fused_score_buffer_is_one_aligned_row_per_source(self) -> None:
        backend = object.__new__(native_combine._NativeCombine)
        backend.hidden, backend.top_k = 5120, 6
        backend.grad_configs = (torch.empty(1), torch.empty(1))
        backend.dummy = torch.empty(1, 1, dtype=torch.bfloat16)
        backend.dummy_long = torch.empty(1, dtype=torch.int64)
        backend.dummy_int = torch.empty(1, dtype=torch.int32)
        backend.dummy_byte = torch.empty(1, dtype=torch.uint8)
        expert = torch.ones(12, 5120, dtype=torch.bfloat16)
        out = torch.empty_like(expert)

        def launch(*args: object) -> None:
            """Fill the padded output while checking the private DMA contract."""
            product = args[13]
            self.assertEqual(product.shape, (12, 16))
            product.fill_(0.5)

        with patch.object(backend, "_record"), patch.object(native_combine.multicore_ops, "mega_moe_grad", launch):
            result = backend.gradients(expert, torch.ones(12, 1, dtype=torch.bfloat16), torch.arange(12),
                                       torch.ones(2, 5120, dtype=torch.bfloat16), out, True)
        self.assertEqual(result.shape, (12,))
        self.assertEqual(result.untyped_storage().nbytes(), 12 * 16 * 2)
        self.assertTrue(torch.equal(result, torch.full((12,), 0.5, dtype=torch.bfloat16)))

    def test_restored_score_mode_requires_the_fused_weight_adjoint(self) -> None:
        """Reject metadata combinations that cannot produce source-ordered FP32 scores."""
        image = native_combine._runtime_image(
            5120, 6, backward=True, reduce_scores=True, restore_scores=True)
        events = int.from_bytes(image[12:16], "little")
        task = native_combine.TaskDescC.from_buffer_copy(image, 64 + events * 4)
        self.assertEqual(task.extra_value_2, 2)
        for reduce_scores, need_weight_grad in [(False, True), (True, False)]:
            with self.subTest(reduce_scores=reduce_scores, need_weight_grad=need_weight_grad):
                with self.assertRaisesRegex(ValueError, "require fused score reduction and weight gradients"):
                    native_combine._runtime_image(
                        5120, 6, True, need_weight_grad,
                        reduce_scores=reduce_scores, restore_scores=True)

    def test_restored_scores_have_compact_fp32_storage(self) -> None:
        """Keep each DMA block private and compact only its valid FP32 element."""
        backend = object.__new__(native_combine._NativeCombine)
        backend.hidden, backend.top_k = 5120, 6
        backend.grad_configs = (torch.empty(1), torch.empty(1))
        backend.restore_config = torch.empty(1)
        backend.dummy = torch.empty(1, 1, dtype=torch.bfloat16)
        backend.dummy_long = torch.empty(1, dtype=torch.int64)
        backend.dummy_int = torch.empty(1, dtype=torch.int32)
        backend.dummy_byte = torch.empty(1, dtype=torch.uint8)
        expert = torch.ones(12, 5120, dtype=torch.bfloat16)
        expected = torch.arange(12, dtype=torch.float32)

        def launch(*args: object) -> None:
            """Fill FP32 values through the BF16 private-schema carrier."""
            self.assertEqual(args[13].shape, (12, 16))
            self.assertIs(args[27], backend.restore_config)
            args[13].view(torch.float32)[:, 0].copy_(expected)

        with patch.object(backend, "_record"), patch.object(native_combine.multicore_ops, "mega_moe_grad", launch):
            result = backend.gradients(
                expert, torch.ones(12, 1, dtype=torch.bfloat16), torch.arange(12),
                torch.ones(2, 5120, dtype=torch.bfloat16), torch.empty_like(expert), True,
                source_ordered=True)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(result.is_contiguous())
        self.assertEqual(result.untyped_storage().nbytes(), 12 * 4)
        self.assertTrue(torch.equal(result, expected))

    def test_restored_fp32_route_uses_the_complete_native_adjoint(self) -> None:
        """Consume complete source scores without another zero-fill or IndexPut."""
        expert = torch.ones(12, 5120, dtype=torch.bfloat16)
        weights = torch.ones(2, 6, dtype=torch.float32)
        expected = torch.arange(12, dtype=torch.float32)
        backend = SimpleNamespace(gradients=Mock(return_value=expected))
        with (
            patch.object(torch_route, "_is_supported", return_value=True),
            patch.object(torch_route, "_kernel", return_value=backend),
            patch.object(torch, "zeros") as zeros,
            patch.object(torch.Tensor, "index_put_") as index_put,
        ):
            result = torch_route._restore_torch_output_grad_out(
                expert, torch.arange(12), weights, torch.ones(2, 5120, dtype=torch.bfloat16),
                torch.empty_like(expert), torch.ones(12, 1, dtype=torch.bfloat16), torch.arange(12) // 6,
                need_weight_grad=True)
        self.assertTrue(backend.gradients.call_args.kwargs["source_ordered"])
        self.assertTrue(torch.equal(result, expected.reshape(2, 6)))
        zeros.assert_not_called()
        index_put.assert_not_called()

    def test_pure_sum_has_a_distinct_forward_runtime_flag(self) -> None:
        for sum_only in (False, True):
            with self.subTest(sum_only=sum_only):
                image = native_combine._runtime_image(5120, 6, sum_only=sum_only)
                events = int.from_bytes(image[12:16], "little")
                task = native_combine.TaskDescC.from_buffer_copy(image, 64 + events * 4)
                self.assertEqual(task.task_type, native_combine.TaskType.TASK_NATIVE_BIT_COMBINE)
                self.assertEqual(task.extra_value_2, int(sum_only))

    def test_pure_sum_rejects_unvalidated_geometry_before_device_work(self) -> None:
        kernel = object.__new__(native_combine._NativeCombine)
        kernel.hidden, kernel.top_k = 16, 1
        with self.assertRaisesRegex(ValueError, "H5120/K6"):
            kernel.sum(torch.empty(0, 16), torch.empty(0, dtype=torch.int64))

    def test_empty_input_sum_returns_without_a_device_launch(self) -> None:
        kernel = object.__new__(native_combine._NativeCombine)
        kernel.hidden, kernel.top_k = 5120, 6
        kernel.dummy = torch.empty(1, dtype=torch.bfloat16)
        result = kernel.sum(torch.empty(0, 5120, dtype=torch.bfloat16), torch.empty(0, dtype=torch.int64))
        self.assertEqual(result.shape, (0, 5120))
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_missing_or_wrong_payload_is_rejected_before_device_work(self) -> None:
        for version in (None, 1, 2, 3, 4, 5, 6, 7, 8):
            namespace = SimpleNamespace()
            if version is not None:
                namespace.mega_moe_native_combine_version = Mock(return_value=version)
            with (
                self.subTest(version=version),
                patch.object(native_combine.multicore_ops, "_load_native"),
                patch.object(torch, "ops", SimpleNamespace(hyper_parallel=namespace)),
            ):
                native_combine._require_payload.cache_clear()
                if version == 8:
                    native_combine._require_payload()
                else:
                    with self.assertRaisesRegex(RuntimeError, "payload ABI 8"):
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

    def test_prepared_inverse_reaches_native_without_regeneration(self) -> None:
        backend = object.__new__(native_combine._NativeCombine)
        backend.hidden, backend.top_k = 16, 2
        backend.config = backend.sum_config = torch.empty(1, dtype=torch.uint8)
        backend.dummy = torch.empty(1, 1, dtype=torch.bfloat16)
        backend.dummy_long = torch.empty(1, dtype=torch.int64)
        backend.dummy_int = torch.empty(1, dtype=torch.int32)
        backend.dummy_byte = torch.empty(1, dtype=torch.uint8)
        source = torch.randn(8, 16, dtype=torch.bfloat16)
        order = torch.tensor([3, 0, 7, 5, 1, 6, 4, 2])
        inverse = torch.argsort(order)
        with (patch.object(native_combine, "_inverse_order") as regenerate,
              patch.object(backend, "_record"),
              patch.object(native_combine.multicore_ops, "mega_moe_grad") as launch):
            result = backend(source, torch.ones(8, dtype=torch.bfloat16), order, inverse_order=inverse)
        self.assertEqual(result.shape, (4, 16))
        regenerate.assert_not_called()
        self.assertIs(launch.call_args.args[1], inverse)
        for invalid in (inverse.int(), inverse[:-1]):
            with self.subTest(shape=tuple(invalid.shape), dtype=invalid.dtype):
                with self.assertRaisesRegex(ValueError, "contiguous INT64 permutation"):
                    backend(source, torch.ones(8, dtype=torch.bfloat16), order, inverse_order=invalid)

    def test_gradient_output_cannot_overwrite_a_source(self) -> None:
        backend = object.__new__(native_combine._NativeCombine)
        expert = torch.randn(4, 16, dtype=torch.bfloat16)
        dy = torch.randn_like(expert)
        scores = torch.ones(4, 1, dtype=torch.bfloat16)
        order = torch.arange(4)
        for output in (expert, dy):
            with self.subTest(alias=output is expert), self.assertRaisesRegex(ValueError, "must not alias"):
                backend.gradients(expert, scores, order, dy, output, True)
