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
"""Torch-compatible source and receive ordering must preserve exact route identities."""

import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import torch_route
from hyper_parallel.core.multicore.modules.mega_moe import route as moe_route
from hyper_parallel.core.multicore.modules.mega_moe import function as moe_function
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import TaskType
from hyper_parallel.distributed.expert_parallel.experts import _aggregate_ep_outputs


class TestTorchRoute(unittest.TestCase):
    """Exercise exact routing without distributed initialization or accelerator kernels."""

    def test_source_order_preserves_repeated_keys_empty_inputs_and_float_sort(self) -> None:
        """Retain literal source indices and the output lease for each sorting policy."""
        generator = torch.Generator().manual_seed(630)
        spec = SimpleNamespace(local_experts=6, num_experts=96, ep_size=16, top_k=6)
        for tokens in (0, 1, 17, 4096):
            ids = torch.randint(spec.num_experts, (tokens, spec.top_k), generator=generator)
            hidden = torch.randn(tokens, 7, generator=generator).bfloat16()
            keys = ids.flatten().div(spec.local_experts, rounding_mode="floor") * spec.num_experts + ids.flatten()
            for float_sort in (False, True):
                with self.subTest(tokens=tokens, float_sort=float_sort), patch.object(
                        torch_route, "_SORT_FP32_ENABLED", float_sort):
                    expected = (keys.float() if float_sort else keys).argsort()
                    destination = torch.full((tokens * spec.top_k + 1, 7), float("nan"), dtype=hidden.dtype)
                    routed, order = torch_route.permute_torch_input(hidden, ids, spec, destination)
                    self.assertTrue(torch.equal(order, expected), f"Expected order {expected}, got {order}")
                    expected_values = hidden[expected // spec.top_k]
                    self.assertTrue(torch.equal(routed, expected_values),
                                    f"Expected values {expected_values}, got {routed}")
                    self.assertEqual(routed.untyped_storage().data_ptr(), destination.untyped_storage().data_ptr())
                    self.assertTrue(destination[-1].isnan().all(), f"Guard row overwritten: {destination[-1]}")

    def test_order_preparation_captures_fresh_events_and_reuses_workspace_stream(self) -> None:
        """Capture each count producer without submitting the backward-only row mapping."""
        counts = Mock()
        workspace = SimpleNamespace(weight_grad_stream=None)
        stream, current = Mock(), Mock()
        events = [Mock(), Mock()]
        with (
            patch.object(torch.npu, "current_stream", return_value=current),
            patch.object(torch.npu, "Stream", return_value=stream) as create_stream,
            patch.object(torch.npu, "Event", side_effect=events),
            patch.object(moe_route, "received_torch_order") as build,
        ):
            for event in events:
                actual, pending, ready = moe_route._prepare_weight_grad_order(
                    counts, 7, SimpleNamespace(route_policy="torch"), workspace)
                self.assertIsNone(actual)
                self.assertIs(pending, stream)
                self.assertIs(ready, event)
                event.record.assert_called_once_with(current)
        create_stream.assert_called_once_with(device=counts.device)
        build.assert_not_called()
        stream.wait_stream.assert_not_called()
        counts.record_stream.assert_not_called()

    def test_order_stream_is_not_created_for_cann_or_without_workspace(self) -> None:
        """Keep the CANN path unchanged and retain standalone synchronous preparation."""
        counts = torch.tensor([[0, 2], [1, 0]])
        expected = torch_route.received_torch_order(counts, 3)
        with patch.object(torch.npu, "Stream") as create_stream:
            actual, stream, ready = moe_route._prepare_weight_grad_order(
                counts, 3, SimpleNamespace(route_policy="cann"), SimpleNamespace())
            self.assertIsNone(actual)
            self.assertIsNone(stream)
            self.assertIsNone(ready)
            actual, stream, ready = moe_route._prepare_weight_grad_order(
                counts, 3, SimpleNamespace(route_policy="torch"), None)
        create_stream.assert_not_called()
        self.assertIsNone(stream)
        self.assertIsNone(ready)
        self.assertTrue(torch.equal(actual, expected))

    def test_cann_empty_pull_skips_native_permutation(self) -> None:
        """Policy selection must retain the existing empty-source CANN bypass."""
        hidden = torch.empty(0, 4, dtype=torch.bfloat16)
        ids = torch.empty(0, 2, dtype=torch.int64)
        output = torch.empty(8, 4, dtype=torch.bfloat16)[:0]
        spec = SimpleNamespace(route_policy="cann")
        with patch.object(moe_route.multicore_ops, "moe_token_permute_out") as permute:
            routed, mapping = moe_route._permute_for_policy(hidden, ids, spec, output)
        permute.assert_not_called()
        self.assertIs(routed, output)
        self.assertEqual(mapping.shape, (0,))
        self.assertEqual(mapping.dtype, ids.dtype)

    def test_receive_sort_retains_literal_keys_dtype_and_contiguous_layout(self) -> None:
        """Keep the backend sort input unchanged, including duplicate-key tie behavior."""
        matrices = (torch.zeros(2, 3, dtype=torch.int32), torch.tensor([[0, 0], [1, 0]]),
                    torch.tensor([[3, 0, 2], [1, 4, 0]], dtype=torch.int32),
                    torch.arange(96).remainder(7).reshape(8, 12))
        original_sort = torch_route._sort_keys
        for counts in matrices:
            peers, experts = counts.shape
            expected = torch.repeat_interleave(torch.arange(experts).repeat(peers), counts.reshape(-1).long())
            captured = []

            def inspect(keys: torch.Tensor, bound: int) -> torch.Tensor:
                self.assertEqual(keys.dtype, expected.dtype)
                self.assertTrue(keys.is_contiguous())
                self.assertEqual(bound, experts)
                self.assertTrue(torch.equal(keys, expected))
                captured.append(True)
                return original_sort(keys, bound)

            with self.subTest(counts=counts.tolist()), patch.object(torch_route, "_sort_keys", side_effect=inspect):
                torch_route.received_torch_order(counts, max(1, expected.numel()))
            self.assertEqual(captured, [True])

    def test_receive_order_matches_native_sort_for_ragged_and_empty_sources(self) -> None:
        """Recover the second unstable sort using counts alone, including empty destinations."""
        generator = torch.Generator().manual_seed(51)
        matrices = [torch.randint(0, 9, (peers, experts), generator=generator)
                    for peers, experts in ((1, 8), (2, 3), (16, 6))]
        matrices.extend((torch.zeros(2, 3, dtype=torch.int64), torch.tensor([[0, 0], [1, 0]]),
                         torch.tensor([[0, 0, 0], [3, 1, 2]])))
        for counts in matrices:
            for use_float32 in (False, True):
                with self.subTest(counts=counts.tolist(), float_sort=use_float32), patch.object(
                        torch_route, "_SORT_FP32_ENABLED", use_float32):
                    local_ids = torch.repeat_interleave(torch.arange(counts.shape[1]).repeat(counts.shape[0]),
                                                        counts.flatten())
                    raw_tags = torch.arange(local_ids.numel())
                    transported = torch.cat([raw_tags[local_ids == expert] for expert in range(counts.shape[1])])
                    order = torch_route.received_torch_order(counts, max(1, raw_tags.numel()))
                    expected = raw_tags[torch_route._sort_keys(local_ids, counts.shape[1])]
                    self.assertTrue(torch.equal(transported[order], expected))

    def test_source_permutation_and_combine_match_native_outputs_and_gradients(self) -> None:
        """Source order and BF16 round boundaries preserve the native gather/combine graph."""
        torch.manual_seed(42)
        spec = SimpleNamespace(local_experts=2, num_experts=4, ep_size=2, top_k=2)
        ids = torch.tensor([[3, 0], [1, 2], [2, 0]], dtype=torch.int64)
        hidden = torch.randn(3, 4).bfloat16()
        expected_order = ((ids.flatten() // 2) * 4 + ids.flatten()).argsort()
        routed, order = torch_route.permute_torch_input(hidden, ids, spec)
        self.assertTrue(torch.equal(order, expected_order))
        self.assertTrue(torch.equal(routed, hidden[expected_order // 2]))
        destination = torch.empty_like(routed)
        direct, _ = torch_route.permute_torch_input(hidden, ids, spec, destination)
        self.assertEqual(direct.data_ptr(), destination.data_ptr())
        self.assertTrue(torch.equal(direct, routed))
        values = torch.randn(6, 4).bfloat16()
        weights = torch.rand(3, 2)
        upstream = torch.randn(3, 4).bfloat16()
        results = []
        for native in (True, False):
            value = values.clone().requires_grad_()
            score = weights.clone().requires_grad_()
            if native:
                result = _aggregate_ep_outputs(value, score.flatten().bfloat16(),
                    torch.arange(3).repeat_interleave(2), expected_order, (1, 3, 4)).reshape(3, 4)
            else:
                result = torch_route.restore_torch_output(value, order, score)
            result.backward(upstream)
            results.append((result, value.grad, score.grad))
        for actual, expected in zip(results[1], results[0]):
            self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))

    def test_direct_combine_adjoint_matches_native_autograd_bits(self) -> None:
        """Preserve BF16 products, broadcast reductions, unique scatter and signed zeros."""
        generator = torch.Generator().manual_seed(715)
        for expert_dtype in (torch.float32, torch.bfloat16):
            for dtype in (torch.float32, torch.bfloat16):
                for tokens, width in ((0, 7), (1, 5120), (3, 17), (9, 5120)):
                    with self.subTest(dtype=dtype, expert_dtype=expert_dtype, tokens=tokens, width=width):
                        order = torch.randperm(tokens * 3, generator=generator)
                        values = torch.randn(tokens * 3, width, generator=generator).to(expert_dtype)
                        weights = torch.randn(tokens, 3, generator=generator).to(dtype)
                        if tokens:
                            values[0, :4] = torch.tensor([0.0, -0.0, 1.0, -1.0]).bfloat16()
                            weights[0, 0] = -0.0
                        values.requires_grad_()
                        weights.requires_grad_()
                        grad_output = torch.randn(tokens, width, generator=generator).to(expert_dtype)
                        reference = torch_route.restore_torch_output(values, order, weights)
                        reference.backward(grad_output)
                        storage = torch.full((tokens * 3 + 1, width), float("nan"), dtype=values.dtype)
                        with torch.no_grad():
                            _, scores, token_ids = torch_route._restore_torch_output_with_state(values, order, weights)
                        actual = torch_route._restore_torch_output_grad_out(
                            values.detach(), order, weights.detach(), grad_output, storage[:-1],
                            scores, token_ids, need_weight_grad=True)
                        self.assertTrue(torch.equal(storage[:-1].view(torch.uint8), values.grad.view(torch.uint8)))
                        self.assertTrue(torch.equal(actual.view(torch.uint8), weights.grad.view(torch.uint8)))
                        self.assertTrue(storage[-1].isnan().all())
                        self.assertIsNone(torch_route._restore_torch_output_grad_out(
                            values.detach(), order, weights.detach(), grad_output, storage[:-1],
                            scores, token_ids, need_weight_grad=False))

    def test_empty_combine_retains_input_and_router_gradient_edges(self) -> None:
        """Zero local tokens remain differentiable while another rank may receive experts."""
        values = torch.empty(0, 4, dtype=torch.bfloat16, requires_grad=True)
        weights = torch.empty(0, 2, requires_grad=True)
        result = torch_route.restore_torch_output(values, torch.empty(0, dtype=torch.int64), weights)
        result.sum().backward()
        self.assertEqual(values.grad.shape, values.shape)
        self.assertEqual(weights.grad.shape, weights.shape)

    def test_ordered_weight_gradients_disable_only_original_weight_gemm_tasks(self) -> None:
        """Preserve every dependency and event trigger when deferring ordered wgrad GEMMs."""
        spec = MegaMoeSpec(local_num_tokens=128, hidden_size=5120, intermediate_size=2304,
            num_experts=8, top_k=6, initial_capacity_factor=None, receive_capacity=1024,
            ep_size=1, ep_group=None, rank_id=0, num_cube_cores=24)
        baseline = _build_runtime_artifacts(spec)[3]
        ordered = _build_runtime_artifacts(replace(spec, route_policy="torch"))[3]
        self.assertEqual(baseline.task_num, ordered.task_num)
        skipped = 0
        for index in range(baseline.task_num):
            before, after = baseline.all_tasks[index], ordered.all_tasks[index]
            self.assertEqual((before.dependent_event, before.trigger_event),
                             (after.dependent_event, after.trigger_event))
            if before.task_type == TaskType.TASK_GROUPED_MATMUL and before.tiling_data_position in (22, 23):
                self.assertEqual(after.task_type, TaskType.TASK_BEGIN_TASK_GRAPH)
                skipped += 1
            else:
                self.assertEqual(before.task_type, after.task_type)
        self.assertEqual(skipped, 2 * spec.local_experts * spec.num_cube_cores)

    def test_ordered_wgrad_uses_two_gemms_and_no_original_gradient_buffers(self) -> None:
        """Each ordered gradient is produced once and returned directly without a full-size placeholder."""
        spec = SimpleNamespace(route_policy="torch", hidden_size=4, intermediate_size=2)
        first, second = torch.empty(2, 4, 4), torch.empty(2, 2, 4)
        gradients = moe_function._allocate_backward_intermediates(spec, 3, torch.ones(3, 4), first, second)
        self.assertEqual(gradients.grad_weight1.numel(), 1)
        self.assertEqual(gradients.grad_weight2.numel(), 1)
        gradients.swiglu_grad.copy_(torch.arange(12).reshape(3, 4))
        saved = SimpleNamespace(dispatch=torch.arange(12).reshape(3, 4).float(),
            activation=torch.arange(6).reshape(3, 2).float(), group_list=torch.tensor([0, 3]),
            weight1=first, weight2=second)
        execution = moe_function._BackwardExecution(torch.ones(3, 4), torch.ones(4, 4), None, None, None, gradients)
        outputs = [torch.ones_like(first), torch.ones_like(second)]
        order = torch.tensor([2, 0, 1])
        with patch.object(
            moe_function.load_torch_npu(), "npu_grouped_matmul", side_effect=[[item] for item in outputs],
        ) as gmm:
            actual = moe_function._ordered_weight_gradients(saved, execution, order)
        self.assertEqual(gmm.call_count, 2)
        self.assertTrue(torch.equal(gmm.call_args_list[0].kwargs["x"][0], saved.dispatch[order].T))
        self.assertTrue(torch.equal(gmm.call_args_list[1].kwargs["x"][0], saved.activation[order].T))
        self.assertIs(actual.intermediates.grad_weight1, outputs[0])
        self.assertIs(actual.intermediates.grad_weight2, outputs[1])
        with patch.object(moe_function.load_torch_npu(), "npu_grouped_matmul") as gmm:
            empty = moe_function._ordered_weight_gradients(saved, execution, order[:0])
        gmm.assert_not_called()
        self.assertTrue(torch.equal(empty.intermediates.grad_weight1, torch.zeros_like(first)))
        self.assertTrue(torch.equal(empty.intermediates.grad_weight2, torch.zeros_like(second)))

    def test_ordered_wgrad_returns_native_eoi_storage(self) -> None:
        """Transpose only external GMM results while keeping empty EOI experts exact zero."""
        raw = torch.arange(12).reshape(3, 4).float()
        saved = SimpleNamespace(dispatch=raw, activation=raw[:, :2].contiguous(),
                                group_list=torch.tensor([0, 3]),
                                weight1=torch.empty(2, 6, 4), weight2=torch.empty(2, 4, 2))
        execution = moe_function._BackwardExecution(raw, None, None, None, None,
            moe_function._BackwardIntermediates(None, None, None, raw, None))
        outputs = [torch.arange(48).reshape(2, 4, 6), torch.arange(16).reshape(2, 2, 4)]
        with patch.object(moe_function.load_torch_npu(), "npu_grouped_matmul", side_effect=[[x] for x in outputs]):
            actual = moe_function._ordered_weight_gradients(
                saved, execution, torch.tensor([2, 0, 1]), expert_weight_layout="EOI")
        for value, expected in zip((actual.intermediates.grad_weight1, actual.intermediates.grad_weight2), outputs):
            self.assertTrue(torch.equal(value, expected.transpose(-1, -2)))
            self.assertTrue(value.is_contiguous())
        with patch.object(moe_function.load_torch_npu(), "npu_grouped_matmul") as gmm:
            empty = moe_function._ordered_weight_gradients(
                saved, execution, torch.empty(0, dtype=torch.int64), expert_weight_layout="EOI")
        gmm.assert_not_called()
        for value, weight in zip((empty.intermediates.grad_weight1, empty.intermediates.grad_weight2),
                                 (saved.weight1, saved.weight2)):
            self.assertTrue(torch.equal(value, torch.zeros_like(weight)))

    def test_preordered_dispatch_skips_only_the_redundant_input_gather(self) -> None:
        """Keep both GEMM operands identical while reducing gathers from four to three."""
        raw = torch.arange(12).reshape(3, 4).float()
        order = torch.tensor([2, 0, 1])
        saved = SimpleNamespace(dispatch=raw, activation=raw[:, :2].contiguous(),
                                group_list=torch.tensor([0, 3]))
        execution = moe_function._BackwardExecution(raw + 20, None, None, None, None,
            moe_function._BackwardIntermediates(None, None, None, raw + 10, None))
        calls = []
        for ordered in (False, True):
            saved.dispatch = raw[order] if ordered else raw
            with (
                patch.object(moe_function.load_torch_npu(), "npu_grouped_matmul", side_effect=[[raw], [raw]]) as gmm,
                patch.object(torch, "index_select", wraps=torch.index_select) as gather,
            ):
                moe_function._ordered_weight_gradients(saved, execution, order, dispatch_ordered=ordered)
            self.assertEqual(gather.call_count, 3 if ordered else 4)
            calls.append(gmm.call_args_list)
        for baseline, candidate in zip(*calls):
            torch.testing.assert_close(baseline.kwargs["x"][0], candidate.kwargs["x"][0])
            torch.testing.assert_close(baseline.kwargs["weight"][0], candidate.kwargs["weight"][0])

    def test_first_gemm_input_is_released_before_second_gemm(self) -> None:
        """A temporary gather must not increase the peak while W2 gradients are produced."""
        raw = torch.arange(12).reshape(3, 4).float()
        order = torch.tensor([2, 0, 1])
        saved = SimpleNamespace(dispatch=raw, activation=raw[:, :2].contiguous(),
                                group_list=torch.tensor([0, 3]))
        execution = moe_function._BackwardExecution(raw + 20, None, None, None, None,
            moe_function._BackwardIntermediates(None, None, None, raw + 10, None))
        original_select = torch.index_select
        captured = []
        calls = []

        def select(*args: object, **kwargs: object) -> torch.Tensor:
            result = original_select(*args, **kwargs)
            if not captured:
                captured.append(weakref.ref(result))
            return result

        def grouped(*, x: list[torch.Tensor], **kwargs: object) -> list[torch.Tensor]:
            del x, kwargs
            if calls:
                self.assertIsNone(captured[0]())
            else:
                self.assertIsNotNone(captured[0]())
            calls.append(True)
            return [raw]

        with (
            patch.object(torch, "index_select", new=select),
            patch.object(moe_function.load_torch_npu(), "npu_grouped_matmul", new=grouped),
        ):
            moe_function._ordered_weight_gradients(saved, execution, order)
        self.assertEqual(len(calls), 2)

    def test_input_restore_is_the_native_indexing_adjoint(self) -> None:
        """Restore duplicate input routes without allocating a dummy differentiable gather."""
        order = torch.tensor([6, 0, 4, 2, 7, 1, 5, 3])
        source = torch.zeros(4, 4, dtype=torch.bfloat16, requires_grad=True)
        derivative = torch.arange(32).reshape(8, 4).bfloat16()
        native, = torch.autograd.grad(source[order // 2], source, derivative)
        spec = SimpleNamespace(route_policy="torch", top_k=2, hidden_size=4)
        context = SimpleNamespace(needs_input_grad=(True,), has_permutation=True, source_rows=8,
                                  token_rows=4, plan=SimpleNamespace(spec=spec))
        actual = moe_function._restore_input_gradient(context, derivative, (order,))
        self.assertTrue(torch.equal(actual.view(torch.int16), native.view(torch.int16)))
