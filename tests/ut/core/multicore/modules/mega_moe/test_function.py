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
"""CPU storage-lifetime tests for the MegaMoe autograd bridge."""

import unittest
import weakref
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.core.multicore.diagnostics import MegaKernelDiagnostics
from hyper_parallel.core.multicore.modules.mega_moe import function as function_module
from hyper_parallel.core.multicore.modules.mega_moe.function import _MegaMoeFunction
from hyper_parallel.core.multicore.modules.mega_moe.torch_route import (
    _restore_torch_output_with_state, restore_torch_output,
)
from tests.common.mark_utils import arg_mark


# The private autograd bridge and saved execution buffers are the contracts under test.
class TestMegaMoeFunction(unittest.TestCase):
    """Exercise real autograd contexts with mocked communication and kernels."""

    # pylint: disable=protected-access

    def test_forward_joins_order_after_launch_and_before_saved_tensor_hooks(self) -> None:
        """Enqueue expert work before joining and protect cross-stream saved-tensor reads."""
        rows = torch.ones(4, 4)
        order, side, current = Mock(), Mock(), Mock()
        context = SimpleNamespace(needs_input_grad=(True,) * 9)
        spec = SimpleNamespace(dispatch_mode="pull", top_k=2)
        plan = SimpleNamespace(spec=spec)
        workspace = Mock(in_use=True)
        execution = SimpleNamespace(dispatch=rows, source=rows, combine=rows,
                                    profile_call=Mock(), intermediates=SimpleNamespace(
                                        up_proj=rows, activation=rows))
        metadata = SimpleNamespace(expert_capacity=4)
        trace = []
        current.wait_stream.side_effect = lambda stream: trace.append(("wait", stream))
        order.record_stream.side_effect = lambda stream: trace.append(("record", stream))
        permutation = (rows, torch.zeros(2, 2, dtype=torch.int64), torch.arange(4), order, side)
        with (
            patch.object(function_module, "_prepare_forward_execution", return_value=execution),
            patch.object(function_module, "_launch_forward_kernel",
                         side_effect=lambda *args: trace.append(("kernel", None))),
            patch.object(function_module, "_save_forward_state",
                         side_effect=lambda *args: trace.append(("save", args[-1][1]))),
            patch.object(torch.npu, "current_stream", return_value=current),
        ):
            output = _MegaMoeFunction.forward(context, rows, rows, rows, metadata, plan,
                                              workspace, permutation, None, True)
        self.assertTrue(torch.equal(output, rows))
        self.assertEqual(trace, [("kernel", None), ("wait", side), ("record", current), ("save", order)])

    def test_forward_submits_deferred_order_after_kernel_and_waits_before_save(self) -> None:
        """Preserve count readiness and allocator lifetimes across the deferred stream."""
        rows = torch.ones(4, 4)
        order, side, current, counts, ready = Mock(), Mock(), Mock(), Mock(), Mock()
        context = SimpleNamespace(needs_input_grad=(True,) * 9)
        spec = SimpleNamespace(dispatch_mode="pull", top_k=2)
        plan = SimpleNamespace(spec=spec)
        workspace = Mock(in_use=True)
        execution = SimpleNamespace(dispatch=rows, source=rows, combine=rows,
                                    profile_call=Mock(), intermediates=SimpleNamespace(
                                        up_proj=rows, activation=rows))
        metadata = SimpleNamespace(expert_capacity=4)
        trace = []
        side.wait_event.side_effect = lambda event: trace.append(("counts_ready", event))
        counts.record_stream.side_effect = lambda stream: trace.append(("counts_lifetime", stream))
        current.wait_stream.side_effect = lambda stream: trace.append(("wait", stream))
        order.record_stream.side_effect = lambda stream: trace.append(("order_lifetime", stream))
        permutation = (rows, torch.zeros(2, 2, dtype=torch.int64), torch.arange(4), None, side, counts, ready)

        def build_order(actual_counts: Any, capacity: int) -> Any:
            self.assertIs(actual_counts, counts)
            self.assertEqual(capacity, 4)
            trace.append(("build", side))
            return order

        with (
            patch.object(function_module, "_prepare_forward_execution", return_value=execution),
            patch.object(function_module, "_launch_forward_kernel",
                         side_effect=lambda *args: trace.append(("kernel", None))),
            patch.object(function_module, "received_torch_order", side_effect=build_order),
            patch.object(function_module, "_save_forward_state",
                         side_effect=lambda *args: trace.append(("save", args[-1][1]))),
            patch.object(torch.npu, "current_stream", return_value=current),
            patch.object(torch.npu, "stream", side_effect=lambda stream: nullcontext(stream)),
        ):
            output = _MegaMoeFunction.forward(context, rows, rows, rows, metadata, plan,
                                              workspace, permutation, None, True)
        self.assertTrue(torch.equal(output, rows))
        self.assertEqual(trace, [("kernel", None), ("counts_ready", ready), ("counts_lifetime", side),
                                 ("build", side), ("wait", side), ("order_lifetime", current), ("save", order)])

    def test_push_snapshot_preserves_final_row_order_and_empty_capacity(self) -> None:
        """Save owned, ordered X without changing empty receive storage required by the ABI."""
        rows = torch.arange(16).reshape(4, 4).float()
        for capacity, order in ((4, torch.tensor([2, 0, 3, 1])), (1, torch.empty(0, dtype=torch.int64))):
            with self.subTest(capacity=capacity):
                context = SimpleNamespace(needs_input_grad=(True,) * 9)
                plan = SimpleNamespace(spec=SimpleNamespace(dispatch_mode="push", top_k=2, route_policy="torch"))
                workspace = Mock(in_use=True)
                snapshot = torch.full((capacity, 4), float("nan"))
                execution = SimpleNamespace(dispatch=rows[:capacity], source=rows, combine=rows,
                                            profile_call=Mock(), intermediates=SimpleNamespace(
                                                up_proj=rows, activation=rows, down_proj=snapshot))
                permutation = (rows, torch.zeros(2, 2, dtype=torch.int64), torch.arange(4), order)
                with (
                    patch.object(function_module, "_prepare_forward_execution", return_value=execution),
                    patch.object(function_module, "_launch_forward_kernel"),
                    patch.object(function_module, "_save_forward_state") as save,
                ):
                    _MegaMoeFunction.forward(context, rows, rows, rows, SimpleNamespace(expert_capacity=capacity),
                                              plan, workspace, permutation, None, True)
                self.assertTrue(context.dispatch_ordered)
                saved = save.call_args.args[3]
                self.assertIs(saved, snapshot)
                self.assertEqual(saved.shape, (capacity, 4))
                self.assertNotEqual(saved.data_ptr(), execution.dispatch.data_ptr())
                if order.numel():
                    torch.testing.assert_close(saved, rows[order])
                else:
                    self.assertTrue(saved.isnan().all())
                    self.assertNotEqual(saved.data_ptr(), 0)

    def test_torch_pull_combine_adjoint_writes_the_leased_source(self) -> None:
        """Torch combine backward owns no separate expert-gradient staging tensor."""
        values = torch.arange(24).reshape(6, 4).bfloat16()
        mapping = torch.tensor([4, 1, 5, 0, 3, 2])
        weights = torch.tensor([[0.25, -0.0], [1.5, -1.0], [0.125, 2.0]])
        gradient = torch.arange(12).reshape(3, 4).bfloat16()
        buffer = torch.full((8, 4), float("nan"), dtype=torch.bfloat16)
        context = SimpleNamespace(plan=SimpleNamespace(spec=SimpleNamespace(
            dispatch_mode="pull", route_policy="torch")), workspace=SimpleNamespace(source_buffer=buffer),
            source_rows=6, has_unpermute=True, needs_input_grad=(True,) * 9)
        reference_values, reference_weights = values.clone().requires_grad_(), weights.clone().requires_grad_()
        reference = restore_torch_output(reference_values, mapping, reference_weights)
        reference.backward(gradient)
        _, scores, token_ids = _restore_torch_output_with_state(values, mapping, weights)
        with patch.object(function_module.multicore_ops, "mega_moe_unpermute_grad_out") as cann:
            source, grad_weights = function_module._stage_backward_source(
                context, gradient, (mapping, torch.arange(6), values, weights, scores, token_ids))
        cann.assert_not_called()
        self.assertEqual(source.data_ptr(), buffer.data_ptr())
        self.assertTrue(torch.equal(source.view(torch.uint8), reference_values.grad.view(torch.uint8)))
        self.assertTrue(torch.equal(grad_weights.view(torch.uint8), reference_weights.grad.view(torch.uint8)))
        self.assertTrue(buffer[6:].isnan().all())

    def test_ragged_pull_uses_reserved_source_prefix(self) -> None:
        """Forward and backward retain SHMEM addresses while exposing only actual rows."""
        buffer = torch.full((256, 4), float("nan"))
        workspace = SimpleNamespace(source_buffer=buffer)
        spec = SimpleNamespace(dispatch_mode="pull", hidden_size=4)
        rows = torch.arange(12).reshape(3, 4).float()
        dispatch, source = function_module._dispatch_and_source(spec, workspace, rows, 7)
        self.assertEqual(tuple(dispatch.shape), (7, 4))
        self.assertEqual(source.data_ptr(), buffer.data_ptr())
        torch.testing.assert_close(source, rows)
        self.assertTrue(torch.isnan(buffer[3:]).all())
        context = SimpleNamespace(plan=SimpleNamespace(spec=spec), workspace=workspace,
                                  source_rows=3, has_unpermute=False)
        staged, probabilities = function_module._stage_backward_source(context, rows + 1, ())
        torch.testing.assert_close(staged, rows + 1)
        self.assertIsNone(probabilities)
        context.source_rows = 0
        context.has_unpermute = True
        context.needs_input_grad = (False,) * 7 + (True,)
        context.topk_dtype = torch.float32
        with patch.object(function_module.multicore_ops, "mega_moe_unpermute_grad_out") as unpermute:
            empty, gradient = function_module._stage_backward_source(
                context, rows[:0], (torch.empty(0, dtype=torch.int32), rows[:0], torch.empty(0, 2)))
        unpermute.assert_not_called()
        self.assertEqual(tuple(empty.shape), (0, 4))
        self.assertEqual(tuple(gradient.shape), (0, 2))
        self.assertEqual(function_module._native_source(empty).data_ptr(), buffer.data_ptr())

    def test_empty_native_source_preserves_symmetric_storage(self) -> None:
        """Empty pull ranks must retain the symmetric base used to address remote ranks."""
        storage = torch.arange(32).reshape(8, 4)
        empty_source = storage[:0]
        native_source = function_module._native_source(empty_source)
        self.assertEqual(native_source.data_ptr(), storage.data_ptr())
        torch.testing.assert_close(native_source, storage[:1])
        self.assertEqual(tuple(empty_source.shape), (0, 4))
        ordinary_empty = torch.empty(0, 4)
        placeholder = function_module._native_source(ordinary_empty)
        self.assertEqual(tuple(placeholder.shape), (1, 4))
        self.assertNotEqual(placeholder.data_ptr(), 0)

    def test_eoi_weight_operands_and_gradients_keep_contiguous_layout(self) -> None:
        """Native EOI operands and gradients already match the parameter storage layout."""
        spec = SimpleNamespace(
            expert_weight_layout="EOI",
            local_experts=2,
            hidden_size=4,
            intermediate_size=3,
        )
        plan = SimpleNamespace(spec=spec)
        gate_up = torch.empty(2, 6, 4)
        down = torch.empty(2, 4, 3)
        self.assertIs(function_module._native_weight_operand(gate_up, plan), gate_up)
        self.assertIs(function_module._native_weight_operand(down, plan), down)

        gradients = function_module._allocate_backward_intermediates(
            spec,
            8,
            torch.empty(8, 4),
            gate_up,
            down,
        )
        for weight, gradient in ((gate_up, gradients.grad_weight1), (down, gradients.grad_weight2)):
            self.assertEqual(gradient.shape, weight.shape,
                             msg=f"Gradient shape: expected={weight.shape}, actual={gradient.shape}")
            self.assertEqual(gradient.stride(), weight.stride(),
                             msg=f"Gradient stride: expected={weight.stride()}, actual={gradient.stride()}")
            self.assertEqual(torch.count_nonzero(gradient).item(), 0,
                             msg=f"Empty-expert gradients: expected=0, actual={gradient}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backward_unpacks_saved_tensors_once(self) -> None:
        """Feature: MegaKernel diagnostics.
        Description: Checkpoint unpack hooks must only be consumed once per backward call.
        Expectation: Backward unpacks saved tensors once.
        """
        tensors = tuple(torch.ones(1) for _ in range(13))
        context = Mock()
        gradient = torch.ones(1)
        context.plan.spec.dispatch_mode = "push"
        context.plan.native_weight_gradients = False
        context.has_source_inverse = False
        execution = function_module._BackwardExecution(
            dispatch=torch.ones(1), grad_x=torch.ones(1), event_counters=Mock(),
            gmm_workspace=None, swiglu_workspace=None,
            intermediates=function_module._BackwardIntermediates(*(torch.ones(1) for _ in range(5))),
        )
        with (
            patch.object(type(context), "saved_tensors", create=True,
                         new_callable=PropertyMock, side_effect=[tensors, RuntimeError("duplicate unpack")]) as unpack,
            patch.object(function_module, "_prepare_backward_execution", return_value=execution),
            patch.object(function_module, "_launch_backward_kernel"),
            patch.object(function_module, "_restore_input_gradient", return_value=gradient) as restore,
            patch.object(function_module, "_release_completed_graph"),
        ):
            result = _MegaMoeFunction.backward(context, gradient)
        unpack.assert_called_once_with()
        restore.assert_called_once_with(context, execution.grad_x, tensors[12:], None)
        self.assertIs(result[0], gradient)
        self.assertIs(result[1], execution.intermediates.grad_weight1)
        self.assertIs(result[2], execution.intermediates.grad_weight2)
        context.workspace.release.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_deferred_backward_keeps_local_owned_storage_after_workspace_reuse(self) -> None:
        """Feature: Deferred backward storage.
        Description: Reuse the workspace between deferred forward and backward.
        Expectation: Backward retains its own local tensors.
        """
        self._check_deferred_backward(permuted=False)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_permutation_gradient_consumes_workspace_before_release(self) -> None:
        """Feature: Deferred backward storage.
        Description: Run the permutation gradient while tracking workspace release.
        Expectation: Consume the receive buffer before releasing its lease.
        """
        self._check_deferred_backward(permuted=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_frozen_tokens_do_not_launch_a_permutation_gradient(self) -> None:
        """Feature: Deferred backward storage.
        Description: Backpropagate with frozen token activations.
        Expectation: Skip the token permutation gradient.
        """
        self._check_deferred_backward(permuted=True, input_grad=False)

    def test_dynamic_tokens_restore_each_deferred_forward_length(self) -> None:
        """Use per-call token counts despite later forwards reusing a larger source buffer."""
        self._check_deferred_backward(permuted=True, vary_tokens=True)

    def test_cann_pull_releases_combine_values_before_backward_scratch(self) -> None:
        """Default CANN execution must not retain full combine tensors through backward."""
        output = torch.ones(4, 4)
        probabilities = torch.ones(2, 2)
        output_ref, probabilities_ref = weakref.ref(output), weakref.ref(probabilities)
        workspace = SimpleNamespace(source_buffer=torch.empty(4, 4), claim=Mock(), release=Mock())
        context = SimpleNamespace(
            plan=SimpleNamespace(spec=SimpleNamespace(dispatch_mode="pull", route_policy="cann")),
            workspace=workspace, source_rows=4, has_unpermute=True, topk_dtype=torch.float32,
            needs_input_grad=(False,) * 7 + (True, False),
            saved_tensors=(torch.empty(0),) * 12 + (torch.arange(4), output, probabilities),
        )
        context.maybe_clear_saved_tensors = lambda: setattr(context, "saved_tensors", None)
        del output, probabilities

        def stop_before_allocation(*_args: Any) -> None:
            self.assertIsNone(output_ref())
            self.assertIsNone(probabilities_ref())
            raise RuntimeError("checked staging ownership")

        def unpermute_without_retaining_arguments(*_args: Any) -> None:
            return None

        with (
            patch.object(function_module.multicore_ops, "mega_moe_unpermute_grad_out",
                         new=unpermute_without_retaining_arguments),
            patch.object(function_module, "_prepare_backward_execution", new=stop_before_allocation),
            self.assertRaisesRegex(RuntimeError, "checked staging ownership"),
        ):
            _MegaMoeFunction.backward(context, torch.empty(2, 4))
        workspace.release.assert_called_once()

    def test_torch_pull_releases_combine_values_before_backward_scratch(self) -> None:
        """Release integrated combine activations before allocating expert backward scratch."""
        output = torch.ones(4, 4, dtype=torch.bfloat16)
        probabilities = torch.ones(2, 2)
        output_ref, probabilities_ref = weakref.ref(output), weakref.ref(probabilities)
        workspace = SimpleNamespace(source_buffer=torch.empty_like(output), claim=Mock(), release=Mock())
        context = SimpleNamespace(
            plan=SimpleNamespace(spec=SimpleNamespace(dispatch_mode="pull", route_policy="torch")),
            workspace=workspace, source_rows=4, has_unpermute=True,
            needs_input_grad=(False,) * 7 + (True, False),
            saved_tensors=(torch.empty(0),) * 12 + (torch.arange(4), torch.arange(4), output, probabilities,
                                                    probabilities.reshape(4, 1).bfloat16(), torch.arange(4) // 2),
        )
        context.maybe_clear_saved_tensors = lambda: setattr(context, "saved_tensors", None)
        del output, probabilities

        def stop_before_allocation(*_args: Any) -> None:
            self.assertIsNone(output_ref())
            self.assertIsNone(probabilities_ref())
            raise RuntimeError("checked staging ownership")

        with (
            patch.object(function_module, "_prepare_backward_execution", new=stop_before_allocation),
            self.assertRaisesRegex(RuntimeError, "checked staging ownership"),
        ):
            _MegaMoeFunction.backward(context, torch.ones(2, 4, dtype=torch.bfloat16))
        workspace.release.assert_called_once()

    def test_non_reentrant_checkpoint_unpacks_saved_tensors_once(self) -> None:
        """Checkpoint recomputation must not fail when backward reads its permutation state."""
        # CPU-only inputs must not select an installed accelerator for autocast replay.
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        self._check_deferred_backward(permuted=False, replay=True)

    def _check_deferred_forward(
        self, source: Any, weight1: Any, weight2: Any, route: Any, plan: Any, workspace: Any,
        down_pointers: list[int],
        *, permuted: bool, vary_tokens: bool, replay: bool,
    ) -> Any:
        """Run one forward and check that its autograd storage survives workspace reuse."""
        tokens = source.shape[0]
        if permuted:
            ids = torch.tensor([[0, 1]] * tokens, dtype=torch.int32)
            prepared = SimpleNamespace(
                routed_tokens=source.detach().repeat_interleave(2, dim=0),
                metadata=route,
                weight_grad_order=None, weight_grad_order_stream=None, weight_grad_ready_event=None,
                unpermute_mapping=(torch.arange(tokens * 2, dtype=torch.int32) if vary_tokens
                                   else torch.tensor([0, 2, 1, 3], dtype=torch.int32)),
            )
            output = function_module.execute_mega_moe_with_permutation(
                source, ids, weight1, weight2, prepared, plan, workspace
            )
        elif replay:
            operation = partial(function_module.execute_mega_moe, route=route, plan=plan, workspace=workspace)
            output = checkpoint(operation, source, weight1, weight2, use_reentrant=False)
        else:
            output = function_module.execute_mega_moe(source, weight1, weight2, route, plan, workspace)
        if not replay:
            saved = output.grad_fn.saved_tensors
            self.assertEqual(len(saved), 13 if permuted and source.requires_grad else 12)
            self.assertEqual(saved[0].data_ptr(), down_pointers[-1])
            self.assertNotEqual(saved[0].data_ptr(), workspace.expert_buffer.data_ptr())
            self.assertEqual(saved[0].untyped_storage().nbytes(), route.expert_capacity * 4 * source.element_size())
        self.assertEqual(tuple(output.shape), (tokens * 2 if permuted else 4, 4))
        return output

    def _check_deferred_backward(self, *, permuted: bool, input_grad: bool = True,
                                vary_tokens: bool = False, replay: bool = False) -> None:
        """Exercise delayed backward with an optional input permutation boundary."""
        spec = SimpleNamespace(hidden_size=4, intermediate_size=2, rank_id=0,
                               ep_size=2, num_experts=4, local_num_tokens=2, top_k=2,
                               dispatch_mode="push", expert_weight_layout="EIO", auto_tokens=False)
        plan = SimpleNamespace(spec=spec, reuse_backward_dispatch=False)
        for name in ("up_proj", "swiglu", "down_proj", "act_grad", "gate_grad",
                     "w1_grad", "w2_grad", "swiglu_grad"):
            setattr(plan, f"{name}_tiling", None)
        runtime = SimpleNamespace(normal_tensor=None, kernel_name="mega_moe", rank=0)
        plan.fwd_diagnostics = MegaKernelDiagnostics(runtime)
        plan.bwd_diagnostics = MegaKernelDiagnostics(runtime)
        plan.fwd_runtime_config = plan.bwd_runtime_config = None
        workspace = Mock(
            expert_capacity=128,
            expert_buffer=torch.empty(128, 4),
            routed_buffer=torch.empty(8 if vary_tokens else 4, 4),
            forward_event_counters=torch.empty(16),
            backward_event_counters=torch.empty(16),
            gmm_workspace=torch.empty(1),
            swiglu_grad_workspace=torch.empty(1),
        )
        weight1 = torch.ones(2, 4, 4, requires_grad=True)
        weight2 = torch.ones(2, 2, 4, requires_grad=True)
        down_pointers = []
        backward_capacities = []

        def release_workspace() -> None:
            """Make accidental reads after the lease immediately observable."""
            workspace.expert_buffer.fill_(float("nan"))
            workspace.routed_buffer.fill_(float("nan"))

        workspace.release.side_effect = release_workspace

        def permutation_gradient(grad: Any, mapping: Any, token_rows: int, top_k: int) -> Any:
            """Require direct consumption of shared rows and return owned token rows."""
            self.assertEqual(grad.data_ptr(), workspace.routed_buffer.data_ptr())
            expected_rows = (1, 3, 2, 4)[int(grad[0, 0]) - 1] if vary_tokens else 2
            self.assertEqual((token_rows, top_k), (expected_rows, 2))
            self.assertEqual(grad.shape[0], token_rows * top_k)
            self.assertEqual(mapping.tolist(), list(range(token_rows * 2)) if vary_tokens else [0, 2, 1, 3])
            self.assertTrue(torch.isfinite(grad).all())
            return grad.reshape(token_rows, top_k, 4).sum(dim=1)

        def forward_kernel(*args: Any) -> None:
            """Fill every output with a call-specific sentinel."""
            dispatch, source = args[0], args[2]
            up_proj, activation, down_proj, combine = args[7], args[8], args[11], args[12]
            tag = source[0, 0].item()
            capacity = up_proj.shape[0]
            self.assertEqual(tuple(down_proj.shape), (capacity, 4))
            self.assertEqual(tuple(activation.shape), (capacity, 2))
            self.assertEqual(tuple(dispatch.shape), (128, 4))
            dispatch.fill_(tag)
            up_proj.fill_(tag + 10)
            activation.fill_(tag + 20)
            down_proj.fill_(tag + 30)
            combine.fill_(tag + 40)
            down_pointers.append(down_proj.data_ptr())

        def backward_kernel(*args: Any) -> None:
            """Check saved values before emulating the in-place gradient outputs."""
            saved_dispatch = args[17]
            capacity = saved_dispatch.shape[0]
            backward_capacities.append(capacity)
            tag = saved_dispatch[0, 0].item()
            self.assertTrue(torch.equal(saved_dispatch, torch.full_like(saved_dispatch, tag)))
            self.assertTrue(torch.equal(args[9], torch.full_like(args[9], tag + 10)))
            self.assertTrue(torch.equal(args[5], torch.full_like(args[5], tag + 20)))
            for tensor, width in ((args[8], 2), (args[10], 4), (args[12], 4)):
                self.assertEqual(tuple(tensor.shape), (capacity, width))
            self.assertEqual(tuple(args[13].shape), (8 if vary_tokens else 4, 4))
            self.assertEqual(torch.count_nonzero(args[6]).item(), 0)
            self.assertEqual(torch.count_nonzero(args[18]).item(), 0)
            args[13].fill_(tag)
            args[6].fill_(tag)
            args[18].fill_(tag)

        capacities = (7, 2, 1, 5)
        outputs = []
        sources = []
        with (
            patch.object(MegaKernelDiagnostics, "launch",
                         side_effect=AssertionError("ordinary path must bypass launch")),
            patch.object(
                function_module.multicore_ops,
                "mega_moe",
                side_effect=forward_kernel,
            ),
            patch.object(
                function_module.multicore_ops,
                "mega_moe_grad",
                side_effect=backward_kernel,
            ),
            patch.object(function_module.multicore_ops, "moe_token_permute_grad",
                         side_effect=permutation_gradient) as mock_permutation,
        ):
            for tag, capacity in enumerate(capacities, start=1):
                tokens = (1, 3, 2, 4)[tag - 1] if vary_tokens else 2
                source = torch.full((tokens if permuted else 4, 4), float(tag), requires_grad=input_grad)
                route = SimpleNamespace(expert_capacity=capacity, group_list=torch.tensor([0, capacity]))
                for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size",
                             "combine_src_off", "combine_target_off", "combine_size"):
                    setattr(route, name, torch.zeros(4, dtype=torch.int64))
                output = self._check_deferred_forward(
                    source, weight1, weight2, route, plan, workspace, down_pointers,
                    permuted=permuted, vary_tokens=vary_tokens, replay=replay,
                )
                outputs.append(output)
                sources.append(source)
            for tag in range(len(outputs), 0, -1):
                self.assertTrue(torch.equal(outputs[tag - 1], torch.full_like(outputs[tag - 1], float(tag + 40))))
                outputs[tag - 1].sum().backward()
                source = sources[tag - 1]
                expected = torch.full_like(source, float(tag * (2 if permuted else 1)))
                if input_grad:
                    self.assertTrue(torch.equal(source.grad, expected))
                else:
                    self.assertIsNone(source.grad)
        self.assertTrue(torch.equal(weight1.grad, torch.full_like(weight1, 10.0)))
        self.assertTrue(torch.equal(weight2.grad, torch.full_like(weight2, 10.0)))
        self.assertEqual(mock_permutation.call_count, len(outputs) if permuted and input_grad else 0)
        self.assertEqual(backward_capacities, list(reversed(capacities)))
        self.assertEqual(workspace.claim.call_count, 12 if replay else 8)
        self.assertEqual(workspace.release.call_count, 12 if replay else 8)


if __name__ == "__main__":
    unittest.main()
