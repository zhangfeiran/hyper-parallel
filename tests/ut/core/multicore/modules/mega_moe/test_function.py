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

import struct
import unittest
import weakref
from itertools import product
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import function as function_module

from tests.common.mark_utils import arg_mark


class TestMegaMoeFunction(unittest.TestCase):
    """Exercise real autograd contexts with mocked communication and kernels."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_static_runtime_cache_reuses_image_and_bounds_replacement(self) -> None:
        """
        Feature: Static runtime caching
        Description: Reuse writable scratch, then change base identity and suffix pointers.
        Expectation: Reuse identical images and replace only changed entries.
        """
        base = torch.arange(16, dtype=torch.uint8)
        cache = {}
        with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream", autospec=True):
            first = function_module._split_runtime(base, None, 6, home_only=True, cache=cache)
            first[base.numel() - 1] = 93
            repeated = function_module._split_runtime(base, None, 6, home_only=True, cache=cache)
            self.assertIs(repeated, first)
            self.assertEqual(repeated[base.numel() - 1].item(), 93)
            self.assertEqual(base[-1].item(), 15)
            weights, gradients = (torch.empty(1), torch.empty(2)), (torch.empty(3), torch.empty(4))
            pool = SimpleNamespace(weights=weights, gradients=gradients, weight_ready=None, projection_ready=None)
            active = function_module._split_runtime(base, pool, 6, cache=cache)
            self.assertIsNot(active, first)
            self.assertIs(function_module._split_runtime(base, pool, 6, cache=cache), active)
            backward = function_module._split_runtime(base, pool, 6, backward=True, cache=cache)
            self.assertIsNot(backward, active)
            self.assertEqual(len(cache), 1)
            self.assertEqual(struct.unpack("<II5Q", bytes(active[base.numel():].tolist()))[-2:], (0, 0))
            self.assertEqual(struct.unpack("<II5Q", bytes(backward[base.numel():].tolist()))[-2:],
                             tuple(t.data_ptr() for t in gradients))
            other_base = base.clone()
            other = function_module._split_runtime(other_base, pool, 6, cache=cache)
            self.assertIsNot(other, active)
            self.assertEqual(len(cache), 2)
            self.assertIs(cache[id(base)][0], base)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_dynamic_runtime_descriptors_update_suffix_without_resetting_scratch(self) -> None:
        """
        Feature: Dynamic runtime ownership
        Description: Advance v3, v4 and v5 epochs while reusing a serial workspace image.
        Expectation: Update the suffix while retaining the kernel's writable scratch.
        """
        base = torch.arange(16, dtype=torch.uint8)
        weights, gradients = (torch.empty(1), torch.empty(2)), (torch.empty(3), torch.empty(4))
        descriptor = torch.empty(16, dtype=torch.uint8)
        for version in (3, 4, 5):
            cache = {}
            ready = (4096, 1) if version == 3 else None
            projection = None if version == 3 else ((4096, 8192), 1)
            pool = SimpleNamespace(weights=weights, gradients=gradients, weight_ready=ready,
                                   projection_ready=projection)
            options = {"backward": True, "gradient_return": descriptor} if version == 5 else {}
            with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream", autospec=True):
                first = function_module._split_runtime(base, pool, 6, cache=cache, **options)
                first[base.numel() - 1] = 93
                if version == 3:
                    pool.weight_ready = (4096, 2)
                else:
                    pool.projection_ready = ((4096, 8192), 2)
                if version == 5:
                    options["gradient_return"] = descriptor.clone()
                second = function_module._split_runtime(base, pool, 6, cache=cache, **options)
                independent = function_module._split_runtime(base, pool, 6, **options)
            self.assertIs(first, second)
            self.assertEqual(second[base.numel() - 1].item(), 93)
            self.assertEqual(len(cache), 1)
            torch.testing.assert_close(second[base.numel():], independent[base.numel():])
            self.assertEqual(independent[base.numel() - 1].item(), 15)
            if version == 5:
                encoded = struct.unpack("<II9Q", bytes(second[base.numel():].tolist()))
                self.assertEqual(encoded[-2:], (2, options["gradient_return"].data_ptr()))

    def test_dynamic_runtime_reallocates_when_descriptor_layout_changes(self) -> None:
        """An ABI suffix-size change cannot overwrite the next allocation or stale fields."""
        base, cache = torch.arange(16, dtype=torch.uint8), {}
        pool = SimpleNamespace(weights=(torch.empty(1), torch.empty(2)),
                               gradients=(torch.empty(3), torch.empty(4)),
                               weight_ready=(4096, 1), projection_ready=None)
        with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream"):
            first = function_module._split_runtime(base, pool, 6, cache=cache)
            pool.weight_ready, pool.projection_ready = None, ((4096, 8192), 2)
            second = function_module._split_runtime(base, pool, 6, cache=cache)
        self.assertIsNot(first, second)
        self.assertEqual(second.numel() - first.numel(), 8)
        self.assertEqual(len(cache), 1)

    def test_home_only_runtime_keeps_split_addressing_without_guest_pointers(self) -> None:
        """A no-transfer B>0 graph still addresses home expert matrices separately."""
        base = torch.arange(16, dtype=torch.uint8)
        for backward in (False, True):
            with self.subTest(backward=backward), patch.object(torch.npu, "current_stream"), \
                    patch.object(torch.Tensor, "record_stream", autospec=True):
                result = function_module._split_runtime(base, None, 6, backward=backward, home_only=True)
            self.assertTrue(torch.equal(result[:base.numel()], base))
            self.assertEqual(struct.unpack("<II5Q", bytes(result[base.numel():].tolist())),
                             (0x53505754, 2, 6, 0, 0, 0, 0))
        self.assertIs(function_module._split_runtime(base, None, 6), base)

    def test_split_runtime_preserves_v2_and_encodes_deferred_ready_v3(self) -> None:
        """Only deferred consumers receive the per-slot device ready base and epoch."""
        base = torch.arange(16, dtype=torch.uint8)
        weights = (torch.empty(1), torch.empty(2))
        gradients = (torch.empty(3), torch.empty(4))
        for ready, backward in product((None, (4096, 17)), (False, True)):
            with self.subTest(ready=ready, backward=backward):
                pool = SimpleNamespace(weights=weights, gradients=gradients, weight_ready=ready, projection_ready=None)
                with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream"):
                    result = function_module._split_runtime(  # pylint: disable=protected-access
                        base, pool, 6, backward=backward)
                torch.testing.assert_close(result[:16], base)
                data = bytes(result[16:].tolist())
                values = struct.unpack("<II5Q" if ready is None else "<II7Q", data)
                pointers = tuple(g.data_ptr() for g in gradients) if backward else (0, 0)
                expected = (0x53505754, 2 if ready is None else 3, 6, *(w.data_ptr() for w in weights), *pointers)
                self.assertEqual(values, expected + (() if ready is None else ready))

    def test_split_runtime_encodes_independent_projection_ready_v4(self) -> None:
        """Pass distinct W13/W2 cache-line bases and one epoch to forward/backward."""
        base = torch.arange(16, dtype=torch.uint8)
        weights = (torch.empty(1), torch.empty(2))
        gradients = (torch.empty(3), torch.empty(4))
        pool = SimpleNamespace(weights=weights, gradients=gradients, weight_ready=None,
                               projection_ready=((4096, 8192), 23))
        for backward in (False, True):
            with self.subTest(backward=backward), patch.object(torch.npu, "current_stream"), \
                    patch.object(torch.Tensor, "record_stream"):
                result = function_module._split_runtime(base, pool, 6, backward=backward)
            torch.testing.assert_close(result[:16], base)
            pointers = tuple(g.data_ptr() for g in gradients) if backward else (0, 0)
            self.assertEqual(struct.unpack("<II8Q", bytes(result[16:].tolist())),
                             (0x53505754, 4, 6, *(w.data_ptr() for w in weights), *pointers, 4096, 8192, 23))

    def test_split_runtime_v5_retains_projection_prefix_and_gradient_descriptor(self) -> None:
        """Backward adds one invocation descriptor without changing existing projection addresses."""
        base = torch.arange(16, dtype=torch.uint8)
        weights = (torch.empty(1), torch.empty(2))
        gradients = (torch.empty(3), torch.empty(4))
        descriptor = torch.empty(128, dtype=torch.uint8)
        pool = SimpleNamespace(weights=weights, gradients=gradients, weight_ready=None,
                               projection_ready=((4096, 8192), 23))
        with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream"):
            result = function_module._split_runtime(base, pool, 6, backward=True, gradient_return=descriptor)
        torch.testing.assert_close(result[:16], base)
        self.assertEqual(struct.unpack("<II9Q", bytes(result[16:].tolist())),
                         (0x53505754, 5, 6, *(w.data_ptr() for w in weights),
                          *(g.data_ptr() for g in gradients), 4096, 8192, 23, descriptor.data_ptr()))

    def test_kernel_return_requires_backward_projection_readiness(self) -> None:
        """Reject a descriptor that the consumer cannot safely interpret before allocation."""
        base, descriptor = torch.empty(16, dtype=torch.uint8), torch.empty(128, dtype=torch.uint8)
        for backward, pool in ((False, SimpleNamespace(projection_ready=((4096, 8192), 1))),
                               (True, SimpleNamespace(projection_ready=None))):
            with self.assertRaisesRegex(ValueError, "projection-ready backward"):
                function_module._split_runtime(base, pool, 6, backward=backward, gradient_return=descriptor)

    def test_backward_selects_runtime_from_global_replica_presence(self) -> None:
        """Repeated empty/nonempty routes choose schedules without mutating the plan."""
        regular, overlap = object(), object()
        plan = SimpleNamespace(spec=SimpleNamespace(dispatch_mode="push"), bwd_runtime=overlap,
                               bwd_runtime_no_replica=regular, reuse_backward_dispatch=False,
                               bwd_runtime_no_replica_reuse_dispatch=True)
        saved = SimpleNamespace(dispatch=torch.empty(3, 4), weight1=None, weight2=None)
        workspace = Mock(expert_buffer=torch.empty(3, 4), routed_buffer=torch.empty(3, 4),
                         gmm_workspace=torch.empty(1), swiglu_grad_workspace=torch.empty(1))
        with patch.object(function_module, "_allocate_backward_intermediates") as allocate, \
                patch.object(function_module, "prepare_mega_kernel_call") as prepare:
            for transfers, w13 in ((False, True), (True, True), (True, False), (False, False), (True, True)):
                function_module._prepare_backward_execution(
                    workspace, plan, saved, torch.empty(3, 4), has_replica_transfers=transfers, overlap_w13=w13)
                self.assertIs(prepare.call_args.args[0], overlap if transfers and w13 else regular)
                if transfers and w13:
                    self.assertIsNone(allocate.call_args.args[-1])
                else:
                    self.assertEqual(allocate.call_args.args[-1].data_ptr(), workspace.expert_buffer.data_ptr())
            plan.bwd_runtime_no_replica = None
            function_module._prepare_backward_execution(
                workspace, plan, saved, torch.empty(3, 4), has_replica_transfers=False)
            self.assertIs(prepare.call_args.args[0], overlap)

    def test_deferred_backward_keeps_local_owned_storage_after_workspace_reuse(self) -> None:
        """Retain each route's data and capacity across grow/shrink and reverse backward."""
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                self._check_deferred_backward(permuted=False, reuse=reuse)

    def test_permutation_gradient_consumes_workspace_before_release(self) -> None:
        """Keep token gradients intact when release immediately poisons shared rows."""
        for reuse, pull in product((False, True), repeat=2):
            with self.subTest(reuse=reuse, pull=pull):
                self._check_deferred_backward(permuted=True, reuse=reuse, pull=pull)

    def test_frozen_tokens_do_not_launch_a_permutation_gradient(self) -> None:
        """Compute expert weight gradients without an unused token reduction."""
        self._check_deferred_backward(permuted=True, input_grad=False, reuse=True)

    def test_pull_dispatch_skips_only_the_exact_source_self_copy(self) -> None:
        """Use pre-permuted SHMEM directly while preserving ordinary-source staging."""
        spec = SimpleNamespace(dispatch_mode="pull", hidden_size=4)
        source = torch.empty(4, 4)
        workspace = SimpleNamespace(source_buffer=source)
        with patch.object(torch.Tensor, "copy_", autospec=True) as copy:
            dispatch, actual_source = function_module._dispatch_and_source(spec, workspace, source, 3)
            copy.assert_not_called()
        self.assertIs(actual_source, source)
        self.assertEqual(tuple(dispatch.shape), (3, 4))
        rows = torch.ones(4, 4)
        function_module._dispatch_and_source(spec, workspace, rows, 3)
        torch.testing.assert_close(source, rows)

    def test_empty_replica_plan_skips_leases_through_reversed_backward(self) -> None:
        """B stays enabled while an immutable global plan has no transfers."""
        for pull in (False, True):
            with self.subTest(pull=pull):
                self._check_deferred_backward(permuted=False, pull=pull, empty_replica_route=True)

    def test_saved_parameter_version_is_checked_before_claiming_backward_storage(self) -> None:
        """A mutated saved parameter must abort before any resident replica can be borrowed."""
        self._check_deferred_backward(permuted=False, mutate_before_backward=True)

    def _check_deferred_backward(
        self, *, permuted: bool, input_grad: bool = True, reuse: bool = False, pull: bool = False,
        empty_replica_route: bool = False,
        mutate_before_backward: bool = False,
    ) -> None:
        """Exercise delayed backward with an optional input permutation boundary."""
        spec = SimpleNamespace(replica_slots_per_rank=int(empty_replica_route), hidden_size=4,
                               intermediate_size=2, rank_id=0,
                               ep_size=2, num_experts=4, local_num_tokens=2, top_k=2,
                               dispatch_mode="pull" if pull else "push")
        plan = SimpleNamespace(spec=spec, reuse_backward_dispatch=reuse)
        for name in ("up_proj", "swiglu", "down_proj", "act_grad", "gate_grad",
                     "w1_grad", "w2_grad", "swiglu_grad"):
            setattr(plan, f"{name}_tiling", None)
        runtime = SimpleNamespace(normal_tensor=torch.empty(16, dtype=torch.uint8) if empty_replica_route else None)
        plan.fwd_runtime = runtime
        plan.bwd_runtime = runtime
        workspace = Mock(
            replica_inbox=None,
            replica_provider=None,
            replica_runtime_images={},
            in_use=False,
            expert_capacity=128,
            source_buffer=torch.empty(4, 4),
            expert_buffer=torch.empty(128, 4),
            routed_buffer=torch.empty(4, 4),
            forward_event_counters=torch.empty(16),
            backward_event_counters=torch.empty(16),
            gmm_workspace=torch.empty(1),
            swiglu_grad_workspace=torch.empty(1),
        )
        weight1 = torch.ones(2, 4, 4, requires_grad=True)
        weight2 = torch.ones(2, 2, 4, requires_grad=True)
        down_pointers = []
        dispatch_pointers = []
        backward_capacities = []
        scratch_references = []
        restore_gradient = function_module._restore_input_gradient  # pylint: disable=protected-access

        def restore_without_scratch(*args: Any) -> Any:
            """Require ordinary scratch and the optional SHMEM view to be released."""
            self.assertTrue(scratch_references)
            self.assertTrue(all(reference() is None for reference in scratch_references))
            return restore_gradient(*args)

        def release_workspace() -> None:
            """Make accidental reads after the lease immediately observable."""
            self.assertTrue(workspace.in_use)
            workspace.in_use = False
            workspace.source_buffer.fill_(float("nan"))
            workspace.expert_buffer.fill_(float("nan"))
            workspace.routed_buffer.fill_(float("nan"))

        def claim_workspace() -> None:
            """Reject nested claims while exercising the real autograd bridge."""
            self.assertFalse(workspace.in_use)
            workspace.in_use = True

        workspace.claim.side_effect = claim_workspace
        workspace.release.side_effect = release_workspace

        def permutation_gradient(grad: Any, mapping: Any, token_rows: int, top_k: int) -> Any:
            """Require direct consumption of shared rows and return owned token rows."""
            self.assertEqual(grad.data_ptr(), workspace.routed_buffer.data_ptr())
            self.assertEqual((token_rows, top_k), (2, 2))
            self.assertEqual(mapping.tolist(), [0, 2, 1, 3])
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
            self.assertEqual(tuple(dispatch.shape), (capacity if pull else 128, 4))
            dispatch.fill_(tag)
            up_proj.fill_(tag + 10)
            activation.fill_(tag + 20)
            down_proj.fill_(tag + 30)
            combine.fill_(tag + 40)
            down_pointers.append(down_proj.data_ptr())
            dispatch_pointers.append(dispatch.data_ptr())

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
            self.assertEqual(tuple(args[13].shape), (4, 4))
            self.assertEqual(args[12].data_ptr() == args[0].data_ptr(), reuse)
            self.assertNotEqual(saved_dispatch.data_ptr(), args[0].data_ptr())
            scratch_references[:] = [weakref.ref(args[index]) for index in (8, 10, 12)]
            self.assertEqual(torch.count_nonzero(args[6]).item(), 0)
            self.assertEqual(torch.count_nonzero(args[18]).item(), 0)
            args[13].fill_(tag)
            args[6].fill_(tag)
            args[18].fill_(tag)

        def _forward(source: torch.Tensor, route: Any) -> torch.Tensor:
            """Borrow the caller's lease when routing directly into pull source."""
            if not permuted:
                return function_module.execute_mega_moe(source, weight1, weight2, route, plan, workspace)
            ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
            prepared = SimpleNamespace(
                routed_tokens=source.detach().repeat_interleave(2, dim=0), metadata=route,
                unpermute_mapping=torch.tensor([0, 2, 1, 3], dtype=torch.int32),
            )
            if pull:
                workspace.claim()
                workspace.source_buffer.copy_(prepared.routed_tokens)
                prepared.routed_tokens = workspace.source_buffer
            output = function_module.execute_mega_moe_with_permutation(
                source, ids, weight1, weight2, prepared, plan, workspace, workspace_claimed=pull,
            )
            if pull:
                self.assertTrue(workspace.in_use)
                workspace.release()
            return output

        capacities = (7, 2, 1, 5)
        outputs = []
        sources = []
        with (
            patch.object(
                function_module.multicore_ops,
                "mega_moe_with_profile_buffer",
                side_effect=forward_kernel,
            ),
            patch.object(
                function_module.multicore_ops,
                "mega_moe_grad_with_profile_buffer",
                new=backward_kernel,
            ),
            patch.object(torch.npu, "current_stream"),
            patch.object(torch.Tensor, "record_stream", autospec=True),
            patch.object(function_module, "prefetch_weights", side_effect=AssertionError("Unexpected replica lease")),
            patch.object(function_module, "return_gradients", side_effect=AssertionError("Unexpected replica return")),
            patch.object(function_module, "_restore_input_gradient", new=restore_without_scratch),
            patch.object(function_module.multicore_ops, "moe_token_permute_grad",
                         side_effect=permutation_gradient) as mock_permutation,
        ):
            for tag, capacity in enumerate(capacities, start=1):
                source = torch.full((2 if permuted else 4, 4), float(tag), requires_grad=input_grad)
                replica = SimpleNamespace(plan=SimpleNamespace(transfers=())) if empty_replica_route else None
                route = SimpleNamespace(replica_route=replica, expert_capacity=capacity,
                                        group_list=torch.tensor([0, capacity]))
                for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size",
                             "combine_src_off", "combine_target_off", "combine_size"):
                    setattr(route, name, torch.zeros(4, dtype=torch.int64))
                output = _forward(source, route)
                saved = output.grad_fn.saved_tensors
                self.assertEqual(len(saved), 13 if permuted and input_grad else 12)
                self.assertTrue(all(t.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
                                    for t in saved))
                self.assertEqual(saved[0].data_ptr(), (dispatch_pointers if pull else down_pointers)[-1])
                self.assertNotEqual(saved[0].data_ptr(), workspace.expert_buffer.data_ptr())
                self.assertEqual(saved[0].untyped_storage().nbytes(), capacity * 4 * source.element_size())
                self.assertEqual(tuple(output.shape), (4, 4))
                outputs.append(output)
                sources.append(source)
            if mutate_before_backward:
                claims = workspace.claim.call_count
                with torch.no_grad():
                    weight1.add_(1)
                with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                    outputs[-1].sum().backward()
                self.assertEqual(workspace.claim.call_count, claims)
                return
            for tag in range(len(outputs), 0, -1):
                self.assertTrue(torch.equal(outputs[tag - 1], torch.full((4, 4), float(tag + 40))))
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
        self.assertEqual(workspace.claim.call_count, 8)
        self.assertEqual(workspace.release.call_count, 8)


if __name__ == "__main__":
    unittest.main()
