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
"""Regression tests for execution precision and diagnostic launch contracts."""

import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import (
    function as execution,
)
from hyper_parallel.core.multicore.modules.mega_moe import (
    plan,
    workspace,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import (
    build_multi_wave_backward_graph,
)
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue


class TestReplicaGradientPrecision(unittest.TestCase):
    """Keep physical partials in FP32 until owner reduction finishes."""

    def test_replica_weight_gradients_use_fp32_without_promoting_activations(self) -> None:
        """
        Feature: replica gradient precision
        Description: allocate physical gradients with and without remote replicas
        Expectation: only replica dW uses FP32 while activation tensors remain BF16
        """
        for replicas in (0, 1):
            for multi_wave in (False, True):
                with self.subTest(replicas=replicas, multi_wave=multi_wave):
                    spec = SimpleNamespace(hidden_size=8, intermediate_size=4, replica_slots_per_rank=replicas)
                    grad = torch.empty((16, 8), dtype=torch.bfloat16)
                    weight1 = torch.empty((2 + replicas, 8, 8), dtype=torch.bfloat16)
                    weight2 = torch.empty((2 + replicas, 4, 8), dtype=torch.bfloat16)

                    dw1, dw2, act, swiglu, dx = execution._allocate_backward_intermediates(
                        torch, spec, 16, grad, weight1, weight2,
                        multi_wave=multi_wave, inplace_backward=False,
                    )

                    expected_dtype = torch.float32 if replicas else torch.bfloat16
                    self.assertEqual(dw1.dtype, expected_dtype)
                    self.assertEqual(dw2.dtype, expected_dtype)
                    self.assertEqual(dw1.shape, weight1.shape)
                    self.assertEqual(dw2.shape, weight2.shape)
                    for activation in (act, swiglu, dx):
                        self.assertEqual(activation.dtype, torch.bfloat16)
                    if multi_wave:
                        self.assertEqual(torch.count_nonzero(dw1).item(), 0)
                        self.assertEqual(torch.count_nonzero(dw2).item(), 0)

    def test_fp32_gradient_descriptors_include_later_wave_accumulators(self) -> None:
        """
        Feature: mixed-precision backward descriptor ABI
        Description: promote the two weight-gradient buffers in a multi-wave graph
        Expectation: all references to dW use four-byte elements without promoting other buffers
        """
        values = TaskSplitValue(seq_size=4096, all_expert_num=32, top_k=8, ep=4, tp=1)
        graph = build_multi_wave_backward_graph(values)

        plan._use_fp32_weight_gradients(graph)

        references = 0
        for op in graph.topological_sort():
            for spec in (*op.inputs, *op.outputs):
                if spec.param_position in (6, 18):
                    self.assertEqual(spec.dtype_size, 4)
                    references += 1
                elif spec.param_position in (0, 2, 5, 8, 9, 10, 11, 12, 13, 17):
                    self.assertEqual(spec.dtype_size, 2)
        self.assertGreaterEqual(references, 6)

    def test_fp32_scratch_does_not_overrun_bf16_retained_activation(self) -> None:
        """
        Feature: mixed-precision retained scratch
        Description: fit FP32 matrices inside a BF16 activation buffer
        Expectation: only one W1 matrix fits and remaining matrices use the workspace tail
        """
        scratch = workspace._plan_weight_grad_scratch(
            local_experts=2, hidden_size=8, intermediate_size=4,
            element_size=4, activation_element_size=2, receive_capacity=16,
        )

        self.assertEqual(scratch.retained_w1_experts, 1)
        self.assertEqual(scratch.retained_w2_experts, 0)
        self.assertEqual(scratch.tail_bytes, 512)

    def test_moe_autograd_reduces_unrounded_partials_before_home_dtype_cast(self) -> None:
        """
        Feature: owner-directed MoE autograd boundary
        Description: return a physical FP32 partial that loses its fraction when cast to BF16
        Expectation: owner reduction sees the fraction and only the logical gradient is cast
        """
        hidden = torch.ones((2, 8), dtype=torch.bfloat16, requires_grad=True)
        weight1 = torch.ones((2, 8, 8), dtype=torch.bfloat16, requires_grad=True)
        weight2 = torch.ones((2, 4, 8), dtype=torch.bfloat16, requires_grad=True)
        physical = (
            torch.ones((3, 8, 8), dtype=torch.bfloat16),
            torch.ones((3, 4, 8), dtype=torch.bfloat16),
        )
        pending = Mock()
        pending.wait.return_value = physical

        def reduce_partials(_plan: Any, dw1: torch.Tensor, dw2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Cancel a remote contribution before the BF16 owner leaf receives its gradient."""
            self.assertEqual(dw1.dtype, torch.float32)
            self.assertEqual(dw2.dtype, torch.float32)
            return dw1[:2] - 256.0, dw2[:2] - 256.0

        pending.manager.reduce_replica_gradients.side_effect = reduce_partials
        spec = SimpleNamespace(
            local_num_tokens=2, hidden_size=8, intermediate_size=4, replica_slots_per_rank=1, physical_local_experts=3,
            forward_permute_fused=True, forward_unpermute_fused=True,
            backward_permute_fused=True, backward_unpermute_fused=True,
        )
        execution_runtime = SimpleNamespace(
            spec=spec, inplace_wave_backward=False, reuse_retained_dw_scratch=False, atomic_dw_accumulation=False,
        )
        metadata = SimpleNamespace(
            wave_count=1, expert_capacity=4, receive_capacity=4,
            **{name: torch.zeros(4, dtype=torch.int64) for name in (
                "group_list", "dispatch_src_off", "dispatch_target_off", "dispatch_size",
                "combine_src_off", "combine_target_off", "combine_size",
            )},
        )
        route = SimpleNamespace(
            metadata=metadata, placement=pending.plan, placement_payload=pending, routed_tokens=None,
            unpermute_mapping=torch.arange(2, dtype=torch.int32), permute_src_index=None,
        )
        buffers = SimpleNamespace(
            dispatch=torch.empty((4, 8), dtype=torch.bfloat16),
            down_proj=torch.empty((4, 8), dtype=torch.bfloat16), final_out=hidden.detach().clone(),
            up_proj=torch.empty((4, 8), dtype=torch.bfloat16), activation=torch.empty((4, 4), dtype=torch.bfloat16),
        )
        backward_route = SimpleNamespace(
            grad_topk_weights=None, route_weight=torch.zeros((2, 1)), grad_x=torch.zeros_like(hidden),
        )
        backward_buffers = SimpleNamespace(
            grad_weight1=torch.full(physical[0].shape, 256.5),
            grad_weight2=torch.full(physical[1].shape, 256.5),
        )
        with (
            patch.object(execution, "_prepare_forward_buffers", return_value=buffers),
            patch.object(execution, "_launch_forward") as launch,
            patch.object(execution, "_prepare_backward_route", return_value=backward_route),
            patch.object(execution, "_prepare_backward_buffers", return_value=backward_buffers),
            patch.object(execution, "_launch_backward"),
        ):
            output = execution.execute_mega_moe(
                hidden, torch.zeros((2, 1), dtype=torch.int32), torch.ones((2, 1)),
                weight1, weight2, route, execution_runtime, Mock(),
            )
            output.sum().backward()

        pending.wait.assert_called_once_with()
        self.assertIs(launch.call_args.args[4], physical[0])
        self.assertIs(launch.call_args.args[5], physical[1])
        pending.manager.reduce_replica_gradients.assert_called_once()
        torch.testing.assert_close(weight1.grad, torch.full_like(weight1, 0.5), atol=0, rtol=0)
        torch.testing.assert_close(weight2.grad, torch.full_like(weight2, 0.5), atol=0, rtol=0)


class TestBackwardGradientOwnership(unittest.TestCase):
    """Autograd outputs must outlive subsequent directional workspace reuse."""

    @staticmethod
    def _finish_input_gradient(grad_x: torch.Tensor, *, permute_fused: bool) -> torch.Tensor:
        """Exercise the real return boundary without a device kernel or communicator."""
        context = SimpleNamespace(
            plan=SimpleNamespace(spec=SimpleNamespace(local_num_tokens=2)),
            permute_fused=permute_fused, backward_unpermute_fused=True,
            wave_count=1, replica_pending=None,
        )
        saved = SimpleNamespace(topk_weights=torch.ones((2, 2)))
        route = SimpleNamespace(
            grad_x=grad_x, grad_topk_weights=torch.zeros((2, 2)), route_weight=torch.zeros(4),
        )
        buffers = SimpleNamespace(grad_weight1=torch.ones(1), grad_weight2=torch.ones(1))
        return execution._finish_backward(context, torch.ones((2, 3)), saved, route, buffers)[0]

    def test_fused_input_gradient_survives_workspace_reuse(self) -> None:
        """
        Feature: fused backward output ownership
        Description: overwrite shared directional storage after returning token-order dX
        Expectation: the returned gradient retains its values in exact-size owned storage
        """
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                scratch = torch.arange(24, dtype=dtype).reshape(8, 3)
                expected = scratch[:2].clone()

                gradient = self._finish_input_gradient(scratch, permute_fused=True)
                scratch.fill_(-17)

                torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
                self.assertEqual(gradient.untyped_storage().nbytes(), expected.numel() * expected.element_size())
                self.assertNotEqual(gradient.untyped_storage().data_ptr(), scratch.untyped_storage().data_ptr())

    def test_consecutive_fused_input_gradients_have_independent_ownership(self) -> None:
        """
        Feature: delayed autograd consumers
        Description: return two gradients from successive uses of the same workspace
        Expectation: neither later backward writes nor consumer mutations affect the earlier gradient
        """
        scratch = torch.ones((4, 3))
        first = self._finish_input_gradient(scratch, permute_fused=True)
        scratch.fill_(2)
        second = self._finish_input_gradient(scratch, permute_fused=True)

        second.add_(3)

        torch.testing.assert_close(first, torch.ones((2, 3)), rtol=0, atol=0)
        torch.testing.assert_close(second, torch.full((2, 3), 5.0), rtol=0, atol=0)
        torch.testing.assert_close(scratch, torch.full((4, 3), 2.0), rtol=0, atol=0)

    def test_unpermute_only_keeps_already_owned_output_without_copy(self) -> None:
        """
        Feature: independently fused backward endpoints
        Description: return the privately allocated unpermute-only output
        Expectation: no redundant clone or extra storage is introduced
        """
        owned = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)

        gradient = self._finish_input_gradient(owned, permute_fused=False)

        self.assertEqual(gradient.untyped_storage().data_ptr(), owned.untyped_storage().data_ptr())
        torch.testing.assert_close(gradient, owned, rtol=0, atol=0)


class TestForwardCycleProfiling(unittest.TestCase):
    """Keep opt-in native arguments and default allocation behavior consistent."""

    def test_forward_preparation_and_launch_use_the_same_profile_flag(self) -> None:
        """
        Feature: opt-in forward diagnostic ABI
        Description: prepare buffers, then change the environment before launch
        Expectation: the prepared flag and its buffer travel together at fixed input positions
        """
        spec = SimpleNamespace(
            local_num_tokens=2, hidden_size=8, intermediate_size=4, rank_id=0, ep_size=2,
            physical_num_experts=4, local_experts=2, forward_permute_fused=True, forward_unpermute_fused=True,
        )
        execution_runtime = SimpleNamespace(
            spec=spec, up_proj_tiling=object(), swiglu_tiling=object(), down_proj_tiling=object(),
            fwd_runtime_config=object(),
        )
        metadata = SimpleNamespace(
            expert_capacity=4, retained_capacity=4, receive_capacity=4, wave_count=1,
            **{name: object() for name in (
                "group_list", "dispatch_src_off", "dispatch_target_off", "dispatch_size",
                "combine_src_off", "combine_target_off", "combine_size",
            )},
        )
        route = SimpleNamespace(permute_src_index=object(), unpermute_mapping=object())
        hidden = torch.ones((2, 8))
        weights = torch.ones((2, 8, 8))
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.dict(os.environ):
                scratch = workspace.MegaMoeWorkspace(
                    symmetric_memory=Mock(),
                    shared=False, expert_buffer=torch.empty((4, 8)), routed_buffer=torch.empty((4, 8)),
                    gmm_workspace=torch.empty(16, dtype=torch.uint8),
                )
                os.environ[execution._CYCLE_PROFILE_ENV] = str(int(enabled))
                with patch.object(scratch, "prepare_event_counters", return_value=object()):
                    buffers = execution._prepare_forward_buffers(
                        torch, execution_runtime, scratch, metadata, hidden, route,
                    )
                expected = scratch.forward_cycle_profile if enabled else scratch.gmm_workspace
                self.assertIs(buffers.cycle_profile_workspace, expected)
                self.assertIs(buffers.cycle_profile_enabled, enabled)
                os.environ[execution._CYCLE_PROFILE_ENV] = str(int(not enabled))
                with patch.object(execution.multicore_adapter, "mega_moe", autospec=True) as launch:
                    execution._launch_forward(
                        execution_runtime, metadata, hidden, hidden, weights, weights, route, weights, buffers,
                    )
                args = launch.call_args.args
                kwargs = launch.call_args.kwargs
                self.assertIs(kwargs["cycle_profile_workspace"], expected)
                self.assertEqual(args[22:], (0, 2, 4, 8, 2))
                self.assertEqual(kwargs["receive_capacity"], 4)
                self.assertEqual(kwargs["wave_count"], 1)
                self.assertEqual(kwargs["home_experts"], 0)
                self.assertEqual(kwargs["cycle_profile"], int(enabled))
                self.assertIs(args[20], execution_runtime.fwd_runtime_config)

    def test_invalid_cycle_profile_flag_is_rejected(self) -> None:
        """
        Feature: diagnostic flag validation
        Description: provide an unsupported environment value
        Expectation: fail before selecting an unsupported native template key
        """
        with patch.dict(os.environ, {execution._CYCLE_PROFILE_ENV: "yes"}), \
                self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            execution._cycle_profile_enabled()


if __name__ == "__main__":
    unittest.main()
