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
"""Check DSV4.1 replica activation, native routing and MegaMoe configuration."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.core.expert_parallel.hot_replica import native as native_module
from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute
from hyper_parallel.core.expert_parallel.hot_replica.native import (
    _replica_swiglu, _replica_swiglu_backward,
)
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.planner import build_expert_replica_plan
from hyper_parallel.models.deepseek_v41.adapter.distributed import moe_engram_expert_parallel as adapter
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import deepseek_v41_swiglu
from tests.ut.auto_models.models.deepseek_v41.test_megamoe import _source


class TestDeepseekV41HotReplica(unittest.TestCase):
    """Keep the same mathematical model when experts acquire guest copies."""

    def test_clamp_outputs_and_gradients_at_boundaries(self) -> None:
        """Match FP32 source math for random values and inclusive clamp boundaries."""
        torch.manual_seed(19)
        for limit in (0.0, 1.0, 10.0):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(limit=limit, dtype=dtype):
                    data = torch.cat((torch.randn(32, 8) * 12,
                                      torch.tensor([-11., -10., -1., 0., 1., 10., 11., 12.]).repeat(8, 1)))
                    packed = data.to(dtype).requires_grad_()
                    gate, value = packed.chunk(2, dim=-1)
                    expected = deepseek_v41_swiglu(gate, value, limit).to(dtype)
                    gradient = torch.randn_like(expected)
                    expected.backward(gradient)
                    torch.testing.assert_close(_replica_swiglu(packed.detach(), limit), expected.detach())
                    torch.testing.assert_close(
                        _replica_swiglu_backward(gradient, packed.detach(), limit), packed.grad,
                        rtol=0.01 if dtype == torch.bfloat16 else 1e-5,
                        atol=0.01 if dtype == torch.bfloat16 else 1e-5,
                    )

    def test_ep64_budget_six_bounds_extreme_hotspot(self) -> None:
        """Six copies cap even all tokens choosing the same home owner's six experts."""
        counts = [[4096] * 6 + [0] * 378 for _ in range(64)]
        plan = build_expert_replica_plan(counts, 6, minimum_load_reduction=0.0)
        plan.validate()
        self.assertEqual(set(plan.destination_loads), {24576})
        self.assertTrue(plan.transfers)
        self.assertEqual(ExpertReplicaConfig(384, 64, 6).maximum_receive_rows(4096, 6), 24576)
        self.assertGreater(ExpertReplicaConfig(384, 64, 5).maximum_receive_rows(4096, 6), 24576 * 10)

    def test_native_unpermutation_preserves_router_gradients(self) -> None:
        """Run a CPU identity transport to check route order, scores and the child module hook."""
        module = nn.Module()
        module.experts = nn.Identity()
        module.shared_experts = lambda states: states * 0.25
        states = torch.randn(1, 5, 8, requires_grad=True)
        indices = torch.tensor([[2, 0], [1, 3], [3, 0], [2, 1], [0, 3]])
        weights = torch.rand(5, 2, requires_grad=True)
        config = ExpertReplicaConfig(4, 2, 2)
        factors = indices.reshape(-1).sort(stable=True).values + 1
        called = []
        module.experts.register_forward_pre_hook(lambda *_: called.append(True))
        module.experts.forward = lambda values, counts, route: values * factors[:, None]
        state = SimpleNamespace(route=object())
        with (patch.object(adapter, "_router", return_value=(indices, weights)),
              patch.object(adapter, "dispatch_native_replicas", side_effect=lambda inputs, *_args, **_kw:
                           (inputs, state)),
              patch.object(adapter, "combine_native_replicas", side_effect=lambda value, _state: value)):
            actual = adapter._replica_routed_forward(module, states, None, None, config, 0.0)
        expected = states * ((indices + 1) * weights).sum(-1).reshape(1, 5, 1) + states * 0.25
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.sum(), (states, weights), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.sum(), (states, weights))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)
        self.assertEqual(called, [True])

    def test_push_pull_receive_identical_replica_options(self) -> None:
        """Use CPU planning and HCCL P2P for both subgroup-capable executors."""
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                experts = DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=128,
                                                   dispatch_mode=mode, replica_slots_per_rank=2)
                experts.configure(None, 2, 2)
                self.addCleanup(experts.close_runtime)
                executor = experts._executor
                self.assertEqual(executor.replica_slots_per_rank, 2)
                self.assertEqual(executor.replica_minimum_load_reduction, 0.0)
                self.assertEqual(executor.replica_min_rows, 0)
                self.assertEqual(executor.replica_planner, "cpu")
                self.assertEqual(executor.replica_transport, "p2p")
                self.assertEqual(len(list(experts.parameters())), 2)

    def test_native_autograd_clamp_returns_original_owner_gradients(self) -> None:
        """Exercise the actual autograd bridge with CPU grouped GEMM and no network."""
        torch.manual_seed(27)
        plan = build_expert_replica_plan([[3, 2]], 2, minimum_load_reduction=0.0)
        route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), 0, None)
        counts = torch.tensor([3, 2, 0, 0])
        inputs = torch.randn(5, 4, dtype=torch.bfloat16).requires_grad_()
        weights = [torch.randn(*shape, dtype=torch.bfloat16).requires_grad_()
                   for shape in ((2, 4, 6), (2, 3, 4))]

        def grouped(values: torch.Tensor, matrices: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
            """Substitute only the hardware GEMM boundary."""
            ends = [0] + groups.tolist()
            return torch.cat([values[begin:end] @ weight
                              for begin, end, weight in zip(ends[:-1], ends[1:], matrices)])

        with patch.object(native_module, "_gmm", side_effect=grouped):
            actual = native_module._NativeReplicaExperts.apply(inputs, *weights, counts, route, 1.0)
            upstream = torch.randn_like(actual)
            actual_grads = torch.autograd.grad(actual, (inputs, *weights), upstream)
        expected_parts = []
        for index, section in enumerate((slice(0, 3), slice(3, 5))):
            gate, value = (inputs[section] @ weights[0][index]).chunk(2, -1)
            activation = deepseek_v41_swiglu(gate, value, 1.0).to(inputs.dtype)
            expected_parts.append(activation @ weights[1][index])
        expected = torch.cat(expected_parts)
        expected_grads = torch.autograd.grad(expected, (inputs, *weights), upstream)
        torch.testing.assert_close(actual, expected)
        for observed, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(observed, reference, rtol=0.02, atol=0.02)
