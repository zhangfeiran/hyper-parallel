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
"""Conservative admission, native fallback and parameter-gradient contracts."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.components.modules import moe
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from hyper_parallel.core.expert_parallel.hot_replica.admission import (
    default_admission_profile, reduction_basis_points, resolve_replica_admission,
)
from hyper_parallel.core.expert_parallel.hot_replica import planner
from hyper_parallel.core.multicore.modules.mega_moe import module as mega_moe_module
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestReplicaAdmission(unittest.TestCase):
    """Keep workload matching conservative without device-to-host decisions."""

    def setUp(self) -> None:
        """Keep cached architecture matches independent across tests."""
        default_admission_profile.cache_clear()
        self.addCleanup(default_admission_profile.cache_clear)
        self.execution = {"backend": "push", "transport": "shmem_signal_kernel_gradient", "tokens": 4096,
                          "hidden": 5120, "intermediate": 1792, "experts": 24, "ranks": 4, "top_k": 8,
                          "budget": 1, "device": torch.device("npu:0")}

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_matching_profile_and_architecture_are_required(self):
        """
        description: Only complete-step supported shapes and transports can admit optional copies.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        with patch.object(torch.npu, "get_device_name", return_value="Ascend910B4") as device_name:
            self.assertEqual(resolve_replica_admission(None, **self.execution), (0.4, 24576))
            self.assertEqual(resolve_replica_admission(None, **{**self.execution, "tokens": 2048}), (0.4, 16384))
            self.assertEqual(resolve_replica_admission(None, **{**self.execution, "transport": "p2p"}), (0.4, 32768))
            device_name.assert_called()
        with patch.object(torch.npu, "get_device_name", return_value="OtherNPU"):
            default_admission_profile.cache_clear()
            self.assertEqual(resolve_replica_admission(None, **self.execution), (1.0, 0))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_unknown_identity_does_not_query_device(self):
        """
        description: Unmeasured shapes, pull, native and CPU reject optional copies immediately.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        for key, value in (("backend", "native"), ("backend", "pull"), ("tokens", 512), ("hidden", 1024),
                           ("intermediate", 512), ("experts", 96), ("ranks", 8), ("top_k", 2), ("budget", 2),
                           ("device", torch.device("cpu"))):
            with self.subTest(key=key, value=value), patch.object(torch.npu, "get_device_name") as device_name:
                self.assertEqual(resolve_replica_admission(None, **{**self.execution, key: value}), (1.0, 0))
                device_name.assert_not_called()

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_explicit_override_and_validation(self):
        """
        description: Explicit zero retains placement and invalid policies fail at the boundary.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        with patch.object(torch.npu, "get_device_name", side_effect=AssertionError("device query")):
            self.assertEqual(resolve_replica_admission(0.0, **self.execution), (0.0, 0))
            self.assertEqual(resolve_replica_admission(0.35001, **self.execution), (0.35001, 0))
        self.assertEqual(reduction_basis_points(0.35001), 3501)
        for value in (True, -0.1, 1.01, float("nan"), float("inf"), "0.4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reduction_basis_points(value)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_rejected_plan_restores_pristine_home_quotas_after_constructor_fallback(self):
        """
        description: A rejected candidate cannot mutate the home-only fallback through an alias.
        expectation: Late improvement is discarded and all original owner loads are restored.
        feature: Expert hot replica admission rollback.
        """
        counts = [[100, 100, 100, 0]] * 2
        worse = [{0: 200, 1: 200, 2: 200}, {2: 0, 3: 0}]
        with patch.object(planner, "_bounded_copies", return_value=worse):
            result = planner.build_expert_replica_plan(counts, 1, capacity_limit=400,
                                                      minimum_load_reduction=0.9)
        result.validate()
        self.assertEqual(result.transfers, ())
        self.assertEqual(result.destination_loads, (400, 200))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_native_default_uses_ordinary_dispatch_and_combine(self):
        """
        description: B remains a budget while unmatched native runs use the exact B=0 dispatcher.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        for threshold in (None, 1.0):
            with self.subTest(threshold=threshold):
                style = ExpertParallel(replica_slots_per_rank=1, replica_minimum_load_reduction=threshold)
                module, mesh, context = SimpleNamespace(), MagicMock(), object()
                inputs = (torch.randn(3, 8), torch.tensor([1, 2, 0, 0]))
                style._token_dispatcher = MagicMock()  # pylint: disable=protected-access
                style._token_dispatcher.dispatch.return_value = (*inputs, context)  # pylint: disable=protected-access
                with patch("hyper_parallel.core.expert_parallel.expert_parallel.dispatch_native_replicas") as replica:
                    actual = style._token_dispatch(module, inputs, mesh)  # pylint: disable=protected-access
                    style._token_combine(module, actual[0], mesh)  # pylint: disable=protected-access
                    replica.assert_not_called()
                self.assertIs(actual[0], inputs[0])
                style._token_dispatcher.combine.assert_called_once_with(  # pylint: disable=protected-access
                    module, actual[0], mesh, context)
                self.assertEqual(style.replica_slots_per_rank, 1)
        self.assertTrue(ExpertParallel(replica_slots_per_rank=1, replica_minimum_load_reduction=0.0)._replicas_enabled)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_native_no_guest_preserves_ordinary_outputs_and_gradients(self):
        """
        description: An explicitly admitted but empty native plan uses original weights and score gradients.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        torch.manual_seed(57)
        ordinary = moe.GroupedExperts(8, 16, 2)
        candidate = copy.deepcopy(ordinary)
        candidate._hot_replica_dispatch = SimpleNamespace(route=SimpleNamespace(plan=SimpleNamespace(transfers=())))
        inputs = torch.randn(5, 8, requires_grad=True)
        other = inputs.detach().clone().requires_grad_()
        scores = torch.rand(5, requires_grad=True)
        other_scores = scores.detach().clone().requires_grad_()
        expected = ordinary(inputs, torch.tensor([3, 2]), scores)
        with patch.object(moe, "native_replica_experts", side_effect=AssertionError("guest compute")):
            actual = candidate(other, torch.tensor([3, 2, 0]), other_scores)
        gradient = torch.randn_like(actual)
        expected.backward(gradient)
        actual.backward(gradient)
        for actual_value, expected_value in ((actual, expected), (other.grad, inputs.grad),
                                             (other_scores.grad, scores.grad)):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        for actual_param, expected_param in zip(candidate.parameters(), ordinary.parameters()):
            torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=0, atol=0)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_automatic_home_resources_are_resolved_before_heap_accounting(self):
        """
        description: Unsupported automatic workloads bind ordinary resources without changing requested B.
        expectation: Bound and active specifications agree; explicit policies keep their replica resources.
        feature: Expert hot replica resource fallback.
        """
        options = {"local_num_tokens": 128, "hidden_size": 16, "intermediate_size": 8,
                   "num_experts": 4, "top_k": 2, "ep_size": 2, "replica_slots_per_rank": 1,
                   "create_parameters": False}
        automatic = mega_moe_module.MegaMoeExperts(**options)
        explicit = mega_moe_module.MegaMoeExperts(**options, replica_minimum_load_reduction=0.0)
        self.addCleanup(automatic.close)
        self.addCleanup(explicit.close)
        original = automatic._resource_group.specification  # pylint: disable=protected-access
        other = explicit._resource_group.specification  # pylint: disable=protected-access
        with patch.object(mega_moe_module, "_MegaMoeExecutionResources") as create:
            automatic._create_execution_resources(  # pylint: disable=protected-access
                torch.zeros(1), shared=False, active_specifications=(original, other))
        bound = create.call_args.args[0]
        active = create.call_args.kwargs["active_specifications"]
        self.assertEqual(bound["num_experts"], 4)
        self.assertEqual(bound["replica_slots_per_rank"], 0)
        self.assertIs(bound, original)
        self.assertIs(bound, active[0])
        self.assertIs(active[1], other)
        self.assertEqual(automatic.replica_slots_per_rank, 1)
        self.assertIs(mega_moe_module._resolve_replica_specification(original, torch.zeros(1)), original)
        with self.assertRaises(ValueError):
            mega_moe_module.MegaMoeExperts.share_execution_resources([automatic, explicit])
