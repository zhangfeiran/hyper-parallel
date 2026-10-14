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
    reduction_basis_points, resolve_replica_admission,
)
from hyper_parallel.core.expert_parallel.hot_replica import planner
from hyper_parallel.core.multicore.modules.mega_moe import module as mega_moe_module
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestReplicaAdmission(unittest.TestCase):
    """Keep optional admission separate from the topology and capacity contract."""

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_default_admission_does_not_query_device(self):
        """
        description: Resolve a route-level default without constructing a device-specific allowlist.
        expectation: The default is a fractional gate with no absolute-row restriction.
        feature: Expert hot replica admission.
        """
        with patch.object(torch.npu, "get_device_name", side_effect=AssertionError("device query")):
            self.assertEqual(resolve_replica_admission(None), (0.4, 0))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_near_balanced_route_skips_optional_solver(self):
        """
        description: Even perfect balance cannot satisfy the default reduction on a near-balanced route.
        expectation: Return conserved home quotas without constructing optional copies.
        feature: Expert hot replica admission.
        """
        counts = [[120, 100, 100, 100]] * 2
        with patch.object(planner, "_bounded_copies", side_effect=AssertionError("optional solver")):
            result = planner.build_expert_replica_plan(counts, 1, capacity_limit=500,
                                                      minimum_load_reduction=resolve_replica_admission(None)[0])
        result.validate()
        self.assertEqual(result.transfers, ())
        self.assertEqual(result.destination_loads, (440, 400))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_explicit_override_and_validation(self):
        """
        description: Explicit zero retains placement and invalid policies fail at the boundary.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        self.assertEqual(resolve_replica_admission(0.0), (0.0, 0))
        self.assertEqual(resolve_replica_admission(0.35001), (0.35001, 0))
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

        def _small_improvement(copies, *_args):
            copies[0][0] -= 10
            copies[1][0] = 10

        with (patch.object(planner, "_bounded_copies", return_value=worse),
              patch.object(planner, "_improve_copies", side_effect=_small_improvement) as improve):
            result = planner.build_expert_replica_plan(counts, 1, capacity_limit=400,
                                                      minimum_load_reduction=0.2)
        improve.assert_called_once()
        result.validate()
        self.assertEqual(result.transfers, ())
        self.assertEqual(result.destination_loads, (400, 200))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_native_default_uses_ordinary_dispatch_and_combine(self):
        """
        description: An explicit threshold of one disables optional native replicas.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        for threshold in (1.0,):
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_native_default_plans_replicas_for_each_route(self) -> None:
        """
        description: Keep native B=1 active and apply admission to each dispatch's counts.
        expectation: Default dispatch invokes replica planning with the relative threshold.
        feature: Route-based native replica admission.
        """
        style = ExpertParallel(replica_slots_per_rank=1)
        mesh = MagicMock()
        mesh.size.return_value = 2
        module = SimpleNamespace()
        inputs = (torch.randn(3, 8), torch.tensor([1, 2, 0, 0]))
        state = object()
        with patch("hyper_parallel.core.expert_parallel.expert_parallel.dispatch_native_replicas",
                   return_value=(inputs, state)) as dispatch:
            actual = style._token_dispatch(module, inputs, mesh)  # pylint: disable=protected-access
        self.assertIs(actual, inputs)
        self.assertIs(module._hot_replica_dispatch, state)
        self.assertEqual(dispatch.call_args.kwargs["minimum_load_reduction"], 0.4)

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
    def test_default_preserves_replica_resources_for_all_topologies(self):
        """
        description: Default and explicit policies keep their requested physical topology and can share resources.
        expectation: A near-balanced invocation cannot permanently disable future hotspot replicas.
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
        self.assertEqual(bound["num_experts"], 6)
        self.assertEqual(bound["replica_slots_per_rank"], 1)
        self.assertIs(bound, original)
        self.assertIs(bound, active[0])
        self.assertIs(active[1], other)
        self.assertEqual(automatic.replica_slots_per_rank, 1)
        mega_moe_module.MegaMoeExperts.share_execution_resources([automatic, explicit])
        self.assertIs(automatic._resource_group, explicit._resource_group)  # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_hot_route_remains_enabled_with_default_admission(self) -> None:
        """
        description: Admit an EP16/E96 hotspot without a shape or transport allowlist.
        expectation: Optional copies remain active and meet the required load reduction.
        feature: Route-based hotspot replica admission.
        """
        counts = [[4096] * 8 + [0] * 88] * 16
        result = planner.build_expert_replica_plan(counts, 1, target_load=32768,
                                                  minimum_replica_rows=1024, capacity_limit=393216,
                                                  minimum_load_reduction=resolve_replica_admission(None)[0])
        result.validate()
        self.assertTrue(result.transfers)
        self.assertLessEqual(max(result.destination_loads), 393216 * 0.6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_near_balanced_capacity_required_replicas_are_retained(self) -> None:
        """
        description: Apply strict optional admission when a near-balanced home route exceeds capacity.
        expectation: Mandatory copies remain active and obey the receive bound.
        feature: Capacity-safe replica admission.
        """
        result = planner.build_expert_replica_plan([[120, 100, 100, 100]] * 2, 2, capacity_limit=420,
                                                  minimum_load_reduction=resolve_replica_admission(None)[0])
        result.validate()
        self.assertTrue(result.transfers)
        self.assertLessEqual(max(result.destination_loads), 420)
