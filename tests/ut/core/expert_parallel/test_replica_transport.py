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
"""P2P peer mapping and legal deferred-route regression tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import (
    build_expert_replica_plan,
    transport,
)
from hyper_parallel.core.expert_parallel.hot_replica.pool import ReplicaPool
from hyper_parallel.core.expert_parallel.hot_replica.routing import (
    ReplicaRoute,
)
from tests.torch.expert_parallel.hot_replica_checks import replica_forward, route_evidence
from tests.common.mark_utils import arg_mark


class TestReplicaTransport(unittest.TestCase):
    """Exercise both directions with default and noncontiguous group ranks."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_home_route_observation_retains_the_real_planner(self):
        """Feature: home-only route evidence.
        Description: Observe the unchanged planner and inspect its selected home plan.
        Expectation: Planner mismatches and active-copy bypasses still fail.
        """
        plan = build_expert_replica_plan([[1, 0], [0, 1]], 1)
        route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), 0, None, home_only=True)
        output = torch.ones(1, requires_grad=True).square()
        executor = MagicMock()
        prepare = executor._prepare_replica_route
        prepare.return_value = route

        def _forward(value):
            executor._prepare_replica_route(value)
            return output

        executor.side_effect = _forward
        self.assertIs(replica_forward(executor, "counts"), output)
        prepare.assert_called_once_with("counts")
        self.assertIs(executor._prepare_replica_route, prepare)
        self.assertEqual(route_evidence(output, "cpu")["transfers"], 0)
        with self.assertRaisesRegex(AssertionError, "executed cpu"):
            route_evidence(output, "device")
        output._replica_home_route = ReplicaRoute(build_expert_replica_plan([[20, 0], [20, 0]], 1),
                                                 torch.empty(0), torch.empty(0), 0, None, home_only=True)
        with self.assertRaisesRegex(AssertionError, "cannot bypass"):
            route_evidence(output, "cpu")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_native_home_route_evidence_still_checks_planner_and_replica_presence(self):
        """Feature: native home route evidence.
        Description: Inspect native home plans without a replica autograd node.
        Expectation: Active copies require their saved route and the requested planner must match.
        """
        output = torch.ones(1, requires_grad=True).square()
        plan = build_expert_replica_plan([[1, 0], [0, 1]], 1)
        route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), 0, None)
        self.assertEqual(route_evidence(output, "cpu", home_route=route)["transfers"], 0)
        with self.assertRaisesRegex(AssertionError, "executed cpu"):
            route_evidence(output, "device", home_route=route)
        route = ReplicaRoute(build_expert_replica_plan([[20, 0], [20, 0]], 1),
                             torch.empty(0), torch.empty(0), 0, None)
        with self.assertRaisesRegex(AssertionError, "cannot bypass"):
            route_evidence(output, "cpu", home_route=route)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_saved_backward_generation_reaches_only_the_provider_lease(self):
        """Feature: backward weight generation.
        Description: Pass the saved generation to the provider lease.
        Expectation: Forward callers cannot select a backward generation.
        """
        token, view = object(), object()
        provider = SimpleNamespace(overlap_home=True, lease=MagicMock())
        provider.lease.return_value.__enter__.return_value = view
        route = SimpleNamespace(transport=provider)
        weights = (torch.ones(1, 2, 2),)
        with transport.prefetch_weights(weights, route, backward=True, overlap=True,
                                        reuse_generation=token) as result:
            self.assertIs(result, view)
        provider.lease.assert_called_once_with(weights, route, backward=True, overlap=True, reuse_generation=token)
        with self.assertRaisesRegex(ValueError, "saved backward context"):
            with transport.prefetch_weights(weights, route, reuse_generation=token):
                pass

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_weight_and_gradient_peers(self):
        """Feature: replica peer mapping.
        Description: Send weights and gradients over default and noncontiguous process groups.
        Expectation: Every transfer uses global peers and retains its supplied group.
        """
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 2, 1)
        self.assertTrue(plan.transfers)
        subgroup, world = object(), object()
        for group, members in ((None, (0, 1)), (world, (0, 1)), (subgroup, (1, 3))):
            for rank in range(2):
                with self.subTest(group=group, rank=rank):
                    route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), rank, group)
                    weights = (torch.ones(2, 2, 2),)
                    pool = ReplicaPool(weights, 1)
                    operations = []
                    def _operation(op, tensor, peer, actual_group):
                        self.assertIs(actual_group, group)
                        self.assertEqual(peer, members[1 - rank])
                        operations.append(op)
                        return SimpleNamespace(tensor=tensor)
                    def _translate(actual_group, peer):
                        self.assertIsNotNone(actual_group)
                        return members[peer]
                    def _exchange(items):
                        for item in items:
                            item.tensor.fill_(2)
                    with (patch.object(transport.dist, "get_global_rank", side_effect=_translate),
                          patch.object(transport.dist, "is_initialized", return_value=True),
                          patch.object(transport.dist, "P2POp", side_effect=_operation),
                          patch.object(transport, "replica_pool", return_value=pool),
                          patch.object(transport, "_exchange", side_effect=_exchange)):
                        with transport.prefetch_weights(weights, route, backward=True) as leased:
                            transport.return_gradients(weights, route, leased.gradients)
                    self.assertEqual(len(operations), 2)
                    self.assertIn(transport.dist.isend, operations)
                    self.assertIn(transport.dist.irecv, operations)
