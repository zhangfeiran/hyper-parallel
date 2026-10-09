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

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import build_expert_replica_plan, transport
from hyper_parallel.core.expert_parallel.hot_replica.pool import ReplicaPool
from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute, prepare_replica_route
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from tests.torch.expert_parallel.hot_replica_checks import deferred_ids


class TestReplicaTransport(unittest.TestCase):
    """Exercise both directions with default and noncontiguous group ranks."""

    def test_saved_backward_generation_reaches_only_the_provider_lease(self):
        """Retain the forward token while rejecting caller-selected forward reuse."""
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

    def test_weight_and_gradient_peers(self):
        """Every send/receive uses global peers while retaining the supplied group."""
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

    def test_uninitialized_single_rank_needs_no_collective(self):
        """A local route can lease guest storage without creating a process group."""
        weights = (torch.ones(2, 2, 2),)
        with (patch.object(transport.dist, "is_initialized", return_value=False),
              patch.object(transport.dist, "get_global_rank", side_effect=AssertionError("rank lookup")),
              patch.object(transport.dist, "batch_isend_irecv", side_effect=AssertionError("P2P"))):
            route = prepare_replica_route(torch.tensor([[0, 1]]), ExpertReplicaConfig(2, 1, 1))
            with transport.prefetch_weights(weights, route, backward=True) as pool:
                result = transport.return_gradients(weights, route, pool.gradients)
        torch.testing.assert_close(result[0], weights[0])

    def test_deferred_topk_is_distinct_and_changes_actual_plan(self):
        """The regression fixture must reach saved-plan coverage for every supported K."""
        for top_k in (1, 2, 3, 4, 5, 6, 8):
            plans = []
            for invocation in range(2):
                ids = deferred_ids(128, top_k, 24, invocation, torch.device("cpu"))
                self.assertTrue(bool(((ids >= 0) & (ids < 24)).all()))
                self.assertTrue(all(row.unique().numel() == top_k for row in ids))
                counts = torch.bincount(ids.flatten(), minlength=24).tolist()
                plans.append(build_expert_replica_plan([counts] * 4, 1))
            self.assertNotEqual(plans[0].slot_to_logical, plans[1].slot_to_logical)
            self.assertTrue(all(plan.transfers for plan in plans))
