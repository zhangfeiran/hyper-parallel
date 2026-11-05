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
"""Keep rejected uncalibrated quotas out of full source-dispatch construction."""

import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaConfig, routing
from tests.common.mark_utils import arg_mark


class TestRejectedQuotaMemo(unittest.TestCase):
    """Keep negative search reuse separate from current routing and capacity validation."""

    @staticmethod
    def _counts() -> list[int]:
        """Create a diffuse hot owner whose average bound overstates B1/B2 benefit."""
        remaining = 4096 * 8 - 950
        owners = [950] + [remaining // 63 + int(index < remaining % 63) for index in range(63)]
        return [rows // 6 + int(index < rows % 6) for rows in owners for index in range(6)]

    @staticmethod
    def _gathered(counts: list[int]) -> tuple[torch.Tensor, routing._GatheredReplicaCounts]:
        """Create the already-validated native count agreement for identical source histograms."""
        logical = torch.tensor([counts] * 64, dtype=torch.int32)
        ids = torch.repeat_interleave(torch.arange(384), torch.tensor(counts)).reshape(8, 4096).T.contiguous()
        summary = [[0, 4096, 31, 4096 * 8, min(counts), max(counts), 64 * sum(counts[rank * 6:(rank + 1) * 6])]
                   for rank in range(64)]
        return ids, routing._GatheredReplicaCounts(logical, summary, logical.tolist())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_independent_forwards_reuse_only_exact_rejected_quotas(self) -> None:
        """Feature: Pure negative quota memoization.
        Description: Repeat identical independent forwards with freshly gathered count tensors.
        Expectation: One search, one cache hit and distinct current route storage.
        """
        memo = routing._RejectedQuotaMemo()
        outputs = []
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, '_admitted_uncalibrated_copies',
                           wraps=routing._admitted_uncalibrated_copies) as search):
            distributed.is_initialized.return_value = True
            distributed.get_world_size.return_value = 64
            distributed.get_rank.return_value = 0
            for _ in range(2):
                ids, gathered = self._gathered(self._counts())
                forward, _replay = create_recompute_contexts()
                with forward:
                    route = routing.prepare_replica_route(
                        ids, ExpertReplicaConfig(384, 64, 2), compact_home=True,
                        minimum_replica_rows=1024, minimum_load_reduction=.4,
                        dynamic_header=(0, 4096, 31), _auto_capacity_factor=1.25,
                        _capacity_budget=Mock(return_value=300000), _gathered_counts=gathered,
                        _rejected_quota_memo=memo)
                self.assertTrue(route.home_only)
                self.assertEqual(route.counts_by_source.tolist(), gathered.counts.tolist())
                outputs.append(route)
        self.assertEqual(search.call_count, 1)
        self.assertEqual(memo.hits, 1)
        self.assertNotEqual(outputs[0].counts_by_source.data_ptr(), outputs[1].counts_by_source.data_ptr())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_cross_forward_rejection_invalidates_each_search_input(self) -> None:
        """Feature: Rejection-cache invalidation.
        Description: Change each pure-search input between independent forwards.
        Expectation: Every changed input repeats the search instead of reusing rejection.
        """
        changes = {
            'counts': {}, 'admission': {'minimum_load_reduction': .3},
            'minimum_rows': {'minimum_replica_rows': 2048}, 'target': {'target_load': 40000},
            'budget': {}, 'nodes': {'_rank_nodes': tuple(index // 8 for index in range(64))}, 'complete': {},
        }
        for change, overrides in changes.items():
            with self.subTest(change=change):
                memo = routing._RejectedQuotaMemo()
                with (patch.object(routing, 'dist') as distributed,
                      patch.object(routing, '_admitted_uncalibrated_copies', return_value=None) as search,
                      patch.object(routing, '_complete_guest_search_enabled', return_value=False) as complete):
                    distributed.is_initialized.return_value = True
                    distributed.get_world_size.return_value = 64
                    distributed.get_rank.return_value = 0
                    for iteration in range(2):
                        changed = iteration == 1
                        counts = self._counts()
                        if changed and change == 'counts':
                            counts[0] += 1
                            counts[1] -= 1
                        ids, gathered = self._gathered(counts)
                        complete.return_value = changed and change == 'complete'
                        forward, _replay = create_recompute_contexts()
                        with forward:
                            options = {"compact_home": True, "minimum_replica_rows": 1024,
                                       "minimum_load_reduction": .4, "target_load": 40960,
                                       "dynamic_header": (0, 4096, 31), "_auto_capacity_factor": 1.25,
                                       "_capacity_budget": Mock(return_value=300000), "_gathered_counts": gathered,
                                       "_rejected_quota_memo": memo}
                            if changed:
                                options.update(overrides)
                            budget = 1 if changed and change == 'budget' else 2
                            routing.prepare_replica_route(ids, ExpertReplicaConfig(384, 64, budget), **options)
                    self.assertEqual(search.call_count, 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_cross_forward_rejection_cannot_override_capacity_required_copies(self) -> None:
        """Feature: Required capacity takes precedence over cached rejection.
        Description: Reduce the live capacity budget after a rejected optional-replica search.
        Expectation: The full planner runs and produces legal required transfers.
        """
        memo = routing._RejectedQuotaMemo()
        ids, gathered = self._gathered(self._counts())
        routes = []
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, 'build_expert_replica_plan', wraps=routing.build_expert_replica_plan) as build):
            distributed.is_initialized.return_value = True
            distributed.get_world_size.return_value = 64
            distributed.get_rank.return_value = 0
            for capacity in (300000, 60000):
                forward, _replay = create_recompute_contexts()
                with forward:
                    routes.append(routing.prepare_replica_route(
                        ids, ExpertReplicaConfig(384, 64, 2), compact_home=True,
                        minimum_replica_rows=1024, minimum_load_reduction=.4,
                        dynamic_header=(0, 4096, 31), _auto_capacity_factor=1.25,
                        _capacity_budget=Mock(return_value=capacity), _gathered_counts=gathered,
                        _rejected_quota_memo=memo))
        self.assertTrue(routes[0].home_only)
        self.assertTrue(routes[1].plan.transfers)
        self.assertEqual(build.call_count, 1)
        self.assertLessEqual(max(routes[1].plan.destination_loads), 60000)
