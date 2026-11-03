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
from hyper_parallel.core.expert_parallel.hot_replica import planner


class TestQuotaAdmission(unittest.TestCase):
    """Exercise the native-agreement path with exact, realizable EP64 histograms."""

    @staticmethod
    def counts() -> list[int]:
        """Create a diffuse hot owner whose average bound overstates B1/B2 benefit."""
        remaining = 4096 * 8 - 950
        owners = [950] + [remaining // 63 + int(index < remaining % 63) for index in range(63)]
        return [rows // 6 + int(index < rows % 6) for rows in owners for index in range(6)]

    @staticmethod
    def gathered(counts: list[int]) -> tuple[torch.Tensor, routing._GatheredReplicaCounts]:
        """Create the already-validated native count agreement for identical source histograms."""
        logical = torch.tensor([counts] * 64, dtype=torch.int32)
        ids = torch.repeat_interleave(torch.arange(384), torch.tensor(counts)).reshape(8, 4096).T.contiguous()
        summary = [[0, 4096, 31, 4096 * 8, min(counts), max(counts), 64 * sum(counts[rank * 6:(rank + 1) * 6])]
                   for rank in range(64)]
        return ids, routing._GatheredReplicaCounts(logical, summary, logical.tolist())

    def test_rejected_quotas_skip_full_cpu_source_dispatch(self) -> None:
        """A legal owner fallback need not build the full EP64 virtual dispatch matrix."""
        counts = self.counts()
        logical = torch.tensor([counts] * 64, dtype=torch.int32)
        ids = torch.repeat_interleave(torch.arange(384), torch.tensor(counts)).reshape(8, 4096).T.contiguous()
        summary = [[0, 4096, 31, 4096 * 8, min(counts), max(counts), 64 * sum(counts[rank * 6:(rank + 1) * 6])]
                   for rank in range(64)]
        original_cpu = torch.Tensor.cpu

        def compact_only(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            if tensor.shape == logical.shape:
                raise AssertionError('Full count readback is unnecessary for rejected quotas')
            return original_cpu(tensor, *args, **kwargs)

        for budget in (1, 2, 6):
            for host in (None, logical.tolist()):
                with self.subTest(budget=budget, host_counts=host is not None):
                    with (patch.object(routing, 'dist') as distributed,
                          patch.object(routing, 'build_expert_replica_plan',
                                       side_effect=AssertionError('Full source planner must not run')),
                          patch.object(torch.Tensor, 'cpu', compact_only)):
                        distributed.is_initialized.return_value = True
                        distributed.get_world_size.return_value = 64
                        distributed.get_rank.return_value = 0
                        route = routing.prepare_replica_route(
                            ids, ExpertReplicaConfig(384, 64, budget), compact_home=True,
                            minimum_load_reduction=.4, dynamic_header=(0, 4096, 31), _auto_capacity_factor=1.25,
                            _capacity_budget=Mock(return_value=300000),
                            _gathered_counts=routing._GatheredReplicaCounts(logical, summary, host))
                    self.assertTrue(route.home_only and route.plan_bypassed)
                    self.assertEqual(route.plan.destination_loads[0], 60800)
                    self.assertFalse(route.plan.transfers)

    def test_minimum_fragment_pruning_keeps_the_full_planner(self) -> None:
        """Source-sensitive pruning cannot use the quota-only rejection shortcut."""
        ids, gathered = self.gathered(self.counts())
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, '_uncalibrated_copies', side_effect=AssertionError('Quota shortcut')),
              patch.object(routing, 'build_expert_replica_plan', wraps=routing.build_expert_replica_plan) as build):
            distributed.is_initialized.return_value = True
            distributed.get_world_size.return_value = 64
            distributed.get_rank.return_value = 0
            route = routing.prepare_replica_route(
                ids, ExpertReplicaConfig(384, 64, 2), compact_home=True, minimum_replica_rows=1,
                minimum_load_reduction=.4, dynamic_header=(0, 4096, 31), _auto_capacity_factor=1.25,
                _capacity_budget=Mock(return_value=300000), _gathered_counts=gathered)
        self.assertEqual(build.call_count, 1)
        self.assertTrue(route.home_only)
        self.assertFalse(route.plan_bypassed)

    def test_required_capacity_copies_are_not_rejected(self) -> None:
        """An owner that exceeds the hard allowance still reaches full source planning."""
        ids, gathered = self.gathered(self.counts())
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, '_uncalibrated_copies', side_effect=AssertionError('Quota shortcut')),
              patch.object(routing, 'build_expert_replica_plan', wraps=routing.build_expert_replica_plan) as build):
            distributed.is_initialized.return_value = True
            distributed.get_world_size.return_value = 64
            distributed.get_rank.return_value = 0
            route = routing.prepare_replica_route(
                ids, ExpertReplicaConfig(384, 64, 2), compact_home=True,
                minimum_load_reduction=.4, dynamic_header=(0, 4096, 31), _auto_capacity_factor=1.25,
                _capacity_budget=Mock(return_value=60000), _gathered_counts=gathered)
        self.assertEqual(build.call_count, 1)
        self.assertTrue(route.plan.transfers)
        self.assertLessEqual(max(route.plan.destination_loads), 60000)

    def test_profitable_copies_keep_the_exact_full_plan(self) -> None:
        """Quota pre-admission must preserve every dispatch quota when replicas remain useful."""
        remainder = 4096 * 7
        counts = [4096] + [remainder // 383 + int(index < remainder % 383) for index in range(383)]
        ids, gathered = self.gathered(counts)
        for budget in (1, 2, 6):
            with self.subTest(budget=budget):
                expected = routing.build_expert_replica_plan(
                    gathered.host_counts, budget, target_load=40960, capacity_limit=300000,
                    receive_capacity=300000, minimum_load_reduction=.4, adaptive=True)
                with (patch.object(routing, 'dist') as distributed,
                      patch.object(planner, '_uncalibrated_copies',
                                   side_effect=AssertionError('Duplicate quota search')),
                      patch.object(routing, 'build_expert_replica_plan',
                                   wraps=routing.build_expert_replica_plan) as build):
                    distributed.is_initialized.return_value = True
                    distributed.get_world_size.return_value = 64
                    distributed.get_rank.return_value = 0
                    route = routing.prepare_replica_route(
                        ids, ExpertReplicaConfig(384, 64, budget), compact_home=True,
                        minimum_load_reduction=.4, dynamic_header=(0, 4096, 31), _auto_capacity_factor=1.25,
                        _capacity_budget=Mock(return_value=300000), _gathered_counts=gathered)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(route.plan, expected)
                self.assertTrue(route.plan.transfers)

    def test_prepared_quotas_preserve_recompute_plan_reuse(self) -> None:
        """Invocation-local quotas must not become part of the checkpoint plan-cache key."""
        counts = [4096] + [4096 * 7 // 383 + int(index < 4096 * 7 % 383) for index in range(383)]
        ids, gathered = self.gathered(counts)
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, 'build_expert_replica_plan', wraps=routing.build_expert_replica_plan) as build):
            distributed.is_initialized.return_value = True
            distributed.get_world_size.return_value = 64
            distributed.get_rank.return_value = 0
            forward, replay = create_recompute_contexts()
            options = dict(compact_home=True, minimum_load_reduction=.4, dynamic_header=(0, 4096, 31),
                           _auto_capacity_factor=1.25, _capacity_budget=Mock(return_value=300000),
                           _gathered_counts=gathered)
            with forward:
                first = routing.prepare_replica_route(ids, ExpertReplicaConfig(384, 64, 2), **options)
            with replay:
                second = routing.prepare_replica_route(ids, ExpertReplicaConfig(384, 64, 2), **options)
        self.assertEqual(build.call_count, 1)
        self.assertIs(first.plan, second.plan)
        self.assertTrue(second.plan_reused)

    def test_changed_policy_or_totals_recompute_prepared_quotas(self) -> None:
        """A previously affordable proposal cannot override a changed target or source histogram."""
        counts = ((100, 0, 0, 0), (0, 0, 0, 0))
        config = ExpertReplicaConfig(4, 2, 1)
        for changed_counts, target, resident in ((counts, 75, 100), (counts, 50, 80),
                                                 (((80, 20, 0, 0), (0, 0, 0, 0)), 50, 100)):
            with self.subTest(counts=changed_counts, target=target, resident=resident):
                copies = planner._uncalibrated_copies([100, 0, 0, 0], [100, 0], config, 100, 100, 50)
                proposal = planner._UncalibratedCandidate(config, (100, 0, 0, 0), copies, 100, 100, 50)
                options = dict(target_load=target, capacity_limit=100, receive_capacity=resident,
                               minimum_load_reduction=.4, adaptive=True)
                expected = routing.build_expert_replica_plan(changed_counts, 1, **options)
                with patch.object(planner, '_uncalibrated_copies', wraps=planner._uncalibrated_copies) as search:
                    actual = routing.build_expert_replica_plan(
                        changed_counts, 1, **options, _uncalibrated_candidate=proposal)
                self.assertEqual(search.call_count, 1)
                self.assertEqual(actual, expected)

    def test_quota_rejection_keeps_checkpoint_home_execution(self) -> None:
        """Forward and recomputation both bypass materialization with the same native metadata."""
        ids, gathered = self.gathered(self.counts())
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, 'build_expert_replica_plan', side_effect=AssertionError('Full source planner'))):
            distributed.is_initialized.return_value = True
            distributed.get_world_size.return_value = 64
            distributed.get_rank.return_value = 0
            forward, replay = create_recompute_contexts()
            options = dict(compact_home=True, minimum_load_reduction=.4, dynamic_header=(0, 4096, 31),
                           _auto_capacity_factor=1.25, _capacity_budget=Mock(return_value=300000),
                           _gathered_counts=gathered)
            with forward:
                first = routing.prepare_replica_route(ids, ExpertReplicaConfig(384, 64, 6), **options)
            with replay:
                second = routing.prepare_replica_route(ids, ExpertReplicaConfig(384, 64, 6), **options)
        self.assertTrue(first.plan_bypassed and second.plan_bypassed)
        self.assertTrue(first.home_only and second.home_only)
        self.assertEqual(first.dynamic_summary, second.dynamic_summary)


if __name__ == '__main__':
    unittest.main()
