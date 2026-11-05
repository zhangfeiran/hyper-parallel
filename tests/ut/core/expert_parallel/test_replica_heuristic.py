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
"""CPU oracles for uncalibrated admission and communication-aware consolidation."""

from collections.abc import Sequence
from copy import deepcopy
import random
from typing import Any
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaConfig, planner, routing
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel


class TestHeuristicReplicaRouting(unittest.TestCase):
    """Reject impossible optional gains before constructing source dispatch quotas."""

    @staticmethod
    def route(counts: Sequence[Sequence[int]], *, native_counts: bool = False,
              host_counts: bool = False, **options: Any) -> routing.ReplicaRoute:
        """Exercise the real route entry with an agreed, variable-length CPU histogram."""
        size, experts = len(counts), len(counts[0])
        config = ExpertReplicaConfig(experts, size, 1)
        logical = torch.tensor(counts, dtype=torch.int64)
        ids = torch.repeat_interleave(torch.arange(experts), logical[0]).reshape(-1, 1).int()
        loads = logical.reshape(size, size, config.home_experts).sum((0, 2)).tolist()
        headers = [[0, sum(row), 31] for row in counts]
        matrix = torch.tensor([list(row) + [0] + header for row, header in zip(counts, headers)])
        if native_counts:
            summary = [header + [sum(row), min(row), max(row), load]
                       for header, row, load in zip(headers, counts, loads)]
            options['_gathered_counts'] = routing._GatheredReplicaCounts(
                logical, summary, logical.tolist() if host_counts else None)
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, '_gather_replica_counts', return_value=matrix) as gather):
            distributed.is_initialized.return_value = True
            distributed.get_world_size.return_value = size
            distributed.get_rank.return_value = 0
            result = routing.prepare_replica_route(ids, config, compact_home=True,
                                                   dynamic_header=tuple(headers[0]), **options)
        if native_counts:
            gather.assert_not_called()
        return result

    def test_impossible_gain_skips_full_planning_with_unequal_and_empty_sources(self) -> None:
        """A compact integer bound rejects impossible gains for raw and native count agreements."""
        cases = ([[5, 3, 0, 0], [1, 0, 4, 3]], [[0, 0, 0, 0], [9, 0, 7, 0]],
                 [[3, 2, 0, 0], [0, 0, 2, 2]])
        for counts in cases:
            for native, host in ((False, False), (True, False), (True, True)):
                with self.subTest(counts=counts, native=native, host=host):
                    with patch.object(routing, 'build_expert_replica_plan', side_effect=AssertionError('full plan')):
                        route = self.route(counts, native_counts=native, host_counts=host,
                                           receive_capacity=32, minimum_load_reduction=0.4)
                    self.assertTrue(route.plan_bypassed)
                    self.assertTrue(route.home_only)
                    self.assertFalse(route.plan.transfers)
                    self.assertEqual(route.plan.logical_counts, tuple(map(tuple, counts)))
                    torch.testing.assert_close(route.counts_by_source, torch.tensor(counts, dtype=torch.int32))
                    expected = tuple((0, sum(row), 31, sum(row), min(row), max(row), load)
                                     for row, load in zip(counts, route.plan.destination_loads))
                    self.assertEqual(route.dynamic_summary, expected)

    def test_integer_admission_boundary_and_absolute_floor(self) -> None:
        """Equality remains admissible, and fractional rounding matches the full planner."""
        counts = [[6, 2, 0, 0], [2, 0, 3, 3]]
        for fraction, absolute, bypass in ((0.2, 0, False), (0.20001, 0, True),
                                            (0.0, 3, True), (0.0, 2, False), (0.0, 0, False)):
            with self.subTest(fraction=fraction, absolute=absolute):
                with patch.object(routing, 'build_expert_replica_plan',
                                  wraps=planner._build_expert_replica_plan) as build:
                    route = self.route(counts, target_load=8, receive_capacity=32,
                                       minimum_load_reduction=fraction, minimum_saved_rows=absolute)
                self.assertEqual(route.plan_bypassed, bypass)
                self.assertEqual(build.call_count, int(not bypass))
                if not bypass:
                    route.plan.validate()

    def test_native_summary_needs_no_histogram_readback_for_rejected_gain(self) -> None:
        """An existing native agreement supplies enough host statistics to skip all count readback."""
        with (patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('count readback')),
              patch.object(routing, 'build_expert_replica_plan', side_effect=AssertionError('full plan'))):
            route = self.route([[5, 3, 0, 0], [1, 0, 4, 3]], native_counts=True,
                               receive_capacity=32, minimum_load_reduction=0.4)
        self.assertTrue(route.plan_bypassed)

    def test_invalid_runtime_credit_fails_before_compact_admission(self) -> None:
        """A compact fallback cannot bypass validation of the internal transfer budget."""
        for credit in (-1, True, 1.5):
            with self.subTest(credit=credit), self.assertRaisesRegex(ValueError, 'Replica transfer row credit'):
                self.route([[5, 3, 0, 0], [1, 0, 4, 3]], receive_capacity=32,
                           minimum_load_reduction=0.4, _replica_transfer_row_credit=credit)

    def test_capacity_required_copies_and_recompute_keep_the_original_graph(self) -> None:
        """A larger replay allowance cannot replace the original mandatory guest graph."""
        counts = [[4, 4, 0, 0], [4, 4, 0, 0]]
        forward, replay = create_recompute_contexts()
        with patch.object(routing, 'build_expert_replica_plan', wraps=planner._build_expert_replica_plan) as build:
            with forward:
                original = self.route(counts, target_load=8, receive_capacity=8, minimum_load_reduction=0.9,
                                      _replica_transfer_row_credit=0)
            with replay:
                repeated = self.route(counts, target_load=8, receive_capacity=32, minimum_load_reduction=0.9,
                                      _replica_transfer_row_credit=8)
        original.plan.validate()
        self.assertTrue(original.plan.transfers)
        self.assertFalse(original.plan_bypassed)
        self.assertTrue(repeated.plan_reused)
        self.assertIs(repeated.plan, original.plan)
        self.assertEqual(build.call_count, 1)
        torch.testing.assert_close(original.physical_ids, repeated.physical_ids)

    def test_a_matching_cost_model_keeps_its_own_admission(self) -> None:
        """A measured profitable layout is not rejected by the uncalibrated row threshold."""
        model = ExpertReplicaCostModel(
            calibration_version=2, backend='push', hidden_size=16, intermediate_size=8, ep_size=2,
            transport='p2p', forward_ms=((0, 0), (100, 100)), backward_ms=((0, 0), (100, 100)),
            weight_ms=0, gradient_ms=0)
        for fraction, copies in ((0.0, True), (0.99, False)):
            with self.subTest(fraction=fraction):
                route = self.route([[6, 2, 0, 0], [2, 0, 3, 3]], target_load=8, receive_capacity=32,
                                   minimum_load_reduction=fraction, _cost_model=model)
                route.plan.validate()
                self.assertFalse(route.plan_bypassed)
                self.assertEqual(bool(route.plan.transfers), copies)


class TestSparseCompletedQuotas(unittest.TestCase):
    """Retain useful completed quotas without calibrated latency coefficients."""

    @staticmethod
    def counts(hot: list[int], top_k: int) -> tuple[tuple[int, ...], ...]:
        """Construct distinct-TopK histograms with one overloaded owner."""
        tokens, experts, ranks = 4096, 192, 32
        remaining = tokens * top_k - sum(hot)
        quotient, remainder = divmod(remaining, experts - len(hot))
        source = tuple(hot) + tuple(quotient + (index < remainder) for index in range(experts - len(hot)))
        return (source,) * ranks

    def test_completed_single_guest_quotas_rescue_rejected_two_slot_layouts(self) -> None:
        """Complete thin seeds before discarding an affordable strong-hotspot layout."""
        for hot in ([397, 396, 396, 396, 396, 396], [1457, 184, 184, 184, 184, 184],
                    [1068, 262, 262, 262, 262, 261]):
            with self.subTest(hot=hot):
                plan = planner._build_expert_replica_plan(
                    self.counts(hot, 6), 2, target_load=30720, capacity_limit=786432,
                    receive_capacity=786432, minimum_replica_rows=1024, minimum_load_reduction=.4,
                    adaptive=True, _rank_nodes=(0,) * 16 + (1,) * 16)
                plan.validate()
                self.assertTrue(plan.transfers)
                self.assertLessEqual(max(plan.destination_loads), 40000)
                self.assertLessEqual(len(plan.transfers), 8)

    def test_sparse_quota_trades_bounded_receive_rows_for_fewer_matrix_edges(self) -> None:
        """An extreme single expert need not replicate onto every idle rank."""
        counts = self.counts([4096], 8)
        with patch.object(planner, '_balanced_edges', wraps=planner._balanced_edges) as edges:
            plan = planner._build_expert_replica_plan(
                counts, 6, target_load=40960, capacity_limit=786432, receive_capacity=786432,
                minimum_replica_rows=1024, minimum_load_reduction=.4, adaptive=True,
                _rank_nodes=(0,) * 16 + (1,) * 16)
        plan.validate()
        self.assertEqual(edges.call_count, 1)
        self.assertLessEqual(len(plan.transfers), 15)
        self.assertLessEqual(max(plan.destination_loads), 2 * 32768)
        self.assertLessEqual(max(max(row) for row in plan.destination_counts), 32768)

    def test_sparse_quota_respects_a_tighter_receive_limit(self) -> None:
        """A communication preference cannot replace a required feasible layout."""
        plan = planner._build_expert_replica_plan(
            self.counts([4096], 8), 6, target_load=32768, capacity_limit=32768,
            receive_capacity=32768, minimum_replica_rows=1024, minimum_load_reduction=.4,
            adaptive=True, _rank_nodes=(0,) * 16 + (1,) * 16)
        plan.validate()
        self.assertEqual(max(plan.destination_loads), 32768)

    def test_native_agreement_admits_and_replays_completed_quotas(self) -> None:
        """The default-fragment runtime path must reuse the newly admitted quotas."""
        counts = self.counts([1068, 262, 262, 262, 262, 261], 6)
        logical = torch.tensor(counts, dtype=torch.int32)
        ids = torch.repeat_interleave(torch.arange(192), logical[0]).reshape(6, 4096).T.contiguous().int()
        summary = [[0, 4096, 31, sum(row), min(row), max(row), 32 * sum(row[rank * 6:(rank + 1) * 6])]
                   for rank, row in enumerate(counts)]
        gathered = routing._GatheredReplicaCounts(logical, summary, counts)
        forward, replay = create_recompute_contexts()
        options = {'compact_home': True, 'minimum_replica_rows': 1024, 'minimum_load_reduction': .4,
                   'dynamic_header': (0, 4096, 31), '_auto_capacity_factor': 1.25,
                   '_capacity_budget': lambda _tokens: 786432, '_gathered_counts': gathered,
                   '_rank_nodes': (0,) * 16 + (1,) * 16}
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, 'build_expert_replica_plan', wraps=routing.build_expert_replica_plan) as build):
            distributed.is_initialized.return_value = True
            distributed.get_world_size.return_value = 32
            distributed.get_rank.return_value = 0
            with forward:
                first = routing.prepare_replica_route(ids, ExpertReplicaConfig(192, 32, 2), **options)
            with replay:
                repeated = routing.prepare_replica_route(ids, ExpertReplicaConfig(192, 32, 2), **options)
        first.plan.validate()
        self.assertFalse(first.home_only)
        self.assertTrue(first.plan.transfers)
        self.assertLessEqual(max(first.plan.destination_loads), 40000)
        self.assertTrue(repeated.plan_reused)
        self.assertIs(first.plan, repeated.plan)
        self.assertEqual(build.call_count, 1)


class TestHeuristicReplicaConsolidation(unittest.TestCase):
    """Remove complete matrix transfers without raising receive peaks or the traffic budget."""

    def test_large_redundant_fragments_are_removed_through_the_runtime_planner(self) -> None:
        """Copies above the small-fragment threshold can still be dominated by fewer edges."""
        fixtures = (([40960, 20480, 40960, 40960, 57344, 40960, 57344, 24576, 28672, 12288, 0, 8192],
                     104790, 273408),
                    ([4096, 16384, 12288, 49152, 12288, 12288, 45056, 16384, 20480, 40960, 4096, 4096],
                     62123, 180907))
        for totals, original_peak, original_remote in fixtures:
            counts = (tuple(totals),) + ((0,) * 12,) * 3
            with self.subTest(totals=totals):
                plan = planner._build_expert_replica_plan(
                    counts, 1, adaptive=True, minimum_replica_rows=1024,
                    capacity_limit=max(sum(totals[start:start + 3]) for start in range(0, 12, 3)))
                plan.validate()
                self.assertLessEqual(len(plan.transfers), 2)
                self.assertLessEqual(max(plan.destination_loads), original_peak)
                remote = sum(value for source, row in enumerate(plan.dispatch_counts)
                             for physical, value in enumerate(row) if physical // plan.config.slots_per_rank != source)
                self.assertLessEqual(remote, original_remote)

    def test_consolidation_preserves_rank_and_node_token_locality(self) -> None:
        """Deleting a weight edge must not move more source tokens over the fabric."""
        fixtures = ((ExpertReplicaConfig(4, 2, 1), ((2, 2, 3, 0), (0, 0, 3, 0)),
                     [{0: 2, 1: 2, 2: 3}, {2: 3, 3: 0}], ()),
                    (ExpertReplicaConfig(4, 4, 1), ((0, 0, 4, 0), (0, 4, 0, 0), (0, 0, 0, 0), (0, 0, 0, 8)),
                     [{0: 0}, {1: 4, 2: 4}, {2: 0}, {3: 8}], (0, 0, 1, 1)))
        for config, counts, copies, nodes in fixtures:
            with self.subTest(nodes=nodes):
                before = deepcopy(copies)
                planner._consolidate_copies(copies, counts, config, nodes)
                self.assertEqual(copies, before)
                planner._materialize(counts, copies, config, rank_nodes=nodes).validate()

    def test_cross_node_copy_is_removed_before_a_competing_local_copy(self) -> None:
        """Limited owner headroom is spent on eliminating a cross-node matrix edge first."""
        config = ExpertReplicaConfig(4, 4, 1)
        counts = ((10, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 8))
        copies = [{0: 6}, {1: 0, 0: 2}, {2: 0, 0: 2}, {3: 8}]
        nodes = (0, 0, 1, 1)
        planner._consolidate_copies(copies, counts, config, nodes)
        plan = planner._materialize(counts, copies, config, capacity_limit=8, rank_nodes=nodes)
        plan.validate()
        self.assertEqual(tuple(edge.target_rank for edge in plan.transfers), (1,))
        self.assertEqual(max(plan.destination_loads), 8)

    def test_shape_credit_bounds_the_loss_of_source_local_tokens(self) -> None:
        """A removed matrix may pay for bounded activation traffic on its own link class."""
        config = ExpertReplicaConfig(4, 4, 1)
        counts = ((0, 0, 4, 0), (0, 4, 0, 0), (0, 0, 0, 0), (0, 0, 0, 8))
        original = [{0: 0}, {1: 4, 2: 4}, {2: 0}, {3: 8}]
        for credit, nodes, expected_edges in ((3, (0, 0, 1, 1), 1), (4, (0, 0, 1, 1), 0),
                                                (4, (0, 1, 1, 0), 0)):
            with self.subTest(credit=credit, nodes=nodes):
                copies = deepcopy(original)
                planner._consolidate_copies(copies, counts, config, nodes, credit)
                plan = planner._materialize(counts, copies, config, capacity_limit=8, rank_nodes=nodes)
                plan.validate()
                self.assertEqual(len(plan.transfers), expected_edges)

    def test_random_consolidation_preserves_traffic_bounds_and_edge_ownership(self) -> None:
        """Actual materialized dispatch agrees with the incremental locality guards for random quotas."""
        generator = random.Random(52103)
        nodes = (7, 2, 7, 2)
        config = ExpertReplicaConfig(12, 4, 3)
        for iteration in range(100):
            credit = generator.choice((0, 8, 32))
            counts = tuple(tuple(generator.randrange(20) for _ in range(12)) for _ in range(4))
            copies = [{expert: sum(row[expert] for row in counts)
                       for expert in range(rank * 3, (rank + 1) * 3)} for rank in range(4)]
            for rank in range(4):
                for expert in generator.sample([expert for expert in range(12) if expert // 3 != rank], 3):
                    owner = expert // 3
                    moved = generator.randrange(copies[owner][expert] + 1)
                    copies[owner][expert] -= moved
                    copies[rank][expert] = moved
            original = deepcopy(copies)
            before = planner._materialize(counts, copies, config, rank_nodes=nodes)
            planner._consolidate_copies(copies, counts, config, nodes, credit)
            after = planner._materialize(counts, copies, config, rank_nodes=nodes)
            repeated = deepcopy(original)
            planner._consolidate_copies(repeated, counts, config, nodes, credit)
            with self.subTest(iteration=iteration):
                before.validate()
                after.validate()
                self.assertEqual(copies, repeated)
                self.assertLessEqual(max(after.destination_loads), max(before.destination_loads))
                old_edges = {(edge.owner_rank, edge.logical_expert, edge.target_rank) for edge in before.transfers}
                new_edges = {(edge.owner_rank, edge.logical_expert, edge.target_rank) for edge in after.transfers}
                self.assertTrue(new_edges.issubset(old_edges))
                for cross_node in (False, True):
                    traffic = []
                    for plan in (before, after):
                        traffic.append(sum(
                            rows for source, row in enumerate(plan.dispatch_counts) for slot, rows in enumerate(row)
                            if (nodes[source] != nodes[slot // config.slots_per_rank] if cross_node
                                else source != slot // config.slots_per_rank)))
                    removed = len(before.transfers) - len(after.transfers)
                    if cross_node:
                        removed = sum(nodes[edge.owner_rank] != nodes[edge.target_rank] for edge in before.transfers)
                        removed -= sum(nodes[edge.owner_rank] != nodes[edge.target_rank] for edge in after.transfers)
                    self.assertLessEqual(traffic[1] - traffic[0], credit * removed)
