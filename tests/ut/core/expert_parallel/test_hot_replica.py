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
"""Bounded planner conservation and logical-route contract tests."""

import itertools
import random
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaConfig, build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute, prepare_replica_route
from hyper_parallel.core.expert_parallel.hot_replica import native, transport
from hyper_parallel.core.multicore.modules.mega_moe.spec import initial_receive_capacity
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestHotReplica(unittest.TestCase):
    """Check capacity against independent distinct-TopK route generation."""

    def test_native_guest_wait_follows_home_matmul(self):
        """Preserve exact row ordering while home computation precedes the guest wait."""
        for home_rows, guest_rows in ((2, 3), (0, 3), (2, 0)):
            for transpose in (False, True):
                with self.subTest(home=home_rows, guest=guest_rows, transpose=transpose):
                    route = SimpleNamespace(rank=0, plan=SimpleNamespace(
                        config=SimpleNamespace(home_experts=1, slots_per_rank=2),
                        dispatch_counts=((home_rows, guest_rows),)))
                    inputs = torch.arange((home_rows + guest_rows) * 2).reshape(-1, 2).float()
                    home, guest = torch.eye(2).unsqueeze(0), (torch.eye(2) * 3).unsqueeze(0)
                    events = []

                    def _gmm(values, weights, groups):
                        events.append("home" if weights.data_ptr() == home.data_ptr() else "guest")
                        self.assertEqual(int(groups[-1]), len(values))
                        return values @ weights[0]

                    prefetch = SimpleNamespace(wait_weights=lambda index: events.append(("wait", index)),
                                               projection_ready=((4096, 8192), 17))
                    with patch.object(native, "_gmm", _gmm):
                        result = native._split_gmm(  # pylint: disable=protected-access
                            inputs, home, guest, torch.tensor([home_rows, home_rows + guest_rows]),
                            route, transpose=transpose, prefetch=prefetch, matrix_index=int(transpose))
                    expected = inputs.clone()
                    expected[home_rows:] *= 3
                    torch.testing.assert_close(result, expected)
                    self.assertEqual(events, (["home"] if home_rows else []) +
                                     ([("wait", int(transpose)), "guest"] if guest_rows else []))

    def test_native_legacy_provider_keeps_its_no_argument_wait(self):
        """An eager or whole-slot provider need not implement projection metadata."""
        route = SimpleNamespace(rank=0, plan=SimpleNamespace(
            config=SimpleNamespace(home_experts=1, slots_per_rank=2), dispatch_counts=((1, 1),)))
        waited = []
        prefetch = SimpleNamespace(wait_weights=lambda: waited.append(True))
        inputs = torch.ones(2, 2)
        weights = torch.eye(2).unsqueeze(0)
        with patch.object(native, "_gmm", lambda values, weight, _groups: values @ weight[0]):
            result = native._split_gmm(inputs, weights, weights, torch.tensor([1, 2]), route,
                                      prefetch=prefetch, matrix_index=1)
        torch.testing.assert_close(result, inputs)
        self.assertEqual(waited, [True])

    def test_copy_threshold_removes_small_balancing_transfers(self):
        """Nearly balanced routes do not copy full experts for one and nine rows."""
        row = [0] * 24
        for index in range(512 * 8):
            row[index % 24] += 1
        original = build_expert_replica_plan([row] * 4, 1)
        plan = build_expert_replica_plan([row] * 4, 1, minimum_replica_rows=16)
        self.assertEqual(len(original.transfers), 2)
        self.assertEqual(plan.transfers, ())
        self.assertEqual(plan.destination_loads, (4104, 4104, 4096, 4080))
        self.assertEqual(plan.logical_counts, original.logical_counts)

    def test_copy_threshold_keeps_mandatory_work_and_consolidates_fragments(self):
        """One existing copy absorbs necessary work before another small copy is removed."""
        counts = [[512] * 8 + [0] * 16] * 4
        config = ExpertReplicaConfig(24, 4, 1)
        upper = config.maximum_receive_rows(512, 8, alignment=1)
        for target in (None, 5120, 7168, 11008):
            with self.subTest(target=target):
                plan = build_expert_replica_plan(counts, 1, target_load=min(target, upper) if target else None,
                                                minimum_replica_rows=4096, capacity_limit=upper)
                plan.validate()
                self.assertEqual(len(plan.transfers), 1)
                self.assertLessEqual(max(plan.destination_loads), upper)
                self.assertGreater(max(plan.destination_loads), 8192)

    def test_histogram_inference_allows_optional_single_hot_copies_to_stay_home(self):
        """Infer a conservative shape bound from distinct-TopK source histograms."""
        row = [512] + [0] * 23
        for index in range(512 * 7):
            row[1 + index % 23] += 1
        plan = build_expert_replica_plan([row] * 4, 1, minimum_replica_rows=1024)
        self.assertEqual(plan.transfers, ())
        self.assertEqual(plan.destination_loads, (5168, 3744, 3744, 3728))
        self.assertLessEqual(max(plan.destination_loads), plan.config.maximum_receive_rows(512, 8, alignment=1))

    def test_copy_threshold_preserves_capacity_across_valid_topk_shapes(self):
        """Consolidation keeps all integer source quotas, budgets, and receive limits."""
        rng = random.Random(9428)
        for _ in range(80):
            ranks, home, tokens = rng.randint(2, 6), rng.randint(1, 6), rng.randint(1, 40)
            experts = ranks * home
            top_k = rng.randint(1, experts)
            counts = [[0] * experts for _ in range(ranks)]
            for row in counts:
                for _ in range(tokens):
                    for expert in rng.sample(range(experts), top_k):
                        row[expert] += 1
            original_peak = max(sum(sum(row[rank * home:(rank + 1) * home]) for row in counts)
                                for rank in range(ranks))
            for budget in (0, 1, 2, home, home + 1):
                original = build_expert_replica_plan(counts, budget)
                upper = original.config.maximum_receive_rows(tokens, top_k, alignment=1)
                for minimum, limit in itertools.product((0, 16, 4096), (None, upper)):
                    plan = build_expert_replica_plan(counts, budget, minimum_replica_rows=minimum,
                                                    capacity_limit=limit)
                    plan.validate()
                    self.assertLessEqual(max(plan.destination_loads), min(upper, original_peak))
                    self.assertLessEqual(len(plan.transfers), len(original.transfers))
                    if not minimum:
                        self.assertEqual(plan, original)

    def test_copy_threshold_rejects_invalid_policy_or_infeasible_capacity(self):
        """Invalid thresholds and impossible receive limits fail before execution."""
        for threshold in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "minimum_replica_rows"):
                build_expert_replica_plan([[2, 0], [2, 0]], 1, minimum_replica_rows=threshold)
        with self.assertRaisesRegex(ValueError, "capacity_limit"):
            build_expert_replica_plan([[2, 0], [2, 0]], 1, capacity_limit=1)

    def test_exhaustive_small_routes(self):
        """All two-rank token routes conserve counts and obey the B bound."""
        choices = list(itertools.combinations(range(4), 2))
        for selected in itertools.product(choices, repeat=4):
            counts = [[0] * 4 for _ in range(2)]
            for token, experts in enumerate(selected):
                for expert in experts:
                    counts[token // 2][expert] += 1
            for budget in (0, 1, 2, 3):
                plan = build_expert_replica_plan(counts, budget)
                plan.validate()
                self.assertLessEqual(max(plan.destination_loads),
                                     plan.config.maximum_receive_rows(2, 2, alignment=1))
                self.assertEqual(sum(plan.destination_loads), 8)

    def test_random_routes(self):
        """Exercise different EP degrees, home counts, skew and spare slots."""
        rng = random.Random(718)
        for _ in range(240):
            ranks, home, tokens = rng.randint(2, 6), rng.randint(1, 6), rng.randint(1, 32)
            experts = ranks * home
            top_k = rng.randint(1, experts)
            counts = [[0] * experts for _ in range(ranks)]
            for row in counts:
                for _ in range(tokens):
                    for expert in rng.sample(range(experts), top_k):
                        row[expert] += 1
            for budget in (0, 1, home, home + 1):
                plan = build_expert_replica_plan(counts, budget)
                self.assertEqual(plan, build_expert_replica_plan(counts, budget))
                plan.validate()
                self.assertLessEqual(max(plan.destination_loads),
                                     plan.config.maximum_receive_rows(tokens, top_k, alignment=1))

    def test_single_hot_expert_balances_with_one_slot(self):
        """One hot expert uses all ranks without changing its parameter owner."""
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        self.assertEqual(plan.destination_loads, (100, 100, 100, 100))
        self.assertTrue(all(transfer.owner_rank == 0 for transfer in plan.transfers))

    def test_no_copy_below_target(self):
        """A fitting route avoids paying weight-copy cost."""
        plan = build_expert_replica_plan([[4, 0, 0, 0], [0, 0, 2, 2]], 1, target_load=4)
        self.assertEqual(plan.transfers, ())

    def test_initial_allocation_stays_bounded_and_dynamic(self):
        """Initial factors remain effective until they reach the maximum bound."""
        specification = {"initial_capacity_factor": 1.0, "local_num_tokens": 128, "top_k": 1,
                         "ep_size": 4, "logical_num_experts": 16, "replica_slots_per_rank": 1}
        self.assertEqual(initial_receive_capacity(specification), 128)
        specification["initial_capacity_factor"] = 100.0
        self.assertEqual(initial_receive_capacity(specification), 512)
        specification["replica_slots_per_rank"] = 4
        self.assertEqual(initial_receive_capacity(specification), 128)

    def test_invalid_config_and_counts(self):
        """Reject impossible topology and non-integer route counts."""
        for args in ((4, 3, 1), (4, 2, -1), (4, 2, True)):
            with self.assertRaises(ValueError):
                ExpertReplicaConfig(*args)
        for counts in (([1, -1],), ([True, 1],), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                build_expert_replica_plan(counts, 1)

    @patch("hyper_parallel.core.expert_parallel.hot_replica.routing.dist.is_initialized", return_value=False)
    def test_route_keeps_topk_slot_order(self, _initialized):
        """Physical remapping reconstructs logical IDs at the original positions."""
        ids = torch.tensor([[2, 0], [1, 2]])
        route = prepare_replica_route(ids, ExpertReplicaConfig(3, 1, 1))
        logical = torch.tensor(route.plan.physical_to_logical)[route.physical_ids]
        torch.testing.assert_close(logical, ids)
        with self.assertRaisesRegex(ValueError, "distinct"):
            prepare_replica_route(torch.tensor([[1, 1]]), ExpertReplicaConfig(3, 1, 1))
        with self.assertRaisesRegex(ValueError, "in-range"):
            prepare_replica_route(torch.tensor([[0, 3]]), ExpertReplicaConfig(3, 1, 1))


    def test_gradient_return_keeps_fp32_and_bounded_inbox(self):
        """High fan-in returns are summed before any owner dtype conversion."""
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), 0, None)
        gradient = torch.ones(2, 2, 2, dtype=torch.float32)
        batches = []

        def _operation(op, tensor, peer, group):
            return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

        def _exchange(operations):
            batches.append(len(operations))
            for pending in operations:
                self.assertEqual(pending.tensor.dtype, torch.float32)
                pending.tensor.fill_(0.0001)

        with patch.object(transport.dist, "P2POp", side_effect=_operation), \
                patch.object(transport.dist, "get_global_rank", side_effect=lambda _group, peer: peer), \
                patch.object(transport, "_exchange", side_effect=_exchange):
            result, = transport.return_gradients((gradient,), route)
        self.assertEqual(max(batches), 1)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.full((1, 2, 2), 1.0003), rtol=0, atol=2e-7)
