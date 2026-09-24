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
from hyper_parallel.core.expert_parallel.hot_replica import native, planner, transport
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

    def test_owned_p2p_return_preserves_guest_storage_and_fp32_order(self):
        """Consume only home rows while preserving guest slices and cancellation order."""
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), 0, None)
        gradient = torch.full((2, 2, 2), 5.0)
        gradient[0].fill_(0.5)
        expected = gradient[:1].clone()
        values = (0.0, float(2**24), 1.0, -float(2**24))
        for peer in range(1, 4):
            expected.add_(values[peer])

        def _operation(op, tensor, peer, group):
            return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

        def _exchange(operations):
            for pending in operations:
                pending.tensor.fill_(values[pending.peer])

        with patch.object(transport.dist, "P2POp", side_effect=_operation), \
                patch.object(transport.dist, "get_global_rank", side_effect=lambda _group, peer: peer), \
                patch.object(transport, "_exchange", side_effect=_exchange):
            result, = transport.return_gradients((gradient,), route, consume=True)
        self.assertEqual(result.data_ptr(), gradient.data_ptr())
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(gradient[1], torch.full((2, 2), 5.0))

    def test_owned_return_falls_back_to_legacy_provider_without_extra_arguments(self):
        """Old providers retain their call contract even when the caller owns its buffers."""
        home = (torch.ones(1, 2, 2),)
        calls = []

        def _legacy(gradients, guests, route):
            calls.append((guests, route))
            return tuple(value.clone() for value in gradients)

        provider = SimpleNamespace(return_gradients=_legacy)
        route = SimpleNamespace(transport=provider)
        result = transport.return_gradients(home, route, consume=True)
        self.assertEqual(calls, [(None, route)])
        self.assertNotEqual(result[0].data_ptr(), home[0].data_ptr())
        torch.testing.assert_close(result[0], home[0])
        provider.return_gradients_owned = lambda gradients, _guests, _route: gradients
        self.assertIs(transport.return_gradients(home, route, consume=True), home)
        borrowed = transport.return_gradients(home, route)
        self.assertNotEqual(borrowed[0].data_ptr(), home[0].data_ptr())
        self.assertEqual(len(calls), 2)

    def test_owned_return_rejects_non_fp32_or_autograd_owned_inputs(self):
        """An ownership opt-in must not silently convert or mutate saved differentiable inputs."""
        plan = build_expert_replica_plan([[1]], 0)
        route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), 0, None)
        for value in (torch.ones(1, 2, 2, dtype=torch.bfloat16), torch.ones(1, 2, 2, requires_grad=True)):
            with self.assertRaisesRegex(ValueError, "detached FP32"), patch.object(transport, "_exchange") as exchange:
                transport.return_gradients((value,), route, consume=True)
            exchange.assert_not_called()

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

    def test_retained_copy_uses_home_rows_below_capacity_target(self):
        """A mandatory copy absorbs remaining expert rows without another weight transfer."""
        counts = [[512] * 8 + [0] * 16] * 4
        config = ExpertReplicaConfig(24, 4, 1)
        upper = config.maximum_receive_rows(512, 8, alignment=1)
        options = {"target_load": upper, "minimum_replica_rows": 4096, "capacity_limit": upper}
        with patch.object(planner, "_rebalance_existing_copies"):
            original = build_expert_replica_plan(counts, 1, **options)
        plan = build_expert_replica_plan(counts, 1, **options)
        plan.validate()
        self.assertEqual(original.destination_loads, (10926, 4096, 0, 1362))
        self.assertEqual(plan.destination_loads, (10240, 4096, 0, 2048))
        self.assertEqual(plan.transfers, original.transfers)
        self.assertEqual(plan.slot_to_logical, original.slot_to_logical)
        self.assertEqual(plan.logical_counts, original.logical_counts)

    def test_existing_copy_stops_at_pair_balance_and_keeps_odd_row(self):
        """A large expert must not make a lighter receiver the new load peak."""
        config = ExpertReplicaConfig(4, 2, 1)
        for total in (20, 21):
            with self.subTest(total=total):
                copies = [{0: total - 1, 1: 0}, {2: 0, 3: 0, 0: 1}]
                planner._rebalance_existing_copies(copies, config)
                self.assertEqual([sum(row.values()) for row in copies], [(total + 1) // 2, total // 2])
                self.assertEqual([set(row) for row in copies], [{0, 1}, {0, 2, 3}])

    def test_existing_copy_does_not_pull_from_a_lighter_home(self):
        """Keep existing guest contributions and avoid moving guest work through another rank."""
        config = ExpertReplicaConfig(4, 2, 1)
        copies = [{0: 2, 1: 0}, {2: 3, 3: 0, 0: 5}]
        expected = [dict(row) for row in copies]
        planner._rebalance_existing_copies(copies, config)
        self.assertEqual(copies, expected)

    def test_retained_copy_refinement_preserves_placements_for_skewed_routes(self):
        """Independent histograms retain source quotas and cannot exceed the old receive peak."""
        rng = random.Random(8157)
        for _ in range(120):
            ranks, home = rng.randint(2, 6), rng.randint(1, 6)
            experts, tokens = ranks * home, rng.randint(1, 80)
            top_k = rng.randint(1, experts)
            popular = rng.sample(range(experts), top_k)
            counts = [[0] * experts for _ in range(ranks)]
            for row in counts:
                for _ in range(tokens):
                    selected = popular if rng.random() < 0.85 else rng.sample(range(experts), top_k)
                    for expert in selected:
                        row[expert] += 1
            budget = rng.randint(0, home + 1)
            config = ExpertReplicaConfig(experts, ranks, budget)
            upper = config.maximum_receive_rows(tokens, top_k, alignment=1)
            options = {"minimum_replica_rows": rng.choice((1, 16, 4096)), "capacity_limit": upper,
                       "target_load": rng.choice((None, tokens * top_k, upper))}
            with patch.object(planner, "_rebalance_existing_copies"):
                original = build_expert_replica_plan(counts, budget, **options)
            plan = build_expert_replica_plan(counts, budget, **options)
            plan.validate()
            self.assertEqual(plan.slot_to_logical, original.slot_to_logical)
            self.assertEqual(plan.transfers, original.transfers)
            self.assertEqual(plan.logical_counts, original.logical_counts)
            self.assertLessEqual(max(plan.destination_loads), min(upper, max(original.destination_loads)))
            self.assertEqual(plan, build_expert_replica_plan(counts, budget, **options))

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
