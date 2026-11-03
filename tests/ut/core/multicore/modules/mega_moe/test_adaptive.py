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
"""Adaptive replica resource bounds, public opt-in and replay integration."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaConfig, ExpertReplicaCostModel, routing
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe import module as module_api
from hyper_parallel.core.multicore.modules.mega_moe import route as route_api
from hyper_parallel.core.multicore.modules.mega_moe.spec import (
    MegaMoeSpec, _grow_receive_capacity, initial_receive_capacity,
)


class TestAdaptiveExecution(unittest.TestCase):
    """Keep physical addresses fixed while admitting wider zero-copy receive layouts."""

    @staticmethod
    def model(**options: object) -> ExpertReplicaCostModel:
        """Create a matching advisory model that rejects optional copies."""
        values = dict(calibration_version=2, backend='push', hidden_size=16, intermediate_size=8,
                      ep_size=4, transport='p2p', forward_ms=((0, 0), (4096, 4096)),
                      backward_ms=((0, 0), (4096, 4096)), weight_ms=10000, gradient_ms=10000)
        return ExpertReplicaCostModel(**(values | options))

    def test_receive_reservation_is_independent_of_the_slot_cap(self) -> None:
        """Bmax equal to home experts must not force balanced receive storage in adaptive mode."""
        values = dict(local_num_tokens=128, top_k=2, ep_size=4, num_experts=24, logical_num_experts=12,
                      replica_slots_per_rank=3, initial_capacity_factor=2)
        self.assertEqual(initial_receive_capacity(values), 512)
        spec = MegaMoeSpec(local_num_tokens=128, top_k=2, ep_size=4, num_experts=24, logical_num_experts=12,
                           replica_slots_per_rank=3, initial_capacity_factor=2, receive_capacity=512,
                           hidden_size=16, intermediate_size=8, ep_group=None, rank_id=0, num_cube_cores=24)
        self.assertEqual(spec.maximum_receive_capacity, 512)
        self.assertEqual(_grow_receive_capacity(spec, 256, 400), (512, 512))
        self.assertEqual(replace(spec, dispatch_mode='pull').maximum_receive_capacity, 1024)

    def test_bounded_receive_growth_admits_owner_without_changing_target(self) -> None:
        """An affordable owner can avoid fixed copy costs without a second count exchange."""
        config, group = ExpertReplicaConfig(4, 2, 1), object()
        ids = torch.tensor([[0, 1]] * 88 + [[0, 2]] * 40)
        matrix = torch.tensor([[128, 88, 40, 0, 0, 0, 128, 31]] * 2)
        for compact in (False, True):
            for allowed, guests in ((384, True), (512, False)):
                with (self.subTest(compact=compact, allowed=allowed),
                      patch.object(routing, 'dist') as distributed,
                      patch.object(routing, '_gather_replica_counts', return_value=matrix) as gather,
                      patch.object(routing, 'build_expert_replica_plan',
                                   wraps=routing.build_expert_replica_plan) as build):
                    distributed.is_initialized.return_value = True
                    distributed.get_rank.return_value = 0
                    distributed.get_world_size.return_value = 2
                    budget = Mock(return_value=allowed)
                    options = dict(compact_home=compact, dynamic_header=(0, 128, 31),
                                   _auto_capacity_factor=1.25, _capacity_budget=budget,
                                   _cost_model=self.model(ep_size=2, replica_overhead_ms=10000))
                    forward, replay = create_recompute_contexts()
                    with forward:
                        first = routing.prepare_replica_route(ids, config, group, **options)
                    with replay:
                        second = routing.prepare_replica_route(ids, config, group, **options)
                self.assertEqual(gather.call_count, 2)
                self.assertEqual(bool(first.plan.transfers), guests)
                self.assertEqual(bool(second.plan.transfers), guests)
                self.assertEqual(first.dynamic_summary, second.dynamic_summary)
                self.assertLessEqual(max(first.plan.destination_loads), allowed)
                self.assertTrue(all(call.args == (128, 384) for call in budget.call_args_list))
                if compact and not guests:
                    self.assertEqual(build.call_count, 0)
                    self.assertTrue(first.plan_bypassed and second.plan_bypassed)
                else:
                    first.plan.validate()
                    self.assertEqual(build.call_count, 1)
                    self.assertTrue(second.plan_reused)
                    self.assertEqual(build.call_args.kwargs['target_load'], 384)

    def test_larger_replay_allowance_keeps_the_original_replica_graph(self) -> None:
        """A newly affordable owner cannot change checkpoint metadata after a replicated forward."""
        config, group = ExpertReplicaConfig(4, 2, 1), object()
        ids = torch.tensor([[0, 1]] * 88 + [[0, 2]] * 40)
        matrix = torch.tensor([[128, 88, 40, 0, 0, 0, 128, 31]] * 2)
        budget = Mock(return_value=384)
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, '_gather_replica_counts', return_value=matrix),
              patch.object(routing, 'build_expert_replica_plan', wraps=routing.build_expert_replica_plan) as build):
            distributed.is_initialized.return_value = True
            distributed.get_rank.return_value = 0
            distributed.get_world_size.return_value = 2
            options = dict(compact_home=True, dynamic_header=(0, 128, 31), _auto_capacity_factor=1.25,
                           _capacity_budget=budget, _cost_model=self.model(ep_size=2, replica_overhead_ms=10000))
            forward, replay = create_recompute_contexts()
            with forward:
                first = routing.prepare_replica_route(ids, config, group, **options)
            budget.return_value = 512
            with replay:
                second = routing.prepare_replica_route(ids, config, group, **options)
        self.assertTrue(first.plan.transfers)
        self.assertTrue(second.plan_reused)
        self.assertIs(first.plan, second.plan)
        self.assertEqual(first.home_only, second.home_only)
        self.assertEqual(build.call_count, 1)

    def test_native_agreement_keeps_compact_cost_admission_without_full_readback(self) -> None:
        """Reused Native counts must retain bounded owner admission and avoid another gather."""
        config = ExpertReplicaConfig(4, 2, 1)
        ids = torch.tensor([[0, 1]] * 88 + [[0, 2]] * 40, dtype=torch.int32)
        logical = torch.tensor([[128, 88, 40, 0]] * 2, dtype=torch.int32)
        summary = [[0, 128, 31, 256, 0, 128, 432], [0, 128, 31, 256, 0, 128, 80]]
        original_cpu = torch.Tensor.cpu

        def compact_only(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            """Reject accidental materialization of the full already-gathered table."""
            if tensor.shape == logical.shape:
                raise AssertionError('Full count table readback')
            return original_cpu(tensor, *args, **kwargs)

        for host in (None, logical.tolist()):
            gathered = routing._GatheredReplicaCounts(logical, summary, host)
            budget = Mock(return_value=512)
            with (self.subTest(host_available=host is not None), patch.object(routing, 'dist') as distributed,
                  patch.object(routing, '_gather_replica_counts', side_effect=AssertionError('Duplicate gather')),
                  patch.object(routing, 'build_expert_replica_plan', side_effect=AssertionError('CPU planner')),
                  patch.object(torch.Tensor, 'cpu', compact_only)):
                distributed.is_initialized.return_value = True
                distributed.get_rank.return_value = 0
                distributed.get_world_size.return_value = 2
                route = routing.prepare_replica_route(
                    ids, config, compact_home=True, dynamic_header=(0, 128, 31),
                    _auto_capacity_factor=1.25, _capacity_budget=budget, _gathered_counts=gathered,
                    _cost_model=self.model(ep_size=2, replica_overhead_ms=10000))
            self.assertTrue(route.home_only and route.plan_bypassed)
            self.assertEqual(route.plan.destination_loads, (432, 80))
            self.assertEqual(route.dynamic_summary, tuple(map(tuple, summary)))
            budget.assert_called_once_with(128, 384)

    def test_public_constructor_needs_only_the_resident_cap(self) -> None:
        """No calibration or adaptive toggle is needed for an unmeasured shape."""
        expert = MegaMoeExperts(hidden_size=16, intermediate_size=8,
            num_experts=12, top_k=2, ep_size=4, replica_slots_per_rank=3)
        try:
            self.assertFalse(hasattr(expert, 'replica_adaptive'))
            self.assertFalse(hasattr(expert, 'replica_cost_model'))
            self.assertEqual(expert.replica_config.physical_experts, 24)
        finally:
            expert.close()

    def test_adaptive_route_replays_with_one_gather_and_fixed_physical_ids(self) -> None:
        """A feasible owner peak larger than the Bmax balance bound survives recomputation."""
        config = ExpertReplicaConfig(12, 4, 3)
        ids = (torch.arange(100) % 3).reshape(-1, 1)
        matrix = torch.tensor([[34, 33, 33] + [0] * 10] * 4)
        forward, replay = create_recompute_contexts()
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, '_gather_replica_counts', return_value=matrix) as gather,
              patch.object(routing, 'build_expert_replica_plan', wraps=routing.build_expert_replica_plan) as build):
            distributed.is_initialized.return_value = True
            distributed.get_rank.return_value = 0
            distributed.get_world_size.return_value = 4
            options = dict(_cost_model=self.model(), receive_capacity=400, compact_home=True)
            with forward:
                first = routing.prepare_replica_route(ids, config, **options)
            with replay:
                second = routing.prepare_replica_route(ids, config, **options)
        self.assertEqual(gather.call_count, 2)
        self.assertEqual(build.call_count, 1)
        self.assertTrue(first.home_only)
        self.assertTrue(second.plan_reused)
        self.assertIs(first.plan, second.plan)
        self.assertEqual(max(first.plan.destination_loads), 400)
        self.assertEqual(first.plan.config, config)

    def test_forward_passes_adaptive_policy_without_extra_collectives(self) -> None:
        """Both dispatch modes keep the physical reservation while using the chosen logical graph."""
        for mode in ('push', 'pull'):
            with self.subTest(mode=mode):
                expert = MegaMoeExperts(hidden_size=16, intermediate_size=8,
                    num_experts=12, top_k=2, ep_size=4, dispatch_mode=mode, replica_slots_per_rank=3,
                    ).bfloat16()
                physical = SimpleNamespace(rank_id=0, replica_slots_per_rank=3, auto_tokens=False)
                home = SimpleNamespace(rank_id=0, replica_slots_per_rank=0)
                selected = SimpleNamespace(spec=home)
                resources = SimpleNamespace(spec=physical, home_spec=home, plan=None, home_plan=None,
                    rank_nodes=(0, 1, 1, 0),
                    workspace=Mock(capacity_floor=512, in_use=True), ensure_capacity=Mock(),
                    plan_for_tokens=Mock(return_value=selected))
                hidden = torch.ones(64, 16, dtype=torch.bfloat16)
                ids, probabilities = torch.zeros(64, 2, dtype=torch.int32), torch.full((64, 2), .5)
                route = SimpleNamespace(unpermute_mapping=object(), maximum_received_slots=320,
                                        metadata=SimpleNamespace(plan_tokens=128))
                try:
                    with (patch.object(torch.Tensor, 'is_npu', new_callable=PropertyMock, return_value=True),
                          patch.object(torch.npu, 'is_current_stream_capturing', return_value=False),
                          patch.object(expert, '_get_execution_resources', return_value=resources),
                          patch.object(module_api, '_runtime_cost_model', return_value=self.model(backend=mode)),
                          patch.object(module_api, 'prepare_replica_route') as prepare,
                          patch.object(module_api, 'prepare_topk_route', return_value=route),
                          patch.object(module_api, 'execute_mega_moe_with_permutation', return_value=hidden),
                          patch.object(module_api, 'restore_topk_output', return_value=hidden)):
                        prepare.return_value = SimpleNamespace(physical_ids=ids, home_only=True,
                                                              counts_by_source=torch.zeros(4, 12))
                        torch.testing.assert_close(expert(hidden, ids, probabilities), hidden)
                    self.assertIsNotNone(prepare.call_args.kwargs['_cost_model'])
                    self.assertEqual(prepare.call_args.kwargs['_rank_nodes'], resources.rank_nodes)
                    self.assertEqual(prepare.call_args.kwargs['minimum_load_reduction'], 0)
                finally:
                    expert.close()

    def test_unmatched_runtime_credits_only_required_expert_gradients(self) -> None:
        """Frozen projections and inference contribute no unverified matrix-transfer credit."""
        expert = MegaMoeExperts(hidden_size=16, intermediate_size=24, num_experts=4, top_k=1,
                               ep_size=2, replica_slots_per_rank=1, create_parameters=False)
        resources = SimpleNamespace(spec=SimpleNamespace(auto_tokens=True), home_spec=object(), rank_nodes=(0, 1))
        try:
            for mask, credit in ((31, 12), (5, 8), (9, 4), (1, 0), (12, 0)):
                with self.subTest(mask=mask):
                    with patch.object(module_api, 'prepare_replica_route') as prepare:
                        expert._prepare_replica_route(torch.zeros(8, 1, dtype=torch.int32), resources,
                                                       torch.empty(8, 16, dtype=torch.bfloat16), mask, None)
                    self.assertEqual(prepare.call_args.kwargs['_replica_transfer_row_credit'], credit)
                    self.assertIsNone(prepare.call_args.kwargs['_cost_model'])
            forward, _ = create_recompute_contexts()
            with forward, patch.object(module_api, 'prepare_replica_route') as prepare:
                expert._prepare_replica_route(torch.zeros(8, 1, dtype=torch.int32), resources,
                                               torch.empty(8, 16, dtype=torch.bfloat16), 12, None)
            self.assertEqual(prepare.call_args.kwargs['_replica_transfer_row_credit'], 12)
        finally:
            expert.close()

    def test_checkpoint_keeps_alternating_placements_owned_by_each_call(self) -> None:
        """Cold and hot calls share a checkpoint without overwriting the earlier decision."""
        config, group = ExpertReplicaConfig(12, 4, 3), object()
        cold = [[25] * 12] + [[0] * 12] * 3
        hot = [[100] * 3 + [0] * 9] + [[0] * 12] * 3
        forward, replay = create_recompute_contexts()
        options = dict(adaptive=True, cost_model=self.model(weight_ms=0, gradient_ms=0), capacity_limit=300)
        with forward:
            original = [routing._plan_replica_route(counts, config, group, 0, **options)[0]
                        for counts in (cold, hot, cold)]
        self.assertEqual([bool(plan.transfers) for plan in original], [False, True, False])
        with replay:
            repeated = [routing._plan_replica_route(counts, config, group, 0, **options)
                        for counts in (cold, hot, cold)]
        self.assertTrue(all(hit and plan is expected for (plan, hit), expected in zip(repeated, original)))
        self.assertEqual([bool(plan.transfers) for plan in original], [False, True, False])

    def test_automatic_push_uses_actual_group_size_and_reuses_the_checkpoint_plan(self) -> None:
        """Group token sizes must control both compact bypass and cached CPU replay."""
        config, group = ExpertReplicaConfig(4, 2, 1), object()
        ids = torch.tensor([[0, 2]] * 128)
        matrix = torch.tensor([[128, 0, 128, 0, 0, 0, 128, 31],
                               [257, 0, 257, 0, 0, 0, 257, 31]])
        for compact in (False, True):
            forward, replay = create_recompute_contexts()
            with (self.subTest(compact=compact), patch.object(routing, 'dist') as distributed,
                  patch.object(routing, '_gather_replica_counts', return_value=matrix) as gather,
                  patch.object(routing, '_plan_replica_route', wraps=routing._plan_replica_route) as plan,
                  patch.object(routing, 'build_expert_replica_plan', wraps=routing.build_expert_replica_plan) as build):
                distributed.is_initialized.return_value = True
                distributed.get_rank.return_value = 0
                distributed.get_world_size.return_value = 2
                options = dict(compact_home=compact, dynamic_header=(0, 128, 31), _auto_capacity_factor=1.25)
                with forward:
                    first = routing.prepare_replica_route(ids, config, group, receive_capacity=320, **options)
                with replay:
                    second = routing.prepare_replica_route(ids, config, group, receive_capacity=4096, **options)
            self.assertEqual(gather.call_count, 2)
            self.assertEqual(first.plan.destination_loads, (385, 385))
            self.assertEqual(second.plan.destination_loads, (385, 385))
            self.assertEqual(first.dynamic_summary, second.dynamic_summary)
            if compact:
                self.assertEqual(build.call_count, 0)
                self.assertEqual(plan.call_count, 0)
                self.assertTrue(first.plan_bypassed and second.plan_bypassed)
                self.assertFalse(second.plan_reused)
                self.assertTrue(first.home_only and second.home_only)
                self.assertEqual(first.counts_by_source.tolist(), matrix[:, :4].tolist())
            else:
                self.assertEqual(build.call_count, 1)
                self.assertTrue(second.plan_reused)
                self.assertIs(first.plan, second.plan)
                for call in plan.call_args_list:
                    self.assertEqual(call.kwargs['receive_capacity'], 768)
                    self.assertEqual(call.kwargs['target_load'], 768)

    def test_compact_cost_bound_only_skips_plans_that_choose_owner_execution(self) -> None:
        """Early admission must agree with the complete planner, including capacity pressure."""
        counts = [[4, 4, 0, 0], [3, 3, 1, 1]]
        matrix = torch.tensor([row + [0, 0, 4, 31] for row in counts])
        ids = torch.tensor([[0, 1]] * 4)
        config = ExpertReplicaConfig(4, 2, 1)
        for capacity in (8, 16):
            for overhead in (0., 10000.):
                with self.subTest(capacity=capacity, overhead=overhead):
                    model = self.model(ep_size=2, replica_overhead_ms=overhead)
                    route = routing._try_compact_home_route(
                        matrix, ids, config, object(), 0, target_load=None,
                        receive_capacity=capacity, factor=None, minimum_load_reduction=0.,
                        minimum_saved_rows=0, cost_model=model, rank_nodes=())
                    if capacity == 8 or not overhead:
                        self.assertIsNone(route)
                    else:
                        plan = routing.build_expert_replica_plan(
                            counts, 1, adaptive=True, capacity_limit=16, receive_capacity=capacity, cost_model=model)
                        self.assertFalse(plan.transfers)
                        self.assertTrue(route.plan_bypassed)
                        self.assertEqual(route.plan.destination_loads, plan.destination_loads)
                        self.assertEqual(route.counts_by_source.tolist(), counts)

    def test_compact_markers_preserve_mixed_checkpoint_call_order(self) -> None:
        """A skipped call must not consume the cached plan belonging to the next hot call."""
        config, group = ExpertReplicaConfig(4, 2, 1), object()
        ids = torch.tensor([[0], [0], [2], [2]])
        cold = torch.tensor([[2, 0, 2, 0, 0, 0, 4, 31], [3, 0, 3, 0, 0, 0, 6, 31]])
        hot = torch.tensor([[4, 0, 0, 0, 0, 0, 4, 31], [6, 0, 0, 0, 0, 0, 6, 31]])
        forward, replay = create_recompute_contexts()
        with (patch.object(routing, 'dist') as distributed,
              patch.object(routing, '_gather_replica_counts', side_effect=[cold, hot, cold] * 2)):
            distributed.is_initialized.return_value = True
            distributed.get_rank.return_value = 0
            distributed.get_world_size.return_value = 2
            options = dict(compact_home=True, dynamic_header=(0, 4, 31), target_load=5, receive_capacity=8)
            with forward:
                original = [routing.prepare_replica_route(ids, config, group, **options) for _ in range(3)]
            with replay:
                repeated = [routing.prepare_replica_route(ids, config, group, **options) for _ in range(3)]
        self.assertEqual([route.plan_bypassed for route in original], [True, False, True])
        self.assertEqual([route.plan_bypassed for route in repeated], [True, False, True])
        self.assertEqual([route.plan_reused for route in repeated], [False, True, False])
        self.assertIs(original[1].plan, repeated[1].plan)
        self.assertTrue(repeated[1].plan.transfers)

    def test_automatic_replica_route_grows_before_pull_without_another_count_exchange(self) -> None:
        """Reuse gathered controls for uneven and empty sources before publishing Pull rows."""
        for mode in ('push', 'pull'):
            for local_tokens, peer_tokens in ((128, 257), (0, 257), (0, 0)):
                with self.subTest(mode=mode, tokens=(local_tokens, peer_tokens)):
                    logical = [[local_tokens, 0, local_tokens, 0], [peer_tokens, 0, peer_tokens, 0]]
                    plan = routing.build_expert_replica_plan(logical, 1)
                    counts = torch.tensor(plan.dispatch_counts, dtype=torch.int32)
                    tokens = torch.empty(local_tokens, 16, dtype=torch.bfloat16)
                    ids = torch.tensor([[0, 3]] * local_tokens, dtype=torch.int32).reshape(local_tokens, 2)
                    summary = tuple((0, size, 31, size * 2, 0, size, load)
                                    for size, load in zip((local_tokens, peer_tokens), plan.destination_loads))
                    replica = routing.ReplicaRoute(plan, ids, counts, 0, None, dynamic_summary=summary)
                    spec = MegaMoeSpec(128, 16, 8, 6, 2, 1.25, 384, 2, None, 0, 24,
                                       dispatch_mode=mode, auto_tokens=True, replica_slots_per_rank=1,
                                       logical_num_experts=4)
                    workspace, order = Mock(in_use=False, source_buffer=None), []

                    def prepare(rows: int, _received: int) -> None:
                        """Materialize the committed source view before Pull permutation."""
                        order.append('capacity')
                        workspace.source_buffer = torch.empty(rows * 2, 16, dtype=tokens.dtype)
                        workspace.in_use = mode == 'pull'

                    def permute(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                        """Record permutation order without invoking an NPU operator."""
                        order.append('permute')
                        output = args[2] if len(args) == 3 else tokens.repeat_interleave(2, dim=0)
                        return output, torch.arange(local_tokens * 2, dtype=torch.int32)

                    with (patch.object(route_api, '_start_dynamic_gather', side_effect=AssertionError('Extra gather')),
                          patch.object(route_api, '_dynamic_route_payload', side_effect=AssertionError('Extra upload')),
                          patch.object(route_api, '_permute_topk_input', side_effect=permute),
                          patch.object(route_api, '_permute_topk_input_out', side_effect=permute)):
                        result = route_api._prepare_dynamic_route(
                            tokens, ids, ids.reshape(-1), spec, counts[0], workspace, 31,
                            replica_route=replica, prepare_workspace=prepare)
                    expected = ['capacity', 'permute'] if mode == 'pull' else ['permute', 'capacity']
                    self.assertEqual(order, ['capacity'] if mode == 'pull' and not local_tokens else expected)
                    self.assertIs(result.metadata.replica_route, replica)
                    self.assertEqual(result.metadata.local_num_tokens, local_tokens)
                    self.assertEqual(result.maximum_received_slots, max(plan.destination_loads))

    def test_mixed_adaptive_configuration_fails_before_symmetric_allocation(self) -> None:
        """Ranks cannot use different capacity policies for one physical resource group."""
        resources = module_api._MegaMoeExecutionResources.__new__(module_api._MegaMoeExecutionResources)
        resources.spec = SimpleNamespace(ep_size=4, ep_group=object())

        def gather(result: list, manifest: tuple, **_kwargs: object) -> None:
            """Expose one peer with the same topology but a different adaptive flag."""
            row = list(manifest[2][0])
            row[-3] = False
            peer = manifest[:2] + ((tuple(row),),)
            result[:] = [manifest, peer, manifest, manifest]

        with (patch.object(module_api.dist, 'all_gather_object', side_effect=gather),
              self.assertRaisesRegex(ValueError, 'configurations differ')):
            resources._validate_distributed_configuration(({'_replica_cost_model_enabled': True},))
