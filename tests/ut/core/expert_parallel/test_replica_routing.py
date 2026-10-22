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
"""CPU oracles for packed route metadata and stable no-copy remapping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.core.activation_memory import checkpoint
from hyper_parallel.core.expert_parallel.hot_replica import (
    ExpertReplicaConfig,
    build_expert_replica_plan,
    routing,
)
from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts, get_recompute_state


class TestReplicaRouting(unittest.TestCase):
    """Compare both remap paths with explicit per-expert occurrence quotas."""

    def test_checkpoint_reuses_only_matching_cpu_plans_and_clears_after_replay(self) -> None:
        """A changed resident capacity cannot change the original invocation's replay placement."""
        config, group = ExpertReplicaConfig(4, 2, 1), object()
        counts = [[10, 4, 0, 0]] * 2
        forward, replay = create_recompute_contexts()
        with patch.object(routing, "build_expert_replica_plan", wraps=build_expert_replica_plan) as build:
            with forward:
                first, hit = routing._plan_replica_route(counts, config, group, 0,
                                                        capacity_limit=28, receive_capacity=14)
                self.assertFalse(hit)
                cache = get_recompute_state().get_resource(routing._RecomputedReplicaPlans, lambda: None)
            with replay:
                second, hit = routing._plan_replica_route(counts, config, group, 0,
                                                         capacity_limit=28, receive_capacity=28)
                self.assertTrue(hit)
                self.assertIs(first, second)
            self.assertEqual(build.call_count, 1)
            self.assertFalse(cache.entries)
            third, hit = routing._plan_replica_route(counts, config, group, 0,
                                                    capacity_limit=28, receive_capacity=28)
            self.assertFalse(hit)
            self.assertIsNot(first, third)
            self.assertEqual(build.call_count, 2)

    def test_replay_rebuilds_changed_inputs_configuration_and_retained_graphs(self) -> None:
        """Counts, invocation order and process-group identity guard every cache lookup."""
        config, group = ExpertReplicaConfig(4, 2, 1), object()
        counts = [[10, 4, 0, 0]] * 2
        for change in ("counts", "group", "target"):
            with self.subTest(change=change):
                forward, replay = create_recompute_contexts()
                with forward:
                    first, _ = routing._plan_replica_route(counts, config, group, 0, target_load=14)
                changed = [[9, 5, 0, 0]] * 2 if change == "counts" else counts
                with replay:
                    plan, hit = routing._plan_replica_route(changed, config, object() if change == "group" else group,
                                                            0, target_load=15 if change == "target" else 14)
                    self.assertFalse(hit)
                    self.assertEqual(plan.logical_counts, tuple(map(tuple, changed)))
                    self.assertIsNot(plan, first)
                with replay:
                    _, hit = routing._plan_replica_route(counts, config, group, 0, target_load=14)
                    self.assertFalse(hit)

    def test_replay_cache_is_ordered_and_released_on_exceptions(self) -> None:
        """Multiple calls in a region replay in order and partial failures release all CPU plans."""
        config, group = ExpertReplicaConfig(4, 2, 1), object()
        forward, replay = create_recompute_contexts()
        counts = ([[10, 4, 0, 0]] * 2, [[0, 0, 10, 4]] * 2)
        with forward:
            plans = [routing._plan_replica_route(value, config, group, 0)[0] for value in counts]
            cache = get_recompute_state().get_resource(routing._RecomputedReplicaPlans, lambda: None)
        with self.assertRaisesRegex(RuntimeError, "consumer failed"), replay:
            for value, plan in zip(counts, plans):
                actual, hit = routing._plan_replica_route(value, config, group, 0)
                self.assertTrue(hit)
                self.assertIs(actual, plan)
            raise RuntimeError("consumer failed")
        self.assertFalse(cache.entries)
        self.assertIsNone(get_recompute_state())

    def test_multistep_checkpoint_preserves_collectives_remapping_and_gradients(self) -> None:
        """Replay dynamic and empty routes while rebuilding only CPU plans once per invocation."""
        previous = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("cpu")
        self.addCleanup(DefaultDeviceType.set_device_type, previous)
        config = ExpertReplicaConfig(4, 2, 1)
        group, observations = object(), []
        for tokens, peer_tokens in ((4, 4), (0, 4), (3, 0), (0, 0), (7, 2), (2, 7)):
            with self.subTest(tokens=tokens, peer_tokens=peer_tokens):
                ids = torch.zeros((tokens, 1), dtype=torch.int64)
                matrix = torch.tensor([[tokens, 0, 0, 0, 0], [peer_tokens, 0, 0, 0, 0]])
                capacity = [1]
                value = torch.arange(tokens, dtype=torch.float32).requires_grad_()

                def operation(source: torch.Tensor) -> torch.Tensor:
                    """Exercise full route construction followed by saved-tensor recomputation."""
                    with torch.no_grad():
                        route = routing.prepare_replica_route(ids, config, group, receive_capacity=capacity[0])
                        observations.append((route.plan_reused, route.plan, route.counts_by_source))
                    return source.sin().square()

                with (patch.object(routing, "dist") as distributed,
                      patch.object(routing, "_gather_replica_counts", side_effect=lambda *_: matrix.clone()) as gather,
                      patch.object(routing, "build_expert_replica_plan", wraps=build_expert_replica_plan) as build):
                    distributed.is_initialized.return_value = True
                    distributed.get_rank.return_value = 0
                    distributed.get_world_size.return_value = 2
                    result = checkpoint(operation, value, use_reentrant=False)
                    capacity[0] = 100
                    result.sum().backward()
                self.assertEqual(gather.call_count, 2)
                self.assertEqual(build.call_count, 1)
                before, after = observations[-2], observations[-1]
                self.assertFalse(before[0])
                self.assertTrue(after[0])
                self.assertIs(before[1], after[1])
                self.assertNotEqual(before[2].data_ptr(), after[2].data_ptr())
                torch.testing.assert_close(value.grad, 2*value.detach().sin()*value.detach().cos())

    def test_count_payload_retains_global_validation_flags(self) -> None:
        """Build payloads for legal, duplicate, out-of-range and empty routes."""
        cases = (([[0, 1], [2, 3]], [1, 1, 1, 1, 0]),
                 ([[0, 0], [2, 3]], [2, 0, 1, 1, 1]),
                 ([[-1, 4], [2, 3]], [1, 0, 1, 2, 1]))
        for ids, expected in cases:
            payload = routing._replica_count_payload(torch.tensor(ids), 4)
            torch.testing.assert_close(payload, torch.tensor(expected))
        empty = routing._replica_count_payload(torch.empty((0, 2), dtype=torch.int64), 4)
        torch.testing.assert_close(empty, torch.zeros(5, dtype=torch.int64))

    def test_gather_preserves_rank_order_flags_and_retained_invocations(self) -> None:
        """Complete delayed rank-major collectives and retain their outputs."""
        expected = torch.tensor([[10, 11, 0], [20, 21, 1], [30, 31, 0], [40, 41, 0]])
        group, waited = object(), []

        def _gather(output, payload, **options):
            self.assertIs(options["group"], group)
            self.assertTrue(options["async_op"])
            self.assertTrue(output.is_contiguous())
            self.assertEqual(output.dtype, payload.dtype)

            def _wait():
                output.copy_(expected.flatten())
                waited.append(True)

            return SimpleNamespace(wait=_wait)

        with patch.object(routing.dist, "all_gather_into_tensor", side_effect=_gather):
            first = routing._gather_replica_counts(expected[0], 4, group)
            self.assertEqual(len(waited), 1)
            torch.testing.assert_close(first, expected)
            expected.add_(100)
            second = routing._gather_replica_counts(expected[0], 4, group)
            torch.testing.assert_close(second, expected)
            torch.testing.assert_close(first, expected - 100)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_remap_matches_occurrence_oracle_and_dispatch_counts(self):
        """Preserve every TopK position for zero, one and multiple guest budgets."""
        generator = torch.Generator().manual_seed(829)
        for budget in (0, 1, 6):
            for hot in (False, True):
                ids = [torch.stack([torch.randperm(8 if hot else 24, generator=generator)[:4]
                                    for _ in range(16)]) for _ in range(4)]
                counts = [torch.bincount(row.flatten(), minlength=24).tolist() for row in ids]
                plan = build_expert_replica_plan(counts, budget)
                for rank, original in enumerate(ids):
                    with self.subTest(budget=budget, hot=hot, rank=rank):
                        slots, lengths, dispatch = routing._upload_route_metadata(plan, rank, torch.device("cpu"))
                        self.assertEqual(dispatch.dtype, torch.int32)
                        self.assertTrue(dispatch.is_contiguous())
                        self.assertEqual(dispatch.tolist(), [list(row) for row in plan.dispatch_counts])
                        self.assertEqual(slots.untyped_storage().data_ptr(), dispatch.untyped_storage().data_ptr())
                        self.assertEqual(lengths.untyped_storage().data_ptr(), dispatch.untyped_storage().data_ptr())
                        expected = torch.empty_like(original.flatten())
                        for expert in range(24):
                            positions = (original.flatten() == expert).nonzero().flatten()
                            offset = 0
                            for physical, logical in enumerate(plan.physical_to_logical):
                                if logical == expert:
                                    rows = plan.dispatch_counts[rank][physical]
                                    expected[positions[offset:offset + rows]] = physical
                                    offset += rows
                            self.assertEqual(offset, positions.numel())
                        actual = routing._remap_replica_ids(original.T.contiguous().T, plan, slots, lengths)
                        torch.testing.assert_close(actual, expected.reshape_as(original))
                        torch.testing.assert_close(torch.tensor(plan.physical_to_logical)[actual], original)
