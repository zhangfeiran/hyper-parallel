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

from hyper_parallel.core.expert_parallel.hot_replica import (
    ExpertReplicaConfig,
    build_expert_replica_plan,
    routing,
)


class TestReplicaRouting(unittest.TestCase):
    """Compare both remap paths with explicit per-expert occurrence quotas."""

    def test_ragged_and_empty_sources_use_one_shared_receive_bound(self) -> None:
        """An empty source must not derive a zero receive bound for its nonempty peers."""
        config = ExpertReplicaConfig(4, 2, 1)
        matrix = torch.tensor([[0, 0, 0, 0, 0], [3, 3, 0, 0, 0]], dtype=torch.int64)
        limits = []
        for rank, tokens in enumerate((0, 3)):
            ids = torch.tensor([[0, 1]] * tokens, dtype=torch.int64).reshape(tokens, 2)
            with (
                patch.object(routing.dist, "is_initialized", return_value=True),
                patch.object(routing.dist, "get_rank", return_value=rank),
                patch.object(routing.dist, "get_world_size", return_value=2),
                patch.object(routing, "_gather_replica_counts", return_value=matrix),
                patch.object(routing, "build_expert_replica_plan", wraps=build_expert_replica_plan) as build,
            ):
                route = routing.prepare_replica_route(ids, config, max_local_num_tokens=128)
            route.plan.validate()
            self.assertEqual(tuple(route.physical_ids.shape), (tokens, 2))
            limits.append(build.call_args.kwargs["capacity_limit"])
        self.assertEqual(limits, [config.maximum_receive_rows(128, 2, alignment=1)] * 2)

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
