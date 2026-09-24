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
from unittest.mock import patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.plan import ExpertExecutionPlan
from hyper_parallel.core.expert_parallel.hot_replica import routing


class TestReplicaRouting(unittest.TestCase):
    """Compare both remap paths with explicit per-expert occurrence quotas."""

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

    def test_home_only_mapping_avoids_expansion_and_rank_holes(self):
        """A configured B does not require sorting when no replica is used."""
        for budget in (0, 1, 6):
            plan = build_expert_replica_plan([[2] * 24] * 4, budget)
            self.assertFalse(plan.transfers)
            ids = torch.tensor([[23, 6, 0], [12, 5, 18]])
            slots, lengths, _ = routing._upload_route_metadata(plan, 0, torch.device("cpu"))
            self.assertEqual(slots.numel(), 0)
            with (patch.object(routing, "stable_expert_order", side_effect=AssertionError("unneeded sort")),
                  patch.object(torch, "repeat_interleave", side_effect=AssertionError("unneeded expansion")),
                  patch.object(torch.Tensor, "scatter_", side_effect=AssertionError("unneeded scatter"))):
                actual = routing._remap_replica_ids(ids, plan, slots, lengths)
            torch.testing.assert_close(torch.tensor(plan.physical_to_logical)[actual], ids)
            self.assertEqual(int(actual[0, 0]), 23 + 3 * budget)

    def test_cached_host_rows_and_uploaded_buffers_have_invocation_ownership(self):
        """Caller mutations and later uploads cannot corrupt saved graph metadata."""
        original = build_expert_replica_plan([[40, 0, 0, 0]] * 2, 1)
        logical = [list(row) for row in original.logical_counts]
        dispatch = [list(row) for row in original.dispatch_counts]
        plan = ExpertExecutionPlan(original.config, logical, original.slot_to_logical, dispatch)
        self.assertIs(plan.destination_loads, plan.destination_loads)
        self.assertIs(plan.transfers, plan.transfers)
        self.assertIs(plan.source_runs, plan.source_runs)
        logical[0][0] = dispatch[0][0] = 999
        self.assertEqual(plan, original)
        first = routing._upload_route_metadata(plan, 0, torch.device("cpu"))[2]
        second = routing._upload_route_metadata(plan, 0, torch.device("cpu"))[2]
        self.assertNotEqual(first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr())
        first.zero_()
        self.assertEqual(second.tolist(), [list(row) for row in original.dispatch_counts])
