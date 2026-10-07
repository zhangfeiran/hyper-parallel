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

from hyper_parallel.core.expert_parallel.hot_replica import build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.plan import ExpertExecutionPlan
from hyper_parallel.core.expert_parallel.hot_replica import routing
from tests.common.mark_utils import arg_mark


class TestReplicaRouting(unittest.TestCase):
    """Compare both remap paths with explicit per-expert occurrence quotas."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_count_payload_retains_global_validation_flags(self) -> None:
        """
        Feature: Collective count and validation payload
        Description: Build payloads for legal, duplicate, out-of-range and empty routes.
        Expectation: Preserve exact clamped histograms and report invalid routes before transfer.
        """
        cases = (([[0, 1], [2, 3]], [1, 1, 1, 1, 0]),
                 ([[0, 0], [2, 3]], [2, 0, 1, 1, 1]),
                 ([[-1, 4], [2, 3]], [1, 0, 1, 2, 1]))
        for ids, expected in cases:
            payload = routing._replica_count_payload(torch.tensor(ids), 4)
            torch.testing.assert_close(payload, torch.tensor(expected))
        empty = routing._replica_count_payload(torch.empty((0, 2), dtype=torch.int64), 4)
        torch.testing.assert_close(empty, torch.zeros(5, dtype=torch.int64))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_count_payload_matches_integer_oracle_across_layouts(self) -> None:
        """
        Feature: Fixed-size integer route histogram
        Description: Count strided, empty and invalid routes using an independent scalar oracle.
        Expectation: All occurrences and global validation flags remain exact int64 values.
        """
        generator = torch.Generator().manual_seed(9027)
        for experts in (1, 7, 24):
            for shape in ((0, 8), (3, 0), (32, 1), (16, 8)):
                for dtype in (torch.int32, torch.int64):
                    ids = torch.randint(-2, experts + 2, shape, generator=generator, dtype=dtype)
                    if ids.numel():
                        ids[0, 0] = torch.iinfo(dtype).min
                        ids[-1, -1] = torch.iinfo(dtype).max
                    ids = ids.T.contiguous().T.to(torch.int64)
                    expected = [0] * (experts + 1)
                    for row in ids.tolist():
                        if len(set(row)) != len(row):
                            expected[-1] = 1
                        for expert in row:
                            if not 0 <= expert < experts:
                                expected[-1] = 1
                            expected[min(max(expert, 0), experts - 1)] += 1
                    with self.subTest(experts=experts, shape=shape, dtype=dtype):
                        actual = routing._replica_count_payload(ids, experts)
                        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.int64))
                        self.assertEqual(actual[:-1].sum().item(), ids.numel())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_gather_preserves_rank_order_flags_and_retained_invocations(self) -> None:
        """
        Feature: Invocation-owned count gather
        Description: Complete delayed rank-major collectives and retain their outputs.
        Expectation: Counts and validation flags remain exact, independent and ordered after wait.
        """
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_local_count_gather_owns_storage_without_distributed_setup(self) -> None:
        """
        Feature: Single-rank count ownership
        Description: Gather a local integer payload without an initialized process group.
        Expectation: The returned row preserves values and survives later input mutation.
        """
        payload = torch.tensor([3, 5, 0], dtype=torch.int32)
        with patch.object(routing.dist, "all_gather_into_tensor", side_effect=AssertionError("Unexpected collective")):
            result = routing._gather_replica_counts(payload, 1, None)
        payload.zero_()
        torch.testing.assert_close(result, torch.tensor([[3, 5, 0]], dtype=torch.int32))

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
