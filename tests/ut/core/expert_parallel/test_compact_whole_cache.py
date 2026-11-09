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
"""Check bounded complete-expert admission reuse after fresh count agreement."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import routing
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from tests.common.mark_utils import arg_mark


class TestCompactWholeCache(unittest.TestCase):
    """Retain exact admission semantics without retaining mutable quotas or device tensors."""

    def setUp(self) -> None:
        """Use an independently bounded four-rank skew with two nonempty experts."""
        routing._cached_complete_admission.cache_clear()
        self.addCleanup(routing._cached_complete_admission.cache_clear)
        config = ExpertReplicaConfig(8, 4, 2)
        self.decision = routing._RejectedReplicaQuotas(config, (400, 400, 0, 0, 0, 0, 0, 0),
                                                       (800, 0, 0, 0), 200, 4000, 0, 4)
        self.signature = (self.decision, 1000, 1000, False, (0, 0, 1, 1), ())

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_repeated_admission_returns_private_mutable_copies(self) -> None:
        """Feature: Exact reusable admission.
        Description: Reuse the same fresh-count signature after mutating a previous caller's dictionaries.
        Expectation: The planner runs once and subsequent callers receive unchanged independent quotas.
        """
        with patch.object(routing, '_admitted_uncalibrated_copies',
                          wraps=routing._admitted_uncalibrated_copies) as planner:
            first = routing._complete_admission_copies(self.signature)
            expected = [dict(row) for row in first]
            first[0].clear()
            second = routing._complete_admission_copies(self.signature)
            self.assertEqual(second, expected)
            self.assertIsNot(first, second)
            self.assertEqual(planner.call_count, 1)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_capacity_topology_and_policy_changes_invalidate_cached_quotas(self) -> None:
        """Feature: Complete admission cache identity.
        Description: Change each capacity, topology, search and benefit-policy input separately.
        Expectation: Every changed input is evaluated by the original admission algorithm.
        """
        replacements = [replace(self.decision, target=201), replace(self.decision, minimum_rows=5),
                        replace(self.decision, admission=4001), replace(self.decision, saved_rows=1),
                        replace(self.decision, config=ExpertReplicaConfig(8, 4, 1)),
                        replace(self.decision, totals=(400, 396, 4, 0, 0, 0, 0, 0), loads=(796, 4, 0, 0))]
        changed = [(value, *self.signature[1:]) for value in replacements]
        changed += [(self.decision, 1001, 1000, False, (0, 0, 1, 1), ()),
                    (self.decision, 1000, 999, False, (0, 0, 1, 1), ()),
                    (self.decision, 1000, 1000, True, (0, 0, 1, 1), ()),
                    (self.decision, 1000, 1000, False, (0, 1, 0, 1), ()),
                    (self.decision, 1000, 1000, False, (0, 0, 1, 1), (0, 0, 1, 1))]
        with patch.object(routing, '_admitted_uncalibrated_copies',
                          wraps=routing._admitted_uncalibrated_copies) as planner:
            routing._complete_admission_copies(self.signature)
            for signature in changed:
                with self.subTest(signature=signature):
                    routing._complete_admission_copies(signature)
            self.assertEqual(planner.call_count, 1 + len(changed))

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_cache_eviction_recomputes_the_oldest_admission(self) -> None:
        """Feature: Bounded admission state lifetime.
        Description: Fill more than sixty-four receive-bound signatures and revisit the oldest one.
        Expectation: The oldest quotas are recomputed exactly instead of being retained without a bound.
        """
        with patch.object(routing, '_admitted_uncalibrated_copies',
                          wraps=routing._admitted_uncalibrated_copies) as planner:
            first = routing._complete_admission_copies(self.signature)
            for limit in range(1001, 1065):
                routing._complete_admission_copies((self.decision, limit, *self.signature[2:]))
            self.assertEqual(routing._complete_admission_copies(self.signature), first)
            self.assertEqual(planner.call_count, 66)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_invalid_minimum_rows_cannot_hit_a_numerically_equal_integer_key(self) -> None:
        """Feature: Cache input-type validation.
        Description: Warm an integer key, then pass the same numeric value with an invalid type.
        Expectation: Cached metadata cannot turn malformed options into a valid route.
        """
        routing._complete_admission_copies(self.signature)
        for value in (4.0, True, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                routing._complete_admission_copies((replace(self.decision, minimum_rows=value), *self.signature[1:]))

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_only_complete_routes_reuse_admission_and_every_call_keeps_current_counts(self) -> None:
        """Feature: Complete-only production reachability.
        Description: Prepare repeated routes through the real public CPU planner with each policy.
        Expectation: Shared partial planning is unchanged; complete planning reuses only the pure CPU decision.
        """
        counts = torch.tensor([[100, 100, 0, 0, 0, 0, 0, 0]] * 4, dtype=torch.int32)
        summary = [[0, 200, 31, 200, 0, 100, load] for load in (800, 0, 0, 0)]
        gathered = routing._GatheredReplicaCounts(counts, summary, counts.tolist())
        ids = torch.tensor([[0]] * 100 + [[1]] * 100, dtype=torch.int32)
        for complete in (False, True):
            routing._cached_complete_admission.cache_clear()
            with (self.subTest(complete=complete), patch.object(routing, 'dist') as distributed,
                  patch.object(routing, '_admitted_uncalibrated_copies',
                               wraps=routing._admitted_uncalibrated_copies) as planner):
                distributed.is_initialized.return_value = True
                distributed.get_world_size.return_value = 4
                distributed.get_rank.return_value = 0
                routes = [routing.prepare_replica_route(
                    current, self.decision.config, minimum_replica_rows=4, minimum_load_reduction=.4,
                    target_load=200, receive_capacity=1000, compact_home=True, dynamic_header=(0, 200, 31),
                    _gathered_counts=gathered, _complete_experts=complete) for current in (ids, ids.flip(0))]
                self.assertEqual(planner.call_count, 1 if complete else 2)
                self.assertEqual(routes[0].plan, routes[1].plan)
                self.assertFalse(torch.equal(routes[0].physical_ids, routes[1].physical_ids))
                torch.testing.assert_close(routes[0].counts_by_source, routes[1].counts_by_source)
