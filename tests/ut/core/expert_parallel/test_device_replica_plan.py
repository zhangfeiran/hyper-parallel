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
"""Device plan conservation and parity with the existing constructive CPU oracle."""

import random
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.device import build_device_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.planner import _bounded_copies, _materialize


class TestDeviceReplicaPlan(unittest.TestCase):
    """Validate all device outputs rather than reconstructing a plan from host counts."""

    def test_constructive_plan_matches_cpu_oracle(self):
        """Include asymmetric counts, empty ranks and guest budgets larger than H/E."""
        rng = random.Random(916)
        for ranks in (1, 2, 4):
            for home in (1, 2, 6):
                for budget in (0, 1, home, ranks * home + 1):
                    with self.subTest(ranks=ranks, home=home, budget=budget):
                        config = ExpertReplicaConfig(ranks * home, ranks, budget)
                        counts = tuple(tuple(rng.randrange(40) if rng.randrange(3) else 0
                                             for _ in range(config.num_experts)) for _ in range(ranks))
                        totals = [sum(row[e] for row in counts) for e in range(config.num_experts)]
                        loads = [sum(totals[r * home:(r + 1) * home]) for r in range(ranks)]
                        copies = _bounded_copies(loads, totals, config)
                        expected = _materialize(counts, copies, config)
                        tensor = torch.tensor(counts, dtype=torch.int64)
                        # Any accidental scalar conversion would make device execution
                        # depend on a host roundtrip, even if values were correct.
                        with patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar read")), \
                             patch.object(torch.Tensor, "tolist", side_effect=AssertionError("list read")), \
                             patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("scalar branch")):
                            actual = build_device_expert_replica_plan(tensor, config, capacity_limit=sum(loads))
                        inferred = build_device_expert_replica_plan(tensor, config)
                        self.assertEqual(inferred.host_summary().destination_counts, expected.destination_counts)
                        summary = actual.host_summary()
                        self.assertEqual(summary.slot_to_logical, expected.slot_to_logical)
                        self.assertEqual(summary.destination_counts, expected.destination_counts)
                        self.assertEqual(summary.transfers, expected.transfers)
                        torch.testing.assert_close(actual.dispatch_counts, torch.tensor(expected.dispatch_counts))
                        for rank in range(ranks):
                            slots, lengths = actual.source_runs(rank)
                            ids = torch.repeat_interleave(slots, lengths)
                            logical = actual.slot_to_logical.flatten()[ids]
                            torch.testing.assert_close(logical, torch.repeat_interleave(
                                torch.arange(config.num_experts), tensor[rank]))

    def test_distinct_topk_counts_obey_theoretical_capacity(self):
        """Use actual unique TopK routes to check the shared bound independently."""
        generator = torch.Generator().manual_seed(93)
        for budget in (0, 1, 2, 6):
            config = ExpertReplicaConfig(24, 4, budget)
            for top_k in (1, 2, 8, 24):
                ids = torch.rand(4, 32, 24, generator=generator).argsort(-1)[..., :top_k]
                counts = torch.stack([torch.bincount(row.flatten(), minlength=24) for row in ids])
                limit = config.maximum_receive_rows(32, top_k, alignment=1)
                result = build_device_expert_replica_plan(counts, config, capacity_limit=limit)
                self.assertLessEqual(max(result.host_summary().destination_loads), limit)

    def test_invalid_device_values_are_reported_by_control_summary(self):
        """No malformed count or capacity silently reaches execution consumers."""
        config = ExpertReplicaConfig(4, 2, 1)
        for counts, limit in (([[1, -1, 0, 0]] * 2, 100), ([[10, 0, 0, 0]] * 2, 1)):
            result = build_device_expert_replica_plan(torch.tensor(counts), config, capacity_limit=limit)
            with self.assertRaisesRegex(ValueError, "Device replica plan"):
                result.host_summary()
