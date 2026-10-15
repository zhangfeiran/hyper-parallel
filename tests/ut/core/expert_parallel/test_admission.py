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
"""Conservative admission, native fallback and parameter-gradient contracts."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica import planner
from hyper_parallel.core.expert_parallel.hot_replica.admission import (
    reduction_basis_points,
    resolve_replica_admission,
)
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestReplicaAdmission(unittest.TestCase):
    """Keep optional admission separate from the topology and capacity contract."""

    def test_default_admission_does_not_query_device(self):
        """Resolve a route-level default without constructing a device-specific allowlist."""
        with patch.object(torch.npu, "get_device_name", side_effect=AssertionError("device query")):
            self.assertEqual(resolve_replica_admission(None), (0.4, 0))

    def test_near_balanced_route_skips_optional_solver(self):
        """Even perfect balance cannot satisfy the default reduction on a near-balanced route."""
        counts = [[120, 100, 100, 100]] * 2
        with patch.object(planner, "_bounded_copies", side_effect=AssertionError("optional solver")):
            result = planner.build_expert_replica_plan(counts, 1, capacity_limit=500,
                                                      minimum_load_reduction=resolve_replica_admission(None)[0])
        result.validate()
        self.assertEqual(result.transfers, ())
        self.assertEqual(result.destination_loads, (440, 400))

    def test_explicit_override_and_validation(self):
        """Explicit zero retains placement and invalid policies fail at the boundary."""
        self.assertEqual(resolve_replica_admission(0.0), (0.0, 0))
        self.assertEqual(resolve_replica_admission(0.35001), (0.35001, 0))
        self.assertEqual(reduction_basis_points(0.35001), 3501)
        for value in (True, -0.1, 1.01, float("nan"), float("inf"), "0.4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reduction_basis_points(value)

    def test_rejected_plan_restores_pristine_home_quotas_after_constructor_fallback(self):
        """A rejected candidate cannot mutate the home-only fallback through an alias."""
        counts = [[100, 100, 100, 0]] * 2
        worse = [{0: 200, 1: 200, 2: 200}, {2: 0, 3: 0}]

        def _small_improvement(copies, *_args):
            copies[0][0] -= 10
            copies[1][0] = 10

        with (patch.object(planner, "_bounded_copies", return_value=worse),
              patch.object(planner, "_improve_copies", side_effect=_small_improvement) as improve):
            result = planner.build_expert_replica_plan(counts, 1, capacity_limit=400,
                                                      minimum_load_reduction=0.2)
        improve.assert_called_once()
        result.validate()
        self.assertEqual(result.transfers, ())
        self.assertEqual(result.destination_loads, (400, 200))

    def test_hot_route_remains_enabled_with_default_admission(self) -> None:
        """Admit an EP16/E96 hotspot without a shape or transport allowlist."""
        counts = [[4096] * 8 + [0] * 88] * 16
        result = planner.build_expert_replica_plan(counts, 1, target_load=32768,
                                                  minimum_replica_rows=1024, capacity_limit=393216,
                                                  minimum_load_reduction=resolve_replica_admission(None)[0])
        result.validate()
        self.assertTrue(result.transfers)
        self.assertLessEqual(max(result.destination_loads), 393216 * 0.6)

    def test_near_balanced_capacity_required_replicas_are_retained(self) -> None:
        """Apply strict optional admission when a near-balanced home route exceeds capacity."""
        result = planner.build_expert_replica_plan([[120, 100, 100, 100]] * 2, 2, capacity_limit=420,
                                                  minimum_load_reduction=resolve_replica_admission(None)[0])
        result.validate()
        self.assertTrue(result.transfers)
        self.assertLessEqual(max(result.destination_loads), 420)
