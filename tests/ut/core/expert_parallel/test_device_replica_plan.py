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
"""Exact policy parity of the portable device solver and shared launch contract."""

import ctypes
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.common.mark_utils import arg_mark

from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaConfig, build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica import device_kernel
from hyper_parallel.core.expert_parallel.hot_replica.device import (
    build_device_expert_replica_plan, validate_planner_backend,
)


class TestDeviceReplicaPlan(unittest.TestCase):
    """Compile the same integer solver for CPU; mock only the NPU launch boundary."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the portable solver without requiring a device toolkit."""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            header = Path(device_kernel.__file__).parent / "_device_kernel" / "algorithm.h"
            wrapper = folder / "solver.cpp"
            wrapper.write_text(
                '#include "algorithm.h"\n#include <vector>\n'
                'extern "C" int launch_planner(void*, const int64_t* counts, int64_t* control, int32_t* dispatch, '
                'int r, int e, int b, int64_t capacity, int64_t target, int64_t minimum, '
                'int64_t admission, int64_t saved_rows) {\n'
                'std::vector<int64_t> scratch(5*r*e+3*e+4*r);\n'
                'for (int i=0;i<r*e;++i) scratch[i]=counts[i];\n'
                'hot::Solver(scratch.data(),r,e,b).Run(control,dispatch,capacity,target,minimum,admission,saved_rows); '
                'return 0; }\n'
            )
            subprocess.run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', '-I', str(header.parent),
                            str(wrapper), '-o', str(folder / 'solver.so')], check=True, capture_output=True)
            cls.library = ctypes.CDLL(str(folder / 'solver.so'))
            cls.library.launch_planner.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_int64] * 5
            cls.library.launch_planner.restype = ctypes.c_int

    def setUp(self) -> None:
        """Replace only device context, lifetime recording and library loading."""
        self.enterContext(patch.object(device_kernel, "_validate_device"))
        self.enterContext(patch.object(device_kernel, "_planner_library", return_value=self.library))
        self.enterContext(patch.object(torch.npu, "device"))
        stream = self.enterContext(patch.object(torch.npu, "current_stream"))
        stream.return_value.npu_stream = 0
        self.enterContext(patch.object(torch.Tensor, "record_stream"))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_fused_policy_matches_cpu_oracle(self):
        """
        description: Compare slots and every source quota, including target and minimum-row policy.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        rng = random.Random(1732)
        for ranks in (1, 2, 3, 4, 8):
            for home in (1, 2, 6):
                for budget in (0, 1, home, ranks * home + 1):
                    config = ExpertReplicaConfig(ranks * home, ranks, budget)
                    for iteration in range(20):
                        with self.subTest(ranks=ranks, home=home, budget=budget, iteration=iteration):
                            counts = torch.tensor([[rng.randrange(40) if rng.randrange(3) else 0
                                                    for _ in range(config.num_experts)] for _ in range(ranks)])
                            if iteration % 5 == 0:
                                counts[:, home:] = 0
                            loads = counts.sum(0).reshape(ranks, home).sum(1)
                            target = (None, int(loads.max()), int(loads.sum() // ranks))[iteration % 3]
                            minimum = (0, 10, 40, 1000)[iteration % 4]
                            capacity = None if iteration % 2 else int(loads.max())
                            options = {"target_load": target, "minimum_replica_rows": minimum,
                                       "capacity_limit": capacity,
                                       "minimum_load_reduction": (0.0, 0.35, 0.4, 0.5001, 1.0)[iteration % 5],
                                       "minimum_saved_rows": (0, 20, 1000)[iteration % 3]}
                            expected = build_expert_replica_plan(counts.tolist(), budget, **options)
                            with patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar read")), \
                                 patch.object(torch.Tensor, "tolist", side_effect=AssertionError("list read")), \
                                 patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("scalar branch")):
                                actual = build_device_expert_replica_plan(counts, config, **options)
                            summary = actual.host_summary()
                            self.assertEqual(actual.control.numel(), 1 + 2 * config.physical_experts + ranks * ranks)
                            logical = actual.slot_to_logical.flatten()
                            expected_order = torch.where(logical >= 0, logical, config.num_experts).argsort(stable=True)
                            torch.testing.assert_close(actual.source_order, expected_order)
                            torch.testing.assert_close(actual.source_counts,
                                                       actual.dispatch_counts[:, expected_order].to(torch.int64))
                            self.assertEqual(summary.slot_to_logical, expected.slot_to_logical)
                            self.assertEqual(summary.destination_counts, expected.destination_counts)
                            self.assertEqual(summary.transfers, expected.transfers)
                            torch.testing.assert_close(actual.dispatch_counts,
                                                       torch.tensor(expected.dispatch_counts, dtype=torch.int32))
                            self.assertEqual(summary.rank_splits, tuple(tuple(row) for row in
                                             actual.dispatch_counts.reshape(ranks, ranks, -1).sum(2).tolist()))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_retained_outputs_and_source_order(self):
        """
        description: Later plans and reverse consumption cannot overwrite earlier quotas.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        config = ExpertReplicaConfig(24, 4, 1)
        counts = torch.zeros(4, 24, dtype=torch.int64)
        counts[:, :6] = 20
        retained = []
        for shift in (0, 6, 12, 18, 0):
            source = counts.roll(shift, 1)
            plan = build_device_expert_replica_plan(source, config, minimum_replica_rows=40)
            retained.append((source, plan))
        self.assertEqual(len({plan.control.data_ptr() for _, plan in retained}), len(retained))
        for source, plan in reversed(retained):
            for rank in range(4):
                slots, rows = plan.source_runs(rank)
                ids = torch.repeat_interleave(slots, rows)
                torch.testing.assert_close(plan.slot_to_logical.flatten()[ids],
                                           torch.repeat_interleave(torch.arange(24), source[rank]))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_invalid_values_are_reported_by_control(self):
        """
        description: Negative, overflowing and infeasible counts never yield a usable summary.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        config = ExpertReplicaConfig(4, 2, 1)
        for counts, limit in (([[1, -1, 0, 0]] * 2, 100), ([[10, 0, 0, 0]] * 2, 1),
                              ([[torch.iinfo(torch.int64).max, 0, 0, 0]] * 2, None),
                              ([[2**32, 0, 0, 0]] * 2, None)):
            result = build_device_expert_replica_plan(torch.tensor(counts), config, capacity_limit=limit)
            with self.assertRaisesRegex(ValueError, "Device replica plan"):
                result.host_summary()

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_boundary_validation_and_cost_policy(self):
        """
        description: Configuration errors fail before any native kernel is launched.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        config = ExpertReplicaConfig(4, 2, 1)
        for counts in (torch.ones(2, 4), torch.ones(1, 4, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                build_device_expert_replica_plan(counts, config)
        for options in ({"target_load": -1}, {"minimum_replica_rows": True}, {"capacity_limit": 2**63},
                        {"minimum_load_reduction": float("nan")}, {"minimum_load_reduction": True}):
            with self.assertRaises(ValueError):
                build_device_expert_replica_plan(torch.ones(2, 4, dtype=torch.int64), config, **options)
        validate_planner_backend("device", 4096, None)
        with self.assertRaisesRegex(ValueError, "host cost model"):
            validate_planner_backend("device", 0, object())

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_integer_admission_boundary_and_large_counts(self):
        """
        description: CPU and device include the threshold and reject its first larger basis point.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        config = ExpertReplicaConfig(4, 4, 1)
        for rows in (100, 2**28):
            counts = torch.tensor([[rows, rows, 0, 0]] * 4, dtype=torch.int64)
            for threshold, transfers in ((0.5, True), (0.5001, False), (1.0, False)):
                with self.subTest(rows=rows, threshold=threshold):
                    options = {"capacity_limit": rows * 4, "minimum_load_reduction": threshold}
                    expected = build_expert_replica_plan(counts.tolist(), 1, **options)
                    actual = build_device_expert_replica_plan(counts, config, **options).host_summary()
                    self.assertEqual(bool(expected.transfers), transfers)
                    self.assertEqual(actual.transfers, expected.transfers)
                    self.assertEqual(actual.destination_counts, expected.destination_counts)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_admission_cannot_remove_capacity_required_copy(self):
        """
        description: Strict admission preserves required copies and the distinct-TopK receive bound.
        expectation: Admission, integer conservation and ordinary compute contracts hold.
        feature: Expert hot replica admission.
        """
        config = ExpertReplicaConfig(24, 4, 1)
        counts = torch.tensor([[512] * 8 + [0] * 16] * 4)
        limit = config.maximum_receive_rows(512, 8, alignment=1)
        options = {"target_load": limit, "capacity_limit": limit, "minimum_load_reduction": 1.0,
                   "minimum_saved_rows": 2**60}
        expected = build_expert_replica_plan(counts.tolist(), 1, **options)
        actual = build_device_expert_replica_plan(counts, config, **options).host_summary()
        self.assertTrue(actual.transfers)
        self.assertLessEqual(max(actual.destination_loads), limit)
        self.assertEqual(actual.transfers, expected.transfers)
        self.assertEqual(actual.destination_counts, expected.destination_counts)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_absolute_saved_rows_rejects_small_optional_reduction(self):
        """
        description: Fractional and absolute gains must both meet their boundaries.
        expectation: Equal saved rows pass, one row above fails, required copies remain.
        feature: Expert hot replica admission.
        """
        config = ExpertReplicaConfig(4, 4, 1)
        counts = torch.tensor([[100, 100, 0, 0]] * 4)
        for saved_rows, active in ((200, True), (201, False)):
            options = {"capacity_limit": 400, "minimum_load_reduction": 0.4, "minimum_saved_rows": saved_rows}
            expected = build_expert_replica_plan(counts.tolist(), 1, **options)
            actual = build_device_expert_replica_plan(counts, config, **options).host_summary()
            self.assertEqual(bool(expected.transfers), active)
            self.assertEqual(actual.transfers, expected.transfers)
            self.assertEqual(actual.destination_counts, expected.destination_counts)
        for saved_rows in (-1, True, 2**64):
            with self.subTest(saved_rows=saved_rows), self.assertRaises(ValueError):
                build_device_expert_replica_plan(counts, config, minimum_saved_rows=saved_rows)
