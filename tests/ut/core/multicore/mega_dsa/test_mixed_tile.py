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
"""Mixed SFA schedule admission and independent member-evidence rejection."""

import unittest

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule,
    validate_mixed_trace,
)


class TestMixedSfaSchedule(unittest.TestCase):
    """CPU contracts cannot substitute for executing the actual collaborative kernel."""

    def test_reserved_group_and_round_bounds(self):
        """Always retain progress capacity and reject bools and lossy scalar conversions."""
        for value in (0, 20, -1, True, 2.0, "2"):
            with self.subTest(groups=value), self.assertRaises(ValueError):
                MixedSfaSchedule(value)
        for value in (0, 65, False, 1.0):
            with self.subTest(rounds=value), self.assertRaises(ValueError):
                MixedSfaSchedule(1, value)

    def test_versioned_config_and_independent_trace(self):
        """Prepared buffers have stable integer ABI and do not alias across invocations."""
        schedule = MixedSfaSchedule(7, 3)
        self.assertEqual(schedule.runtime_config("cpu").tolist(), [0x48504453414D4958, 1, 7, 3])
        first, second = schedule.new_trace("cpu"), schedule.new_trace("cpu")
        self.assertEqual(first.shape, (20, 64))
        self.assertEqual(first.dtype, torch.int64)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(first.stride(0) * first.element_size(), 512)

    @staticmethod
    def _complete_trace():
        # Independently enumerated seven-group traversal over 20 logical partitions.
        records = ((9, 72, 14), (9, 81, 15), (9, 90, 16),
                   (9, 99, 17), (9, 108, 18), (9, 117, 19), (6, 63, 13))
        trace = torch.zeros(20, 64, dtype=torch.int64)
        for group, record in enumerate(records):
            trace[group, 0] = record[-1]
            for offset in (0, 16, 32):
                trace[group, offset + 1:offset + 4] = torch.tensor(record)
        return trace

    def test_complete_three_member_evidence(self):
        """All members independently account for the repeated full logical grid."""
        report = validate_mixed_trace(self._complete_trace(), MixedSfaSchedule(7, 3))
        self.assertEqual(report, {"logical_tasks": 60, "compute_groups": 7, "members_per_group": 3})

    def test_missing_member_and_stale_ticket_rejected(self):
        """A cube-only success record, wrong checksum and stale dispatch all fail admission."""
        for group, word in ((0, 17), (6, 34), (2, 0)):
            trace = self._complete_trace()
            trace[group, word] -= 1
            with self.subTest(group=group, word=word), self.assertRaises(ValueError):
                validate_mixed_trace(trace, MixedSfaSchedule(7, 3))

    def test_reserved_compute_and_invalid_snapshot_rejected(self):
        """Reserved groups may not run SFA; the validator requires an explicit typed snapshot."""
        trace = self._complete_trace()
        trace[19, 1] = 1
        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_mixed_trace(trace, MixedSfaSchedule(7, 3))
        for invalid in (trace.int(), trace[:19], torch.empty(20, 64, dtype=torch.int64, device="meta")):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                validate_mixed_trace(invalid, MixedSfaSchedule(7, 3))
