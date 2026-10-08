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
"""Admission, buffer ownership and independent three-phase completion evidence."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import fused_forward
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


class TestFusedForward(unittest.TestCase):
    """Require completed LI/merge/SFA invocations with fresh control storage."""

    @staticmethod
    def _traces(groups):
        phases = []
        for epoch in (1, 2, 3):
            trace = torch.zeros(20, 64, dtype=torch.int64)
            for group in range(groups):
                tasks = tuple(range(group, 20, groups))
                trace[group, 0] = tasks[-1]
                for offset in (0, 16, 32):
                    trace[group, offset + 1:offset + 5] = torch.tensor(
                        [len(tasks), sum(task + 1 for task in tasks), tasks[-1], int(epoch < 3)])
                    trace[group, offset + 6] = epoch
            trace[groups, [6, 21, 22, 24, 38]] = epoch
            trace[groups, 25] = 3 * groups
            phases.append(trace)
        return tuple(phases)

    def test_phase_handoff_requires_all_members_and_distinct_epochs(self):
        """Independently reject early SFA, stale arrivals, missing work and idle-group writes."""
        schedule = MixedSfaSchedule(7)
        traces = self._traces(7)
        originals = tuple(trace.clone() for trace in traces)
        evidence = fused_forward.validate_fused_dsa_traces(traces, schedule, require_ld=True)
        self.assertEqual(evidence["phase_order"], ["li_main", "li_merge", "sfa"])
        self.assertEqual(evidence["arrivals_per_phase"], 21)
        for actual, expected in zip(traces, originals):
            torch.testing.assert_close(actual, expected)
        for phase, row, word, value in ((2, 7, 24, 2), (1, 7, 25, 20), (0, 3, 38, 0),
                                        (2, 4, 17, 0), (2, 4, 20, 1), (2, 8, 1, 1)):
            broken = tuple(trace.clone() for trace in traces)
            broken[phase][row, word] = value
            with self.subTest(phase=phase, row=row, word=word), self.assertRaises(ValueError):
                fused_forward.validate_fused_dsa_traces(broken, schedule, require_ld=True)
        with self.assertRaisesRegex(ValueError, "phase snapshots"):
            fused_forward.validate_fused_dsa_traces(traces[:2], schedule, require_ld=True)

    def test_three_phase_trace_reuse_and_single_launch(self):
        """Each complete invocation owns outputs and control records while scratch can be reused."""
        query = torch.empty(1, 32, 512, dtype=torch.bfloat16)
        main = (query, torch.empty(1, 512), torch.empty(1, 32, 64), torch.empty(1, 64))
        retained = torch.empty(64, dtype=torch.uint8)
        traces = []

        def _execute(*args):
            trace, scratch = args[9], args[10]
            self.assertIs(scratch, retained)
            self.assertEqual(trace.shape, (3, 20, 64))
            self.assertFalse(bool(trace.any()))
            self.assertEqual(args[3].shape, (1, 1, 512))
            self.assertEqual(args[5].shape, (1, 1, 64))
            for tensor in args[12:]:
                tensor.fill_(1)
            trace.fill_(17)
            traces.append(trace)

        with patch.object(fused_forward, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_forward_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_forward_out", side_effect=_execute,
                             create=True) as launch:
            first = fused_forward.fused_dsa_forward_probe((None, None, None), main, None, 0.1,
                                                          MixedSfaSchedule(1), retained)
            second = fused_forward.fused_dsa_forward_probe((None, None, None), main, None, 0.1,
                                                           MixedSfaSchedule(19), retained)
        self.assertEqual(launch.call_count, 2)
        self.assertNotEqual(traces[0].data_ptr(), traces[1].data_ptr())
        self.assertIs(first[-1], retained)
        self.assertEqual(first[-2][2][0, 0].item(), 17)
        for index in range(5):
            self.assertNotEqual(first[index].data_ptr(), second[index].data_ptr())

    def test_incompatible_abi_and_repeats_reject_before_allocation(self):
        """Old payloads and partial repeats must never launch or allocate mutable state."""
        with patch.object(fused_forward, "_load_native") as load:
            with self.assertRaisesRegex(ValueError, "complete invocations"):
                fused_forward.fused_dsa_forward_probe(None, None, None, 0.1, MixedSfaSchedule(1, 2))
            load.assert_not_called()
        with patch.object(fused_forward, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_forward_version", return_value=0, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_forward_out", create=True) as launch:
            with self.assertRaisesRegex(RuntimeError, "ABI 1"):
                fused_forward.fused_dsa_forward_probe(None, None, None, 0.1, MixedSfaSchedule(1))
            launch.assert_not_called()
