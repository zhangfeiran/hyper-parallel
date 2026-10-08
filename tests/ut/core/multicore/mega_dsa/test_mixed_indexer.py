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
"""Two-phase LI lifetime, completion ordering and failure propagation contracts."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import mixed_indexer
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


class TestMixedIndexerProbe(unittest.TestCase):
    """Mock native execution while keeping the real buffer ownership boundary."""

    def test_repeat_requires_complete_pair(self):
        """Reject per-phase repeats before loading native code or allocating scratch."""
        with patch.object(mixed_indexer, "_load_native") as loader:
            with self.assertRaisesRegex(ValueError, "complete main/merge"):
                mixed_indexer.mixed_indexer_forward_probe(None, None, None, None, MixedSfaSchedule(1, 3))
            loader.assert_not_called()

    def test_merge_sees_main_partials_and_reuse_preserves_owner(self):
        """The merge receives the exact main buffers and live retained storage."""
        query = torch.empty(1, 64, 128, dtype=torch.bfloat16)
        key, weights = torch.empty(1, 1, 128), torch.empty(1, 64)
        lengths = torch.tensor([1], dtype=torch.int32)
        retained = torch.empty(mixed_indexer.MIXED_INDEXER_WORKSPACE_BYTES, dtype=torch.uint8)
        calls = []

        def _execute(_query, _key, _weights, _actual_q, _actual_k, _config,
                     trace, scratch, phase, indices, values):
            self.assertIs(scratch, retained)
            if phase == 0:
                self.assertEqual(indices[0, 0, 0].item(), -2)
                indices[0, 0, 0] = 123
                scratch[0] = 37
            else:
                self.assertEqual(indices[0, 0, 0].item(), 123)
                self.assertEqual(scratch[0].item(), 37)
                indices[0, 0, 0] = 0
            calls.append((phase, trace, indices, values))

        with patch.object(mixed_indexer, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_mixed_indexer_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_mixed_indexer_out", side_effect=_execute, create=True):
            first = mixed_indexer.mixed_indexer_forward_probe(query, key, weights, lengths,
                                                            MixedSfaSchedule(7), retained)
            second = mixed_indexer.mixed_indexer_forward_probe(query, key, weights, lengths,
                                                             MixedSfaSchedule(1), retained)
        self.assertEqual([call[0] for call in calls], [0, 1, 0, 1])
        self.assertIs(calls[0][2], calls[1][2])
        self.assertIs(calls[0][3], calls[1][3])
        self.assertIsNot(calls[0][1], calls[1][1])
        self.assertNotEqual(first[0].data_ptr(), second[0].data_ptr())
        self.assertIs(first[3], retained)
        self.assertEqual(first[0][0, 0, 0].item(), 0)

    def test_failed_main_never_launches_merge(self):
        """Surface launch failures without consuming incomplete partials."""
        query = torch.empty(1, 64, 128, dtype=torch.bfloat16)
        with patch.object(mixed_indexer, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_mixed_indexer_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_mixed_indexer_out",
                             side_effect=RuntimeError("main failed"), create=True) as launch:
            with self.assertRaisesRegex(RuntimeError, "main failed"):
                mixed_indexer.mixed_indexer_forward_probe(query, None, None, None, MixedSfaSchedule(1))
            self.assertEqual(launch.call_count, 1)

    def test_ld_evidence_requires_every_member_and_both_phases(self):
        """Detect a skipped merge and disagreeing LD ownership independently of task counts."""
        main = torch.zeros(20, 64, dtype=torch.int64)
        main[0, 0] = 19
        for offset in (0, 16, 32):
            main[0, offset + 1:offset + 5] = torch.tensor([20, 210, 19, 2])
        schedule = MixedSfaSchedule(1)
        result = mixed_indexer.validate_mixed_indexer_traces((main, main.clone()), schedule, require_ld=True)
        self.assertEqual(result["ld_partitions"], 2)
        for word, value in ((20, 1), (36, 0), (4, 21)):
            merge = main.clone()
            merge[0, word] = value
            with self.subTest(word=word), self.assertRaises(ValueError):
                mixed_indexer.validate_mixed_indexer_traces((main, merge), schedule, require_ld=True)
        missing = main.clone()
        missing[0, [4, 20, 36]] = 0
        with self.assertRaisesRegex(ValueError, "did not execute"):
            mixed_indexer.validate_mixed_indexer_traces((missing, missing.clone()), schedule, require_ld=True)


class TestFusedIndexerProbe(unittest.TestCase):
    """Check ABI admission, owned phase records and independent progress evidence."""

    @staticmethod
    def _traces(schedule):
        traces = []
        for epoch in (1, 2):
            trace = torch.zeros(20, 64, dtype=torch.int64)
            for group in range(schedule.compute_groups):
                tasks = tuple(range(group, 20, schedule.compute_groups))
                trace[group, 0] = tasks[-1]
                for offset in (0, 16, 32):
                    trace[group, offset + 1:offset + 5] = torch.tensor(
                        [len(tasks), sum(task + 1 for task in tasks), tasks[-1], 1])
                    trace[group, offset + 6] = epoch
            trace[schedule.compute_groups, [6, 21, 22, 24, 38]] = epoch
            trace[schedule.compute_groups, 25] = 3 * schedule.compute_groups
            traces.append(trace)
        return tuple(traces)

    def test_arrival_and_reserved_release_are_required(self):
        """Reject incomplete arrivals, false release counts and additional idle-group work."""
        schedule = MixedSfaSchedule(7)
        traces = self._traces(schedule)
        original = tuple(trace.clone() for trace in traces)
        evidence = mixed_indexer.validate_fused_indexer_traces(traces, schedule, require_ld=True)
        self.assertEqual(evidence["progress_group"], 7)
        self.assertEqual(evidence["arrivals_per_phase"], 21)
        for actual, expected in zip(traces, original):
            torch.testing.assert_close(actual, expected)
        for phase, row, word, value in ((0, 3, 22, 0), (1, 7, 24, 1), (0, 7, 25, 20),
                                        (1, 7, 38, 0), (1, 7, 0, 3), (0, 8, 1, 1)):
            broken = tuple(trace.clone() for trace in traces)
            broken[phase][row, word] = value
            with self.subTest(phase=phase, row=row, word=word), self.assertRaises(ValueError):
                mixed_indexer.validate_fused_indexer_traces(broken, schedule, require_ld=True)

    def test_old_payload_is_rejected_before_allocation(self):
        """Phase 2 must never reach a payload that only implements host phase closure."""
        with patch.object(mixed_indexer, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_mixed_indexer_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_mixed_indexer_out", create=True) as launch:
            with self.assertRaisesRegex(RuntimeError, "ABI 2"):
                mixed_indexer.mixed_indexer_fused_forward_probe(None, None, None, None, MixedSfaSchedule(1))
            launch.assert_not_called()

    def test_single_launch_reuse_has_fresh_phase_records(self):
        """Reuse only retained scratch; each invocation starts with independent zeroed arrivals."""
        query = torch.empty(1, 64, 128, dtype=torch.bfloat16)
        retained = torch.empty(64, dtype=torch.uint8)
        observed = []

        def _execute(*args):
            trace, scratch, phase = args[-5], args[-4], args[-3]
            indices, values = args[-2], args[-1]
            self.assertIs(scratch, retained)
            self.assertEqual(phase, 2)
            self.assertEqual(trace.shape, (2, 20, 64))
            self.assertFalse(bool(trace.any()))
            trace.fill_(17)
            indices.fill_(0)
            values.fill_(3)
            observed.append(trace)

        with patch.object(mixed_indexer, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_mixed_indexer_version", return_value=2, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_mixed_indexer_out", side_effect=_execute,
                             create=True) as op:
            first = mixed_indexer.mixed_indexer_fused_forward_probe(query, None, None, None,
                                                                  MixedSfaSchedule(1), retained)
            second = mixed_indexer.mixed_indexer_fused_forward_probe(query, None, None, None,
                                                                   MixedSfaSchedule(19), retained)
        self.assertEqual(op.call_count, 2)
        self.assertNotEqual(observed[0].data_ptr(), observed[1].data_ptr())
        self.assertNotEqual(first[0].data_ptr(), second[0].data_ptr())
        self.assertEqual(first[2][1][0, 0].item(), 17)

    def test_fused_repeats_reject_before_native(self):
        """Retained partials require a fresh complete invocation per traversal."""
        with patch.object(mixed_indexer, "_load_native") as load:
            with self.assertRaisesRegex(ValueError, "complete invocations"):
                mixed_indexer.mixed_indexer_fused_forward_probe(None, None, None, None, MixedSfaSchedule(1, 2))
            load.assert_not_called()
