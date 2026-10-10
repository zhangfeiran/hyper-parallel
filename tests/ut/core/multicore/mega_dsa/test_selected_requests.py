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
"""Independent selected descriptors, bounded counts and device dependency contracts."""

from dataclasses import replace
import unittest

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.selected_requests import (
    SELECTED_REQUESTS_MAGIC, selected_request_reference, selected_request_rows,
    validate_selected_requests, validate_selected_training_traces,
)


class TestSelectedRequests(unittest.TestCase):
    """Check canonical positions, owner-local permutations and incomplete request releases."""

    @staticmethod
    def _fixture():
        meta = replace(DsaBatchMeta.packed((3, 4)), token_owners=(0, 0, 0, 1, 1, 1, 1),
                       token_local_offsets=(0, 2, 1, 0, 1, 3, 2), cp_ranks=(5, 3), root_pes=(0, 1),
                       q_global_ids=(2, 0, 1), kv_global_ids=(1, 2, 0))
        indices = torch.tensor([[0, 0, -1, 2], [0, -1, 2, 3], [2, 2, -1, 3],
                                [0, 1, -1, 99], [1, 1, 0, -1], [3, -1, -1, -1],
                                [3, 0, 9, -1]], dtype=torch.int32)[:, None]
        return meta, indices

    def test_packed_padding_duplicates_and_holes_keep_owner_offsets(self):
        """Consecutive sources across an unselected destination never merge into one run."""
        meta, indices = self._fixture()
        actual = selected_request_reference(indices, meta)
        torch.testing.assert_close(actual.requests, torch.tensor([[0, 0, 0, 1], [0, 1, 2, 1],
                                                                 [1, 0, 3, 2], [1, 2, 6, 1]]))
        torch.testing.assert_close(actual.membership, torch.tensor([3, 0, 2, 3, 2, 0, 1, 0], dtype=torch.int32))
        self.assertEqual((actual.selected_slots, actual.remote_keys, actual.remote_runs), (11, 3, 2))
        rows = selected_request_rows(meta, "cpu")
        torch.testing.assert_close(rows[:, 0], torch.tensor([0, 0, 0, 3, 3, 3, 3]))
        torch.testing.assert_close(rows[:, 1:], torch.tensor([[0, 0], [0, 2], [0, 1], [1, 0],
                                                            [1, 1], [1, 3], [1, 2]]))

    def test_empty_and_worst_case_descriptor_capacity(self):
        """Empty selected rows emit no requests; interleaved owners can consume the entire capacity."""
        meta, indices = self._fixture()
        empty = selected_request_reference(torch.full_like(indices, -1), meta)
        self.assertEqual(empty.requests.shape, (0, 4))
        self.assertEqual((empty.selected_slots, empty.remote_keys, empty.remote_runs), (0, 0, 0))
        self.assertFalse(bool(empty.membership.any()))
        owners = tuple(key % 2 for key in range(7))
        meta = replace(meta, token_owners=owners, token_local_offsets=tuple(key // 2 for key in range(7)),
                       cp_ranks=(5, 3, 7), root_pes=(0, 1, 2), cp_rank=2, q_global_ids=(), kv_global_ids=())
        indices = torch.tensor([0, 1, 2, 0, 1, 2, 3], dtype=torch.int32)[:, None, None]
        full = selected_request_reference(indices, meta)
        self.assertEqual(len(full.requests), 7)
        self.assertEqual((full.selected_slots, full.remote_keys, full.remote_runs), (7, 7, 7))
        torch.testing.assert_close(full.requests[:, 2], torch.arange(7))
        torch.testing.assert_close(full.membership, torch.tensor([1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.int32))

    def test_count_and_membership_snapshots_reject_stale_or_truncated_records(self):
        """Numerical output cannot replace actual descriptor/count and generation evidence."""
        meta, indices = self._fixture()
        oracle = selected_request_reference(indices, meta)
        requests = torch.full((7, 4), -9, dtype=torch.int64)
        requests[:4] = oracle.requests
        counts = torch.tensor([SELECTED_REQUESTS_MAGIC, 3, 4, 5, 11, 3, 2, 0, 100, 110, 0, 0, 0, 0, 0, 0])
        evidence = validate_selected_requests(indices, meta, requests, counts, oracle.membership, 3)
        self.assertEqual(evidence["duplicate_slots"], 6)
        self.assertEqual(evidence["request_capacity"], 7)
        for word, value in ((1, 2), (2, 3), (3, 4), (4, 10), (6, 1), (9, 100), (10, 1)):
            broken = counts.clone()
            broken[word] = value
            with self.subTest(word=word), self.assertRaises(ValueError):
                validate_selected_requests(indices, meta, requests, broken, oracle.membership, 3)
        for field in (0, 1):
            broken = requests.clone() if field == 0 else oracle.membership.clone()
            broken[0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_selected_requests(indices, meta, broken if field == 0 else requests,
                                           counts, broken if field == 1 else oracle.membership, 3)

    @staticmethod
    def _traces(groups):
        traces = []
        for epoch in range(1, 10):
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
            traces.append(trace)
        return tuple(traces)

    def test_selected_request_handoffs_precede_sfa_and_kl(self):
        """All three membership/request stages close before the math consumers can start."""
        traces = self._traces(7)
        schedule = MixedSfaSchedule(7)
        evidence = validate_selected_training_traces(traces, schedule, require_ld=True)
        self.assertEqual(evidence["phase_order"][2:6],
                         ["membership_init", "membership_build", "request_pack_pull", "sfa"])
        for phase, row, word, value in ((2, 7, 24, 2), (3, 4, 38, 3), (4, 7, 25, 20),
                                       (5, 4, 20, 1), (8, 8, 1, 1)):
            broken = tuple(trace.clone() for trace in traces)
            broken[phase][row, word] = value
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                validate_selected_training_traces(broken, schedule, require_ld=True)

    def test_invalid_reference_shapes_and_dtypes_do_not_admit_device_tensors(self):
        """Offline validation never performs an implicit transfer or shape reinterpretation."""
        meta, indices = self._fixture()
        for value in (indices.float(), indices[:, 0], indices[:6], indices.expand(7, 2, 4),
                      torch.empty((7, 1, 0), dtype=torch.int32)):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                selected_request_reference(value, meta)
