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
"""CPU trace measurements with explicit remote ownership and packed masking."""

import json
import unittest
from dataclasses import replace

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.trace import profile_index_trace


class TestIndexTrace(unittest.TestCase):
    """Test trace accounting against small manually counted access patterns."""

    def test_remote_histograms_union_reuse_and_storage_runs(self):
        """Owner runs follow owner-local addresses, including reordered storage."""
        meta = DsaBatchMeta((0, 6), (2, 3, 5), (2, 3, 5), (1, 1, 0, 0, 1, 0),
                            (1, 0, 2, 0, 2, 1), cp_ranks=(7, 2), root_pes=(3, 0))
        indices = torch.tensor([[0, 2, -1], [0, 1, 3], [1, 3, 5]], dtype=torch.int32)
        report = profile_index_trace(indices, meta, kv_bytes_per_token=1152, tile_sizes=(2,))
        summary = report["profiles"][0]
        self.assertEqual(summary["selected_references"], 8)
        self.assertEqual(summary["tile_distinct_reads"], 7)
        self.assertEqual(summary["invocation_distinct_keys"], 5)
        self.assertEqual(summary["inter_tile_repeated_reads"], 2)
        first, second = summary["tiles"]
        self.assertEqual(first["owner_reference_histogram"], [2, 3])
        self.assertEqual(first["owner_distinct_histogram"], [2, 2])
        self.assertEqual(first["owner_storage_run_lengths"], [1, 1, 2])
        self.assertEqual(first["remote_fraction"], 0.5)
        self.assertEqual(first["remote_payload_bytes"], 2304)
        self.assertEqual(second["owner_storage_run_lengths"], [2, 1])
        self.assertEqual(second["previous_tile_reused_keys"], 2)
        self.assertEqual(second["previous_tile_reuse_fraction"], 2 / 3)
        self.assertEqual(report["cp_ranks"], [7, 2])
        self.assertEqual(report["padding_slots"], 1)
        json.dumps(report)

    def test_tile_deduplication_and_reordered_queries(self):
        """Tile unions follow Q storage order rather than sorted global IDs."""
        meta = replace(DsaBatchMeta.packed((4,)), q_global_ids=(3, 1, 2, 0))
        indices = torch.tensor([[0, 2], [0, 1], [0, 2], [0, -1]], dtype=torch.int32)
        profiles = profile_index_trace(indices, meta, kv_bytes_per_token=2, tile_sizes=(1, 2, 4))["profiles"]
        self.assertEqual([item["tile_distinct_reads"] for item in profiles], [7, 5, 3])
        self.assertEqual([item["intra_tile_repeated_reads"] for item in profiles], [0, 2, 4])
        self.assertEqual(profiles[1]["tiles"][1]["previous_tile_reused_keys"], 2)
        self.assertEqual(profiles[2]["deduplicated_payload_bytes"], 6)

    def test_packed_future_and_cross_sequence_slots_are_masked(self):
        """Illegal known IDs do not inflate reads; empty tiles have zero fractions."""
        meta = DsaBatchMeta.packed((2, 2))
        indices = torch.tensor([[1, -1], [0, 1], [0, 3], [1, -1]], dtype=torch.int32)
        report = profile_index_trace(indices, meta, kv_bytes_per_token=8, tile_sizes=(2,))
        self.assertEqual(report["masked_slots"], 4)
        self.assertEqual(report["empty_queries"], 3)
        empty = report["profiles"][0]["tiles"][1]
        self.assertEqual(empty["distinct_selected_keys"], 0)
        self.assertEqual(empty["remote_fraction"], 0)
        self.assertEqual(empty["previous_tile_reuse_fraction"], 0)
        self.assertEqual(empty["owner_storage_run_lengths"], [])

    def test_unknown_and_duplicate_ids_fail(self):
        """Invalid snapshots must not be silently interpreted as sparse padding."""
        meta = DsaBatchMeta.packed((2,))
        for values in ([[0, 0], [0, 1]], [[0, -2], [0, 1]], [[0, 2], [0, 1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                profile_index_trace(torch.tensor(values, dtype=torch.int32), meta, kv_bytes_per_token=1)

    def test_snapshot_and_configuration_validation(self):
        """Only explicit host int32 snapshots and positive declared sizes are accepted."""
        meta = DsaBatchMeta.packed((2,))
        for tensor in (torch.zeros(2, 1), torch.zeros(1, 2, dtype=torch.int32),
                       torch.zeros(2, 0, dtype=torch.int32), torch.zeros(2, 1, 1, dtype=torch.int32),
                       torch.empty(2, 1, dtype=torch.int32, device="meta")):
            with self.subTest(shape=tensor.shape, device=tensor.device), self.assertRaises(ValueError):
                profile_index_trace(tensor, meta, kv_bytes_per_token=1)
        indices = torch.zeros(2, 1, dtype=torch.int32)
        for size in (0, -1, True, 1.5):
            with self.subTest(bytes=size), self.assertRaises(ValueError):
                profile_index_trace(indices, meta, kv_bytes_per_token=size)
        for tiles in ((), (0,), (True,), (1, 1), (1.5,), ([],), [1]):
            with self.subTest(tiles=tiles), self.assertRaises(ValueError):
                profile_index_trace(indices, meta, kv_bytes_per_token=1, tile_sizes=tiles)

    def test_empty_local_query_shard_is_serializable(self):
        """A rank without queries contributes no artificial reads or payload."""
        meta = replace(DsaBatchMeta.packed((2,)), q_global_ids=())
        report = profile_index_trace(torch.empty(0, 2048, dtype=torch.int32), meta, kv_bytes_per_token=1152)
        for profile in report["profiles"]:
            self.assertEqual(profile["selected_references"], 0)
            self.assertEqual(profile["tiles"], [])
        self.assertEqual(report["query_count"], 0)
        json.dumps(report)
