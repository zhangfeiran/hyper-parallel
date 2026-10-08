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
"""CPU checks for raw selection admission and explicit cutoff-tie certificates."""

import unittest
from dataclasses import replace

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_indexer_validate import (
    _analytic_expected,
    _fixture,
    _global_samples,
    _oracle_samples,
    _raw_contract,
    _score_certificate,
    _zero_tie_proof,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta


class TestIndexerValidate(unittest.TestCase):
    """Exercise selection certificates without constructing any native device context."""

    def test_raw_checks_reject_invalid_ids_duplicates_counts_and_leading_holes(self):
        """Raw validation detects errors that a global-ID export could otherwise filter."""
        meta = DsaBatchMeta.packed((4, 3))
        raw = torch.full((7, 1, 2048), -1, dtype=torch.int32)
        for query in range(7):
            count = meta.sequence_position(query)[1] + 1
            raw[query, 0, :count] = torch.arange(count, dtype=torch.int32)
        self.assertTrue(_raw_contract(raw, meta)["passed"])
        cases = []
        invalid = raw.clone()
        invalid[0, 0, 2] = -2
        cases.append((invalid, "legal_or_padding"))
        future = raw.clone()
        future[4, 0, 0] = 4
        cases.append((future, "legal_or_padding"))
        duplicate = raw.clone()
        duplicate[3, 0, 0] = 1
        cases.append((duplicate, "unique"))
        short = raw.clone()
        short[3, 0, 3] = -1
        cases.append((short, "native_cardinality"))
        hole = raw.clone()
        hole[3, 0, 0], hole[3, 0, 10] = -1, 0
        cases.append((hole, "trailing_padding"))
        for snapshot, check in cases:
            with self.subTest(check=check):
                report = _raw_contract(snapshot, meta)
                self.assertFalse(report["passed"])
                self.assertFalse(report["checks"][check])
        global_ids = _global_samples(raw, meta, (0, 4, 6))
        self.assertEqual(global_ids[1, :2].tolist(), [4, -1])
        self.assertEqual(global_ids[2, :4].tolist(), [4, 5, 6, -1])

    def test_cutoff_ties_allow_only_boundary_substitution(self):
        """Ties may change the cutoff subset but cannot omit a strictly better key."""
        meta = replace(DsaBatchMeta.packed((8,)), q_global_ids=(7,))
        scores = torch.tensor([[9.0, 8.0, 6.0, 6.0, 6.0, 2.0, 1.0, 0.0]])
        tied = torch.tensor([[0, 1, 4]], dtype=torch.int32)
        report = _score_certificate(scores, tied, meta, sparse_count=3)
        self.assertTrue(report["passed"])
        self.assertEqual(report["queries"]["7"]["strictly_better_count"], 2)
        self.assertEqual(report["queries"]["7"]["cutoff_tie_count"], 3)
        for ids in ([1, 2, 3], [0, 1, 5], [0, 1, 1], [0, 1, -1]):
            with self.subTest(ids=ids):
                self.assertFalse(_score_certificate(scores, torch.tensor([ids], dtype=torch.int32),
                                                    meta, sparse_count=3)["passed"])

    def test_long_history_signed_and_concentrated_fixtures_use_all_candidates(self):
        """Unique signed ranks reverse the winner range; the dominant partition retains all K winners."""
        meta = DsaBatchMeta.packed((2112, 2176))
        queries = (2111, 4287)
        for case in ("increasing", "signed_decreasing", "concentrated", "all_tied"):
            with self.subTest(case=case):
                values = _fixture(case, meta)
                sample_meta, scores, expected = _oracle_samples(values, meta, queries)
                self.assertTrue(_score_certificate(scores, expected, sample_meta, sparse_count=2048)["passed"])
                for row, query in enumerate(queries):
                    start = meta.global_cu_seqlens[meta.sequence_position(query)[0]]
                    begin = query - 2047 if case == "increasing" else start
                    self.assertEqual(set(expected[row].tolist()), set(range(begin, begin + 2048)))
                if case != "all_tied":
                    analytic = _global_samples(_analytic_expected(case, meta), meta, queries)
                    for actual, oracle in zip(analytic, expected):
                        self.assertEqual(set(actual.tolist()), set(oracle.tolist()))
                if case == "concentrated":
                    report = _score_certificate(scores, expected, sample_meta, sparse_count=2048)
                    self.assertEqual(report["queries"]["2111"]["hypothetical_prefix_partition_counts"], [2048, 0])
                    self.assertEqual(report["queries"]["2111"]["cutoff_gap"], 3)

    def test_packed_certificate_excludes_other_sequence_even_if_its_scores_are_higher(self):
        """Packed global IDs, rather than relative query row numbers, determine the candidate domain."""
        meta = replace(DsaBatchMeta.packed((4, 4)), q_global_ids=(5,))
        scores = torch.tensor([[100.0, 90.0, 80.0, 70.0, 2.0, 1.0, 50.0, 60.0]])
        valid = torch.tensor([[4, 5, -1]], dtype=torch.int32)
        self.assertTrue(_score_certificate(scores, valid, meta, sparse_count=3)["passed"])
        for ids in ([0, 4, -1], [4, 6, -1]):
            self.assertFalse(_score_certificate(scores, torch.tensor([ids], dtype=torch.int32),
                                                meta, sparse_count=3)["passed"])

    def test_zero_tie_proof_requires_inactive_heads_and_valid_cutoff_membership(self):
        """A strict negative dot margin certifies ReLU zero; cancellation and invalid membership do not."""
        query = -torch.ones(6, 8, 128, dtype=torch.bfloat16)
        keys = torch.ones(6, 128, dtype=torch.bfloat16)
        weights = torch.ones(6, 8, dtype=torch.bfloat16)
        first = torch.tensor([0, 1, 2], dtype=torch.int32)
        second = torch.tensor([0, 1, 3], dtype=torch.int32)
        certificate = {"passed": True, "cutoff": 0.0}
        proof = _zero_tie_proof((query, keys, weights), 5, first, second, certificate, certificate)
        self.assertTrue(proof["certified"])
        self.assertGreater(proof["minimum_negative_margin"], 0)
        query[:, 0].neg_()
        self.assertFalse(_zero_tie_proof((query, keys, weights), 5, first, second,
                                         certificate, certificate)["certified"])
        self.assertFalse(_zero_tie_proof((query, keys, weights), 5, first, second,
                                         {"passed": False, "cutoff": 0.0}, certificate)["certified"])
