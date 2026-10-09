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
"""CPU invariants for the migrated route distribution and original sweep."""

import unittest

import torch

from tests.torch.expert_parallel.dirichlet_routes import RouteShape, make_dirichlet_route, route_pairs


class TestDirichletRoutes(unittest.TestCase):
    """Verify route legality and reproducibility independently of the executor."""

    def test_original_sweep_conserves_distinct_slots(self):
        """Every EP16/E96 point is legal, including saturated tiny-alpha routes."""
        shape = RouteShape()
        for alpha, seed in route_pairs():
            with self.subTest(alpha=alpha, seed=seed):
                ids, scores, counts = make_dirichlet_route(shape, alpha, seed)
                self.assertEqual(int(counts.sum()), 4096 * 8, "Every TopK slot must be conserved")
                self.assertLessEqual(int(counts.max()), 4096, "Each expert can occur at most once per token")
                actual = torch.bincount(ids.long().flatten(), minlength=96)
                torch.testing.assert_close(actual, counts.long(), rtol=0, atol=0)
                self.assertFalse(bool((ids.sort(dim=1).values.diff(dim=1) == 0).any()),
                                 "A token must select eight distinct logical experts")
                torch.testing.assert_close(scores.sum(dim=1), torch.ones(4096))

    def test_sampling_preserves_rng_and_route(self):
        """Generating routes neither perturbs initialization nor changes on replay."""
        before = torch.random.get_rng_state().clone()
        first = make_dirichlet_route(RouteShape(), 0.005, 943)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        second = make_dirichlet_route(RouteShape(), 0.005, 943)
        for actual, expected in zip(first, second):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_concentration_changes_realized_rank_load(self):
        """Tiny alpha produces greater rank imbalance than a concentrated sample."""
        shape = RouteShape()
        balanced = make_dirichlet_route(shape, 10000, 42)[2].reshape(16, 6).sum(1)
        skewed = make_dirichlet_route(shape, 0.005, 943)[2].reshape(16, 6).sum(1)
        self.assertGreater(float(skewed.max()), float(balanced.max()), "Alpha must control realized rank imbalance")

    def test_invalid_inputs(self):
        """Reject impossible TopK dimensions and invalid distribution parameters."""
        for alpha in (0, 0.0001, float("nan"), float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                make_dirichlet_route(RouteShape(), alpha, 42)
        with self.assertRaises(ValueError):
            RouteShape(top_k=97)
        with self.assertRaises(ValueError):
            make_dirichlet_route(RouteShape(), 1, -1)

    def test_original_sweep_size_and_order(self):
        """All four variants replay the same 55 unique distribution profiles."""
        pairs = route_pairs()
        self.assertEqual(len(pairs), 55, "The original experiment measured 55 routes")
        self.assertEqual(len(set(pairs)), 55, "A route must not be measured twice in a sweep")
        self.assertEqual(pairs, route_pairs(), "Fresh processes must retain identical measurement order")
