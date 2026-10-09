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
"""Deterministic, lossless Dirichlet routes migrated from megamoe-sun.

The sampling, water filling, integer apportionment and token-major construction
preserve the original algorithm. CPU construction keeps sampling outside device
timing and permits validating the same saved routes for every executor.
"""

from dataclasses import dataclass
import math
import random

import torch

ALPHAS = (10000, 3000, 1000, 300, 100, 30, 10, 3, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
EXTRA_PAIRS = ((0.01, 1942), (0.01, 1708), (0.01, 0), (0.005, 943), (0.01, 2104),
               (0.01, 4050), (0.01, 2486), (0.01, 1164), (0.01, 29), (0.02, 1942))
SOURCE_REVISION = "046c2861ff60c7187ef0c58f8ac60252b0c4e5a4"


@dataclass(frozen=True)
class RouteShape:
    """Token and expert dimensions independent of any execution backend."""

    local_num_tokens: int = 4096
    num_experts: int = 96
    top_k: int = 8

    def __post_init__(self) -> None:
        """Reject dimensions that cannot represent a distinct TopK route."""
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (self.local_num_tokens, self.num_experts, self.top_k)):
            raise ValueError("tokens, experts and top_k must be positive integers")
        if self.top_k > self.num_experts:
            raise ValueError("top_k must not exceed num_experts")


def dirichlet_expert_counts(shape: RouteShape, alpha: float, seed: int) -> torch.Tensor:
    """Apportion the original seeded Dirichlet profile into exact TopK slots.

    Args:
        shape: Token and expert dimensions.
        alpha: Dirichlet concentration, finite and at least 0.001.
        seed: Nonnegative CPU sampling seed.

    Returns:
        CPU int64 expert counts, each bounded by the number of local tokens.
    """
    if not math.isfinite(alpha) or alpha < 0.001:
        raise ValueError("Dirichlet alpha must be finite and at least 0.001")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    total_slots = shape.local_num_tokens * shape.top_k
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        concentration = torch.full((shape.num_experts,), alpha, dtype=torch.float64)
        shares = torch.distributions.Dirichlet(concentration).sample()
    target = torch.zeros(shape.num_experts, dtype=torch.float64)
    remaining = torch.ones(shape.num_experts, dtype=torch.bool)
    remaining_slots = total_slots
    while remaining_slots:
        active = remaining.nonzero(as_tuple=False).flatten()
        active_shares = shares[active]
        share_sum = active_shares.sum()
        if not bool(share_sum > 0):
            active_shares = torch.ones_like(active_shares)
            share_sum = active_shares.sum()
        proposed = active_shares / share_sum * remaining_slots
        saturated = proposed > shape.local_num_tokens
        if not bool(saturated.any()):
            target[active] = proposed
            break
        saturated_experts = active[saturated]
        target[saturated_experts] = shape.local_num_tokens
        remaining[saturated_experts] = False
        remaining_slots -= shape.local_num_tokens * saturated_experts.numel()
    counts = target.floor().to(torch.int64)
    residual = total_slots - int(counts.sum())
    if residual:
        fractions = target - counts
        fractions[counts >= shape.local_num_tokens] = -1.0
        order = torch.argsort(fractions, descending=True, stable=True)
        counts[order[:residual]] += 1
    if int(counts.sum()) != total_slots or bool((counts > shape.local_num_tokens).any()):
        raise RuntimeError("Dirichlet apportionment violated its slot constraints")
    return counts


def make_dirichlet_route(shape: RouteShape, alpha: float, seed: int) -> tuple[torch.Tensor, ...]:
    """Build the original token-major TopK route and descending routing scores.

    Args:
        shape: Token and expert dimensions.
        alpha: Dirichlet concentration.
        seed: CPU sampling seed.

    Returns:
        CPU int32 IDs, float32 scores and int32 expert counts.
    """
    counts = dirichlet_expert_counts(shape, alpha, seed)
    expert_ids = torch.repeat_interleave(torch.arange(shape.num_experts), counts)
    ids = expert_ids.view(shape.top_k, shape.local_num_tokens).T.contiguous()
    ordered = ids.sort(dim=-1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise RuntimeError("Dirichlet route selected one expert twice for a token")
    weights = torch.arange(shape.top_k, 0, -1, dtype=torch.float32)
    weights = (weights / weights.sum()).expand(shape.local_num_tokens, -1).clone()
    return ids.to(torch.int32), weights, counts.to(torch.int32)


def route_pairs(shuffle_seed: int = 20260903) -> list[tuple[float, int]]:
    """Return the original 55 alpha/seed pairs in their seeded measurement order."""
    points = [(float(alpha), seed) for alpha in ALPHAS for seed in (7, 42, 83)] + list(EXTRA_PAIRS)
    random.Random(shuffle_seed).shuffle(points)
    return points
