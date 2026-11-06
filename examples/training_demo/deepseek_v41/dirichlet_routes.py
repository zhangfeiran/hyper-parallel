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
"""Construct repeatable exact Top-K routes for controlled load experiments.

The water-fill apportionment and striped Top-K construction follow the
megamoe-sun-261027 expert-layer benchmark. Profiles are shared by all sources;
independent source profiles would average away destination imbalance.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch


def make_route(tokens: int, experts: int, top_k: int, alpha: float | None,
               seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU int64 IDs and exact counts without changing training RNG.

    Args:
        tokens: Local input rows.
        experts: Global logical expert count.
        top_k: Distinct experts selected for each token.
        alpha: Positive Dirichlet concentration, or None for exact balance.
        seed: Private reproducible profile seed shared by source ranks.

    Returns:
        Token-major expert IDs and their expert histogram on CPU.
    """
    if tokens < 1 or not 1 <= top_k <= experts or seed < 0:
        raise ValueError('Require positive tokens, 1 <= TopK <= experts, and a nonnegative seed')
    if alpha is not None and (not math.isfinite(alpha) or alpha < 0.001):
        raise ValueError('Dirichlet alpha must be finite and at least 0.001')
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        shares = (torch.ones(experts, dtype=torch.float64) if alpha is None else
                  torch.distributions.Dirichlet(torch.full((experts,), alpha, dtype=torch.float64)).sample())
        target = torch.zeros(experts, dtype=torch.float64)
        remaining = torch.ones(experts, dtype=torch.bool)
        slots = tokens * top_k
        while slots:
            active = remaining.nonzero(as_tuple=False).flatten()
            weights = shares[active]
            if not bool(weights.sum() > 0):
                weights = torch.ones_like(weights)
            proposed = weights / weights.sum() * slots
            saturated = proposed > tokens
            if not bool(saturated.any()):
                target[active] = proposed
                break
            full = active[saturated]
            target[full] = tokens
            remaining[full] = False
            slots -= tokens * full.numel()
        counts = target.floor().long()
        residual = tokens * top_k - int(counts.sum())
        fractions = target - counts
        fractions[counts >= tokens] = -1
        counts[torch.argsort(fractions, descending=True, stable=True)[:residual]] += 1
        ids = torch.repeat_interleave(torch.arange(experts), counts).view(top_k, tokens).T.contiguous()
        sorted_ids = ids.sort(dim=-1).values
        if (int(counts.sum()) != tokens * top_k or bool((counts > tokens).any())
                or bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any())):
            raise RuntimeError('Synthetic route violated exact-count or distinct-TopK constraints')
        return ids, counts


def route_metadata(ids: torch.Tensor, counts: torch.Tensor, ep: int) -> dict[str, Any]:
    """Describe the counterfactual owner-only load for one shared source profile."""
    if counts.numel() % ep:
        raise ValueError('EP must divide the global expert count')
    destination = (counts.reshape(ep, -1).sum(dim=1) * ep).tolist()
    return {'tokens': ids.shape[0], 'top_k': ids.shape[1], 'experts': counts.numel(), 'ep': ep,
            'route_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            'source_expert_counts': counts.tolist(), 'owner_destination_rows': destination,
            'owner_skew': max(destination) / (sum(destination) / ep)}
