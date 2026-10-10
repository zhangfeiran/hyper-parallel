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
"""CPU FP64 projection VJPs at explicitly supplied Native activation and output values."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.reference import selected_kl_reference, sparse_attention_reference
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import gradient_error_metrics


@dataclass(frozen=True)
class ProjectionGeometry:
    """Dimensions of a bounded projection fixture, independent of Native admission limits."""

    heads: int = 32
    compressed: int = 512
    rope: int = 64
    index_heads: int = 64
    index_dim: int = 128
    hidden: int = 4
    output: int = 4

    def __post_init__(self) -> None:
        for dimension in (self.heads, self.compressed, self.rope, self.index_heads, self.index_dim,
                          self.hidden, self.output):
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
                raise ValueError("projection geometry dimensions must be positive integers")

    @property
    def widths(self) -> tuple[int, ...]:
        """Return the seven input projection widths."""
        return (self.heads*self.compressed, self.compressed, self.heads*self.rope, self.rope,
                self.index_heads*self.index_dim, self.index_dim, self.index_heads)

    @property
    def tails(self) -> tuple[tuple[int, ...], ...]:
        """Return activation shapes excluding the query or key row dimension."""
        return ((self.heads, self.compressed), (self.compressed,), (self.heads, self.rope), (self.rope,),
                (self.index_heads, self.index_dim), (self.index_dim,), (self.index_heads,))


def projection_fixture(tokens: int, geometry: ProjectionGeometry, *, seed: int = 20261012) -> tuple:
    """Prepare reproducible CPU FP32 hidden values, eight parameter matrices and output targets."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(tokens, geometry.hidden, generator=generator)*.25
    weights = tuple(torch.randn(width, geometry.hidden, generator=generator)*(.3 if field < 4 else .4)
                    for field, width in enumerate(geometry.widths))
    output_weight = torch.randn(geometry.output, geometry.heads*geometry.compressed, generator=generator)
    output_weight = output_weight/(geometry.heads*geometry.compressed)**.5
    target = torch.randn(tokens, geometry.output, generator=generator)*.1
    return hidden, (*weights, output_weight), target


def projection_vjp_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Require both pointwise and normalized error for nonzero projection VJPs.

    A small nonzero oracle cannot pass solely because it falls below the absolute
    pointwise tolerance. Zero references use the separate absolute rule. These
    fixed limits are declared before Native candidates are observed.
    """
    left, right = actual.detach().cpu().double(), expected.detach().cpu().double()
    metric = gradient_error_metrics(left, right)
    pointwise = bool(torch.allclose(left, right, rtol=.02, atol=2e-5))
    normalized = metric["finite"] and (metric["relative_l2"] <= .02 if metric["expected_norm"]
                                        else metric["max_abs"] <= 2e-5)
    return {**metric, "pointwise_pass": pointwise,
            "passed": metric["finite"] and pointwise and normalized,
            "rtol": .02, "atol": 2e-5, "nonzero_relative_l2_limit": .02}


def project_local_states(hidden: torch.Tensor, weights: tuple[torch.Tensor, ...], geometry: ProjectionGeometry,
                         query_rows: torch.Tensor, key_rows: torch.Tensor,
                         weight_dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Project actual query/owner-key rows; detach hidden before all three index projections."""
    if len(weights) != 8:
        raise ValueError("projection fixture requires seven input matrices and one output matrix")
    states = []
    for field, (weight, tail) in enumerate(zip(weights[:7], geometry.tails)):
        source = hidden.detach() if field >= 4 else hidden
        rows = key_rows if field in (1, 3, 5) else query_rows
        projected = (source.index_select(0, rows) @ weight.T).reshape(len(rows), *tail)
        dtype = weight_dtype if field == 6 else torch.bfloat16
        states.append(projected.to(dtype))
    return tuple(states)


def projection_vjp_reference(hidden: torch.Tensor, weights: tuple[torch.Tensor, ...], targets: torch.Tensor,
                             states: tuple[torch.Tensor, ...], output: torch.Tensor, indices: torch.Tensor,
                             meta: DsaBatchMeta, geometry: ProjectionGeometry, *, mode: str,
                             coefficient: float, auxiliary: float, attention_scale: float) -> tuple:
    """Evaluate one global mean objective and every FP64 projection VJP at fixed Native values.

    Supplied activations and compressed output set the forward values. Their
    Jacobians come from independent CPU linear projections and selected SFA/KL.
    This contract checks Native/autograd VJP composition and detached projection
    inputs; it does not measure BF16 projection or output composition error.
    """
    if mode not in ("lm_only", "kl_only", "lm_plus_kl"):
        raise ValueError("projection oracle requires an explicit supported objective mode")
    if (len(states) != 7 or len(weights) != 8
            or meta.q_global_ids != tuple(range(meta.global_valid_queries))
            or meta.kv_global_ids != tuple(range(meta.global_valid_queries))):
        raise ValueError("projection reference requires seven complete global-packed states and eight matrices")
    if not math.isfinite(coefficient) or coefficient < 0 or not math.isfinite(auxiliary):
        raise ValueError("projection reference requires a nonnegative finite coefficient and finite upstream")
    leaf_hidden = hidden.detach().cpu().double().requires_grad_()
    leaves = tuple(weight.detach().cpu().double().requires_grad_() for weight in weights)
    fixed = []
    for field, (weight, tail, value) in enumerate(zip(leaves[:7], geometry.tails, states)):
        source = leaf_hidden.detach() if field >= 4 else leaf_hidden
        projected = (source @ weight.T).reshape(meta.global_valid_queries, *tail)
        fixed.append(value.detach().cpu().double()+(projected-projected.detach()))
    normalization = DsaLossNormalization(meta.global_valid_queries)
    selected = indices.detach().cpu()
    query, key, query_rope, key_rope = fixed[0], fixed[1], fixed[2], fixed[3]
    iq, ik, merge = fixed[4], fixed[5], fixed[6]
    main, _ = sparse_attention_reference(query, key, query_rope, key_rope, selected, meta,
                                          attention_scale=attention_scale)
    observed = output.detach().cpu().double()+(main-main.detach())
    predicted = observed.flatten(1) @ leaves[7].T
    main_loss = (predicted-targets.detach().cpu().double()).square().sum()*normalization.local_sum_scale
    auxiliary_loss = leaf_hidden.new_zeros(())
    if mode != "lm_only":
        auxiliary_loss = selected_kl_reference(iq, ik, merge, query, key, query_rope, key_rope, selected, meta,
                                               attention_scale=attention_scale, normalization=normalization,
                                               loss_coeff=coefficient)
    objective = auxiliary_loss*auxiliary if mode == "kl_only" else main_loss
    if mode == "lm_plus_kl":
        objective = objective+auxiliary_loss*auxiliary
    gradients = torch.autograd.grad(objective, (leaf_hidden, *leaves), allow_unused=True)
    return objective.detach(), auxiliary_loss.detach(), gradients
