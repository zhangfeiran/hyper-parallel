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
"""Declarative dense SwiGLU replacement using the existing Trainer/checkpoint surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.checkpoint import ConcatenateWithSections
from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter
from hyper_parallel.core.multicore.modules.mega_ffn.module import MegaFFN
from hyper_parallel.models.replacement import module_replacement


def _projections(module):
    projections = tuple(getattr(module, name, None) for name in ("gate_proj", "up_proj", "down_proj"))
    if not all(isinstance(projection, nn.Linear) for projection in projections):
        raise TypeError("MegaFFN replacement requires gate_proj/up_proj/down_proj Linear modules")
    if any(projection.bias is not None for projection in projections):
        raise ValueError("MegaFFN replacement currently requires bias-free projections")
    gate, up, down = projections
    if (gate.in_features != up.in_features or gate.out_features != up.out_features
            or down.in_features != gate.out_features or down.out_features != gate.in_features):
        raise ValueError("MegaFFN source projection dimensions disagree")
    return projections


def _validate_weights(projections):
    gate, up, _ = projections
    weights = tuple(projection.weight for projection in projections)
    if (len({id(weight) for weight in weights}) != 3
            or len({(weight.device, weight.dtype) for weight in weights}) != 1
            or gate.weight.requires_grad != up.weight.requires_grad):
        raise ValueError("MegaFFN source weights require distinct, matching storage and gate/up training policies")


@module_replacement
class MegaFFNAdapter(MegaFFN):
    """Convert separate source projections before parameter sharding/optimizer creation."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Pack a bias-free SiLU FFN once and preserve checkpoint conversion metadata.

        Args:
            module: Source MLP with gate_proj, up_proj and down_proj Linear modules.
            module_fqn: Source FQN supplied by the existing replacement executor.
            context: Standard replacement context; mega_ffn_execution may select a dense backend.
        """
        del module_fqn
        execution = None if context is None else context.get("mega_ffn_execution")
        projections = _projections(module)
        _validate_weights(projections)
        gate, up, down = projections
        config = getattr(module, "config", None)
        activation = getattr(config, "hidden_act", getattr(config, "hidden_activation", None))
        if activation is not None and activation not in ("silu", "swiglu"):
            raise ValueError("MegaFFN replacement requires a SiLU/SwiGLU source activation")
        super().__init__(gate.in_features, gate.out_features, create_parameters=False, execution=execution)
        self.config = config
        with torch.no_grad():
            self.gate_up = nn.Parameter(torch.cat((gate.weight, up.weight), dim=0).t().contiguous(),
                                        requires_grad=gate.weight.requires_grad)
            self.down = nn.Parameter(down.weight.t().contiguous(), requires_grad=down.weight.requires_grad)
        self.train(module.training)

    def make_transforms(self) -> list[WeightConverter]:
        """Expose reversible HF projection-to-packed checkpoint layout transforms."""
        return [
            WeightConverter(source_patterns=["gate_proj.weight", "up_proj.weight"], target_patterns="gate_up",
                            operations=[ConcatenateWithSections((self.intermediate_size, self.intermediate_size)),
                                        Transpose()]),
            WeightConverter(source_patterns="down_proj.weight", target_patterns="down", operations=[Transpose()]),
        ]
