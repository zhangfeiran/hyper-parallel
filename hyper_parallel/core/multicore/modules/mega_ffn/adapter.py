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
"""通过现有 Trainer 与检查点接口声明稠密 FFN 替换。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.checkpoint import ConcatenateWithSections
from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter, WeightRenaming
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
    weights = tuple(projection.weight for projection in projections)
    if (len({id(weight) for weight in weights}) != 3
            or len({(weight.device, weight.dtype) for weight in weights}) != 1):
        raise ValueError("MegaFFN source weights require distinct, matching storage")


@module_replacement
class MegaFFNAdapter(MegaFFN):
    """在参数分片与优化器创建前，按语义程序转换原始三个投影。"""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """初始化无偏置 SiLU FFN，并保存检查点布局转换信息。

        Args:
            module: 包含 gate_proj、up_proj 与 down_proj Linear 子模块的源 MLP。
            module_fqn: 由现有替换执行器提供的源模块全名。
            context: 标准替换上下文，通过 mega_ffn_program 与 mega_ffn_execution 选择程序和后端。
        """
        del module_fqn
        execution = None if context is None else context.get("mega_ffn_execution")
        program = None if context is None else context.get("mega_ffn_program")
        projections = _projections(module)
        _validate_weights(projections)
        gate, up, down = projections
        config = getattr(module, "config", None)
        activation = getattr(config, "hidden_act", getattr(config, "hidden_activation", None))
        if activation is not None and activation not in ("silu", "swiglu"):
            raise ValueError("MegaFFN replacement requires a SiLU/SwiGLU source activation")
        super().__init__(gate.in_features, gate.out_features, create_parameters=False, execution=execution,
                         program=program)
        if len(self.weight_names) == 2 and gate.weight.requires_grad != up.weight.requires_grad:
            raise ValueError("Packed MegaFFN requires matching gate/up training policies")
        self.config = config
        with torch.no_grad():
            if len(self.weight_names) == 3:
                for name, projection in zip(self.weight_names, projections):
                    setattr(self, name, nn.Parameter(projection.weight.detach().clone(),
                                                    requires_grad=projection.weight.requires_grad))
                self.train(module.training)
                return
            gate_up = torch.cat((gate.weight, up.weight), dim=0)
            down_weight = down.weight.detach().clone()
            if self.weight_layout == "kn":
                gate_up, down_weight = gate_up.t().contiguous(), down_weight.t().contiguous()
            self.gate_up = nn.Parameter(gate_up,
                                        requires_grad=gate.weight.requires_grad)
            self.down = nn.Parameter(down_weight, requires_grad=down.weight.requires_grad)
        self.train(module.training)

    def make_transforms(self) -> list[WeightConverter | WeightRenaming]:
        """声明可逆的 HF 投影检查点变换；独立 NK 投影只需要改名。"""
        if len(self.weight_names) == 3:
            return [WeightRenaming(source_patterns=f"{name}_proj.weight", target_patterns=name)
                    for name in self.weight_names]
        transpose = [Transpose()] if self.weight_layout == "kn" else []
        return [
            WeightConverter(source_patterns=["gate_proj.weight", "up_proj.weight"], target_patterns="gate_up",
                            operations=[ConcatenateWithSections((self.intermediate_size, self.intermediate_size)),
                                        *transpose]),
            WeightConverter(source_patterns="down_proj.weight", target_patterns="down", operations=transpose)
            if transpose else WeightRenaming(source_patterns="down_proj.weight", target_patterns="down"),
        ]
