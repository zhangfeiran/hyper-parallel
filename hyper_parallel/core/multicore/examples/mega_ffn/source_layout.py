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
"""原始 NK 数值契约的完整模型替换与训练状态映射。"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from hyper_parallel.components.modules import SwiGLUMLP
from hyper_parallel.core.multicore.examples.mega_ffn.qwen_dense_model import DenseMLP, QwenDenseConfig, QwenDenseModel
from hyper_parallel.core.multicore.frontend.examples.common_ffn import common_ffn
from hyper_parallel.core.multicore.frontend.examples.linear_ffn import linear_ffn
from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.core.multicore.modules.mega_ffn.module import MegaFFN
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)


def build_source_model(config: QwenDenseConfig, baseline: str, *, recompute: bool = False,
                       execution: DenseExecutionConfig | None = None) -> QwenDenseModel:
    """在普通模型迁移设备、分片和创建优化器前替换所有稠密 FFN。

    Args:
        config: 原始完整 Qwen 稠密模型的结构与激活配置。
        baseline: common 使用分步 BF16 激活，packed 使用一次 BF16 舍入的打包激活。
        recompute: 是否通过标准 Torch 接口对 FFN 进行非重入重计算。
        execution: 候选的稠密执行配置，传入现有模块替换上下文。

    Returns:
        参数独立、保留原始 NK 布局及检查点映射的完整模型。
    """
    programs = {"common": common_ffn, "packed": linear_ffn}
    if baseline not in programs:
        raise ValueError("Source-NK model requires the original common or packed numerical contract")
    model = QwenDenseModel(config, "common", recompute=recompute)
    rules = [ModuleReplacementSpec(("layers.*.mlp",), MegaFFNAdapter, DenseMLP)]
    apply_module_replacements(model, compile_module_replacements(model, rules),
                              weights_mapping=model.weights_mapping,
                              context={"mega_ffn_program": programs[baseline], "mega_ffn_execution": execution})
    model.backend = "mega_ffn"
    # 随机初始化参数已由本次替换完成转换，逆向保存复用实际使用的检查点映射。
    model._weight_conversions = list(model.weights_mapping)  # pylint: disable=protected-access
    return model


def _packed_projections(packed, down, intermediate):
    if packed.ndim == 0:
        gate, up = packed, packed
    else:
        gate, up = packed.split(intermediate, dim=0)
    return {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down}


def source_layout_tensors(model: nn.Module, values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """将完整参数、梯度、主参数或优化器状态映射到三个原始 NK 投影。

    Args:
        model: 原始 common/packed 模块、完整模型或显式 Source-NK 候选。
        values: 覆盖全部命名参数的张量映射；每个值必须是原参数形状或标量计数。

    Returns:
        以原始三个投影名称为键的视图映射，不复制矩阵。

    Raises:
        ValueError: 状态键或形状不完整，或候选仍使用旧 KN 参数布局。
    """
    parameters = dict(model.named_parameters())
    if set(values) != set(parameters):
        raise ValueError("Source-NK state must cover every named model parameter exactly")
    for name, parameter in parameters.items():
        value = values[name]
        if (not isinstance(value, torch.Tensor)
                or value.ndim != 0 and tuple(value.shape) != tuple(parameter.shape)):
            raise ValueError(f"Source-NK state has an invalid tensor shape: {name}")
    result = dict(values)
    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        if isinstance(module, SwiGLUMLP):
            keys = (prefix + "linear_fc1.weight", prefix + "linear_fc2.weight")
            projections = _packed_projections(*(result.pop(key) for key in keys), module.intermediate_size)
        elif isinstance(module, MegaFFN):
            if module.weight_layout != "nk":
                raise ValueError("Source-NK state mapping requires original NK candidate parameters")
            weights = {role: result.pop(prefix + role) for role in module.weight_names}
            if module.weight_names == ("gate_up", "down"):
                projections = _packed_projections(weights["gate_up"], weights["down"], module.intermediate_size)
            else:
                projections = {role + "_proj.weight": weights[role] for role in module.weight_names}
        else:
            continue
        result.update({prefix + role: value for role, value in projections.items()})
    return result
