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
"""HF-native complete LongCat Flash construction and optional optimized replacements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn  # pylint: disable=forbidden-backend-import

from transformers import LongcatFlashConfig, LongcatFlashForCausalLM
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashExperts, LongcatFlashMLA

from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements, module_replacement,
)


def build_longcat_flash(config: LongcatFlashConfig | dict[str, Any], *,
                       expert_backend: str | None = None, optimized_attention: bool = False,
                       dispatch_mode: str = "push") -> LongcatFlashForCausalLM:
    """Build the entire source architecture, then apply explicitly selected replacements.

    Args:
        config: Complete HF configuration or serialized configuration mapping.
        expert_backend: None retains HF experts; native selects portable EP math;
            megamoe selects the BF16 NPU executor.
        optimized_attention: Select generic fused-latent MLA with LongCat semantics.
        dispatch_mode: MegaMoE transport used when expert_backend is megamoe.

    Returns:
        The complete CausalLM, including both attention/dense sublayers per block.
    """
    if isinstance(config, dict):
        config = LongcatFlashConfig.from_dict(config)
    config._attn_implementation = "eager"
    model = LongcatFlashForCausalLM(config)
    rules = []
    if expert_backend is not None:
        @module_replacement
        def replace_experts(*, module: nn.Module, module_fqn: str,
                            context: Mapping[str, Any]) -> nn.Module:
            """Construct the selected backend using the replacement executor's context."""
            return LongcatFlashMegaMoeExperts(module=module, module_fqn=module_fqn, context=context,
                                             backend=expert_backend, dispatch_mode=dispatch_mode)

        rules.append(ModuleReplacementSpec(match=("model.layers.*.mlp.experts",), factory=replace_experts,
                                           module_type=LongcatFlashExperts))
    if optimized_attention:
        from hyper_parallel.models.longcat_flash.adapter.attention import (  # pylint: disable=C0415
            LongcatFlashMLAAttention,
        )

        rules.append(ModuleReplacementSpec(match=("model.layers.*.self_attn.[01]",), factory=LongcatFlashMLAAttention,
                                           module_type=LongcatFlashMLA))
    if rules:
        conversions = []
        apply_module_replacements(model, compile_module_replacements(model, rules), weights_mapping=conversions)
        model._weight_conversions = conversions  # pylint: disable=protected-access
    return model
