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
"""Independent HF oracles for the complete LongCat Flash integration."""

from typing import Any

# Native tensor APIs are the project backend contract.
# pylint: disable=forbidden-backend-import
import torch
from transformers import LongcatFlashConfig, LongcatFlashForCausalLM
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashTopkRouter

from hyper_parallel.models.longcat_flash.modeling_longcat_flash import build_longcat_flash
from hyper_parallel.models.validation_spec import (
    ModelValidationSpec, ModuleParityCase, ObservationSpec, ParameterProbeSpec,
)


def cropped_config(*, num_layers: int = 28) -> dict[str, Any]:
    """Reduce parameter widths while preserving released routing cardinalities."""
    return {"vocab_size": 128, "hidden_size": 32, "ffn_hidden_size": 64, "num_layers": num_layers,
                "num_attention_heads": 4, "num_key_value_heads": 4, "q_lora_rank": 16, "kv_lora_rank": 16,
                "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8, "head_dim": 8,
                "expert_ffn_hidden_size": 32, "moe_topk": 12, "n_routed_experts": 512,
                "zero_expert_num": 256, "max_position_embeddings": 128, "use_cache": False}


def build_validation_model(*, optimized: bool = False, num_layers: int = 28) -> torch.nn.Module:
    """Construct a complete HF model; optimized selects the portable candidate."""
    return build_longcat_flash(cropped_config(num_layers=num_layers),
                              expert_backend="native" if optimized else None,
                              optimized_attention=False)


def _reference(context: Any) -> torch.nn.Module:
    del context
    config = LongcatFlashConfig.from_dict(cropped_config())
    config._attn_implementation = "eager"
    return LongcatFlashForCausalLM(config).model.layers[0].mlp.experts


def _candidate(context: Any) -> torch.nn.Module:
    del context
    return build_validation_model(optimized=True)


def _inputs(context: Any) -> dict[str, Any]:
    del context
    hidden = torch.randn(4, 32, requires_grad=True)
    router = LongcatFlashTopkRouter(LongcatFlashConfig.from_dict(cropped_config()))
    routing = router(hidden)
    weights, ids = routing[-2:]
    return {"hidden_states": hidden, "top_k_index": ids,
            "top_k_weights": weights.detach().requires_grad_()}


def _copy_weights(reference: Any, candidate: Any, context: Any) -> None:
    del context
    source = reference.state_dict()
    target = candidate.state_dict()
    for name in target:
        if ".linear_qkv." in name:
            q_name = name.replace("linear_qkv", "q_a_proj")
            kv_name = name.replace("linear_qkv", "kv_a_proj_with_mqa")
            target[name] = torch.cat((source[q_name], source[kv_name]))
        else:
            target[name] = source[name]
    candidate.load_state_dict(target, strict=True)


def get_validation_spec() -> ModelValidationSpec:
    """Declare full-depth portable parity; native NPU acceptance is separate."""
    return ModelValidationSpec(
        module_cases=(ModuleParityCase(
            name="native_real_and_identity_experts", candidate_selector="model.layers.0.mlp.experts",
            candidate_builder=_candidate, reference_builder=_reference,
            input_builder=_inputs, weight_adapter=_copy_weights,
            observations=(ObservationSpec("weighted_experts", reference_getter=lambda value: value.output,
                                          candidate_getter=lambda value: value.output,
                                          atol=2e-5, rtol=2e-4),),
            objective=lambda value: value.float().square().mean(),
            required_dtypes=("float32",),
        ),),
        parameter_probes=(ParameterProbeSpec("model.layers.*.mlp.router.classifier.weight"),
                          ParameterProbeSpec("model.layers.*.mlp.experts.*")),
        metadata={"architecture": "HF-native LongcatFlashForCausalLM", "depth": 28,
                  "portable_parity_is_native_npu_acceptance": False},
    )
