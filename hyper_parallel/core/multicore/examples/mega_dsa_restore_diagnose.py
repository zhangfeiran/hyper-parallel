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
"""Output-restoration VJP replay for the explicit offline DSA layer diagnostic."""

import copy

import torch

# Reuse the source formula; the model has no public restoration capture API.
from hyper_parallel.components.modules.dsa_attention import (
    _restore_attention_projection,
)


def _value_restore(model, sparse, hidden_shape):
    weight = model.kv_b_proj.weight.view(model.num_heads, -1, model.kv_lora_rank)
    return _restore_attention_projection(
        sparse.reshape(*hidden_shape[:2], model.num_heads, model.kv_lora_rank),
        weight[:, model.qk_nope_head_dim:].transpose(1, 2), num_heads=model.num_heads,
        batch_size=hidden_shape[0], seq_length=hidden_shape[1], kv_lora_rank=model.kv_lora_rank,
        value_head_dim=model.v_head_dim,
    )


def _restoration_replay(model, capture, phase):
    """Replay detached identical inputs with a fixed captured output cotangent.

    The value phase stops before o_proj; the output phase starts after value
    restoration. The combined phase composes both using the same sparse input.
    Only restoration parameters are reachable, independent of the main/index
    projection graphs. This is a diagnostic, not a production precision policy.
    """
    if phase not in ("value", "output", "combined"):
        raise ValueError("restoration phase must be value, output or combined")
    anchor = model.o_proj.weight
    source = capture["restored_input"] if phase == "output" else capture["sparse_input"]
    inputs = source.to(device=anchor.device, dtype=anchor.dtype).detach().requires_grad_()
    if phase == "output":
        output = model.o_proj(inputs)
    else:
        output = _value_restore(model, inputs, capture["hidden_shape"])
        if phase == "combined":
            output = model.o_proj(output)
    cotangent = capture["restored_cotangent"] if phase == "value" else capture["output_cotangent"]
    cotangent = cotangent.to(device=anchor.device, dtype=anchor.dtype)
    names = ("input_cotangent", *dict(model.named_parameters()))
    gradients = torch.autograd.grad(output, (inputs, *model.parameters()), grad_outputs=cotangent, allow_unused=True)
    result = {name: None if value is None else value.detach().float().cpu()
              for name, value in zip(names, gradients)}
    result["output"] = output.detach().float().cpu()
    projection = result["kv_b_proj.weight"]
    if projection is not None:
        projection = projection.reshape(model.num_heads, model.qk_nope_head_dim + model.v_head_dim, -1)
        result["W_UK"] = projection[:, :model.qk_nope_head_dim]
        result["W_UV"] = projection[:, model.qk_nope_head_dim:]
    return result


def _restoration_diagnosis(template, native, device, compare):
    source = copy.deepcopy(template).to(device)
    cpu = copy.deepcopy(template).float()
    capture = native["restoration"]
    result = {}
    combined = None
    for phase in ("value", "output", "combined"):
        measured = _restoration_replay(source, capture, phase)
        oracle = _restoration_replay(cpu, capture, phase)
        result[phase] = compare(measured, oracle)
        if phase == "combined":
            combined = measured
    expected = {"input_cotangent": native["sparse_cotangent"],
                "o_proj.weight": native["full"]["o_proj.weight"]}
    weight = native["full"]["kv_b_proj.weight"].reshape(source.num_heads, -1, source.kv_lora_rank)
    expected["W_UV"] = weight[:, source.qk_nope_head_dim:]
    result["combined_vs_captured_boundary"] = compare({name: combined[name] for name in expected}, expected)
    result["main_key_slice_zero"] = bool((combined["W_UK"] == 0).all())
    result["scope"] = "identical quantized weights/inputs and fixed cotangents; BF16 NPU vs FP32 CPU; " \
                      "restoration replay is not calibrated model acceptance"
    return result
