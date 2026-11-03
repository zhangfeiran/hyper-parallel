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
"""NPU regression for explicit BF16 SiLU/multiply cast boundaries."""

import torch
from torch.nn import functional as F

from hyper_parallel.components.functional.grouped_matmul import grouped_matmul
from hyper_parallel.core.multicore import MegaMoeExperts
from tests.torch.multicore import _test_mega_moe as baseline


def test_hf_bf16_swiglu_rounding() -> None:
    """Feature: optional HF BF16 activation compatibility.

    Description: Isolate gate and up derivatives using identity projections,
        including gate=5.9375 where sigmoid multiplication changes BF16 rounding.
    Expectation: Outputs, input gradients and both weight gradients are exact.
    """
    device = baseline.DEVICE
    layer = MegaMoeExperts(hidden_size=256, intermediate_size=128, num_experts=1,
                           top_k=1, ep_size=1, expert_weight_layout="EOI", swiglu_hf_bf16=True)
    layer = layer.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.gate_up_weight.copy_(torch.eye(256, device=device, dtype=torch.bfloat16).unsqueeze(0))
        layer.down_weight.zero_()
        layer.down_weight[0, :128].copy_(torch.eye(128, device=device, dtype=torch.bfloat16))
    torch.manual_seed(618)
    values = (torch.arange(4096, device=device, dtype=torch.float32).remainder(3072) / 128 - 12).bfloat16()
    hidden = torch.randn(4096, 256, device=device, dtype=torch.bfloat16)
    hidden[:, :128] = values.reshape(-1, 1)
    hidden.requires_grad_()
    actual_hidden = hidden.detach().clone().requires_grad_()
    gate_weight = layer.gate_up_weight.detach().clone().requires_grad_()  # pylint: disable=not-callable
    down_weight = layer.down_weight.detach().clone().requires_grad_()  # pylint: disable=not-callable
    group_list = torch.tensor([4096], device=device, dtype=torch.int64)
    gate, up = grouped_matmul(hidden, gate_weight.transpose(1, 2), group_list=group_list).chunk(2, -1)
    expected = grouped_matmul(F.silu(gate) * up, down_weight.transpose(1, 2), group_list=group_list)
    ids = torch.zeros(4096, 1, device=device, dtype=torch.int64)
    probabilities = torch.ones(4096, 1, device=device, dtype=torch.float32)
    upstream = torch.randn_like(expected)
    try:
        actual = layer(actual_hidden, ids, probabilities)
        expected.backward(upstream)
        actual.backward(upstream)
        torch.npu.synchronize()
        for observed, reference in ((actual, expected), (actual_hidden.grad, hidden.grad),
                                    (layer.gate_up_weight.grad, gate_weight.grad),
                                    (layer.down_weight.grad, down_weight.grad)):
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)
    finally:
        layer.close()
