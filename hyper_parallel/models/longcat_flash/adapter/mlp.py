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
"""LongCat dense MLP with HF BF16 activation and projection boundaries."""

# Native tensor APIs are the model adapter contract.
# pylint: disable=forbidden-backend-import
import torch
from torch.nn import functional as F

from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class LongcatFlashSwiGLUMLP(SwiGLUMLP):
    """Retain packed parameter ownership and HF's separate autograd branches."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply separate BF16 Gate/Up projections, SiLU and multiply.

        Args:
            x: Input token states.
        """
        gate_weight, up_weight = self.linear_fc1.weight.chunk(2, dim=0)
        bias = self.linear_fc1.bias
        gate_bias, up_bias = (None, None) if bias is None else bias.chunk(2, dim=0)
        gate = F.linear(x, gate_weight, gate_bias)  # pylint: disable=not-callable
        up = F.linear(x, up_weight, up_bias)  # pylint: disable=not-callable
        return self.linear_fc2(F.silu(gate) * up)
