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
"""CPU contracts for the adapter over the upstream MegaMoe parameter API."""
"""CPU contracts for the explicit fixed-shape DSV route fusion adapter."""

import unittest

from torch import nn

from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from tests.common.mark_utils import arg_mark


def _source() -> nn.Module:
    """Use meta parameters to verify configuration without native initialization."""
    module = nn.Module()
    module.num_experts, module.hidden_dim, module.intermediate_dim = 4, 8, 4
    module.limit, module.act_fn = 10.0, nn.SiLU()
    module.gate_up_proj = nn.Parameter(nn.Linear(8, 32, device="meta").weight.reshape(4, 8, 8))
    module.down_proj = nn.Parameter(nn.Linear(8, 16, device="meta").weight.reshape(4, 8, 4))
    return module


class TestDeepseekV41Fusion(unittest.TestCase):
    """Validate static and dynamic contracts before allocating a native executor."""

    # Executor configuration is intentionally inspected without initializing NPU state.
    # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_static_tokens_enable_explicit_route_fusion(self) -> None:
        """Feature: DSV MegaMoe fusion.
        Description: Select fixed rows and both forward and backward endpoints.
        Expectation: The executor binds the exact token count and requested modes.
        """
        experts = DeepseekV41MegaMoeExperts(
            module=_source(), local_num_tokens=128, dynamic_tokens=False,
            forward_fusion_mode="both", backward_fusion_mode="both",
        )
        self.addCleanup(experts.close_runtime)
        experts.configure(None, 1, 2)
        executor = experts._executor
        self.assertIsNone(executor.max_local_num_tokens)
        self.assertEqual(executor.local_num_tokens, 128)
        self.assertEqual(executor._resource_group.specification["forward_fusion_mode"], "both")
        self.assertEqual(executor._resource_group.specification["backward_fusion_mode"], "both")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_dynamic_tokens_reject_static_fusion_without_coercion(self) -> None:
        """Feature: DSV MegaMoe fusion.
        Description: Request static fusion under the variable-length adapter contract.
        Expectation: Configuration rejects unsupported fusion and non-boolean switches.
        """
        with self.assertRaisesRegex(ValueError, "dynamic_tokens must be boolean"):
            DeepseekV41MegaMoeExperts(module=_source(), dynamic_tokens="false")
        experts = DeepseekV41MegaMoeExperts(module=_source(), forward_fusion_mode="both")
        self.addCleanup(experts.close_runtime)
        with self.assertRaisesRegex(ValueError, "static tokens"):
            experts.configure(None, 1, 2)
