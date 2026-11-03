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
"""Lightweight LongCat Flash distributed acceptance launchers."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_longcat_flash_ep() -> None:
    """
    Feature: LongCat Flash model integration.
    Description: Run two-rank complete-model MegaMoE parity with the production recipe.
    Expectation: Outputs, gradients, and state obey the asserted contract.
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_test_longcat_flash.py")),
                  case_name="test_longcat_flash_ep", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_longcat_empty_routes() -> None:
    """Feature: Empty real-expert sources and identity-only execution.

    Description: Exercise two-rank forward/backward when only a peer has real source assignments.
    Expectation: Both owners receive finite gradients; globally identity-only steps leave expert parameters unused.
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_test_longcat_empty_routes.py")),
                  case_name="test_longcat_empty_routes", num_proc=2)
