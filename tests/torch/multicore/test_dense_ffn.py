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
"""Framework-free launcher for actual Ascend dense FFN training validation."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_dense_ffn_training() -> None:
    """Feature: Native dense FFN training.
    Description: Compare outputs/gradients and checkpoint/close lifecycle on one Ascend NPU.
    Expectation: Every component comparison and pending backward succeeds.
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_ffn_impl.py")),
                  case_name="test_dense_ffn_training", master_port=29617, num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dense_ffn_fully_shard() -> None:
    """Feature: Dense FFN FSDP integration.
    Description: Validate gradients, changed shard weights, recompute and checkpoint restore on two NPUs.
    Expectation: Independent SGD components remain aligned through repeated unshard/reshard cycles.
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_ffn_fsdp_impl.py")),
                  case_name="test_dense_ffn_fully_shard", master_port=29619, num_proc=2)
