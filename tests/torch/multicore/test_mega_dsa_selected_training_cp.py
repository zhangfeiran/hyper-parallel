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
"""Framework-free launchers for device selected requests and complete CP training."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_mega_dsa_training_cp.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_mega_dsa_selected_training_cp1() -> None:
    """Verify LM/KL isolation, loss scaling and seven owner gradients on CP1."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_training_cp", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_selected_training_cp2() -> None:
    """Verify LM/KL isolation, loss scaling and seven owner gradients on CP2."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_training_cp", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_selected_training_cp4() -> None:
    """Verify LM/KL isolation, loss scaling and seven owner gradients on CP4."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_training_cp", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_mega_dsa_selected_training_long_cp1() -> None:
    """Verify real truncated TopK, LM/KL gradients and long-history checkpoint on CP1."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_training_long_cp", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_selected_training_long_cp2() -> None:
    """Verify real truncated TopK, LM/KL gradients and long-history checkpoint on CP2."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_training_long_cp", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_selected_training_long_cp4() -> None:
    """Verify real truncated TopK, LM/KL gradients and long-history checkpoint on CP4."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_training_long_cp", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_mega_dsa_selected_zero_kl_cp1() -> None:
    """Verify zero KL, six phase closures and owner-local gradients on CP1."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_zero_kl_cp", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_selected_zero_kl_cp2() -> None:
    """Verify zero KL, six phase closures and owner-local gradients on CP2."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_zero_kl_cp", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_selected_zero_kl_cp4() -> None:
    """Verify zero KL, six phase closures and owner-local gradients on CP4."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_zero_kl_cp", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_mega_dsa_selected_zero_kl_long_cp1() -> None:
    """Verify real truncated TopK, zero KL and long-history checkpoint on CP1."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_zero_kl_long_cp", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_selected_zero_kl_long_cp2() -> None:
    """Verify real truncated TopK, zero KL and long-history checkpoint on CP2."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_zero_kl_long_cp", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_selected_zero_kl_long_cp4() -> None:
    """Verify real truncated TopK, zero KL and long-history checkpoint on CP4."""
    torchrun_case(_WORKER, "test_mega_dsa_selected_zero_kl_long_cp", num_proc=4)
