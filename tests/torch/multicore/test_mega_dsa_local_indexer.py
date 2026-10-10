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
"""Framework-free launchers for compact local-query LI on Ascend NPUs."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_mega_dsa_local_indexer.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_mega_dsa_local_indexer_cp1() -> None:
    """Verify actual local query shapes, causal positions and two native phase closures."""
    torchrun_case(_WORKER, "test_mega_dsa_local_indexer_cp", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_local_indexer_cp2() -> None:
    """Verify actual local query shapes, causal positions and two native phase closures."""
    torchrun_case(_WORKER, "test_mega_dsa_local_indexer_cp", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_local_indexer_cp4() -> None:
    """Verify actual local query shapes, causal positions and two native phase closures."""
    torchrun_case(_WORKER, "test_mega_dsa_local_indexer_cp", num_proc=4)
