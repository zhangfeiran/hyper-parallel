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
"""Framework-free launchers for local index-gradient owner return."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_mega_dsa_local_index_owner.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_local_index_owner2() -> None:
    """Verify FP32 owner sums and local Q/weight identity on two HCCL ranks."""
    torchrun_case(_WORKER, "test_mega_dsa_local_index_owner", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_mega_dsa_local_index_owner4() -> None:
    """Verify four ranks including empty query tiles and an empty KV owner."""
    torchrun_case(_WORKER, "test_mega_dsa_local_index_owner", num_proc=4)
