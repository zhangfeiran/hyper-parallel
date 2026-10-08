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
"""Lightweight shared-root MoE/DSA NPU acceptance launcher."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark
from tests.torch.multicore._test_env import prepare_multicore_test_environment


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_shared_shmem_root_cp2() -> None:
    """Validate two consumers, delayed MoE backward, streams and both close orders."""
    prepare_multicore_test_environment()
    worker = str(Path(__file__).with_name("_test_shared_shmem_root.py"))
    torchrun_case(worker, "test_shared_shmem_root_coexistence", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_shared_shmem_root_cp4(monkeypatch) -> None:
    """Validate the same ordered CP4/EP4 root and both fixed-capacity transports."""
    prepare_multicore_test_environment()
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", "4")
    worker = str(Path(__file__).with_name("_test_shared_shmem_root.py"))
    torchrun_case(worker, "test_shared_shmem_root_coexistence", num_proc=4)
