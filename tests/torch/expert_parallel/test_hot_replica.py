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
"""Framework-free launchers for final replica transports and planner boundaries."""

from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

TRAINING_CASES = (
    ("native_cpu", "native", "cpu", "p2p"),
    ("native_device", "native", "device", "p2p"),
    *((f"{backend}_{planner}_{transport}", backend, planner, transport)
      for backend in ("push", "pull") for planner in ("cpu", "device")
      for transport in ("p2p", "shmem_signal_kernel_gradient")),
)
_DIRECTORY = Path(__file__).parent


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("case,backend,planner,transport", TRAINING_CASES,
                         ids=[case[0] for case in TRAINING_CASES])
def test_replica_training(case: str, backend: str, planner: str, transport: str) -> None:
    """Check numerical, optimizer, reverse-backward and cross-stream acceptance."""
    del backend, planner, transport
    torchrun_case(str(_DIRECTORY / "_test_hot_replica.py"), f"test_replica_training_npu[{case}]", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("case", ("replica_count_payload", "device_replica_boundaries", "device_route_validation"))
def test_replica_boundaries(case: str) -> None:
    """Retain exact counts, collective route errors and device-planner capacity checks."""
    torchrun_case(str(_DIRECTORY / "_test_replica_boundaries.py"), f"test_{case}_npu", num_proc=4)
