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
"""Launcher for the optional HF BF16 SwiGLU numerical regression."""

import os
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.port_utils import allocate_port
from tests.torch.multicore._case_launcher import run_multicore_case
from tests.torch.multicore._test_env import (
    multicore_adapter_is_available, prepare_multicore_test_environment, without_inherited_rank_environment,
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_hf_bf16_swiglu_rounding() -> None:
    """Feature: optional BF16 intermediate rounding.

    Description: Run a native SiLU/multiply oracle against identity-projection MegaMoE.
    Expectation: Both activation gradient branches retain the HF BF16 cast boundaries.
    """
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("HF BF16 SwiGLU ST requires a Multicore native payload")
    os.environ["HP_MEGA_MOE_WORLD_SIZE"] = "1"
    os.environ["HYPER_PARALLEL_SHMEM_HEAP_SIZE"] = str(64 * 1024 * 1024)
    os.environ["HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT"] = f"tcp://127.0.0.1:{allocate_port()}"
    worker = str(Path(__file__).with_name("_test_swiglu_hf_bf16.py"))
    with without_inherited_rank_environment():
        run_multicore_case(worker, "test_hf_bf16_swiglu_rounding", num_proc=1)
