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
"""Thin launchers for two-rank AST MoE forward/backward acceptance."""

from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run
from tests.common.port_utils import allocate_port
from tests.torch.multicore._test_env import (
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["push", "pull"])
@pytest.mark.parametrize("worker_name,case_name", [
    ("_ast_moe_impl.py", "test_ast_moe_device"),
    ("_ast_moe_lifecycle_impl.py", "test_ast_moe_lifecycle"),
])
def test_ast_moe_device(monkeypatch, tmp_path: Path, mode: str, worker_name: str, case_name: str) -> None:
    """Feature: AST MoE execution.

    Description: Compare routing/clamp numerics and shared/checkpoint/stream lifetimes on two NPUs.
    Expectation: Forward, gradients and updates align; readiness and device profiling remain complete.
    """
    prepare_multicore_test_environment()
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", "2")
    monkeypatch.setenv("HP_AST_MOE_MODE", mode)
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", mode)
    monkeypatch.setenv("HP_AST_MOE_RESULTS", str(tmp_path))
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", str(64 * 1024 * 1024))
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    worker = str(Path(__file__).with_name(worker_name))
    with without_inherited_rank_environment():
        parallel_run([TorchCase(worker, case_name, num_proc=2)], global_num_proc=2)
