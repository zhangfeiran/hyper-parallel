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

"""Lightweight launcher for the DSV4.1 fixed-capacity expert adapter."""

from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run
from tests.common.port_utils import allocate_port
from tests.torch.multicore._test_env import (
    prepare_multicore_test_environment, without_inherited_rank_environment,
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_deepseek_v41_megamoe(monkeypatch: pytest.MonkeyPatch, dispatch_mode: str) -> None:
    """Run EP8/E48 forward and all-gradient comparisons at limit 10."""
    prepare_multicore_test_environment()
    monkeypatch.delenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_DYNAMIC_DISPATCH", dispatch_mode)
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    worker = str(Path(__file__).with_name("_test_deepseek_v41_dynamic_tokens.py"))
    with without_inherited_rank_environment():
        parallel_run([TorchCase(worker, "test_deepseek_v41_dynamic_tokens", num_proc=8)], global_num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_deepseek_v41_megamoe_with_replicas(monkeypatch: pytest.MonkeyPatch, dispatch_mode: str) -> None:
    """Compare empty and unequal B1 sources with independent clipped-expert gradients."""
    monkeypatch.setenv("HP_DSV41_DYNAMIC_REPLICA_SLOTS", "1")
    test_deepseek_v41_megamoe(monkeypatch, dispatch_mode)
