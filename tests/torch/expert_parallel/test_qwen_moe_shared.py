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
"""Framework-free launchers for the optional full Qwen shared-B example."""

import importlib.util
import os
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark
from tests.common.multicore_test_env import prepare_multicore_test_environment, without_inherited_rank_environment
from tests.common.port_utils import allocate_port


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
@pytest.mark.parametrize('case', ['test_shared_qwen_balanced', 'test_shared_qwen_dirichlet',
                                 'test_shared_qwen_learned_router'])
def test_qwen_shared_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    """Run the real eight-rank model without importing its frameworks in this parent."""
    if importlib.util.find_spec('transformers') is None:
        pytest.skip('the Qwen model example requires its optional Transformers dependency')
    prepare_multicore_test_environment()
    monkeypatch.setenv('HP_QWEN_MOE_EVIDENCE_DIR', os.getenv('HP_QWEN_MOE_EVIDENCE_DIR', str(tmp_path)))
    monkeypatch.delenv('SYMMETRIC_MEMORY_HEAP_SIZE', raising=False)
    monkeypatch.setenv('SHMEM_IP_PORT', f'tcp://127.0.0.1:{allocate_port()}')
    worker = str(Path(__file__).with_name('_test_qwen_moe_shared.py'))
    with without_inherited_rank_environment():
        torchrun_case(worker, case, num_proc=8)
