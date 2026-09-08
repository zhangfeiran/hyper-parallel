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
"""Exercise the framework-free example parent, which owns its fresh torchrun workers."""

import importlib.util
import json
from pathlib import Path

import pytest

from examples.torch.multicore.mega_moe import qwen_moe_isolated
from tests.common.mark_utils import arg_mark
from tests.common.multicore_test_env import prepare_multicore_test_environment, without_inherited_rank_environment
from tests.common.port_utils import allocate_port


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
@pytest.mark.parametrize('routing', ['balanced', 'dirichlet', 'topk'])
def test_isolated_qwen_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, routing: str) -> None:
    """Compare CPU references against a second independent EP8 training process group."""
    if importlib.util.find_spec('transformers') is None:
        pytest.skip('the Qwen model example requires its optional Transformers dependency')
    prepare_multicore_test_environment()
    monkeypatch.delenv('SYMMETRIC_MEMORY_HEAP_SIZE', raising=False)
    monkeypatch.setenv('MASTER_PORT', str(allocate_port()))
    monkeypatch.setenv('SHMEM_IP_PORT', f'tcp://127.0.0.1:{allocate_port()}')
    output = tmp_path / 'result.json'
    # Test the actual parent; wrapping it in torchrun_case would create nested EP worlds.
    with without_inherited_rank_environment():
        result = qwen_moe_isolated.main([
            '--order', 'ab', '--artifacts-dir', str(tmp_path), '--output', str(output),
            '--shared-physical-weights', '--replica-slots-per-rank', '1', '--num-layers', '2',
            '--hidden-size', '1024', '--intermediate-size', '256', '--vocab-size', '2048',
            '--sequence-length', '1024', '--expert-capacity-factor', '1.5', '--routing', routing,
            '--router-load-alpha', '0.1', '--warmup-steps', '1', '--measured-steps', '2', '--timeout-seconds', '240',
        ])
    assert result == 0, f'expected successful isolated comparison, got {result}'
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['comparison']['accuracy']['passed'], f'accuracy={payload["comparison"]["accuracy"]}'
    assert len(payload['runs']) == 2, f'expected two fresh runs, got {payload["runs"]}'
    for run in payload['runs']:
        for resident in run['result']['resident_training_per_rank']:
            assert resident['resident_backend_count'] == 1, f'expected one backend, got {resident}'
        if routing == 'dirichlet' and run['result']['backend'] == 'mega_moe':
            for pool in run['result']['shared_pool_per_rank']:
                assert pool['shared_weight_bytes'] == 1572864, f'expected shared B1 weight bytes=1572864, got {pool}'
                assert pool['shared_gradient_bytes'] == 3145728, (
                    f'expected shared B1 gradient bytes=3145728, got {pool}')
