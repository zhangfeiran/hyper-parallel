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
"""Eight-rank full Qwen accuracy and optimizer checks through the actual example."""

import json
import os
import tempfile
from pathlib import Path

from examples.torch.multicore.mega_moe import qwen_moe_benchmark


def _run(routing: str) -> None:
    """Run both complete models and validate the final shared-pool ledger."""
    directory = Path(os.getenv('HP_QWEN_MOE_EVIDENCE_DIR', tempfile.gettempdir()))
    output = directory / f'qwen_shared_{routing}.json'
    result = qwen_moe_benchmark.main([
        '--shared-physical-weights', '--replica-slots-per-rank', '1', '--num-layers', '2',
        '--hidden-size', '1024', '--intermediate-size', '256', '--vocab-size', '2048',
        '--sequence-length', '1024', '--expert-capacity-factor', '1.5', '--routing', routing,
        '--router-load-alpha', '0.1', '--warmup-steps', '1', '--measured-steps', '2', '--output', str(output),
    ])
    assert result == 0, f'benchmark returned {result}, expected 0'
    if int(os.environ['LOCAL_RANK']) == 0:
        payload = json.loads(output.read_text())
        assert payload['configuration']['shared_physical_weights'], f'configuration={payload["configuration"]}'
        assert payload['shape']['sequence_length'] == 1024, f'shape={payload["shape"]}'
        pools = payload['shared_pool_per_rank']
        assert len(pools) == 8, f'expected eight pools, got {pools}'
        for pool in pools:
            assert pool['resource_group_count'] == 1, f'expected one shared resource group, got {pool}'
            if routing == 'dirichlet':
                expected = 3 * 1024 * 256 * 2
                assert pool['shared_weight_bytes'] == expected, f'expected weights={expected}, got {pool}'
                assert pool['shared_gradient_bytes'] == expected * 2, f'expected gradients={expected * 2}, got {pool}'


def test_shared_qwen_balanced() -> None:
    """Compare full-model loss, logits, gradients and AdamW updates on balanced routing."""
    _run('balanced')


def test_shared_qwen_dirichlet() -> None:
    """Exercise actual shared-B replicas in a connected attention/MoE/loss graph."""
    _run('dirichlet')


def test_shared_qwen_learned_router() -> None:
    """Train the learned TopK router alongside the decoder and expert parameters."""
    _run('topk')
