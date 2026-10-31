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
"""Exercise fanout-guided plans against the existing native FP32 training reference."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel
from hyper_parallel.core.multicore.modules.mega_moe import module as module_api
from tests.torch.expert_parallel._test_hot_replica import _run


def main() -> None:
    """Use explicit synthetic EP4 compute costs only to validate planner/datapath integration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('push', 'pull'), required=True)
    parser.add_argument('--result-dir', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=300))
    calls = []

    def resolve(enabled: bool, backend: str, hidden: int, intermediate: int, ep_size: int,
                transport: str, _experts: int, _top_k: int, _device: torch.device) -> ExpertReplicaCostModel | None:
        if not enabled:
            return None
        calls.append(backend)
        return ExpertReplicaCostModel(calibration_version=2, backend=backend, hidden_size=hidden,
            intermediate_size=intermediate, ep_size=ep_size, transport=transport,
            forward_ms=((0, 0), (1, .01), (32768, 10000)), backward_ms=((0, 0), (1, .02), (32768, 20000)),
            weight_ms=.1, gradient_ms=.1, backward_weight_ms=0,
            weight_fanout_ms=((0, 0, 0), (1, .1, .1), (2, .15, .15), (4, .25, .25)))

    try:
        with patch.object(module_api, '_runtime_cost_model', side_effect=resolve):
            _run(args.backend, 6, 'cpu', 'shmem_signal_kernel_gradient_bands', Path(args.result_dir))
        if not calls:
            raise AssertionError('NPU correctness test did not execute the fanout-aware cost path')
        output = Path(args.result_dir) / f'cost-path-rank{dist.get_rank()}.json'
        output.write_text(json.dumps(dict(rank=dist.get_rank(), calls=len(calls), backend=args.backend,
            status='passed', synthetic_costs=True, performance_eligible=False)) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
