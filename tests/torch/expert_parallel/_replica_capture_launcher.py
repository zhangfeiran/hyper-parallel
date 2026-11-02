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
"""Bind calibration workers within allowed NUMA CPUs before importing Torch."""

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import sys

from examples.training_demo.deepseek_v41.deepseek_v41_cpu_worker import parse_cpu_list, placement_plan


def main() -> None:
    """Apply the existing placement policy and retain each rank's affinity."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--worker-module', required=True, choices=(
        'tests.torch.expert_parallel._benchmark_replica_links',
        'tests.torch.expert_parallel._benchmark_replica_contention',
        'tests.torch.expert_parallel._benchmark_replica_critical_path'))
    parser.add_argument('--placement-output', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    local, workers = int(os.environ['LOCAL_RANK']), int(os.environ['LOCAL_WORLD_SIZE'])
    allowed = set(os.sched_getaffinity(0))
    memory_nodes = parse_cpu_list(re.search(r'Mems_allowed_list:\s*(.*)', Path('/proc/self/status').read_text())[1])
    nodes = {int(path.name[4:]): sorted(set(parse_cpu_list((path / 'cpulist').read_text())) & allowed)
             for path in sorted(Path('/sys/devices/system/node').glob('node[0-9]*'))
             if int(path.name[4:]) in memory_nodes}
    plan = placement_plan(nodes, workers)
    if not 0 <= local < workers:
        raise ValueError(f'Invalid local worker: {local}/{workers}')
    if any(name in sys.modules for name in ('torch', 'torch_npu', 'hyper_parallel')):
        raise RuntimeError('Framework imported before CPU placement')
    os.sched_setaffinity(0, plan[local]['cpus'])
    args.placement_output.mkdir(parents=True, exist_ok=True)
    record = dict(global_rank=int(os.environ['RANK']), local_rank=local, pid=os.getpid(),
                  inherited_affinity=sorted(allowed), numa_plan=plan, affinity=sorted(os.sched_getaffinity(0)),
                  placement_before_framework_import=True, memory_policy='inherited_default')
    output = args.placement_output / f'placement-rank{os.environ["RANK"]}.json'
    output.write_text(json.dumps(record, indent=2) + '\n')
    sys.argv = [args.worker_module, *remaining]
    if os.environ.get('REPLICA_PLACEMENT_DRY_RUN') == '1':
        return
    runpy.run_module(args.worker_module, run_name='__main__')
    record['completed'] = True
    record['terminal_affinity'] = sorted(os.sched_getaffinity(0))
    output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
