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
"""Apply rank CPU placement before importing Torch or allocating Host Engram."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import runpy
import sys
from typing import Any


def parse_cpu_list(value: str) -> list[int]:
    """Expand the Linux CPU/node list format, rejecting invalid ranges."""
    result = set()
    for part in value.strip().split(','):
        bounds = [int(item) for item in part.split('-')]
        if len(bounds) == 1:
            first = last = bounds[0]
        elif len(bounds) == 2:
            first, last = bounds
        else:
            raise ValueError(f'Invalid CPU/node list: {value!r}')
        if first < 0 or first > last:
            raise ValueError(f'Invalid CPU/node range: {part!r}')
        result.update(range(first, last + 1))
    return sorted(result)


def placement_plan(nodes: dict[int, list[int]], workers: int) -> list[dict[str, Any]]:
    """Partition allowed CPUs within NUMA nodes without assuming NPU proximity.

    Args:
        nodes: NUMA nodes intersected with the scheduler's allowed CPUs and memory nodes.
        workers: Number of local training workers.

    Returns:
        One disjoint CPU set per local rank, distributed over available NUMA nodes.
    """
    nodes = {node: sorted(set(cpus)) for node, cpus in sorted(nodes.items()) if cpus}
    all_cpus = [cpu for cpus in nodes.values() for cpu in cpus]
    if workers < 1 or len(all_cpus) < workers or len(set(all_cpus)) != len(all_cpus):
        raise ValueError(f'Require disjoint allowed CPUs for {workers} workers: {nodes}')
    assigned = {node: 0 for node in nodes}
    chosen = []
    for _ in range(workers):
        node = max(nodes, key=lambda key: (len(nodes[key]) / (assigned[key] + 1), -key))
        assigned[node] += 1
        chosen.append(node)
    used = {node: 0 for node in nodes}
    plan = []
    for node in chosen:
        index, count, cpus = used[node], assigned[node], nodes[node]
        selected = cpus[index * len(cpus) // count:(index + 1) * len(cpus) // count]
        if not selected:
            raise ValueError(f'Insufficient CPUs on NUMA node {node}: {nodes}')
        plan.append({'numa_node': node, 'cpus': selected})
        used[node] += 1
    return plan


def _snapshot() -> dict[str, Any]:
    """Read page residency and thread affinity outside optimizer timing."""
    pages: dict[str, int] = {}
    for line in Path('/proc/self/numa_maps').read_text().splitlines():
        for node, count in re.findall(r'\bN(\d+)=(\d+)', line):
            pages[node] = pages.get(node, 0) + int(count)
    threads = {}
    for path in Path('/proc/self/task').iterdir():
        try:
            threads[path.name] = sorted(os.sched_getaffinity(int(path.name)))
        except ProcessLookupError:
            # Profiler teardown can retire a thread after its task entry was enumerated.
            continue
    return {'affinity': sorted(os.sched_getaffinity(0)), 'thread_affinities': threads,
            'numa_page_counts': pages,
            'numa_page_count_note': 'Process-wide counts; mapping page sizes may differ; not table-specific bytes'}


def main() -> None:
    """Bind before framework imports, then execute the unchanged training benchmark."""
    mode = os.environ['DSV41_CPU_PLACEMENT']
    if mode not in ('unbound', 'numa'):
        raise ValueError(f'Unknown CPU placement: {mode}')
    rank, workers = int(os.environ['LOCAL_RANK']), int(os.environ['LOCAL_WORLD_SIZE'])
    if not 0 <= rank < workers:
        raise ValueError(f'Invalid local rank {rank}/{workers}')
    allowed = sorted(os.sched_getaffinity(0))
    status = Path('/proc/self/status').read_text()
    memory_nodes = parse_cpu_list(re.search(r'Mems_allowed_list:\s*(.*)', status).group(1))
    nodes = {int(path.name[4:]): sorted(set(parse_cpu_list((path / 'cpulist').read_text())) & set(allowed))
             for path in sorted(Path('/sys/devices/system/node').glob('node[0-9]*'))
             if int(path.name[4:]) in memory_nodes}
    plan = placement_plan(nodes, workers)
    if mode == 'numa':
        os.sched_setaffinity(0, plan[rank]['cpus'])
    if any(name in sys.modules for name in ('torch', 'torch_npu', 'hyper_parallel')):
        raise RuntimeError('Framework imported before CPU placement')
    expected = plan[rank]['cpus'] if mode == 'numa' else allowed
    if sorted(os.sched_getaffinity(0)) != expected:
        raise RuntimeError(f'CPU affinity did not take effect for local rank {rank}: {expected}')
    work = Path(sys.argv[sys.argv.index('--output') + 1]) / f"rank{os.environ['RANK']}"
    work.mkdir(parents=True, exist_ok=True)
    record = {'mode': mode, 'rank': int(os.environ['RANK']), 'local_rank': rank, 'local_workers': workers,
              'pid': os.getpid(), 'inherited_allowed_cpus': allowed, 'allowed_memory_nodes': memory_nodes,
              'numa_allowed_cpus': nodes, 'plan': plan, 'expected_affinity': expected,
              'placement_before_framework_import': True, 'memory_policy': 'inherited_default_no_explicit_membind',
              'npu_distance_used': False, 'visible_devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
              'before': _snapshot()}
    (work / 'cpu-placement.json').write_text(json.dumps(record, indent=2) + '\n')
    if os.environ.get('DSV41_PLACEMENT_DRY_RUN') == '1':
        return
    if os.environ.get('DSV41_CRASH_CAPTURE') == '1':
        module = 'benchmark_deepseek_v41_capture'
    else:
        module = ('benchmark_deepseek_v41_profile' if os.environ.get('CLEAN_PROFILE_TAIL') == '1'
                  else 'benchmark_deepseek_v41_vlm')
    runpy.run_module(f'examples.training_demo.deepseek_v41.{module}', run_name='__main__')
    record['after'] = _snapshot()
    expected_set = set(expected)
    escaped = {tid: cpus for tid, cpus in record['after']['thread_affinities'].items()
               if not set(cpus).issubset(expected_set)}
    record['threads_with_affinity_outside_expected'] = escaped
    (work / 'cpu-placement.json').write_text(json.dumps(record, indent=2) + '\n')
    if escaped:
        raise RuntimeError(f'Rank {record["rank"]}: thread affinity escaped CPU placement: {escaped}')


if __name__ == '__main__':
    main()
