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
"""Compare native and Mega kernels while collecting native fatal stacks on every rank."""

import argparse
import json
import os
from pathlib import Path
import signal
from typing import Any

from examples.training_demo.deepseek_v41.coordinate_deepseek_v41_cluster import run_case
from examples.training_demo.deepseek_v41.deepseek_v41_clean_abba import Case, command_for


def case_matrix() -> tuple[Case, ...]:
    """Use a native baseline before MegaMoe-only and all three Mega kernels."""
    return (Case('01_native', 'off'), Case('02_moe_push', 'moe'), Case('03_all_on', 'on'))


def capture_command(case: Case, index: int) -> list[str]:
    """Reuse the frozen 105-step recipe while replacing profiling with fatal capture."""
    command = command_for(case, index, tail_profile=True, numa_tail=True)
    command[command.index('CLEAN_PROFILE_TAIL=1')] = 'CLEAN_PROFILE_TAIL=0'
    worker = 'examples.training_demo.deepseek_v41.deepseek_v41_cpu_worker'
    command[command.index(worker)] = 'examples.training_demo.deepseek_v41.deepseek_v41_capture_worker'
    node = Path(os.environ['RUN_ROOT'])/'cases'/case.name/'packed/nodes'/f"node_{os.environ['NODE_RANK']}"/case.backend
    command[3:3] = ['DSV41_CRASH_CAPTURE=1',
                    f"DSV41_NATIVE_CAPTURE_LIB={os.environ['DSV41_NATIVE_CAPTURE_LIB']}",
                    f'ASCEND_PROCESS_LOG_PATH={node}/cann-logs', f'ASCEND_WORK_PATH={node}/ascend-work']
    command.insert(command.index('--output'), '--use-fused-attention')
    return command


def publish(root: Path, nodes: int) -> dict[str, Any]:
    """Report failures and capture coverage without treating diagnostic times as speedups."""
    cases = {}
    for case in case_matrix():
        base = root/'cases'/case.name/'packed'
        statuses = []
        for node in range(nodes):
            path = base/'nodes'/f'node_{node}'/case.backend/'case_complete.json'
            statuses.append(json.loads(path.read_text()) if path.exists() else None)
        ranks = {}
        for rank in range(nodes*16):
            folder = base/case.backend/f'rank{rank}'
            path = folder/'native-stacks.log'
            text = path.read_text(errors='replace') if path.exists() else ''
            steps = folder/'steps.jsonl'
            completed_steps = 0
            if steps.exists():
                with steps.open() as stream:
                    completed_steps = sum(1 for _ in stream)
            ranks[str(rank)] = {'native_stack_complete': 'DSV41_NATIVE_STACK_END' in text,
                                'native_stack_started': 'DSV41_NATIVE_STACK_BEGIN' in text,
                                'result_written': (folder/'result.json').exists(),
                                'completed_steps': completed_steps}
        cases[case.name] = {'mega_kernels': case.mega_kernels, 'nodes': statuses, 'ranks': ranks,
                           'training_passed': all(row is not None and row['training_exit_status'] == 0
                                                  and row['cleanup_ok'] for row in statuses)}
    complete = all(all(row is not None for row in case['nodes']) for case in cases.values())
    report = {'collection_complete': complete, 'training_all_passed': all(case['training_passed']
              for case in cases.values()), 'purpose': 'fatal_stack_and_kernel_isolation_not_performance',
              'scope': '8 text layers, 128 dies, EP64, expert FSDP2, B0, 4096 tokens/rank',
              'steps': 105, 'warmup_steps': 20, 'profiler': False, 'blocking': False, 'gc': 'unchanged_auto',
              'attention_backend': 'fused',
              'native_frame_limit': 256, 'cases': cases}
    (root/'capture-summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main() -> None:
    """Continue after captured failures only when every node has cleaned its workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        for index, case in enumerate(case_matrix()):
            print(json.dumps({'case': case.name, 'mode': case.mega_kernels,
                              'command': capture_command(case, index)}))
        return
    root = Path(os.environ['RUN_ROOT'])
    nodes, node_rank = int(os.environ['NNODES']), int(os.environ['NODE_RANK'])

    def interrupted(signum: int, _frame: Any) -> None:
        """Let the coordinator release only this case's owned workers."""
        raise InterruptedError(f'Capture controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    for index, case in enumerate(case_matrix()):
        peers = [root/'cases'/case.name/'packed/nodes'/f'node_{node}'/case.backend for node in range(nodes)]
        folder = peers[node_rank]
        folder.mkdir(parents=True, exist_ok=True)
        command = capture_command(case, index)
        (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        print(json.dumps({'event': 'start', 'case': case.name, 'command': command}), flush=True)
        statuses = run_case(command, folder, peers, 16, int(os.environ.get('HBM_INTERVAL', '1')),
                            float(os.environ.get('CASE_SYNC_TIMEOUT', '600')))
        print(json.dumps({'event': 'finished', 'case': case.name, 'statuses': statuses}), flush=True)
        if node_rank == 0:
            publish(root, nodes)
    if node_rank == 0:
        report = publish(root, nodes)
        if not report['training_all_passed']:
            raise SystemExit('Capture completed with training failures; '
                             'inspect capture-summary.json and native-stacks.log')


if __name__ == '__main__':
    main()
