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
"""Compare CPU placement and MegaMoe with eight mirrored fresh-process runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
from typing import Any

from examples.training_demo.deepseek_v41.coordinate_deepseek_v41_cluster import run_case
from examples.training_demo.deepseek_v41.deepseek_v41_clean_abba import validate_statuses
from examples.training_demo.deepseek_v41 import summarize_deepseek_v41_cluster as summary_base


@dataclass(frozen=True)
class Workload:
    """Keep one fixed model and optimizer horizon across all placement comparisons."""

    layers: int = 8
    experts: int = 384
    ep_size: int = 64
    nodes: int = 8
    cards: int = 16
    steps: int = 100
    warmup: int = 20
    validation_crop: bool = False


@dataclass(frozen=True)
class Case:
    """Specify only the two independent experimental variables."""

    name: str
    placement: str
    mega_kernels: str

    @property
    def backend(self) -> str:
        """Match existing native/push result layout."""
        return 'native' if self.mega_kernels == 'off' else 'push'


def case_matrix() -> tuple[Case, ...]:
    """Mirror all four treatments to balance their position in the job."""
    modes = [('unbound', 'off'), ('unbound', 'moe'), ('numa', 'off'), ('numa', 'moe')]
    order = [(mode, 'a') for mode in modes] + [(mode, 'b') for mode in reversed(modes)]
    return tuple(Case(f'{index:02d}_{placement}_{"native" if mega == "off" else "push"}_{repeat}',
                      placement, mega)
                 for index, ((placement, mega), repeat) in enumerate(order, start=1))


def command_for(case: Case, index: int, workload: Workload) -> list[str]:
    """Set placement in each worker before its first framework import."""
    port = int(os.environ['MASTER_PORT_BASE']) + index * 16
    return ['/usr/bin/env', '-C', os.environ['HYPER_ROOT'],
            f'DSV41_CPU_PLACEMENT={case.placement}',
            f"HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT=tcp://{os.environ['SHMEM_HOST']}:{port + 1}",
            f'HCCL_IF_BASE_PORT={port + 2}', 'PYTHONDONTWRITEBYTECODE=1', sys.executable,
            '-m', 'torch.distributed.run', '--nnodes', str(workload.nodes),
            '--node-rank', os.environ['NODE_RANK'], '--nproc-per-node', str(workload.cards),
            '--master-addr', os.environ['MASTER_ADDR'], '--master-port', str(port), '--rdzv-conf=timeout=7200',
            '-m', 'examples.training_demo.deepseek_v41.deepseek_v41_cpu_worker',
            '--validation-crop' if workload.validation_crop else '--full-model',
            '--experts', str(workload.experts), '--ep-size', str(workload.ep_size), '--edp-shard-size', '2',
            '--steps', str(workload.steps), '--warmup-steps', str(workload.warmup),
            '--packing-mode', 'packed', '--num-hidden-layers', str(workload.layers),
            '--modality', 'text', '--activation-checkpoint', 'full', '--attention-recompute-tier', 'all',
            '--empty-cache-steps', '1', '--engram-storage-backend', 'host', '--random-init-seed', '42',
            '--model-dir', os.environ['MODEL_DIR'], '--engram-assets', os.environ['ENGRAM'],
            '--data', os.environ['DATA'], '--native-lib', os.environ['NATIVE_LIB'],
            '--mega-kernels', case.mega_kernels, '--record-input-identity',
            '--backend', 'owner_ep' if case.backend == 'native' else 'megamoe', '--dispatch-mode', 'push',
            '--output', str(Path(os.environ['RUN_ROOT']) / 'cases' / case.name / 'packed' / case.backend)]


def _identity(folder: Path, case: Case, workload: Workload) -> str:
    """Audit every rank's effective treatment, initialization, inputs and CPU evidence."""
    identities = []
    world = workload.nodes * workload.cards
    expected = {'mega_kernels': case.mega_kernels, 'moe_timing': False, 'profiling_enabled': False,
                'record_input_identity': True, 'steps': workload.steps, 'warmup_steps': workload.warmup,
                'num_hidden_layers': workload.layers, 'experts': workload.experts, 'ep_size': workload.ep_size,
                'edp_shard_size': 2, 'modality': 'text', 'packing_mode': 'packed', 'capacity_factor': None,
                'activation_checkpoint': 'full', 'attention_recompute_tier': 'all',
                'engram_storage_backend': 'host', 'empty_cache_steps': 1, 'random_init_seed': 42,
                'dispatch_mode': 'push', 'full_model': not workload.validation_crop,
                'mega_module_counts': {'mega_moe': 0 if case.mega_kernels == 'off' else workload.layers,
                                       'mega_mhc': 0, 'mega_gate': 0}}
    placements = []
    for rank in range(world):
        work = folder / case.backend / f'rank{rank}'
        result = json.loads((work / 'result.json').read_text())
        differences = {key: (value, result.get(key)) for key, value in expected.items()
                       if result.get(key) != value}
        if differences:
            raise ValueError(f'Unexpected workload at {work}: {differences}')
        placement = json.loads((work / 'cpu-placement.json').read_text())
        cpus = placement['expected_affinity']
        if (placement['mode'] != case.placement or placement['rank'] != rank
                or placement['local_rank'] != rank % workload.cards
                or not placement['placement_before_framework_import']
                or placement['before']['affinity'] != cpus or placement['after']['affinity'] != cpus
                or placement['threads_with_affinity_outside_expected']):
            raise ValueError(f'Invalid CPU placement evidence: {work}')
        placements.append(placement)
        hashes = [row['input_sha256'] for row in result['steps_detail']]
        if len(hashes) != workload.steps or any(len(value) != 64 or
                                               any(char not in '0123456789abcdef' for char in value)
                                               for value in hashes):
            raise ValueError(f'Invalid consumed input identity: {work}')
        identities.append({'initialization': json.loads((work / 'initialization.json').read_text()),
                           'inputs': hashes})
    if case.placement == 'numa':
        for node in range(workload.nodes):
            local = placements[node * workload.cards:(node + 1) * workload.cards]
            cpus = [cpu for row in local for cpu in row['expected_affinity']]
            if len(cpus) != len(set(cpus)):
                raise ValueError(f'CPU sets overlap on node {node}')
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()


def _statistics(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    return {'median_seconds': statistics.median(values), 'mean_seconds': mean,
            'cv_percent': 100 * statistics.pstdev(values) / mean, 'max_seconds': max(values),
            'p90_seconds': sorted(values)[int((len(values) - 1) * 0.9)]}


def publish(root: Path, workload: Workload) -> dict[str, Any]:
    """Publish paired median/mean ratios only with complete, clean, matching evidence."""
    cases = {}
    for case in case_matrix():
        folder = root / 'cases' / case.name / 'packed'
        try:
            validate_statuses([json.loads((folder / 'nodes' / f'node_{node}' / case.backend /
                                          'case_complete.json').read_text()) for node in range(workload.nodes)])
            item, _ = summary_base._summarize_case(folder, folder / 'nodes', workload.nodes, workload.cards,
                                                   workload.steps, workload.warmup, case.backend)
            if item['actual_token_min_max'] != [4096, 4096]:
                raise ValueError(f'Unexpected input rows in {case.name}')
            identity = _identity(folder, case, workload)
            cases[case.name] = {**item, **_statistics(item['measured_rank_max_seconds']),
                                'variables': asdict(case), 'identity_sha256': identity}
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
            cases[case.name] = {'status': 'incomplete_or_failed', 'reason': f'{type(error).__name__}: {error}'}
    comparisons = {}
    groups = {f'{placement}_{backend}': [case.name for case in case_matrix()
                                        if case.placement == placement and case.backend == backend]
              for placement in ('unbound', 'numa') for backend in ('native', 'push')}
    pairs = [('unbound_native', 'unbound_push'), ('numa_native', 'numa_push'),
             ('unbound_native', 'numa_native'), ('unbound_push', 'numa_push')]
    for before, after in pairs:
        names = groups[before] + groups[after]
        complete = all(cases[name]['status'] == 'complete' for name in names)
        matched = complete and len({cases[name]['identity_sha256'] for name in names}) == 1
        eligible = matched and all(cases[name]['hbm_eligible'] for name in names)
        item = {'eligible': eligible, 'initialization_and_inputs_match': matched,
                'baseline_cases': groups[before], 'candidate_cases': groups[after]}
        if eligible:
            baseline = _statistics([value for name in groups[before]
                                    for value in cases[name]['measured_rank_max_seconds']])
            candidate = _statistics([value for name in groups[after]
                                     for value in cases[name]['measured_rank_max_seconds']])
            item.update(baseline=baseline, candidate=candidate,
                        median_speed_ratio=baseline['median_seconds'] / candidate['median_seconds'],
                        mean_speed_ratio=baseline['mean_seconds'] / candidate['mean_seconds'])
        comparisons[f'{before}_vs_{after}'] = item
    report = {'complete': all(case['status'] == 'complete' for case in cases.values()),
              'speedup_eligible': all(item['eligible'] for item in comparisons.values()),
              'workload': asdict(workload), 'gc_policy': 'unchanged_auto', 'profiler': False,
              'module_timing': False, 'measured_steps_inclusive': [workload.warmup + 1, workload.steps],
              'primary_statistic': 'ratio_of_pooled_medians_of_per_step_global_rank_max',
              'case_order': list(cases), 'cases': cases, 'comparisons': comparisons,
              'limits': 'CPU partitions use allowed NUMA CPU lists, not NPU distance. Memory policy is inherited. '
                        'Page residency is process-wide. This does not separate CPU scheduling from first-touch '
                        'memory locality or establish precision parity.'}
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    """Coordinate fresh workers without importing Torch in the controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    workload = Workload(nodes=int(os.environ['NNODES']), cards=int(os.environ['NUM_CARDS']))
    if workload.nodes * workload.cards != 128:
        raise ValueError('Cluster placement comparison requires 128 dies')
    matrix = case_matrix()
    if args.dry_run:
        for index, case in enumerate(matrix):
            print(json.dumps({'index': index, 'case': asdict(case), 'command': command_for(case, index, workload)}))
        return
    root = Path(os.environ['RUN_ROOT'])
    node_rank = int(os.environ['NODE_RANK'])

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f'CPU placement controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if node_rank == 0:
        (root / 'matrix.json').write_text(json.dumps([asdict(case) for case in matrix], indent=2) + '\n')
    for index, case in enumerate(matrix):
        peers = [root / 'cases' / case.name / 'packed/nodes' / f'node_{node}' / case.backend
                 for node in range(workload.nodes)]
        folder, command = peers[node_rank], command_for(case, index, workload)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        (folder / 'variables.json').write_text(json.dumps(asdict(case), indent=2) + '\n')
        print(json.dumps({'event': 'start', 'case': case.name, 'command': command}), flush=True)
        statuses = run_case(command, folder, peers, workload.cards, int(os.environ.get('HBM_INTERVAL', '1')),
                            float(os.environ.get('CASE_SYNC_TIMEOUT', '600')))
        if node_rank == 0:
            publish(root, workload)
        validate_statuses(statuses)
        print(json.dumps({'event': 'finished', 'case': case.name, 'statuses': statuses}), flush=True)
    if node_rank == 0:
        report = publish(root, workload)
        if not report['complete'] or not report['speedup_eligible']:
            raise SystemExit('CPU placement comparison is incomplete or ineligible; inspect summary.json')


if __name__ == '__main__':
    main()
