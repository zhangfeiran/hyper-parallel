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
"""Run clean ABC-CBA optimizer comparisons without importing device frameworks."""

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
from examples.training_demo.deepseek_v41 import summarize_deepseek_v41_cluster as summary_base


_CASE_CACHE: dict[tuple[str, str], tuple[dict, str]] = {}
CLEAN_MEASURED_END_STEP = 35
PROFILE_WARMUP_STEP = 37
PROFILE_START_STEP = 38
PROFILE_END_STEP = 40


@dataclass(frozen=True)
class RunWindow:
    """Keep optimizer, clean measurement and diagnostic windows consistent."""

    steps: int
    warmup: int
    clean_end: int
    profile_warmup: int
    profile_start: int
    profile_end: int


def run_window(numa_tail: bool = False) -> RunWindow:
    """Use a long clean prefix for the fixed CPU-placement comparison."""
    if numa_tail:
        return RunWindow(105, 20, 100, 102, 103, 105)
    return RunWindow(40, 10, CLEAN_MEASURED_END_STEP, PROFILE_WARMUP_STEP,
                     PROFILE_START_STEP, PROFILE_END_STEP)


@dataclass(frozen=True)
class Case:
    """Describe one fresh training process in the mirrored comparison sequence."""

    name: str
    mega_kernels: str
    dispatch_mode: str = 'push'

    @property
    def backend(self) -> str:
        """Keep the existing native/push artifact convention within each case."""
        return 'native' if self.mega_kernels == 'off' else self.dispatch_mode

    @property
    def mode(self) -> str:
        """Keep push and pull distinct when pooling independent repeats."""
        return 'moe_pull' if self.dispatch_mode == 'pull' else self.mega_kernels


def case_matrix(include_pull: bool = False) -> tuple[Case, ...]:
    """Give every mode two independent repeats at mirrored positions."""
    if include_pull:
        modes = (('all_off', 'off', 'push'), ('moe_push', 'moe', 'push'),
                 ('moe_pull', 'moe', 'pull'), ('all_on', 'on', 'push'))
        sequence = [(*mode, 'a') for mode in modes] + [(*mode, 'b') for mode in reversed(modes)]
        return tuple(Case(f'{index:02d}_{label}_{repeat}', mode, dispatch)
                     for index, (label, mode, dispatch, repeat) in enumerate(sequence, start=1))
    return tuple(Case(f'{index:02d}_{label}_{repeat}', mode)
                 for index, (label, mode, repeat) in enumerate((
                     ('all_off', 'off', 'a'), ('moe_only', 'moe', 'a'), ('all_on', 'on', 'a'),
                     ('all_on', 'on', 'b'), ('moe_only', 'moe', 'b'), ('all_off', 'off', 'b'),
                 ), start=1))


def validate_statuses(statuses: list[dict[str, Any]]) -> None:
    """Require clean cross-node termination before accepting any case."""
    if not statuses or not all(row['training_exit_status'] == 0 and row['monitor_exit_status'] == 0
                               and row['cleanup_ok'] and not row['peer_failed']
                               and row.get('artifact_exit_status', 0) == 0 for row in statuses):
        raise RuntimeError(f'Clean ABC-CBA case failed: {statuses}')


def _case_identity(root: Path, case: Case, world: int, tail_profile: bool = False, cards: int = 16,
                   numa_tail: bool = False) -> str:
    """Validate effective modes and hash all ranks' initialization and actual batches."""
    identities = []
    window = run_window(numa_tail)
    assigned_cpus: dict[int, set[int]] = {}
    expected_counts = {'mega_moe': 0 if case.mega_kernels == 'off' else 8,
                       'mega_mhc': 16 if case.mega_kernels == 'on' else 0,
                       'mega_gate': 8 if case.mega_kernels == 'on' else 0}
    expected = {'mega_kernels': case.mega_kernels, 'moe_timing': False, 'profiling_enabled': False,
                'dispatch_mode': case.dispatch_mode, 'capacity_factor': None,
                'record_input_identity': True, 'mega_module_counts': expected_counts,
                'num_hidden_layers': 8, 'experts': 384, 'ep_size': 64, 'edp_shard_size': 2,
                'modality': 'text', 'packing_mode': 'packed', 'activation_checkpoint': 'full',
                'attention_recompute_tier': 'all', 'engram_storage_backend': 'host',
                'empty_cache_steps': 1, 'random_init_seed': 42,
                'steps': window.steps, 'warmup_steps': window.warmup}
    for rank in range(world):
        folder = root / case.backend / f'rank{rank}'
        record = json.loads((folder / 'result.json').read_text())
        wanted = {**expected, 'profiling_enabled': tail_profile and rank % cards == 0}
        if tail_profile:
            profile = json.loads((folder / 'profiling-config.json').read_text())
            if (profile != record.get('tail_profile') or profile['profiler_level'] != 0
                    or profile['clean_steps_inclusive'] != [window.warmup + 1, window.clean_end]
                    or profile['profile_steps_inclusive'] != [window.profile_start, window.profile_end - 1]
                    or profile['profiler_warmup_step'] != window.profile_warmup
                    or profile['profiling_enabled'] != wanted['profiling_enabled']
                    or (wanted['profiling_enabled'] and not profile['trace_exported'])):
                raise ValueError(f'Unexpected tail profiling evidence: {folder}')
        differences = {key: {'expected': value, 'actual': record.get(key)}
                       for key, value in wanted.items() if record.get(key) != value}
        if differences:
            raise ValueError(f'Unexpected workload or instrumentation: {folder}: {differences}')
        if numa_tail:
            placement = json.loads((folder / 'cpu-placement.json').read_text())
            cpus = set(placement['expected_affinity'])
            used = assigned_cpus.setdefault(rank // cards, set())
            if (placement['mode'] != 'numa' or placement['rank'] != rank
                    or placement['local_rank'] != rank % cards or placement['local_workers'] != cards
                    or not placement['placement_before_framework_import'] or not cpus or cpus & used
                    or placement['expected_affinity'] != placement['plan'][rank % cards]['cpus']
                    or placement['before']['affinity'] != placement['expected_affinity']
                    or placement['after']['affinity'] != placement['expected_affinity']
                    or placement['threads_with_affinity_outside_expected']
                    or any(not set(values).issubset(cpus)
                           for values in placement['after']['thread_affinities'].values())):
                raise ValueError(f'Invalid fixed CPU placement evidence: {folder}')
            used.update(cpus)
        hashes = [row['input_sha256'] for row in record['steps_detail']]
        if len(hashes) != window.steps or any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                                              for value in hashes):
            raise ValueError(f'Invalid actual-input identity: {folder}')
        initialization = json.loads((folder / 'initialization.json').read_text())
        identities.append({'initialization': initialization, 'inputs': hashes})
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()


def _compare(cases: dict[str, dict], before: str, after: str, include_pull: bool = False) -> dict[str, Any]:
    """Pool both repeats before taking medians; do not average per-run medians."""
    baseline = [case.name for case in case_matrix(include_pull) if case.mode == before]
    candidate = [case.name for case in case_matrix(include_pull) if case.mode == after]
    names = baseline + candidate
    if any(cases[name]['status'] != 'complete' for name in names):
        return {'eligible': False, 'reason': 'Missing or failed repeat'}
    matching = len({cases[name]['identity_sha256'] for name in names}) == 1
    eligible = matching and all(cases[name]['hbm_eligible'] for name in names)
    result = {'eligible': eligible, 'initialization_and_inputs_match': matching,
              'baseline_cases': baseline, 'candidate_cases': candidate}
    if not eligible:
        return result
    previous = [value for name in baseline for value in cases[name]['measured_rank_max_seconds']]
    current = [value for name in candidate for value in cases[name]['measured_rank_max_seconds']]
    result.update(baseline_mean_seconds=statistics.mean(previous), candidate_mean_seconds=statistics.mean(current),
                  baseline_median_seconds=statistics.median(previous),
                  candidate_median_seconds=statistics.median(current),
                  mean_speed_ratio=statistics.mean(previous) / statistics.mean(current),
                  median_speed_ratio=statistics.median(previous) / statistics.median(current),
                  mean_step_reduction_percent=100 * (1 - statistics.mean(current) / statistics.mean(previous)),
                  median_step_reduction_percent=100 * (1 - statistics.median(current) / statistics.median(previous)),
                  baseline_measured_steps=len(previous), candidate_measured_steps=len(current))
    return result


def _trace_inventory(root: Path, matrix: tuple[Case, ...], world: int, cards: int) -> dict[str, Any]:
    """Require an exported timeline and device-detail artifacts on every selected rank."""
    traces = {}
    for case in matrix:
        traces[case.name] = {}
        for rank in range(0, world, cards):
            folder = root / 'cases' / case.name / 'profiles' / case.backend / f'rank{rank}'
            files = sorted(str(path.relative_to(root)) for path in folder.rglob('*')
                           if path.is_file() and path.stat().st_size)
            timeline = any('trace_view.json' in path or '.trace.json' in path for path in files)
            device = any('kernel_details.csv' in path or path.endswith('.db') for path in files)
            traces[case.name][str(rank)] = {'complete': timeline and device, 'files': files}
    return traces


def publish(root: Path, nodes: int, cards: int, *, include_pull: bool = False,
            tail_profile: bool = False, numa_tail: bool = False) -> dict[str, Any]:
    """Retain complete clean pairs while clearly reporting missing or failed repeats."""
    cases = {}
    matrix = case_matrix(include_pull)
    window = run_window(numa_tail)
    for case in matrix:
        key = str(root) + f':tail={tail_profile}:numa={numa_tail}', case.name
        try:
            if key not in _CASE_CACHE:
                folder = root / 'cases' / case.name / 'packed'
                validate_statuses([json.loads((folder / 'nodes' / f'node_{node}' / case.backend /
                                               'case_complete.json').read_text()) for node in range(nodes)])
                options = {'measured_end_step': window.clean_end} if tail_profile else {}
                item, _signature = summary_base._summarize_case(folder, folder / 'nodes', nodes, cards,
                                                               window.steps, window.warmup, case.backend, **options)
                if item['actual_token_min_max'] != [4096, 4096]:
                    raise ValueError(f'Unexpected actual token rows: {case.name}')
                identity = _case_identity(folder, case, nodes * cards, tail_profile=tail_profile,
                                          cards=cards, numa_tail=numa_tail)
                _CASE_CACHE[key] = item, identity
            item, identity = _CASE_CACHE[key]
            cases[case.name] = {**item, 'variables': asdict(case), 'identity_sha256': identity}
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
            cases[case.name] = {'status': 'incomplete_or_failed', 'reason': f'{type(error).__name__}: {error}'}
    pairs = [('off_vs_moe_only', 'off', 'moe'), ('off_vs_all_on', 'off', 'on'),
             ('moe_only_vs_all_on', 'moe', 'on')]
    if include_pull:
        pairs.extend([('off_vs_moe_pull', 'off', 'moe_pull'), ('moe_push_vs_pull', 'moe', 'moe_pull')])
    comparisons = {name: _compare(cases, before, after, include_pull) for name, before, after in pairs}
    training_complete = all(case['status'] == 'complete' for case in cases.values())
    eligible = training_complete and all(row['eligible'] for row in comparisons.values())
    traces = _trace_inventory(root, matrix, nodes * cards, cards) if tail_profile else {}
    profiling_complete = all(row['complete'] for case in traces.values() for row in case.values())
    complete = training_complete and profiling_complete
    report = {'complete': complete, 'training_complete': training_complete, 'speedup_eligible': eligible,
              'purpose': f'numa_clean{len(matrix)}_tail_level0' if numa_tail else (
                  'tail_level0_profile' if tail_profile else 'clean_abc_cba'),
              'scope': '8 text layers, 128 dies, EP64, expert FSDP2, B0, 4096 tokens/rank',
              'module_timing': False, 'profiler': tail_profile, 'gc_policy': 'unchanged_auto',
              'cpu_placement': 'numa' if numa_tail else 'inherited',
              'executed_steps': window.steps, 'warmup_steps': window.warmup,
              'clean_steps_inclusive': [window.warmup + 1, window.clean_end if tail_profile else window.steps],
              'profiler_level': 0 if tail_profile else None,
              'profile_steps_inclusive': [window.profile_start, window.profile_end - 1] if tail_profile else None,
              'profiling_complete': profiling_complete, 'traces': traces,
              'primary_statistic': 'ratio_of_pooled_medians_of_per_step_global_rank_max',
              'case_order': [case.name for case in matrix], 'cases': cases, 'comparisons': comparisons,
              'limits': 'Two independent fresh-process repeats per mode. Input hashes are collected after step timing. '
                        'Tail profiler warmup, recording and export are excluded from clean step statistics. '
                        'Finite scalars and initialization checks do not establish tensor precision parity.'}
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# DSV4.1 mirrored comparison', '', f'Complete: {complete}',
             f'Training complete: {training_complete}',
             f"Clean optimizer steps (inclusive): {report['clean_steps_inclusive']}",
             f'Level0 tail profiling: {tail_profile}; profiler artifacts complete: {profiling_complete}', '',
             '| Case | Median seconds | Mean seconds | HBM eligible |', '|---|---:|---:|---|']
    for name, row in cases.items():
        if row['status'] == 'complete':
            lines.append(f"| {name} | {row['median_seconds']:.6f} | "
                         f"{row['mean_seconds']:.6f} | {row['hbm_eligible']} |")
        else:
            lines.append(f'| {name} | incomplete | | false |')
    lines.extend(['', '| Comparison | Eligible | Median speed ratio | Mean speed ratio |', '|---|---|---:|---:|'])
    for name, row in comparisons.items():
        ratios = 'unavailable | unavailable'
        if row['eligible']:
            ratios = f"{row['median_speed_ratio']:.6f} | {row['mean_speed_ratio']:.6f}"
        lines.append(f"| {name} | {row['eligible']} | {ratios} |")
    (root / 'REPORT.md').write_text('\n'.join([*lines, '', report['limits'], '']))
    return report


def command_for(case: Case, index: int, tail_profile: bool = False, numa_tail: bool = False) -> list[str]:
    """Use the latest frozen source and safe payload for every mode and repeat."""
    port = int(os.environ['MASTER_PORT_BASE']) + index * 16
    source = os.environ['HYPER_ROOT']
    window = run_window(numa_tail)
    profile_environment = ['CLEAN_PROFILE_TAIL=1'] if tail_profile else []
    module = 'benchmark_deepseek_v41_profile' if tail_profile else 'benchmark_deepseek_v41_vlm'
    if numa_tail:
        if not tail_profile:
            raise ValueError('Fixed NUMA comparison requires tail profiling')
        profile_environment.extend(['CLEAN_NUMA_TAIL=1', 'DSV41_CPU_PLACEMENT=numa'])
        module = 'deepseek_v41_cpu_worker'
    return ['/usr/bin/env', '-C', source,
            f"HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT=tcp://{os.environ['SHMEM_HOST']}:{port + 1}",
            f'HCCL_IF_BASE_PORT={port + 2}', 'PYTHONDONTWRITEBYTECODE=1', *profile_environment, sys.executable,
            '-m', 'torch.distributed.run', '--nnodes', os.environ['NNODES'], '--node-rank', os.environ['NODE_RANK'],
            '--nproc-per-node', os.environ['NUM_CARDS'], '--master-addr', os.environ['MASTER_ADDR'],
            '--master-port', str(port), '--rdzv-conf=timeout=7200',
            '-m', f'examples.training_demo.deepseek_v41.{module}',
            '--full-model', '--experts', '384', '--ep-size', '64', '--edp-shard-size', '2',
            '--steps', str(window.steps), '--warmup-steps', str(window.warmup),
            '--packing-mode', 'packed', '--num-hidden-layers', '8',
            '--modality', 'text', '--activation-checkpoint', 'full', '--attention-recompute-tier', 'all',
            '--empty-cache-steps', '1', '--engram-storage-backend', 'host', '--random-init-seed', '42',
            '--model-dir', os.environ['MODEL_DIR'], '--engram-assets', os.environ['ENGRAM'],
            '--data', os.environ['DATA'],
            '--native-lib', os.environ['NATIVE_LIB'], '--mega-kernels', case.mega_kernels, '--record-input-identity',
            '--backend', 'owner_ep' if case.backend == 'native' else 'megamoe', '--dispatch-mode', case.dispatch_mode,
            '--output', str(Path(os.environ['RUN_ROOT']) / 'cases' / case.name / 'packed' / case.backend)]


def main() -> None:
    """Coordinate fresh workers and publish only clean, identity-matched comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--include-pull', action='store_true')
    parser.add_argument('--tail-profile', action='store_true')
    parser.add_argument('--numa-tail', action='store_true')
    args = parser.parse_args()
    if args.numa_tail and not args.tail_profile:
        parser.error('--numa-tail requires --tail-profile')
    matrix = case_matrix(args.include_pull)
    if args.dry_run:
        for index, case in enumerate(matrix):
            print(json.dumps({'index': index, 'case': asdict(case),
                              'command': command_for(case, index, args.tail_profile, args.numa_tail)}))
        return
    root = Path(os.environ['RUN_ROOT'])
    nodes, node_rank, cards = (int(os.environ[key]) for key in ('NNODES', 'NODE_RANK', 'NUM_CARDS'))

    def interrupted(signum: int, _frame: Any) -> None:
        """Let owned-process cleanup unwind on scheduler termination."""
        raise InterruptedError(f'Clean ABC-CBA controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if node_rank == 0:
        (root / 'matrix.json').write_text(json.dumps([asdict(case) for case in matrix], indent=2) + '\n')
    for index, case in enumerate(matrix):
        peers = [root / 'cases' / case.name / 'packed/nodes' / f'node_{node}' / case.backend for node in range(nodes)]
        folder, command = peers[node_rank], command_for(case, index, args.tail_profile, args.numa_tail)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        (folder / 'variables.json').write_text(json.dumps(asdict(case), indent=2) + '\n')
        print(json.dumps({'event': 'start', 'case': case.name, 'command': command}), flush=True)
        statuses = run_case(command, folder, peers, cards, int(os.environ.get('HBM_INTERVAL', '1')),
                            float(os.environ.get('CASE_SYNC_TIMEOUT', '600')))
        try:
            validate_statuses(statuses)
        except RuntimeError as error:
            print(f'ERROR: {error}; completed comparisons remain available', flush=True)
        if node_rank == 0:
            publish(root, nodes, cards, include_pull=args.include_pull,
                    tail_profile=args.tail_profile, numa_tail=args.numa_tail)
        print(json.dumps({'event': 'finished', 'case': case.name, 'statuses': statuses}), flush=True)
    if node_rank == 0:
        report = publish(root, nodes, cards, include_pull=args.include_pull,
                         tail_profile=args.tail_profile, numa_tail=args.numa_tail)
        if not report['complete'] or not report['speedup_eligible']:
            raise SystemExit('Mirrored comparison is incomplete or ineligible; inspect summary.json')


if __name__ == '__main__':
    main()
