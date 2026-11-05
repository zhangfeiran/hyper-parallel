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
"""Run selectable safe8 DSV replica comparisons without importing device frameworks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
from typing import Any

from examples.training_demo.deepseek_v41.coordinate_deepseek_v41_cluster import run_case, write_json
from examples.training_demo.deepseek_v41.deepseek_v41_clean_abba import validate_statuses
from examples.training_demo.deepseek_v41 import summarize_deepseek_v41_cluster as summary_base
from examples.training_demo.deepseek_v41.replica_graph_controls import install_graph
from examples.training_demo.deepseek_v41.deepseek_v41_route_schedule import RouteSchedule, load_schedule


@dataclass(frozen=True)
class Stage:
    """One backend and adaptive replica cap in an independent training process."""

    backend: str
    budget: int
    graph: str | None = None
    policy: str | None = None

    @property
    def name(self) -> str:
        """Return the stable identifier used for stage selection and artifacts."""
        name = f'{self.backend}_b{self.budget}'
        suffix = self.policy or self.graph
        return name if suffix is None else f'{name}_{suffix}'


def stages(selection: str | None = None, *, schedule_comparison: bool = False,
           calibration_comparison: bool = False, calibration_caps: tuple[int, ...] = (1,),
           complete_guest_comparison: bool = False) -> tuple[Stage, ...]:
    """Validate selected stages before any training process or monitor starts."""
    if complete_guest_comparison and (schedule_comparison or calibration_comparison):
        raise ValueError('Complete guest comparison requires the ordinary no-table schedule')
    matrix = tuple(Stage(backend, budget) for budget in (0, 1, 6) for backend in ('native', 'push', 'pull'))
    if schedule_comparison:
        matrix = (tuple(Stage(backend, 0) for backend in ('native', 'push', 'pull'))
                  + tuple(Stage(backend, budget, graph) for budget in (1, 6) for backend in ('push', 'pull')
                          for graph in ('original', 'suffix')))
    if calibration_comparison:
        if (not calibration_caps or any(type(cap) is not int or cap < 1 for cap in calibration_caps)
                or len(set(calibration_caps)) != len(calibration_caps)):
            raise ValueError('Calibration stage caps must be unique positive integers')
        matrix = (tuple(Stage(backend, 0) for backend in ('native', 'push', 'pull'))
                  + tuple(Stage(backend, cap, policy=policy) for backend in ('push', 'pull') for cap in calibration_caps
                          for policy in ('heuristic', 'calibrated')))
    if selection is None:
        return matrix
    if not schedule_comparison and not calibration_comparison:
        matrix += tuple(Stage(backend, 2) for backend in ('native', 'push', 'pull'))
    if complete_guest_comparison:
        matrix += (Stage('push', 2, policy='complete'), Stage('push', 2, policy='complete_repack'),
                   Stage('push', 6, policy='repack'), Stage('pull', 2, policy='heap2g'))
    names = selection.split(',')
    lookup = {stage.name: stage for stage in matrix}
    if not names or len(set(names)) != len(names) or any(name not in lookup for name in names):
        raise ValueError(f'STAGES requires unique comma-separated identifiers from {tuple(lookup)}: {selection!r}')
    if 'push_b2_complete' in names and not {'push_b2', 'push_b6'}.issubset(names):
        raise ValueError('Complete guest comparison requires stock Push B2 and B6 in the same run')
    if 'push_b2_complete_repack' in names and 'push_b2_complete' not in names:
        raise ValueError('Repacking comparison requires the same complete layout without repacking')
    if 'push_b6_repack' in names and 'push_b6' not in names:
        raise ValueError('B6 repacking comparison requires stock Push B6 in the same run')
    if 'pull_b2_heap2g' in names and 'pull_b2' not in names:
        raise ValueError('Pull heap comparison requires stock Pull B2 in the same run')
    return tuple(lookup[name] for name in names)


def select_calibrated_stages(matrix: tuple[Stage, ...], catalog_path: Path | None
                             ) -> tuple[tuple[Stage, ...], list[str]]:
    """Keep controls and schedule only modeled stages present in the published catalog."""
    files = set()
    if catalog_path is not None:
        catalog = json.loads(catalog_path.read_text())
        if catalog['status'] != 'validated' or not catalog['profiles']:
            raise ValueError('Calibration reported success without a validated profile catalog')
        files = {row['file'] for row in catalog['profiles']}
    skipped = [stage.name for stage in matrix if stage.policy == 'calibrated'
               and f'{stage.backend}-b{stage.budget}.json' not in files]
    selected = tuple(stage for stage in matrix if stage.name not in skipped)
    if not selected:
        raise ValueError('No requested Trainer stage remains after calibration')
    return selected, skipped


def command_for(stage: Stage, index: int) -> list[str]:
    """Keep workload, fused attention and CPU placement fixed across all caps."""
    port = int(os.environ['MASTER_PORT_BASE']) + index * 16
    transport = ('p2p' if stage.backend == 'native' else
                 os.environ.get('REPLICA_TRANSPORT', 'shmem_signal_kernel_gradient_bands'))
    output_root = Path(os.environ.get('DSV41_REPLICA_LOCAL_ROOT', os.environ['RUN_ROOT']))
    packed = output_root / 'cases' / stage.name / 'packed'
    output = packed / stage.backend
    logging_options = []
    control_environment = []
    if os.environ.get('DSV41_REPLICA_LOCAL_ROOT'):
        logging_options = ['--log-dir', str(packed / 'torchrun' / f'node_{os.environ["NODE_RANK"]}'),
                           '--redirects', '3']
        control_environment = ['DSV41_BENCHMARK_CONTROL=cpu']
    route_environment = []
    calibration_environment = []
    if stage.policy in ('heuristic', 'calibrated') or os.environ.get('DSV41_CALIBRATION_E2E') == '1':
        modeled = stage.policy == 'calibrated'
        directory = str(Path(os.environ['RUN_ROOT']) / 'calibration/models') if modeled else ''
        calibration_environment = [f'HYPER_PARALLEL_REPLICA_CALIBRATION_DIR={directory}',
            f'HYPER_PARALLEL_REPLICA_REQUIRE_CALIBRATION={int(modeled)}',
            'HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY=0']
        if stage.budget:
            calibration_environment.append(f'HYPER_PARALLEL_REPLICA_CALIBRATION_AUDIT_DIR={output}')
    if os.environ.get('DSV41_COMPLETE_GUEST_SEARCH_E2E') == '1':
        calibration_environment += [
            'HYPER_PARALLEL_REPLICA_CALIBRATION_DIR=',
            'HYPER_PARALLEL_REPLICA_REQUIRE_CALIBRATION=0',
            'HYPER_PARALLEL_REPLICA_COST_MODEL=0',
            f'HYPER_PARALLEL_SHMEM_HEAP_SIZE={2**31 if stage.policy == "heap2g" else 2**32}',
            'HYPER_PARALLEL_REPLICA_PROJECTED_GUEST_SEARCH=0',
            f'HYPER_PARALLEL_REPLICA_COMPLETE_GUEST_SEARCH={int(stage.policy in ("complete", "complete_repack"))}',
            f'HYPER_PARALLEL_SHMEM_FIXED_HEAP_REPACK={int(stage.policy in ("repack", "complete_repack"))}',
        ]
    if os.environ.get('DSV41_ROUTE_SCHEDULE'):
        selected = load_schedule(Path(os.environ['DSV41_ROUTE_SCHEDULE']))
        if (selected.steps != int(os.environ['TRAIN_ITERS']) or selected.warmup != int(os.environ['WARMUP_STEPS'])
                or os.environ.get('DSV41_ROUTE_SCHEDULE_SHA256', selected.sha256) != selected.sha256):
            raise ValueError('Training horizon differs from the frozen route-window schedule')
        route_environment = [f'DSV41_ROUTE_SCHEDULE={os.environ["DSV41_ROUTE_SCHEDULE"]}',
                             f'DSV41_ROUTE_SCHEDULE_SHA256={selected.sha256}']
        if os.environ.get('DSV41_ROUTE_PROFILE_IDENTITIES'):
            route_environment.append(f'DSV41_ROUTE_PROFILE_IDENTITIES={os.environ["DSV41_ROUTE_PROFILE_IDENTITIES"]}')
    return ['/usr/bin/env', '-C', os.environ['HYPER_ROOT'],
            f"HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT=tcp://{os.environ['SHMEM_HOST']}:{port + 1}",
            f'HCCL_IF_BASE_PORT={port + 2}', 'DSV41_CPU_PLACEMENT=numa', 'CLEAN_PROFILE_TAIL=0',
            'DSV41_CRASH_CAPTURE=0', 'PYTHONDONTWRITEBYTECODE=1',
            'HYPER_PARALLEL_REPLICA_METADATA_REUSE=0', 'HYPER_PARALLEL_REPLICA_COST_MODEL=1',
            'HYPER_PARALLEL_REPLICA_HOST_NATIVE=1', *control_environment, *route_environment,
            *calibration_environment, sys.executable,
            '-m', 'torch.distributed.run', '--nnodes', os.environ['NNODES'], '--node-rank', os.environ['NODE_RANK'],
            '--nproc-per-node', os.environ['NUM_CARDS'], '--master-addr', os.environ['MASTER_ADDR'],
            '--master-port', str(port), '--rdzv-conf=timeout=7200', *logging_options,
            '-m', 'examples.training_demo.deepseek_v41.deepseek_v41_cpu_worker',
            '--full-model', '--experts', '384', '--ep-size', '64', '--edp-shard-size', '2',
            '--steps', os.environ['TRAIN_ITERS'], '--warmup-steps', os.environ['WARMUP_STEPS'],
            '--packing-mode', 'packed', '--num-hidden-layers', '8', '--modality', 'text',
            '--activation-checkpoint', 'full', '--attention-recompute-tier', 'all',
            '--empty-cache-steps', '1', '--engram-storage-backend', 'host', '--random-init-seed', '42',
            '--model-dir', os.environ['MODEL_DIR'], '--engram-assets', os.environ['ENGRAM'],
            '--data', os.environ['DATA'],
            '--native-lib', os.environ['NATIVE_LIB'], '--mega-kernels', 'off' if stage.backend == 'native' else 'moe',
            '--use-fused-attention', '--disable-route-diagnostics',
            '--backend', 'owner_ep' if stage.backend == 'native' else 'megamoe',
            '--dispatch-mode', 'pull' if stage.backend == 'pull' else 'push',
            '--replica-slots-per-rank', str(stage.budget), '--replica-transport', transport,
            '--output', str(output)]


def _identity(folder: Path, stage: Stage, nodes: int, cards: int) -> str:
    """Check every effective training configuration, initialization and NUMA placement."""
    identities, assigned = [], {}
    expected = {
        'backend': 'owner_ep' if stage.backend == 'native' else 'megamoe',
        'dispatch_mode': 'pull' if stage.backend == 'pull' else 'push', 'replica_slots_per_rank': stage.budget,
        'replica_transport': ('p2p' if stage.backend == 'native' else
                              os.environ.get('REPLICA_TRANSPORT', 'shmem_signal_kernel_gradient_bands')),
        'modality': 'text', 'num_hidden_layers': 8, 'experts': 384, 'ep_size': 64, 'edp_shard_size': 2,
        'packing_mode': 'packed', 'activation_checkpoint': 'full', 'attention_recompute_tier': 'all',
        'engram_storage_backend': 'host', 'empty_cache_steps': 1, 'random_init_seed': 42,
        'steps': int(os.environ['TRAIN_ITERS']), 'warmup_steps': int(os.environ['WARMUP_STEPS']),
        'moe_timing': False, 'profiling_enabled': False, 'disable_route_diagnostics': True,
        'record_input_identity': False, 'use_fused_attention': True,
        'mega_kernels': 'off' if stage.backend == 'native' else 'moe',
        'mega_module_counts': {'mega_moe': 0 if stage.backend == 'native' else 8, 'mega_mhc': 0, 'mega_gate': 0},
    }
    if os.environ.get('DSV41_REPLICA_LOCAL_ROOT'):
        expected['benchmark_control_backend'] = 'tcpstore'
    for rank in range(nodes * cards):
        rank_dir = folder / stage.backend / f'rank{rank}'
        result = json.loads((rank_dir / 'result.json').read_text())
        differences = {key: (wanted, result.get(key)) for key, wanted in expected.items() if result.get(key) != wanted}
        if differences:
            raise ValueError(f'Effective replica workload mismatch at rank {rank}: {differences}')
        if os.environ.get('DSV41_REPLICA_LOCAL_ROOT'):
            control = json.loads((rank_dir / 'benchmark-control.json').read_text())
            if control != dict(backend='tcpstore', timing_device='cpu', timeout_seconds=300.0,
                               training_process_group_unchanged=True, world_size=nodes * cards, rank=rank):
                raise ValueError(f'Invalid isolated CPU benchmark control at rank {rank}: {control}')
        placement = json.loads((rank_dir / 'cpu-placement.json').read_text())
        cpus = set(placement['expected_affinity'])
        used = assigned.setdefault(rank // cards, set())
        if (placement['mode'] != 'numa' or placement['rank'] != rank or not cpus or used & cpus
                or not placement['placement_before_framework_import']
                or placement['before']['affinity'] != placement['expected_affinity']
                or placement['after']['affinity'] != placement['expected_affinity']
                or placement['threads_with_affinity_outside_expected']):
            raise ValueError(f'Invalid NUMA placement for rank {rank}')
        used.update(cpus)
        initialization = json.loads((rank_dir / 'initialization.json').read_text())
        if initialization['weight_files'] or initialization['seed'] != 42:
            raise ValueError(f'Expected canonical in-memory weights at rank {rank}')
        identities.append(initialization)
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()


def _graph_identity(folder: Path, stage: Stage, nodes: int) -> dict[str, Any]:
    """Require each node to have installed the same checksumed source graph."""
    variant = stage.graph or 'suffix'
    manifest = json.loads((Path(os.environ['BUNDLE_ROOT']) / 'schedule-controls/manifest.json').read_text())[variant]
    for node in range(nodes):
        path = folder / 'nodes' / f'node_{node}' / stage.backend / 'graph-control.json'
        receipt = json.loads(path.read_text())
        if any(receipt.get(key) != manifest[key] for key in ('source_commit', 'files')):
            raise ValueError(f'Runtime graph identity mismatch at node {node}: {stage.name}')
    return dict(variant=variant, **manifest)


def calibration_identity(folder: Path, stage: Stage, world: int) -> dict[str, Any]:
    """Require every modeled rank to have actually selected the same execution-bound coefficients."""
    identities, soc_names = set(), set()
    for rank in range(world):
        path = folder / stage.backend / f'rank{rank}' / 'calibration-selection.json'
        receipt = json.loads(path.read_text())
        records = receipt['selections']
        expected = 'matched' if stage.policy == 'calibrated' else 'fallback'
        if (receipt['rank'] != rank or not records or any(row['status'] != expected
                or row['backend'] != stage.backend or row['ep_size'] != 64
                or row['hidden_size'] != 5120 or row['intermediate_size'] != 2304
                or row['num_experts'] != 384 or row['top_k'] != 6 or row['checkpointed'] is not True
                or stage.policy == 'calibrated' and (row['scorer'] != 'effective_group'
                   or not isinstance(row['execution_signature'], str) or len(row['execution_signature']) != 64)
                for row in records)):
            raise ValueError(f'Unexpected calibration selection at rank {rank}: {stage.name}')
        soc_names.update(row.get('soc_name') for row in records)
        identities.add(json.dumps([{key: value for key, value in row.items() if key != 'soc_name'}
                                   for row in records], sort_keys=True))
    if len(identities) != 1:
        raise ValueError(f'Calibration coefficients or execution signatures differ across ranks: {stage.name}')
    return dict(status=expected, matched_ranks=world if expected == 'matched' else 0,
                audited_ranks=world, actual_soc_names=sorted(name for name in soc_names if name is not None),
                selections=json.loads(identities.pop()))


def prepare_calibration(*, dry_run: bool = False) -> str:
    """Run collection, fitting and private verification in the same allocation before Trainer controls."""
    root = Path(os.environ['RUN_ROOT'])
    environment = dict(os.environ)
    environment['MASTER_PORT_BASE'] = str(int(os.environ['MASTER_PORT_BASE']) + 2048)
    environment['HYPER_PARALLEL_REPLICA_REQUIRE_CALIBRATION'] = '0'
    environment.pop('HYPER_PARALLEL_REPLICA_CALIBRATION_DIR', None)
    environment.pop('HYPER_PARALLEL_REPLICA_CALIBRATION_AUDIT_DIR', None)
    local = Path(os.environ['DSV41_REPLICA_LOCAL_ROOT']).parent / 'calibration'
    environment['CALIBRATION_NODE_LOCAL_ROOT'] = str(local)
    command = [sys.executable, str(Path(os.environ['HYPER_ROOT']) / 'scripts/replica_autocalibration.py'),
               '--manifest', str(Path(os.environ['BUNDLE_ROOT']) / 'model-shape.json'),
               '--output', str(root / 'calibration')]
    if dry_run:
        print(json.dumps(dict(event='calibrate_before_training', command=command + ['--dry-run'],
                              gate='Publish only after fit and private verification; exit20 skips modeled stages')))
        return 'dry_run'
    local.mkdir(parents=True, exist_ok=True)
    log = local / 'controller.log'
    with log.open('w') as stream:
        process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            status = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    shared = root / 'calibration' / f'controller-node{os.environ["NODE_RANK"]}.log'
    shared.parent.mkdir(parents=True, exist_ok=True)
    temporary = shared.with_suffix('.tmp')
    temporary.write_bytes(log.read_bytes())
    temporary.replace(shared)
    if status not in (0, 20):
        raise RuntimeError(f'Calibration collection/runtime failed: exit={status}; inspect {shared}')
    result = 'validated' if status == 0 else 'rejected'
    write_json(root / 'calibration' / f'launch-node{os.environ["NODE_RANK"]}.json',
               dict(exit_status=status, calibration_status=result))
    if status == 0 and not (root / 'calibration/models/catalog.json').is_file():
        raise RuntimeError('Successful calibration did not publish its validated catalog')
    return result


def validate_route_windows(record: dict[str, Any], selected: RouteSchedule) -> dict[str, dict[str, Any]]:
    """Validate complete step/profile coverage and exact controlled source histograms."""
    if (record['schedule'] != selected.record() or record['schedule_sha256'] != selected.sha256
            or record['completed_steps'] != selected.steps or not record['shared_source_profile']
            or not record['learned_weights_preserved'] or not record['synthetic_performance_only']
            or record['plan_observation_hooks'] or len(record['windows']) != len(selected.windows)):
        raise ValueError('Incomplete or different full-model route schedule')
    result = {}
    for entry, window in zip(record['windows'], selected.windows):
        if entry['window'] != asdict(window) or entry['observed_steps'] != list(range(window.start, window.end)):
            raise ValueError(f'Wrong applied optimizer-step window: {window.name}')
        layers = entry['layers']
        if set(layers) != {str(layer) for layer in range(8)}:
            raise ValueError(f'Incomplete decoder routing coverage: {window.name}')
        skews = []
        for metadata in layers.values():
            counts = metadata['source_expert_counts']
            if (len(counts) != 384 or any(type(value) is not int or not 0 <= value <= 4096 for value in counts)
                    or sum(counts) != 4096 * 6 or metadata['tokens'] != 4096 or metadata['experts'] != 384
                    or metadata['top_k'] != 6 or metadata['ep'] != 64):
                raise ValueError(f'Invalid controlled source histogram: {window.name}')
            owner = [64 * sum(counts[start:start + 6]) for start in range(0, 384, 6)]
            skew = max(owner) / (4096 * 6)
            if metadata['owner_destination_rows'] != owner or not math.isclose(metadata['owner_skew'], skew):
                raise ValueError(f'Invalid owner-only counterfactual: {window.name}')
            skews.append(skew)
        digest = hashlib.sha256(json.dumps(layers, sort_keys=True).encode()).hexdigest()
        result[window.name] = dict(routing_sha256=digest, owner_skew=statistics.median(skews),
                                   owner_skew_max=max(skews))
    return result


def _route_windows(folder: Path, stage: Stage, nodes: int, cards: int) -> dict[str, Any]:
    """Aggregate rank-max medians only inside each declared measured load window."""
    selected = load_schedule(Path(os.environ['DSV41_ROUTE_SCHEDULE']))
    rows, identity = [], None
    for rank in range(nodes * cards):
        rank_dir = folder / stage.backend / f'rank{rank}'
        current = validate_route_windows(json.loads((rank_dir / 'routing-windows.json').read_text()), selected)
        if identity is not None and current != identity:
            raise ValueError(f'Controlled logical routes differ at rank {rank}')
        identity = current
        steps = json.loads((rank_dir / 'result.json').read_text())['steps_detail']
        if len(steps) != selected.steps or any(
                row.get('routing_window') != selected.at(index).name
                or row.get('routing_profile') != selected.at(index).profile for index, row in enumerate(steps)):
            raise ValueError(f'Timing rows differ from applied load windows at rank {rank}')
        rows.append(steps)
    if os.environ.get('DSV41_ROUTE_PROFILE_IDENTITIES'):
        expected = json.loads(Path(os.environ['DSV41_ROUTE_PROFILE_IDENTITIES']).read_text())
        if expected['schedule_sha256'] != selected.sha256 or any(
                identity[window.name] != expected['profiles'][f'{window.profile}/s{window.seed}']
                for window in selected.windows):
            raise ValueError('Frozen controlled profiles differ from the applied route templates')
    result = {}
    for window in selected.windows:
        if not window.measured:
            continue
        times = [max(rank_rows[index]['local_seconds'] for rank_rows in rows) for index in window.measured_indices]
        result[window.name] = dict(profile=window.profile, seed=window.seed, **identity[window.name],
                                   measured_steps=[index + 1 for index in window.measured_indices],
                                   rank_max_seconds=times, median_seconds=statistics.median(times),
                                   p95_seconds=sorted(times)[math.ceil(.95 * len(times)) - 1])
    return result


def publish(root: Path, matrix: tuple[Stage, ...], nodes: int, cards: int) -> dict[str, Any]:
    """Audit all ranks and node termination before publishing median step ratios."""
    cases = {}
    for stage in matrix:
        folder = root / 'cases' / stage.name / 'packed'
        try:
            validate_statuses([json.loads((folder / 'nodes' / f'node_{node}' / stage.backend /
                                           'case_complete.json').read_text()) for node in range(nodes)])
            item, _ = summary_base._summarize_case(
                folder, folder / 'nodes', nodes, cards, int(os.environ['TRAIN_ITERS']),
                int(os.environ['WARMUP_STEPS']), stage.backend)
            cases[stage.name] = {**item, 'initialization_sha256': _identity(folder, stage, nodes, cards)}
            if stage.policy == 'heap2g':
                outputs = (folder / stage.backend / f'rank{rank}' / 'result.json' for rank in range(nodes * cards))
                sizes = [json.loads(path.read_text())['shmem_state']['config']['heap_size_bytes'] for path in outputs]
                if sizes != [2**31] * (nodes * cards):
                    raise ValueError('Pull heap comparison did not use 2 GiB on every rank')
                cases[stage.name]['verified_heap_bytes'] = 2**31
            if os.environ.get('REPLICA_SCHEDULE_E2E') == '1':
                cases[stage.name]['runtime_graph'] = _graph_identity(folder, stage, nodes)
            if stage.policy in ('heuristic', 'calibrated'):
                cases[stage.name]['calibration_selection'] = calibration_identity(folder, stage, nodes * cards)
            if stage.policy in ('complete', 'complete_repack'):
                outputs = (folder / stage.backend / f'rank{rank}' / 'result.json' for rank in range(nodes * cards))
                hits = [json.loads(path.read_text())['complete_guest_search_hits'] for path in outputs]
                if any(type(value) is not int or value <= 0 for value in hits):
                    raise ValueError('Complete guest search did not run on every rank')
                cases[stage.name]['complete_guest_search_hits'] = hits
            if os.environ.get('DSV41_ROUTE_SCHEDULE'):
                cases[stage.name]['route_windows'] = _route_windows(folder, stage, nodes, cards)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            cases[stage.name] = {'status': 'incomplete_or_failed', 'reason': f'{type(error).__name__}: {error}'}
    comparisons = {}
    for stage in matrix:
        baselines = ('native_b0', f'{stage.backend}_b0')
        if stage.graph == 'suffix':
            baselines += (Stage(stage.backend, stage.budget, 'original').name,)
        if stage.policy == 'calibrated':
            baselines += (Stage(stage.backend, stage.budget, policy='heuristic').name,)
        elif stage.policy in ('complete', 'complete_repack'):
            baselines += ('push_b2', 'push_b6')
            if stage.policy == 'complete_repack':
                baselines += ('push_b2_complete',)
        elif stage.policy in ('repack', 'heap2g'):
            baselines += (Stage(stage.backend, stage.budget).name,)
        for baseline in dict.fromkeys(baselines):
            if baseline == stage.name or baseline not in cases:
                continue
            before, after = cases[baseline], cases[stage.name]
            eligible = (before['status'] == after['status'] == 'complete'
                        and before['hbm_eligible'] and after['hbm_eligible']
                        and before['initialization_sha256'] == after['initialization_sha256'])
            names = ([window.name for window in load_schedule(Path(os.environ['DSV41_ROUTE_SCHEDULE'])).windows
                      if window.measured] if os.environ.get('DSV41_ROUTE_SCHEDULE') else [None])
            for name in names:
                first = before['route_windows'][name] if eligible and name is not None else before
                second = after['route_windows'][name] if eligible and name is not None else after
                matched = eligible and (name is None or first['routing_sha256'] == second['routing_sha256'])
                row = {'eligible': matched}
                if matched:
                    row['median_speed_ratio'] = first['median_seconds'] / second['median_seconds']
                    row['median_step_reduction_percent'] = 100 * (
                        1 - second['median_seconds'] / first['median_seconds'])
                key = f'{baseline}_vs_{stage.name}'
                comparisons[key if name is None else f'{name}/{key}'] = row
    skipped_names = os.environ.get('DSV41_CALIBRATION_SKIPPED_STAGES', '')
    skipped = skipped_names.split(',') if skipped_names else []
    report = {'complete': all(case['status'] == 'complete' for case in cases.values()),
              'scope': '8 text layers, 128 dies, EP64, expert FSDP2, Host Engram, full recompute',
              'primary_statistic': 'median_of_per_step_global_rank_max', 'module_timing': False, 'profiler': False,
              'route_diagnostics': False, 'stage_order': [stage.name for stage in matrix],
              'schedule_comparison': os.environ.get('REPLICA_SCHEDULE_E2E') == '1',
              'calibration_comparison': os.environ.get('DSV41_CALIBRATION_E2E') == '1',
              'calibration_status': os.environ.get('DSV41_CALIBRATION_STATUS'),
              'requested_matrix_complete': not skipped, 'skipped_modeled_stages': skipped,
              'load_windows': bool(os.environ.get('DSV41_ROUTE_SCHEDULE')),
              'cases': cases, 'comparisons': comparisons,
              'limits': 'One fresh process per stage. No route observations or input hashes; actual token rows '
                        'and identical consumed batches are not independently observed. Finite scalars do not '
                        'establish tensor precision parity. B is an adaptive cap, not an active-replica count.'}
    if os.environ.get('DSV41_ROUTE_SCHEDULE'):
        report['primary_statistic'] = 'median_of_per_step_global_rank_max_per_load_window'
        report['route_schedule'] = load_schedule(Path(os.environ['DSV41_ROUTE_SCHEDULE'])).record()
        report['aggregate_note'] = ('Whole-case medians include different loads and transition steps. '
                                    'Use only window-prefixed comparisons. HBM retains whole-case high-water demand.')
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    """Run selected stages with the established owned-process cleanup protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    schedule_flag = os.environ.get('REPLICA_SCHEDULE_E2E', '0')
    if schedule_flag not in ('0', '1'):
        raise ValueError('REPLICA_SCHEDULE_E2E must be 0 or 1')
    schedule_comparison = schedule_flag == '1'
    calibrated = os.environ.get('DSV41_CALIBRATION_E2E') == '1'
    complete_guests = os.environ.get('DSV41_COMPLETE_GUEST_SEARCH_E2E') == '1'
    caps = (1,)
    if calibrated:
        manifest = Path(os.environ['BUNDLE_ROOT']) / 'model-shape.json'
        caps = tuple(json.loads(manifest.read_text())['replica_caps'])
    matrix = stages(os.environ.get('STAGES'), schedule_comparison=schedule_comparison,
                    calibration_comparison=calibrated, calibration_caps=caps,
                    complete_guest_comparison=complete_guests)
    if args.dry_run:
        if calibrated:
            prepare_calibration(dry_run=True)
        for index, stage in enumerate(matrix):
            print(json.dumps({'stage': asdict(stage), 'name': stage.name, 'command': command_for(stage, index)}))
        return
    root = Path(os.environ['RUN_ROOT'])
    nodes, rank, cards = (int(os.environ[key]) for key in ('NNODES', 'NODE_RANK', 'NUM_CARDS'))

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f'Replica E2E controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    os.environ.pop('DSV41_CALIBRATION_SKIPPED_STAGES', None)
    if calibrated:
        status = prepare_calibration()
        catalog = root / 'calibration/models/catalog.json' if status == 'validated' else None
        matrix, skipped = select_calibrated_stages(matrix, catalog)
        os.environ['DSV41_CALIBRATION_STATUS'] = 'partial' if skipped and catalog is not None else status
        os.environ['DSV41_CALIBRATION_SKIPPED_STAGES'] = ','.join(skipped)
        if skipped:
            print(json.dumps(dict(event='calibration_stages_skipped', stages=skipped, controls_continue=True)),
                  flush=True)
    if rank == 0:
        (root / 'matrix.json').write_text(json.dumps([asdict(stage) for stage in matrix], indent=2) + '\n')
    for index, stage in enumerate(matrix):
        peers = [root / 'cases' / stage.name / 'packed' / 'nodes' / f'node_{node}' / stage.backend
                 for node in range(nodes)]
        folder = peers[rank]
        folder.mkdir(parents=True, exist_ok=True)
        if schedule_comparison:
            try:
                receipt = install_graph(Path(os.environ['HYPER_ROOT']), Path(os.environ['BUNDLE_ROOT']),
                                        stage.graph or 'suffix')
                (folder / 'graph-control.json').write_text(json.dumps(receipt, indent=2) + '\n')
            except (OSError, ValueError, KeyError, TypeError):
                # Peers must stop promptly even when this node fails before spawning workers.
                write_json(folder / 'exit_status.json', {'training_exit_status': 1})
                write_json(folder / 'case_complete.json', dict(training_exit_status=1, monitor_exit_status=127,
                                                               cleanup_ok=True, peer_failed=False))
                raise
        command = command_for(stage, index)
        (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        print(json.dumps({'event': 'start', 'stage': stage.name, 'command': command}), flush=True)
        local_options = {}
        if os.environ.get('DSV41_REPLICA_LOCAL_ROOT'):
            local_options['local_root'] = Path(os.environ['DSV41_REPLICA_LOCAL_ROOT']) / 'cases' / stage.name / 'packed'
        statuses = run_case(command, folder, peers, cards, int(os.environ.get('HBM_INTERVAL', '1')),
                            float(os.environ.get('CASE_SYNC_TIMEOUT', '600')), **local_options)
        validate_statuses(statuses)
        if rank == 0:
            publish(root, matrix, nodes, cards)
        print(json.dumps({'event': 'finished', 'stage': stage.name}), flush=True)
    if rank == 0 and not publish(root, matrix, nodes, cards)['complete']:
        raise RuntimeError('Replica E2E results are incomplete; inspect summary.json')


if __name__ == '__main__':
    exit_status = 1
    try:
        main()
        exit_status = 0
    finally:
        if (os.environ.get('DRY_RUN') != '1' and '--dry-run' not in sys.argv and '--help' not in sys.argv
                and 'RUN_ROOT' in os.environ and 'NODE_RANK' in os.environ):
            node_dir = Path(os.environ['RUN_ROOT']) / 'nodes' / f"node_{os.environ['NODE_RANK']}"
            node_dir.mkdir(parents=True, exist_ok=True)
            (node_dir / 'exit_status.txt').write_text(f'{exit_status}\n')
