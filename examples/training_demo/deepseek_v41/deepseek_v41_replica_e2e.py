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
import os
from pathlib import Path
import signal
import sys
from typing import Any

from examples.training_demo.deepseek_v41.coordinate_deepseek_v41_cluster import run_case
from examples.training_demo.deepseek_v41.deepseek_v41_clean_abba import validate_statuses
from examples.training_demo.deepseek_v41 import summarize_deepseek_v41_cluster as summary_base


@dataclass(frozen=True)
class Stage:
    """One backend and adaptive replica cap in an independent training process."""

    backend: str
    budget: int

    @property
    def name(self) -> str:
        """Return the stable identifier used for stage selection and artifacts."""
        return f'{self.backend}_b{self.budget}'


def stages(selection: str | None = None) -> tuple[Stage, ...]:
    """Validate selected stages before any training process or monitor starts."""
    matrix = tuple(Stage(backend, budget) for budget in (0, 1, 6) for backend in ('native', 'push', 'pull'))
    if selection is None:
        return matrix
    names = selection.split(',')
    lookup = {stage.name: stage for stage in matrix}
    if not names or len(set(names)) != len(names) or any(name not in lookup for name in names):
        raise ValueError(f'STAGES requires unique comma-separated identifiers from {tuple(lookup)}: {selection!r}')
    return tuple(lookup[name] for name in names)


def command_for(stage: Stage, index: int) -> list[str]:
    """Keep workload, fused attention and CPU placement fixed across all caps."""
    port = int(os.environ['MASTER_PORT_BASE']) + index * 16
    transport = ('p2p' if stage.backend == 'native' else
                 os.environ.get('REPLICA_TRANSPORT', 'shmem_signal_kernel_gradient_bands'))
    output = Path(os.environ['RUN_ROOT']) / 'cases' / stage.name / 'packed' / stage.backend
    return ['/usr/bin/env', '-C', os.environ['HYPER_ROOT'],
            f"HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT=tcp://{os.environ['SHMEM_HOST']}:{port + 1}",
            f'HCCL_IF_BASE_PORT={port + 2}', 'DSV41_CPU_PLACEMENT=numa', 'CLEAN_PROFILE_TAIL=0',
            'DSV41_CRASH_CAPTURE=0', 'PYTHONDONTWRITEBYTECODE=1',
            'HYPER_PARALLEL_REPLICA_METADATA_REUSE=0', 'HYPER_PARALLEL_REPLICA_COST_MODEL=1',
            'HYPER_PARALLEL_REPLICA_HOST_NATIVE=1', sys.executable,
            '-m', 'torch.distributed.run', '--nnodes', os.environ['NNODES'], '--node-rank', os.environ['NODE_RANK'],
            '--nproc-per-node', os.environ['NUM_CARDS'], '--master-addr', os.environ['MASTER_ADDR'],
            '--master-port', str(port), '--rdzv-conf=timeout=7200',
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
    for rank in range(nodes * cards):
        rank_dir = folder / stage.backend / f'rank{rank}'
        result = json.loads((rank_dir / 'result.json').read_text())
        differences = {key: (wanted, result.get(key)) for key, wanted in expected.items() if result.get(key) != wanted}
        if differences:
            raise ValueError(f'Effective replica workload mismatch at rank {rank}: {differences}')
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
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            cases[stage.name] = {'status': 'incomplete_or_failed', 'reason': f'{type(error).__name__}: {error}'}
    comparisons = {}
    for stage in matrix:
        for baseline in dict.fromkeys(('native_b0', f'{stage.backend}_b0')):
            if baseline == stage.name or baseline not in cases:
                continue
            before, after = cases[baseline], cases[stage.name]
            eligible = (before['status'] == after['status'] == 'complete'
                        and before['hbm_eligible'] and after['hbm_eligible']
                        and before['initialization_sha256'] == after['initialization_sha256'])
            row = {'eligible': eligible}
            if eligible:
                row['median_speed_ratio'] = before['median_seconds'] / after['median_seconds']
                row['median_step_reduction_percent'] = 100 * (1 - after['median_seconds'] / before['median_seconds'])
            comparisons[f'{baseline}_vs_{stage.name}'] = row
    report = {'complete': all(case['status'] == 'complete' for case in cases.values()),
              'scope': '8 text layers, 128 dies, EP64, expert FSDP2, Host Engram, full recompute',
              'primary_statistic': 'median_of_per_step_global_rank_max', 'module_timing': False, 'profiler': False,
              'route_diagnostics': False, 'stage_order': [stage.name for stage in matrix],
              'cases': cases, 'comparisons': comparisons,
              'limits': 'One fresh process per stage. No route observations or input hashes; actual token rows '
                        'and identical consumed batches are not independently observed. Finite scalars do not '
                        'establish tensor precision parity. B is an adaptive cap, not an active-replica count.'}
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    """Run selected stages with the established owned-process cleanup protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    matrix = stages(os.environ.get('STAGES'))
    if args.dry_run:
        for index, stage in enumerate(matrix):
            print(json.dumps({'stage': asdict(stage), 'name': stage.name, 'command': command_for(stage, index)}))
        return
    root = Path(os.environ['RUN_ROOT'])
    nodes, rank, cards = (int(os.environ[key]) for key in ('NNODES', 'NODE_RANK', 'NUM_CARDS'))

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f'Replica E2E controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if rank == 0:
        (root / 'matrix.json').write_text(json.dumps([asdict(stage) for stage in matrix], indent=2) + '\n')
    for index, stage in enumerate(matrix):
        peers = [root / 'cases' / stage.name / 'packed' / 'nodes' / f'node_{node}' / stage.backend
                 for node in range(nodes)]
        folder = peers[rank]
        folder.mkdir(parents=True, exist_ok=True)
        command = command_for(stage, index)
        (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        print(json.dumps({'event': 'start', 'stage': stage.name, 'command': command}), flush=True)
        statuses = run_case(command, folder, peers, cards, int(os.environ.get('HBM_INTERVAL', '1')),
                            float(os.environ.get('CASE_SYNC_TIMEOUT', '600')))
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
