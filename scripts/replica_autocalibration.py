#!/usr/bin/env python3
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
"""Scheduler-neutral collection, fitting and model-in-loop calibration coordinator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any


QUALITY_REJECTED_EXIT = 20


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _run(command: list[str], log: Path, timeout: int, peers: list[Path] | None = None) -> int:
    """Own the full child process group through normal completion or interruption."""
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                status = process.poll()
                if status is not None:
                    return status
                if peers and any(json.loads(path.read_text())['exit_status'] for path in peers if path.exists()):
                    raise RuntimeError('A peer calibration node failed')
                time.sleep(.5)
            raise subprocess.TimeoutExpired(command, timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise


def _wait(paths: list[Path], timeout: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        present = [json.loads(path.read_text()) for path in paths if path.exists()]
        if any(record['exit_status'] for record in present):
            raise RuntimeError(f'Calibration stage failed: {present}')
        if len(present) == len(paths):
            return present
        time.sleep(1)
    raise TimeoutError(f'Calibration stage completion missing: {[str(path) for path in paths if not path.exists()]}')


def _terminate(signum: int, _frame: Any) -> None:
    """Propagate scheduler termination through owned child groups and terminal evidence."""
    raise SystemExit(128 + signum)


def _shmem_host(address: str, *, dry_run: bool) -> str:
    """Keep the allocation identity intact while shortening the native bootstrap host."""
    try:
        socket.inet_pton(socket.AF_INET, address)
        return address
    except OSError:
        if dry_run:
            return 'resolved-ipv4-at-launch'
    try:
        resolved = socket.gethostbyname(address)
        socket.inet_pton(socket.AF_INET, resolved)
    except OSError as error:
        raise ValueError(f'Cannot resolve SHMEM bootstrap host to IPv4: {address}') from error
    return resolved


def _shmem_endpoint(host: str, port: int) -> str:
    """Fail before starting workers if the native bootstrap ABI cannot hold the endpoint."""
    endpoint = f'tcp://{host}:{port}'
    if not 1 <= port <= 65535 or len(endpoint.encode()) > 63:
        raise ValueError(f'SHMEM bootstrap endpoint must use a valid port and at most 63 bytes: {endpoint}')
    return endpoint


def main() -> None:
    """Run on every node; publish only after all ranks and both untouched holdouts pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _terminate)
    hosts = [host.strip() for host in os.environ.get('VC_WORKER_HOSTS', '').split(',') if host.strip()]
    nodes = int(os.environ.get('NNODES') or os.environ.get('MA_NUM_HOSTS') or len(hosts) or 1)
    node = int(os.environ.get('NODE_RANK') or os.environ.get('RANK_AFTER_ACC')
               or os.environ.get('VC_TASK_INDEX') or '-1')
    cards = int(os.environ.get('NUM_CARDS') or '16')
    address = os.environ.get('MASTER_ADDR') or (hosts[0].split(':')[0] if hosts else '')
    port = int(os.environ.get('MASTER_PORT_BASE') or '29300')
    timeout = int(os.environ.get('CALIBRATION_PHASE_TIMEOUT_SECONDS') or '3600')
    if nodes < 1 or not 0 <= node < nodes or cards < 1 or not address or not 1024 <= port < 65532 or timeout < 1:
        raise ValueError('Require valid scheduler NNODES/NODE_RANK/MASTER_ADDR and a free rendezvous port')
    manifest = json.loads(args.manifest.read_text())
    if nodes * cards % manifest['ep_size']:
        raise ValueError('Cluster world size must be divisible by the model manifest EP size')
    os.environ['HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY'] = '0'
    os.environ['HYPER_PARALLEL_REPLICA_METADATA_REUSE'] = '0'
    os.environ.update(MASTER_ADDR=address, NNODES=str(nodes), NUM_CARDS=str(cards))
    os.environ.pop('HYPER_PARALLEL_REPLICA_CALIBRATION_DIR', None)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    base = [sys.executable, '-m', 'torch.distributed.run', f'--nnodes={nodes}', f'--node-rank={node}',
            f'--nproc-per-node={cards}', f'--master-addr={address}']
    worker = ['-m', 'tests.torch.expert_parallel._replica_capture_launcher',
              '--worker-module', 'tests.torch.expert_parallel._replica_autocalibration_worker',
              '--manifest', str(args.manifest.resolve())]
    fitter = [sys.executable, '-m', 'hyper_parallel.core.expert_parallel.hot_replica.autocalibration',
              '--manifest', str(args.manifest.resolve()), '--output', str(args.output / 'models')]
    commands = {'collect': base + [f'--master-port={port}'] + worker + ['--output', str(args.output / 'collect'),
                            '--placement-output', str(args.output / 'collect' / 'placement')],
                'fit': fitter + ['--input', str(args.output / 'collect')],
                'verify': base + [f'--master-port={port+2}'] + worker + ['--output', str(args.output / 'verify'),
                                                                 '--candidate-directory', str(args.output / 'models')],
                'publish': fitter + ['--input', str(args.output / 'verify'), '--verify']}
    commands['verify'] += ['--placement-output', str(args.output / 'verify' / 'placement')]
    shmem_host = _shmem_host(address, dry_run=args.dry_run)
    stage_environment = {
        phase: {'HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT': _shmem_endpoint(shmem_host, endpoint_port)}
        for phase, endpoint_port in (('collect', port + 1), ('verify', port + 3))
    }
    if args.dry_run:
        print(json.dumps({'commands': commands, 'model': manifest, 'nodes': nodes, 'node_rank': node,
                          'cards_per_node': cards, 'stage_environment': stage_environment,
                          'runtime_bootstrap': 'caller-owned CANN/native/Python activation'},
                         indent=2))
        return
    node_root = args.output / 'nodes' / f'node_{node}'
    node_root.mkdir(parents=True, exist_ok=True)
    (node_root / 'started').mkdir()
    terminal = 1
    outcome = None
    try:
        for phase in ('collect', 'fit', 'verify', 'publish'):
            if phase in ('collect', 'verify'):
                os.environ.update(stage_environment[phase])
                peers = [args.output / f'{phase}-node{index}.json' for index in range(nodes)]
                try:
                    status = _run(commands[phase], node_root / f'{phase}.log', timeout, peers)
                except BaseException:
                    _write(args.output / f'{phase}-node{node}.json', {'exit_status': 1, 'node': node})
                    raise
                _write(args.output / f'{phase}-node{node}.json', {'exit_status': status, 'node': node})
                _wait([args.output / f'{phase}-node{index}.json' for index in range(nodes)], timeout)
                missing = [rank for rank in range(nodes * cards)
                           if not (args.output / phase / f'status-rank{rank}.json').exists()]
                if missing:
                    raise RuntimeError(f'Missing completed rank evidence: {phase}: {missing}')
            else:
                if node == 0:
                    try:
                        status = _run(commands[phase], node_root / f'{phase}.log', timeout)
                    except BaseException:
                        _write(args.output / f'{phase}-done.json', {'exit_status': 1, 'node': node})
                        raise
                    rejected = status == QUALITY_REJECTED_EXIT
                    _write(args.output / f'{phase}-done.json', {
                        'exit_status': 0 if rejected else status, 'worker_exit_status': status,
                        'quality_rejected': rejected, 'node': node})
                phase_result = _wait([args.output / f'{phase}-done.json'], timeout)[0]
                if phase_result.get('quality_rejected'):
                    if node == 0:
                        if (args.output / 'models/catalog.json').exists():
                            raise RuntimeError('Rejected calibration unexpectedly published a catalog')
                        _write(args.output / 'outcome.json', {
                            'exit_status': 0, 'status': 'rejected', 'collection_complete': True,
                            'catalog_published': False, 'rejected_phase': phase,
                            'worker_exit_status': QUALITY_REJECTED_EXIT,
                            'report': str(args.output / 'models/fit-report.json'),
                            'runtime_environment': {}})
                    outcome = _wait([args.output / 'outcome.json'], timeout)[0]
                    break
        if outcome is None:
            if node == 0:
                if not (args.output / 'models/catalog.json').is_file():
                    raise RuntimeError('Calibration publication completed without a validated catalog')
                _write(args.output / 'outcome.json', {
                    'exit_status': 0, 'status': 'validated', 'collection_complete': True,
                    'catalog_published': True, 'calibration_directory': str(args.output / 'models'),
                    'runtime_environment': {
                        'HYPER_PARALLEL_REPLICA_CALIBRATION_DIR': str(args.output / 'models')}})
            outcome = _wait([args.output / 'outcome.json'], timeout)[0]
        terminal = QUALITY_REJECTED_EXIT if outcome['status'] == 'rejected' else 0
    finally:
        _write(node_root / 'terminal.json', {'exit_status': terminal, 'node': node})
    print(json.dumps(outcome))
    if terminal == QUALITY_REJECTED_EXIT:
        raise SystemExit(QUALITY_REJECTED_EXIT)


if __name__ == '__main__':
    main()
