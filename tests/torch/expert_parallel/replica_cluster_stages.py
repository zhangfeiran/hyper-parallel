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
"""Run identical coordinated stage tables with shared failure and gate receipts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from tests.torch.expert_parallel.replica_cluster_cases import cluster_cases
from tests.torch.expert_parallel.replica_schedule_capture import install_graph, schedule_cases, verify_pair


def write_json(path: Path, value: Any) -> None:
    """Publish a complete receipt atomically on shared storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def terminate(process: subprocess.Popen) -> None:
    """Stop only this launcher's process group after a failed cohort stage."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=10)
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def receipts(root: Path, label: str, nodes: int) -> list[dict[str, Any]]:
    """Read only atomically published records from the current stage."""
    rows = []
    for node in range(nodes):
        directory = root / 'nodes' / f'node_{node}'
        terminal = directory / 'exit_status.txt'
        if terminal.is_file() and terminal.read_text().strip() != '0':
            raise RuntimeError(f'Node {node} already terminated with an error')
        path = directory / f'{label}-stage.json'
        if path.is_file():
            rows.append(json.loads(path.read_text()))
    return rows


def wait_cohort(root: Path, label: str, nodes: int, deadline: float) -> list[dict[str, Any]]:
    """Prevent any node from entering the next collective after a peer failure."""
    while time.monotonic() < deadline:
        rows = receipts(root, label, nodes)
        if any(row['status'] == 'failed' for row in rows):
            raise RuntimeError(f'A peer failed stage {label}')
        if len(rows) == nodes:
            if len({row['table_sha256'] for row in rows}) != 1:
                raise RuntimeError('Nodes disagree on their stage table')
            return rows
        time.sleep(1)
    raise TimeoutError(f'Cohort did not finish stage {label}')


def gate_identity(bundle: Path) -> str:
    """Bind external gate reuse to exactly the same frozen bundle manifest."""
    return hashlib.sha256((bundle / 'SHA256SUMS').read_bytes()).hexdigest()


def gate_passed(root: Path, bundle: Path) -> bool:
    """A missing or mismatched device gate never enables metadata performance."""
    path = root / 'D2-GATE-PASS.json'
    return path.is_file() and json.loads(path.read_text()).get('bundle_sha256') == gate_identity(bundle)


def publish_gate(root: Path, bundle: Path, world: int) -> None:
    """Require both backends and every rank's actual numerical/device checks."""
    for mode in ('push', 'pull'):
        for rank in range(world):
            report = json.loads((root / f'gate-d2-{mode}' / f'rank{rank}.json').read_text())
            required = ('nondefault_streams', 'same_counts_new_ids', 'registry_cleared',
                        'changed_counts_miss', 'changed_policy_miss', 'synthetic_cost_rejected_home')
            probe = report['metadata_device_probe'] or {}
            if not report['all_tensors_identical'] or not all(probe.get(key) is True for key in required):
                raise RuntimeError(f'D2 device gate is incomplete: {mode}/rank{rank}')
            rows = report['variants'][1]['metadata']
            if not any(row['kind'] == 'lookup' and row['hit'] for row in rows):
                raise RuntimeError(f'D2 was enabled without real FSDP descriptor hits: {mode}/rank{rank}')
    write_json(root / 'D2-GATE-PASS.json', dict(bundle_sha256=gate_identity(bundle),
               all_ranks=world, modes=['push', 'pull'], numerical_hashes_identical=True))


def main() -> None:
    """Keep setup once per node and start a fresh Torch process for each control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--phases', default='gate,fsdp,timing,diagnostic,causal')
    parser.add_argument('--order', default='AB')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--schedule-capture', action='store_true')
    args = parser.parse_args()
    nodes, node = int(os.environ['NNODES']), int(os.environ['NODE_RANK'])
    port_base = int(os.environ['MASTER_PORT_BASE'])
    cases = schedule_cases(args.phases, args.order) if args.schedule_capture else cluster_cases(args.phases, args.order)
    outcomes = []
    table_hash = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    output = args.run_root / 'nodes' / f'node_{node}'
    timeout = int(os.environ.get('STAGE_TIMEOUT_SECONDS', '7200'))
    if not args.dry_run:
        write_json(output / 'setup-stage.json', dict(status='passed', table_sha256=table_hash))
        wait_cohort(args.run_root, 'setup', nodes, time.monotonic() + timeout)
        write_json(output / 'stages.json', cases)
    for index, case in enumerate(cases):
        label, port = case['label'], port_base + index * 32
        result = args.run_root / label
        replacements = {'@SOURCE@': str(args.source), '@BUNDLE@': str(args.bundle),
                        '@OUTPUT@': str(result), '@BOOTSTRAP@': str(port + 16)}
        arguments = [replacements.get(value, value.replace('@BUNDLE@', str(args.bundle)))
                     for value in case['arguments']]
        environment = os.environ | {
            'HYPER_PARALLEL_REPLICA_HOST_NATIVE': str(case['host']),
            'HYPER_PARALLEL_REPLICA_METADATA_REUSE': str(case['metadata']),
            'HYPER_PARALLEL_REPLICA_COST_MODEL': str(case['cost']),
            'HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT': f'tcp://{os.environ["SHMEM_HOST"]}:{port + 1}',
            'HCCL_IF_BASE_PORT': str(port + 2), 'PYTHONDONTWRITEBYTECODE': '1'}
        command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes', str(nodes),
                   '--node-rank', str(node), '--nproc-per-node', '16', '--master-addr', os.environ['MASTER_ADDR'],
                   '--master-port', str(port), '--rdzv-conf=timeout=7200', '-m']
        if case['worker'].endswith('._replica_cost_worker'):
            command += [case['worker'], *arguments]
        else:
            command += ['tests.torch.expert_parallel._replica_capture_launcher', '--worker-module', case['worker'],
                        '--placement-output', str(result / 'placement'), *arguments]
        if args.dry_run:
            print(json.dumps(case | dict(command=command, table_sha256=table_hash)))
            continue
        record = dict(case, table_sha256=table_hash, command=command)
        gate_root = Path(os.environ.get('GATE_RUN_ROOT', str(args.run_root)))
        allowed = True
        if case['requires_d2_gate']:
            eligibility = label + '-eligibility'
            write_json(output / f'{eligibility}-stage.json', dict(
                status='passed', table_sha256=table_hash, eligible=gate_passed(gate_root, args.bundle)))
            decisions = wait_cohort(args.run_root, eligibility, nodes, time.monotonic() + timeout)
            allowed = all(row['eligible'] for row in decisions)
        if not allowed:
            record.update(status='skipped', reason='Matching D2 device gate has not passed')
            write_json(output / f'{label}-stage.json', record)
            wait_cohort(args.run_root, label, nodes, time.monotonic() + timeout)
            outcomes.append(dict(label=label, status='skipped', reason=record['reason']))
            continue
        planner = args.source / 'hyper_parallel/core/expert_parallel/hot_replica/planner.py'
        shutil.copyfile(args.bundle / 'controls' / f'planner-{case["planner"]}.py', planner)
        record['planner_sha256'] = hashlib.sha256(planner.read_bytes()).hexdigest()
        if case.get('graph') is not None:
            control = os.environ.get('CONTROL_SOURCE')
            record['graph_source_control'] = install_graph(
                args.source, args.bundle, case['graph'], Path(control) if control else None)
        process = None

        def interrupted(signum: int, _frame: Any) -> None:
            if process is not None:
                terminate(process)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        deadline = time.monotonic() + timeout
        try:
            with (output / f'{label}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=args.source, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    if any(row['status'] == 'failed' for row in receipts(args.run_root, label, nodes)):
                        raise RuntimeError(f'Peer failure during {label}')
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'Stage timeout: {label}')
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError(f'Torch stage {label} exited with {process.returncode}')
            record.update(status='passed', returncode=0)
            write_json(output / f'{label}-stage.json', record)
            wait_cohort(args.run_root, label, nodes, deadline)
            if case.get('graph') == 'suffix':
                pair = verify_pair(args.run_root, case['pair'], nodes * 16)
                pair_label = f'{label}-numeric-pair'
                write_json(output / f'{pair_label}-stage.json', dict(
                    status='passed', table_sha256=table_hash, **pair))
                wait_cohort(args.run_root, pair_label, nodes, deadline)
            outcomes.append(dict(label=label, status='passed'))
            if label == 'gate-d2-pull':
                if node == 0:
                    publish_gate(args.run_root, args.bundle, nodes * 16)
                while not gate_passed(args.run_root, args.bundle):
                    if any(row['status'] == 'failed' for row in receipts(args.run_root, label, nodes)):
                        raise RuntimeError('A peer rejected the D2 gate proof')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Waiting for all-rank D2 gate proof')
                    time.sleep(1)
        except BaseException as error:
            if process is not None:
                terminate(process)
            write_json(output / f'{label}-stage.json', record | dict(status='failed', error=str(error)))
            for remaining in cases[index + 1:]:
                write_json(output / f'{remaining["label"]}-stage.json', remaining | dict(
                    table_sha256=table_hash, status='skipped', reason=f'Prior stage failed: {label}'))
            raise
    if not args.dry_run:
        write_json(output / 'all-stages.json', dict(completed=True, table_sha256=table_hash, selected=args.phases,
                   all_selected_passed=all(row['status'] == 'passed' for row in outcomes), outcomes=outcomes))


if __name__ == '__main__':
    main()
