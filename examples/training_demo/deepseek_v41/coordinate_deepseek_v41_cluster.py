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
"""Run one isolated case and coordinate cleanup before advancing every node."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


def write_json(path: Path, value: Any) -> None:
    """Atomically publish a node status to shared storage."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value) + '\n', encoding='utf-8')
    temporary.replace(path)


def _group_alive(group: int) -> bool:
    # Ignore zombies, which hold no devices and may await an external reaper.
    for entry in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = entry.read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if fields[0] != 'Z' and int(fields[2]) == group:
            return True
    return False


def _stop_group(group: int) -> bool:
    """Signal only the new session created for this case, including its workers."""
    for sig, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 10)):
        if not _group_alive(group):
            return True
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + seconds
        while _group_alive(group) and time.monotonic() < deadline:
            time.sleep(0.1)
    return not _group_alive(group)


def wait_for_cleanup(folders: list[Path], timeout: float) -> list[dict[str, Any]]:
    """Require all nodes to finish the same case before any starts the next."""
    deadline = time.monotonic() + timeout
    while True:
        pending = [folder / 'case_complete.json' for folder in folders]
        if all(path.exists() for path in pending):
            records = [json.loads(path.read_text()) for path in pending]
            if not all(record['cleanup_ok'] for record in records):
                raise RuntimeError(f'Case workers did not clean up on every node: {records}')
            return records
        if time.monotonic() >= deadline:
            missing = [str(path) for path in pending if not path.exists()]
            raise RuntimeError(f'Timed out waiting for peer cleanup: {missing}')
        time.sleep(0.2)


def _snapshot(path: Path) -> None:
    with path.open('w', encoding='utf-8') as stream:
        try:
            status = subprocess.run(['npu-smi', 'info'], stdout=stream, stderr=subprocess.STDOUT,
                                    check=False, timeout=30).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(f'HBM snapshot error: {type(error).__name__}: {error}\n')
            status = 1
    if status:
        print(f'HBM snapshot failed with status {status}: {path}', flush=True)


def run_case(command: list[str], folder: Path, peers: list[Path], cards: int,
             interval: int, timeout: float) -> list[dict[str, Any]]:
    """Abort peer failures promptly, clean owned workers, then allow safe continuation."""
    folder.mkdir(parents=True, exist_ok=True)
    _snapshot(folder / 'npu_before.txt')
    with ExitStack() as stack:
        log = stack.enter_context((folder / 'run.log').open('w', encoding='utf-8'))
        process = stack.enter_context(subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        # The group callback runs before Popen's context can wait for a live child.
        stack.callback(_stop_group, process.pid)
        (folder / 'launcher.pid').write_text(f'{process.pid}\n')
        monitor = None
        peer_failed = False
        try:
            monitor_log = stack.enter_context((folder / 'monitor.log').open('w', encoding='utf-8'))
            monitor_command = [sys.executable, '-m',
                               'examples.training_demo.deepseek_v41.monitor_deepseek_v41_hbm',
                               '--root-pid', str(process.pid), '--num-devices', str(cards),
                               '--interval', str(interval), '--output', str(folder)]
            monitor = stack.enter_context(subprocess.Popen(
                monitor_command, stdout=monitor_log, stderr=subprocess.STDOUT))
            stack.callback(monitor.kill)
            (folder / 'monitor.pid').write_text(f'{monitor.pid}\n')
            while process.poll() is None:
                failures = [peer / 'exit_status.json' for peer in peers]
                if any(path.exists() and json.loads(path.read_text())['training_exit_status'] != 0
                       for path in failures):
                    peer_failed = True
                    break
                time.sleep(0.2)
        finally:
            # Publish failure before waiting for cleanup so peers stop their collectives.
            status = process.poll() if process.poll() is not None else (125 if peer_failed else 130)
            write_json(folder / 'exit_status.json', {'training_exit_status': status})
            (folder / 'exit_status.txt').write_text(f'{status}\n')
            cleanup_ok = _stop_group(process.pid)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cleanup_ok = False
            monitor_status = 127
            if monitor is not None:
                monitor.terminate()
                try:
                    monitor_status = monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor_status = monitor.wait()
            (folder / 'monitor_exit_status.txt').write_text(f'{monitor_status}\n')
            try:
                _snapshot(folder / 'npu_after.txt')
            finally:
                write_json(folder / 'case_complete.json', {
                    'training_exit_status': status, 'monitor_exit_status': monitor_status,
                    'cleanup_ok': cleanup_ok, 'peer_failed': peer_failed,
                })
    return wait_for_cleanup(peers, timeout)


def main() -> None:
    """Run a case without importing the training framework in its controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--packing-mode', choices=('dynamic', 'packed'), required=True)
    parser.add_argument('--label', choices=('native', 'moe_only', 'push', 'pull'), required=True)
    parser.add_argument('--node-rank', type=int, required=True)
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--cards-per-node', type=int, required=True)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--sync-timeout', type=float, default=600)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if (not command or not 0 <= args.node_rank < args.nodes or
            min(args.cards_per_node, args.interval, args.sync_timeout) <= 0):
        parser.error('Invalid command, topology, or timeout')
    peers = [args.run_root / args.packing_mode / 'nodes' / f'node_{node}' / args.label
             for node in range(args.nodes)]

    def interrupted(signum: int, _frame: Any) -> None:
        """Unwind through owned-worker cleanup when the allocation stops."""
        raise InterruptedError(f'Case controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    records = run_case(command, peers[args.node_rank], peers, args.cards_per_node, args.interval, args.sync_timeout)
    print(json.dumps({'all_nodes_cleaned': True, 'node_statuses': records}), flush=True)


if __name__ == '__main__':
    main()
