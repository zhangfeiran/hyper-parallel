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
"""Framework-free contracts for fresh-process Qwen comparisons."""

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from examples.torch.multicore.mega_moe import qwen_moe_isolated as isolated
from examples.torch.multicore.mega_moe.qwen_moe_config import make_config, optimizer_contract, parse_args


class TestIsolatedQwenLauncher(unittest.TestCase):
    """Keep process ordering, backend ownership, and cancellation explicit."""

    def test_import_does_not_load_any_framework(self) -> None:
        """The parent must not initialize a framework just to launch its workers."""
        code = ('import sys; from examples.torch.multicore.mega_moe import qwen_moe_isolated; '
                'print([x for x in ("torch", "torch_npu", "mindspore", "hyper_parallel") if x in sys.modules])')
        result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), '[]')

    def test_default_order_and_worker_configuration(self) -> None:
        """A/B/B/A must use fresh runs while preserving every model and optimizer option."""
        args = isolated.parse_isolated_args(['--shared-physical-weights', '--replica-slots-per-rank', '1',
                                             '--routing', 'dirichlet', '--router-load-alpha', '0.1',
                                             '--hidden-size', '1024', '--sequence-length', '1024'])
        self.assertEqual(isolated.run_order(args.order), [('A1', 'common'), ('B1', 'mega_moe'),
                                                         ('B2', 'mega_moe'), ('A2', 'common')])
        worker = parse_args(isolated.worker_arguments(args))
        self.assertEqual(make_config(worker), make_config(args))
        self.assertEqual((worker.learning_rate, worker.weight_decay, worker.warmup_steps, worker.measured_steps),
                         (args.learning_rate, args.weight_decay, args.warmup_steps, args.measured_steps))

    def test_parent_rejects_worker_only_flags(self) -> None:
        """A compare parent must not silently replace a requested single-backend workflow."""
        with self.assertRaisesRegex(ValueError, 'worker'):
            isolated.parse_isolated_args(['--backend', 'common', '--snapshot-dir', '/tmp/reference'])

    def test_timeout_terminates_only_the_owned_process_group(self) -> None:
        """Cancellation must reap torchrun and its workers, not unrelated device jobs."""
        child = MagicMock(pid=123456)
        child.__enter__.return_value = child
        child.wait.side_effect = [subprocess.TimeoutExpired(['worker'], 1), 0]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(isolated.subprocess, 'Popen', return_value=child), \
                patch.object(isolated.os, 'killpg') as kill_group:
            with self.assertRaises(subprocess.TimeoutExpired):
                isolated.run_process(['worker'], dict(os.environ), Path(directory) / 'run.log', 1)
            kill_group.assert_called_once_with(child.pid, signal.SIGTERM)
            self.assertEqual(child.wait.call_count, 2)

    def test_failed_child_does_not_succeed(self) -> None:
        """A worker precision failure must propagate to the comparison caller."""
        child = MagicMock(pid=123456)
        child.__enter__.return_value = child
        child.wait.return_value = 3
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(isolated.subprocess, 'Popen', return_value=child), \
                patch.object(isolated.os, 'killpg'), \
                self.assertRaises(subprocess.CalledProcessError):
            isolated.run_process(['worker'], dict(os.environ), Path(directory) / 'run.log', 1)

    def test_nested_launcher_is_rejected_before_spawning(self) -> None:
        """Running the parent under torchrun must not recursively launch eight worlds."""
        with patch.dict(os.environ, {'LOCAL_RANK': '0'}), self.assertRaisesRegex(ValueError, 'torchrun'):
            isolated.launch_topology()


class TestIsolatedQwenResults(unittest.TestCase):
    """Reject stale/incomplete evidence and preserve the fresh-process timing statistic."""

    @staticmethod
    def _result(args, backend: str, export: bool, samples: list[float]) -> dict:
        """Build a minimal valid child record without importing a framework."""
        phases = {}
        for name in ('inputs', 'initial_parameters', 'forward', 'gradients', 'updated_parameters'):
            exact = name in ('inputs', 'initial_parameters')
            phases[name] = {'status': 'reference_saved', 'tensor_count': 1} if export else {
                'passed': True, 'tensor_count': 1, 'rtol': 0 if exact else 0.02, 'atol': 0 if exact else 0.002,
            }
        return {'mode': 'isolated_backend', 'backend': backend, 'configuration': asdict(make_config(args)),
                'initialization_seed': args.seed, 'world_size': 8,
                'warmup_steps': args.warmup_steps, 'measured_steps': args.measured_steps,
                'optimizer': optimizer_contract(args), 'accuracy': {'mode': 'export' if export else 'compare',
                                                                  'phases': phases},
                'measurement': {'validation': {'loss_finite': True, 'gradients_finite': True,
                                               'dense_parameter_updated': True, 'expert_parameter_updated': True},
                                'steady_state_optimizer_step_ms': {'samples': samples},
                                'steady_state_memory_per_rank': [{} for _ in range(8)]},
                'resident_training_per_rank': [{'resident_backend_count': 1} for _ in range(8)],
                'runtime': {'source_sha256': {'model': 'fixed'}}}

    def test_invalid_results_cannot_be_summarized_as_passed(self) -> None:
        """Do not accept relaxed tolerances, missing ranks, wrong seeds, or stale optimizer settings."""
        args = isolated.parse_isolated_args(['--measured-steps', '3'])
        result = self._result(args, 'mega_moe', False, [1.0, 2.0, 3.0])
        isolated._validate_result(result, args, 'mega_moe', False)
        for path, value in ((('initialization_seed',), 43), (('optimizer', 'eps'), 0.01),
                            (('accuracy', 'phases', 'gradients', 'rtol'), 0.2),
                            (('accuracy', 'phases', 'forward', 'passed'), False),
                            (('measurement', 'steady_state_optimizer_step_ms', 'samples'), [float('nan')] * 3),
                            (('resident_training_per_rank',), [{'resident_backend_count': 2}] * 8)):
            modified = copy.deepcopy(result)
            target = modified
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                isolated._validate_result(modified, args, 'mega_moe', False)

    def test_summary_averages_run_medians_and_rejects_source_changes(self) -> None:
        """Samples from different process lifetimes must not be pooled into one median."""
        args = isolated.parse_isolated_args(['--measured-steps', '3'])
        samples = [[1.0, 10.0, 100.0], [5.0] * 3, [7.0] * 3, [20.0] * 3]
        records = [{'label': label, 'result': self._result(args, backend, label == 'A1', values)}
                   for (label, backend), values in zip(isolated.run_order('abba'), samples)]
        summary = isolated._summary(args, records, Path('/tmp/unused'))
        performance = summary['comparison']['performance']
        self.assertEqual((performance['common_mean_ms'], performance['mega_moe_mean_ms']), (15.0, 6.0))
        self.assertEqual(performance['mega_over_common'], 0.4)
        records[-1]['result']['runtime'] = {'source_sha256': {'model': 'changed'}}
        with self.assertRaisesRegex(ValueError, 'identity'):
            isolated._summary(args, records, Path('/tmp/unused'))

    def test_parent_runs_four_jobs_serially_and_retains_evidence(self) -> None:
        """The real parent control flow must export once and compare every subsequent run."""
        calls = []

        def run_fake(command: list[str], _environment: dict, _log: Path, _timeout: int) -> None:
            """Write only the requested child record, as a successful worker would."""
            worker_index = next(index for index, value in enumerate(command) if value.endswith('qwen_moe_benchmark.py'))
            worker = parse_args(command[worker_index + 1:])
            calls.append(worker.backend)
            result = self._result(worker, worker.backend, worker.snapshot_dir is not None,
                                  [2.0] * worker.measured_steps)
            Path(worker.output).write_text(json.dumps(result), encoding='utf-8')

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch.object(isolated, 'run_process', side_effect=run_fake):
            output = Path(directory) / 'summary.json'
            result = isolated.main(['--output', str(output)])
            summary = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result, 0)
            self.assertTrue(Path(summary['artifacts_dir']).is_dir())
            self.assertEqual(calls, ['common', 'mega_moe', 'mega_moe', 'common'])
            self.assertEqual(set(summary['comparison']['accuracy']['runs']), {'B1', 'B2', 'A2'})
