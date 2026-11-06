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
"""Verify full Trainer graph switching and median comparisons without devices."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.training_demo.deepseek_v41 import deepseek_v41_replica_e2e as runner
from examples.training_demo.deepseek_v41.replica_graph_controls import GRAPH_FILES


class TestReplicaScheduleE2E(unittest.TestCase):
    """Keep matched graph controls outside the measured Trainer process."""

    @staticmethod
    def _environment(root: Path) -> dict[str, str]:
        return dict(HYPER_ROOT=str(root / 'local/source'), BUNDLE_ROOT=str(root / 'bundle'),
                    RUN_ROOT=str(root / 'run'), MASTER_PORT_BASE='28600', SHMEM_HOST='127.0.0.1',
                    MASTER_ADDR='127.0.0.1', NNODES='8', NODE_RANK='0', NUM_CARDS='16',
                    TRAIN_ITERS='20', WARMUP_STEPS='5', MODEL_DIR=str(root), ENGRAM=str(root / 'engram.json'),
                    DATA=str(root / 'data.jsonl'), NATIVE_LIB=str(root), REPLICA_SCHEDULE_E2E='1')

    @staticmethod
    def _bundle(root: Path) -> None:
        manifest = {}
        for variant in ('original', 'suffix'):
            hashes = {}
            for relative in GRAPH_FILES:
                path = root / 'bundle/schedule-controls' / variant / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(variant)
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest[variant] = dict(source_commit=variant, files=hashes)
        (root / 'bundle/schedule-controls/manifest.json').write_text(json.dumps(manifest))
        (root / 'run').mkdir()

    def test_controller_installs_each_graph_before_the_fresh_training_process(self) -> None:
        """The first B0 control and each adjacent pair receive their declared source bytes."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            observed = []

            def train(command: list[str], folder: Path, *unused: object) -> list[dict]:
                del unused
                receipt = json.loads((folder / 'graph-control.json').read_text())
                variant = receipt['source_commit']
                self.assertTrue(all((root / 'local/source' / relative).read_text() == variant
                                    for relative in GRAPH_FILES))
                self.assertIn('HYPER_PARALLEL_REPLICA_COST_MODEL=1', command)
                self.assertIn('--disable-route-diagnostics', command)
                self.assertNotIn('--force-residency-miss', command)
                observed.append(variant)
                return [dict(training_exit_status=0, monitor_exit_status=0, cleanup_ok=True, peer_failed=False)] * 8

            with (patch.dict('os.environ', self._environment(root), clear=True),
                  patch.object(sys, 'argv', ['controller']), patch.object(runner.signal, 'signal'),
                  patch.object(runner, 'run_case', side_effect=train),
                  patch.object(runner, 'publish', return_value={'complete': True})):
                runner.main()
            self.assertEqual(observed, ['suffix'] * 3 + ['original', 'suffix'] * 4)

    def test_nonleader_graph_receipt_is_required_and_checked(self) -> None:
        """A missing or different node-seven source invalidates the full-model comparison."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            stage = runner.Stage('push', 6, 'suffix')
            folder = root / 'run/cases' / stage.name / 'packed'
            expected = json.loads((root / 'bundle/schedule-controls/manifest.json').read_text())['suffix']
            for node in range(8):
                path = folder / f'nodes/node_{node}/push/graph-control.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(expected))
            with patch.dict('os.environ', self._environment(root), clear=True):
                self.assertEqual(runner._graph_identity(folder, stage, 8)['source_commit'], 'suffix')
                path.write_text(json.dumps(dict(expected, source_commit='original')))
                with self.assertRaisesRegex(ValueError, 'node 7'):
                    runner._graph_identity(folder, stage, 8)

    def test_graph_install_failure_is_published_before_any_worker_starts(self) -> None:
        """A preparation failure must be visible to peers monitoring this stage."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            with (patch.dict('os.environ', self._environment(root), clear=True),
                  patch.object(sys, 'argv', ['controller']), patch.object(runner.signal, 'signal'),
                  patch.object(runner, 'install_graph', side_effect=ValueError('checksum mismatch')),
                  patch.object(runner, 'run_case') as run,
                  self.assertRaisesRegex(ValueError, 'checksum')):
                runner.main()
            run.assert_not_called()
            folder = root / 'run/cases/native_b0/packed/nodes/node_0/native'
            self.assertEqual(json.loads((folder / 'exit_status.json').read_text())['training_exit_status'], 1)
            self.assertTrue(json.loads((folder / 'case_complete.json').read_text())['cleanup_ok'])

    def test_parent_controller_imports_no_framework(self) -> None:
        """Repeated source switches happen in a parent with no imported runtime modules."""
        command = [sys.executable, '-c',
                   'import sys; from examples.training_demo.deepseek_v41 import deepseek_v41_replica_e2e; '
                   'assert not {"torch", "torch_npu", "hyper_parallel"}.intersection(sys.modules)']
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, f'{completed.stderr=}')
