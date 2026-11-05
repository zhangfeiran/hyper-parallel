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
"""Check same-allocation calibration and full Trainer controls without devices."""

import json
import os
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.training_demo.deepseek_v41 import deepseek_v41_replica_e2e as runner


class TestCalibrationE2E(unittest.TestCase):
    """A modeled stage must use published coefficients and prove actual selection."""

    def test_small_matrix_has_adjacent_heuristic_and_model_b1_controls(self) -> None:
        """Collect all three loads with seven Trainer initializations and one graph."""
        stages = runner.stages(calibration_comparison=True)
        self.assertEqual([stage.name for stage in stages], [
            'native_b0', 'push_b0', 'pull_b0', 'push_b1_heuristic', 'push_b1_calibrated',
            'pull_b1_heuristic', 'pull_b1_calibrated'])
        self.assertTrue(all(stage.graph is None for stage in stages))

    def test_partial_catalog_filters_only_unvalidated_modeled_stages(self) -> None:
        """Use the manifest caps and keep every requested control when a profile fails."""
        with TemporaryDirectory() as temp:
            catalog = Path(temp) / 'catalog.json'
            catalog.write_text(json.dumps({'status': 'validated', 'profiles': [{'file': 'pull-b2.json'}]}))
            stages = runner.stages(calibration_comparison=True, calibration_caps=(2, 6))
            selected, skipped = runner.select_calibrated_stages(stages, catalog)
            self.assertEqual([stage.name for stage in selected if stage.policy == 'calibrated'],
                             ['pull_b2_calibrated'])
            self.assertEqual([stage for stage in selected if stage.policy != 'calibrated'],
                             [stage for stage in stages if stage.policy != 'calibrated'])
            self.assertEqual(set(skipped), {'push_b2_calibrated', 'push_b6_calibrated', 'pull_b6_calibrated'})

    def test_modeled_command_has_strict_match_and_control_cannot_inherit_catalog(self) -> None:
        """An inherited catalog cannot turn the heuristic control into a modeled run."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            environment = dict(HYPER_ROOT=str(root), RUN_ROOT=str(root / 'run'), MASTER_PORT_BASE='28600',
                SHMEM_HOST='127.0.0.1', MASTER_ADDR='127.0.0.1', NNODES='8', NODE_RANK='0', NUM_CARDS='16',
                TRAIN_ITERS='35', WARMUP_STEPS='5', MODEL_DIR=str(root), ENGRAM=str(root / 'engram.json'),
                DATA=str(root / 'data.jsonl'), NATIVE_LIB=str(root),
                HYPER_PARALLEL_REPLICA_CALIBRATION_DIR='/inherited/invalid')
            with patch.dict('os.environ', environment, clear=True):
                first = runner.command_for(runner.Stage('push', 1, policy='heuristic'), 3)
                second = runner.command_for(runner.Stage('push', 1, policy='calibrated'), 4)
            self.assertIn('HYPER_PARALLEL_REPLICA_CALIBRATION_DIR=', first)
            self.assertIn('HYPER_PARALLEL_REPLICA_REQUIRE_CALIBRATION=0', first)
            self.assertIn('HYPER_PARALLEL_REPLICA_REQUIRE_CALIBRATION=1', second)
            self.assertIn(f'HYPER_PARALLEL_REPLICA_CALIBRATION_DIR={root}/run/calibration/models', second)
            self.assertIn('--disable-route-diagnostics', second)
            self.assertNotIn('--moe-timing', second)

    def test_all_rank_selection_receipts_are_required(self) -> None:
        """A single rank with fallback or missing evidence invalidates the modeled result."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            stage = runner.Stage('pull', 1, policy='calibrated')
            record = dict(status='matched', backend='pull', hidden_size=5120, intermediate_size=2304,
                          ep_size=64, num_experts=384, top_k=6, checkpointed=True,
                          schedule='fixed_queue_rank_local_guest_communication_suffix_v1',
                          execution_signature='a' * 64, scorer='effective_group', effective_group_ms=[0.] * 6)
            for rank in range(4):
                directory = root / 'pull' / f'rank{rank}'
                directory.mkdir(parents=True)
                (directory / 'calibration-selection.json').write_text(
                    json.dumps(dict(rank=rank, selections=[record])))
            result = runner.calibration_identity(root, stage, 4)
            self.assertEqual(result['matched_ranks'], 4)
            path = root / 'pull/rank3/calibration-selection.json'
            path.write_text(json.dumps(dict(rank=3, selections=[record | {'status': 'fallback'}])))
            with self.assertRaisesRegex(ValueError, 'rank 3'):
                runner.calibration_identity(root, stage, 4)
            path.unlink()
            with self.assertRaises(OSError):
                runner.calibration_identity(root, stage, 4)

    def test_rejected_fit_still_collects_private_verification_without_publication(self) -> None:
        """A rejected table cannot enter training even if its fresh diagnostic capture completes."""
        script = Path(runner.__file__).resolve().parents[3] / 'scripts/replica_autocalibration.py'
        coordinator = runpy.run_path(str(script))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / 'shape.json'
            manifest.write_text(json.dumps(dict(ep_size=2, dispatch_modes=['push'], replica_caps=[1])))
            output = root / 'run'
            calls = []

            def execute(command: list[str], log: Path, *_args: object) -> int:
                phase = log.stem
                calls.append(phase)
                log.write_text('Completed private stage\n')
                if phase == 'fit':
                    models = output / 'models'
                    models.mkdir()
                    (models / 'push-b1.json').write_text('{}')
                    return 20
                folder = Path(command[command.index('--output') + 1])
                folder.mkdir(parents=True, exist_ok=True)
                for rank in range(2):
                    (folder / f'status-rank{rank}.json').write_text(json.dumps(dict(rank=rank, completed=True)))
                return 0

            environment = dict(NNODES='1', NODE_RANK='0', NUM_CARDS='2', MASTER_ADDR='127.0.0.1')
            with (patch.dict(os.environ, environment, clear=True),
                  patch.dict(coordinator['main'].__globals__, {'_run': execute}),
                  patch('sys.argv', ['coordinator', '--manifest', str(manifest), '--output', str(output)])):
                with self.assertRaises(SystemExit) as rejected:
                    coordinator['main']()
            self.assertEqual(rejected.exception.code, 20)
            self.assertEqual(calls, ['collect', 'fit', 'verify'])
            self.assertFalse((output / 'models/catalog.json').exists())
            self.assertTrue(json.loads((output / 'private-verification-only.json').read_text())[
                'private_verification_complete'])

    def test_unidentifiable_design_cannot_start_verification_without_candidates(self) -> None:
        """A valid collection with no fitted model remains quality-rejected rather than missing-file failed."""
        script = Path(runner.__file__).resolve().parents[3] / 'scripts/replica_autocalibration.py'
        coordinator = runpy.run_path(str(script))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / 'shape.json'
            manifest.write_text(json.dumps(dict(ep_size=2, dispatch_modes=['push'], replica_caps=[1])))
            output = root / 'run'
            calls = []

            def execute(command: list[str], log: Path, *_args: object) -> int:
                calls.append(log.stem)
                log.write_text('Completed stage\n')
                if log.stem == 'fit':
                    return 20
                if log.stem == 'verify':
                    raise FileNotFoundError('A private candidate does not exist')
                folder = Path(command[command.index('--output') + 1])
                folder.mkdir(parents=True, exist_ok=True)
                for rank in range(2):
                    (folder / f'status-rank{rank}.json').write_text(json.dumps(dict(rank=rank, completed=True)))
                return 0

            environment = dict(NNODES='1', NODE_RANK='0', NUM_CARDS='2', MASTER_ADDR='127.0.0.1')
            with (patch.dict(os.environ, environment, clear=True),
                  patch.dict(coordinator['main'].__globals__, {'_run': execute}),
                  patch('sys.argv', ['coordinator', '--manifest', str(manifest), '--output', str(output)])):
                with self.assertRaises(SystemExit) as rejected:
                    coordinator['main']()
            self.assertEqual(rejected.exception.code, 20)
            self.assertEqual(calls, ['collect', 'fit'])
            self.assertFalse((output / 'models/catalog.json').exists())
            skipped = json.loads((output / 'private-verification-skipped.json').read_text())
            self.assertFalse(skipped['private_verification_complete'])
            self.assertEqual(skipped['missing_candidates'], ['push-b1.json'])
            self.assertEqual(json.loads((output / 'nodes/node_0/terminal.json').read_text())['exit_status'], 20)


if __name__ == '__main__':
    unittest.main()
