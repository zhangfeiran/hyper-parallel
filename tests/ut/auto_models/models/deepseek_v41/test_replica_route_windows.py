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
"""Check profile changes, checkpoint destinations and window-specific rank-max timing."""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from examples.training_demo.deepseek_v41 import benchmark_deepseek_v41_schedule as worker
from examples.training_demo.deepseek_v41 import deepseek_v41_replica_e2e as runner
from examples.training_demo.deepseek_v41.deepseek_v41_route_schedule import (
    RouteSchedule, RouteWindow, default_schedule,
)
from examples.training_demo.deepseek_v41.dirichlet_routes import make_route, route_metadata


class TestReplicaRouteWindows(unittest.TestCase):
    """Keep every transition outside the declared performance samples."""

    def test_cpu_control_keeps_step_timing_off_the_training_communicator(self) -> None:
        """Elapsed-time reduction must use the CPU control after synchronized training."""
        control = Mock()
        control.maximum.return_value = 3.5
        trainer = SimpleNamespace(base=SimpleNamespace(step_env_metrics={}, step_train_metrics={}),
                                  train_step=lambda _iterator: {'loss': 1., 'grad_norm': 1.})
        diagnostics = SimpleNamespace(clear=lambda: None, token_rows={}, input_identity=None,
                                      route_window=None, control=control)
        with (TemporaryDirectory() as directory,
              patch.object(worker.benchmark.dist, 'barrier') as barrier,
              patch.object(worker.benchmark.dist, 'all_reduce') as reduce,
              patch.object(torch.npu, 'synchronize'), patch.object(torch.npu, 'memory_allocated', return_value=0),
              patch.object(torch.npu, 'memory_reserved', return_value=0),
              patch.object(torch.npu, 'max_memory_allocated', return_value=0),
              patch.object(torch.npu, 'max_memory_reserved', return_value=0),
              patch.object(torch, 'tensor', return_value=torch.zeros(1))):
            result = worker.benchmark._step(trainer, None, 34, Path(directory), diagnostics, 5)
            self.assertEqual(result['max_rank_seconds'], 3.5)
            barrier.assert_not_called()
            reduce.assert_not_called()
            control.barrier.assert_called_once_with(35)
            control.maximum.assert_called_once_with(result['local_seconds'])
            self.assertEqual(control.mark.call_args_list[-1].args, ('step_written', 35))

    def test_real_step_persists_exactly_the_same_route_labels_as_the_result(self) -> None:
        """A scheduled real benchmark step must not mutate metadata after its JSONL append."""
        module = SimpleNamespace(gate=SimpleNamespace(num_experts=24))
        trainer = SimpleNamespace(base=SimpleNamespace(
            model=SimpleNamespace(named_modules=lambda: [('model.layers.0.mlp', module)]),
            step_env_metrics={}, step_train_metrics={}), train_step=lambda _iterator: {'loss': 1., 'grad_norm': 1.})
        diagnostics = SimpleNamespace(clear=lambda: None, token_rows={}, input_identity=None)
        tensor = torch.tensor

        def cpu_tensor(*args: object, **kwargs: object) -> torch.Tensor:
            kwargs['device'] = 'cpu'
            return tensor(*args, **kwargs)

        with (TemporaryDirectory() as directory,
              patch.object(worker.benchmark.dist, 'barrier'), patch.object(worker.benchmark.dist, 'all_reduce'),
              patch.object(torch, 'tensor', side_effect=cpu_tensor), patch.object(torch.npu, 'synchronize'),
              patch.object(torch.npu, 'memory_allocated', return_value=0),
              patch.object(torch.npu, 'memory_reserved', return_value=0),
              patch.object(torch.npu, 'max_memory_allocated', return_value=0),
              patch.object(torch.npu, 'max_memory_reserved', return_value=0)):
            manager = worker.ScheduledRoutes(default_schedule(), 4)
            result = manager.step(trainer, None, 0, Path(directory), diagnostics, 5)
            saved = json.loads((Path(directory) / 'steps.jsonl').read_text())
            self.assertEqual(saved, result)
            self.assertEqual(saved['routing_window'], 'primer')
            self.assertEqual(saved['routing_profile'], '0.1')

    def test_default_has_three_loads_with_eight_samples_each(self) -> None:
        """One initialized model visits balanced, medium and hot loads in 35 steps."""
        selected = default_schedule()
        self.assertEqual((selected.steps, selected.warmup), (35, 5))
        self.assertEqual([window.profile for window in selected.windows[1:]], ['balanced', '1', '0.1'])
        self.assertEqual([list(window.measured_indices) for window in selected.windows[1:]],
                         [list(range(7, 15)), list(range(17, 25)), list(range(27, 35))])
        self.assertEqual([selected.at(index).name for index in range(5)], ['primer'] * 5)

    def test_forward_replay_share_ids_and_counts_describe_replacement_routes(self) -> None:
        """Different profiles keep learned weights, cached replay IDs and current exact counts."""
        selected = RouteSchedule((RouteWindow('primer', '0.1', 42, 0, 1, 0),
                                  RouteWindow('balanced', 'balanced', 42, 1, 1, 1),
                                  RouteWindow('hot', '0.1', 42, 3, 1, 1)))
        module = SimpleNamespace(gate=SimpleNamespace(num_experts=24))
        trainer = SimpleNamespace(base=SimpleNamespace(model=SimpleNamespace(
            named_modules=lambda: [('model.layers.0.mlp', module)])))
        weights = torch.full((128, 6), 1 / 6, requires_grad=True)
        original = (torch.zeros((128, 6), dtype=torch.int32), weights, torch.zeros(24, dtype=torch.int64))
        identities = []
        manager = None

        def train(*_args: object) -> dict:
            first = manager.router(module, torch.zeros(128, 8))
            replay = manager.router(module, torch.zeros(128, 8))
            self.assertIs(first[1], weights)
            self.assertTrue(torch.equal(first[0], replay[0]))
            torch.testing.assert_close(torch.bincount(first[0].flatten().long(), minlength=24), first[2])
            identities.append(first[0])
            return {'step': len(identities)}

        with (patch.object(worker.adapter, '_router', return_value=original),
              patch.object(worker.benchmark, '_step', side_effect=train)):
            manager = worker.ScheduledRoutes(selected, 4)
            for index in range(selected.steps):
                manager.step(trainer, None, index, Path('/unused'))
        self.assertTrue(torch.equal(identities[0], identities[-1]))
        self.assertFalse(torch.equal(identities[0], identities[1]))
        self.assertEqual([entry['observed_steps'] for entry in manager.summary()['windows']], [[0], [1, 2], [3, 4]])
        self.assertEqual(manager.next_step, 5)

    def test_window_medians_exclude_primer_and_transition_outliers(self) -> None:
        """Read two rank reports and retain only each window's eight declared samples."""
        selected = default_schedule()
        ids, counts = make_route(4096, 384, 6, None, 42)
        metadata = route_metadata(ids, counts, 64)
        record = dict(schedule=selected.record(), schedule_sha256=selected.sha256,
                      completed_steps=35, shared_source_profile=True, learned_weights_preserved=True,
                      synthetic_performance_only=True, plan_observation_hooks=False,
                      windows=[dict(window=asdict(window), observed_steps=list(range(window.start, window.end)),
                                    layers={str(layer): metadata for layer in range(8)})
                               for window in selected.windows])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            schedule_path = root / 'schedule.json'
            schedule_path.write_text(json.dumps(selected.record()))
            for rank in range(2):
                folder = root / f'push/rank{rank}'
                folder.mkdir(parents=True)
                rows = []
                for index in range(selected.steps):
                    window = selected.at(index)
                    seconds = 100. if index not in window.measured_indices else (
                        index - window.measured_indices.start + 1 + rank * 2)
                    rows.append(dict(local_seconds=seconds, routing_window=window.name,
                                     routing_profile=window.profile))
                (folder / 'result.json').write_text(json.dumps(dict(steps_detail=rows)))
                (folder / 'routing-windows.json').write_text(json.dumps(record))
            with patch.dict('os.environ', {'DSV41_ROUTE_SCHEDULE': str(schedule_path)}):
                result = runner._route_windows(root, runner.Stage('push', 1), 1, 2)
                self.assertEqual(set(result), {'balanced', 'medium', 'hot'})
                self.assertTrue(all(row['median_seconds'] == 6.5 for row in result.values()))
                self.assertTrue(all(row['rank_max_seconds'] == list(range(3, 11)) for row in result.values()))
                reference = runner.validate_route_windows(record, selected)
                profiles = {f'{window.profile}/s{window.seed}': reference[window.name]
                            for window in selected.windows}
                identity_path = root / 'identities.json'
                identity_path.write_text(json.dumps(dict(schedule_sha256=selected.sha256, profiles=profiles)))
                with patch.dict('os.environ', {'DSV41_ROUTE_PROFILE_IDENTITIES': str(identity_path)}):
                    self.assertEqual(runner._route_windows(root, runner.Stage('push', 1), 1, 2), result)
                    profiles['balanced/s42']['routing_sha256'] = '0' * 64
                    identity_path.write_text(json.dumps(dict(schedule_sha256=selected.sha256, profiles=profiles)))
                    with self.assertRaisesRegex(ValueError, 'Frozen controlled'):
                        runner._route_windows(root, runner.Stage('push', 1), 1, 2)
                bad = json.loads((folder / 'routing-windows.json').read_text())
                bad['windows'][1]['observed_steps'] = [5]
                (folder / 'routing-windows.json').write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'window'):
                    runner._route_windows(root, runner.Stage('push', 1), 1, 2)
