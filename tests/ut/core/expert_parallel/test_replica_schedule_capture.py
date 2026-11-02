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
"""Guard small schedule captures, literal Pull domains and all-rank graph comparisons."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tests.torch.expert_parallel.replica_cluster_fixed_probe import FixedPlanProbe
from hyper_parallel.core.expert_parallel.hot_replica import routing
from hyper_parallel.core.multicore.modules.mega_moe import function as kernels
from tests.torch.expert_parallel.replica_schedule_capture import (
    GRAPH_FILES, install_graph, schedule_cases, verify_pair, weight_submission_summary,
)


class TestReplicaScheduleCapture(unittest.TestCase):
    """Keep source controls and weight lifecycle explicit before any cluster launch."""

    def test_small_matrix_is_literal_cost_off_and_forces_normal_prefetch_miss(self) -> None:
        """Five adjacent graph pairs include the middle regression and diagnostic Pull copies."""
        cases = schedule_cases('schedule', 'AB')
        self.assertEqual(len(cases), 10)
        self.assertEqual(len({row['label'] for row in cases}), 10)
        self.assertEqual(sum(row['diagnostic_pull'] for row in cases), 4)
        for first, second in zip(cases[::2], cases[1::2]):
            self.assertEqual((first['graph'], second['graph']), ('original', 'suffix'))
            self.assertEqual(first['pair'], second['pair'])
            self.assertEqual(first['arguments'], second['arguments'])
            self.assertEqual(first['cost'], 0)
            self.assertIn('--force-residency-miss', first['arguments'])
            self.assertNotIn('--dynamic-plan', first['arguments'])
        with self.assertRaises(ValueError):
            schedule_cases('schedule', 'ABBA')

    def test_pull_capacity_guard_uses_the_declared_none_reservation(self) -> None:
        """A legal Pull fixture must not inherit the Push capacity-factor override."""
        probe = FixedPlanProbe.__new__(FixedPlanProbe)
        probe.fixture = {'model_configuration': {'backend': 'pull'}}
        probe.tokens, probe.top_k, probe.allowance_calls = 4096, 6, 0
        probe.original_options = dict(target_load=None, receive_capacity=None)
        self.assertEqual(probe.capacity_allowance(4096, 6, None, None, None), (None, None))
        with self.assertRaisesRegex(ValueError, 'capacity input'):
            probe.capacity_allowance(4096, 6, None, None, 1.25)
        probe.fixture['model_configuration']['backend'] = 'push'
        probe.original_options = dict(target_load=30720, receive_capacity=38400)
        self.assertEqual(probe.capacity_allowance(4096, 6, None, None, 1.25), (30720, 38400))
        with self.assertRaisesRegex(ValueError, 'capacity input'):
            probe.capacity_allowance(4096, 6, None, None, None)

    def test_uninstrumented_binding_leaves_upload_and_launch_helpers_untouched(self) -> None:
        """Pure samples retain the literal-plan guard without installing timing observation hooks."""
        probe = FixedPlanProbe.__new__(FixedPlanProbe)
        probe.record_observations = True
        original_plan = routing._plan_replica_route
        original_upload = routing._upload_route_metadata
        original_launch = kernels._launch_forward_kernel
        with probe.installed(observations=False):
            self.assertFalse(probe.record_observations)
            self.assertEqual(routing._plan_replica_route, probe.select)
            self.assertIs(routing._upload_route_metadata, original_upload)
            self.assertIs(kernels._launch_forward_kernel, original_launch)
        self.assertTrue(probe.record_observations)
        self.assertIs(routing._plan_replica_route, original_plan)

    def test_weight_summary_uses_actual_submission_records_and_saved_lease_phase(self) -> None:
        """Recompute misses and saved-generation reuse are separate from declared fixture bytes."""
        events = [dict(kind='LeaseBegin', lease_id=index, metadata={'phase': phase})
                  for index, phase in enumerate(('forward', 'recompute', 'backward'))]
        events += [dict(kind='WeightPrefetchSubmit', lease_id=index, metadata={'bytes': 64})
                   for index in (0, 1)]
        events += [dict(kind='WeightGenerationReuse', lease_id=2, metadata={'reuse_hit': True})]
        facts = weight_submission_summary(events)
        self.assertEqual(facts['forward']['bytes'], 64)
        self.assertEqual(facts['recompute']['bytes'], 64)
        self.assertEqual(facts['backward'], dict(count=0, bytes=0, generation_reuse_hits=1))
        with self.assertRaisesRegex(RuntimeError, 'no recorded lease'):
            weight_submission_summary([dict(kind='WeightPrefetchSubmit', lease_id=10, metadata={'bytes': 64})])

    def test_graph_install_checks_all_hashes_before_mutating_the_local_clone(self) -> None:
        """No failed control checksum may leave a partially installed graph."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, source = root / 'bundle', root / 'local/source'
            hashes = {}
            for relative in GRAPH_FILES:
                path = bundle / 'schedule-controls/suffix' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = bundle / 'schedule-controls/manifest.json'
            manifest.write_text(json.dumps({'suffix': dict(source_commit='test', files=hashes)}))
            receipt = install_graph(source, bundle, 'suffix')
            self.assertEqual(receipt['files'], hashes)
            source_paths = {relative: (source / relative).read_bytes() for relative in GRAPH_FILES}
            (bundle / 'schedule-controls/suffix' / GRAPH_FILES[-1]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                install_graph(source, bundle, 'suffix')
            self.assertEqual({relative: (source / relative).read_bytes() for relative in GRAPH_FILES}, source_paths)
            with self.assertRaisesRegex(ValueError, 'node-local'):
                install_graph(bundle / 'source', bundle, 'suffix')

    def test_pair_gate_checks_nonleader_training_hashes_and_weight_lifecycle(self) -> None:
        """A rank-one error or an omitted forced-miss control stops the whole pair."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ('original', 'suffix'):
                folder = root / f'schedule-hot-{variant}'
                folder.mkdir()
                for rank in (0, 1):
                    row = dict(rank=rank, terminal_status=0, tensor_hashes_exact=True,
                               fixed_route_sha256='route', actual_plan_sha256='plan',
                               tensor_sha256={'gradient': 'same'}, native_fingerprint='native',
                               ep_group_global_ranks=[0, 1], fixture_file_sha256='fixture',
                               args={'force_residency_miss': True, 'hidden': 2, 'intermediate': 1},
                               pure_timing=dict(observations=False, profiler=False, warmup=6, iterations=8,
                                    samples=[dict(sample_index=index, local_step_ms=1.) for index in range(8)]),
                               fixed_plan_replay={'transfers': 1},
                               observations=dict(dropped_records=0, host_dropped=0,
                                    weight_submissions={phase: dict(count=2 if rank == 0 and phase != 'backward' else 0,
                                        bytes=12 if rank == 0 and phase != 'backward' else 0,
                                        generation_reuse_hits=int(phase == 'backward'))
                                        for phase in ('forward', 'recompute', 'backward')}))
                    (folder / f'rank{rank}.json').write_text(json.dumps(row))
            self.assertEqual(verify_pair(root, 'hot', 2)['ranks'], 2)
            changed = root / 'schedule-hot-suffix/rank1.json'
            record = json.loads(changed.read_text())
            record['tensor_sha256'] = {'gradient': 'different'}
            changed.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, 'training bytes'):
                verify_pair(root, 'hot', 2)
            record['tensor_sha256'] = {'gradient': 'same'}
            record['args']['force_residency_miss'] = False
            changed.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, 'lifecycle'):
                verify_pair(root, 'hot', 2)
            record['args']['force_residency_miss'] = True
            record['observations']['weight_submissions']['recompute']['bytes'] = 0
            changed.write_text(json.dumps(record))
            self.assertEqual(verify_pair(root, 'hot', 2)['ranks'], 2)
            owner = root / 'schedule-hot-suffix/rank0.json'
            record = json.loads(owner.read_text())
            record['observations']['weight_submissions']['recompute']['bytes'] = 0
            owner.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, 'F/R matrix'):
                verify_pair(root, 'hot', 2)
