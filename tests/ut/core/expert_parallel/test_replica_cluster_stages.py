# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Check stage coverage, feature gates and cohort failure before any device launch."""

import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe.function import _MegaMoeFunction
from tests.torch.expert_parallel.replica_cluster_observer import Ledger
from tests.torch.expert_parallel.replica_cluster_cases import PHASES, cluster_cases
from tests.torch.expert_parallel.replica_cluster_metadata import device_metadata_gate
from tests.torch.expert_parallel.replica_cluster_stages import (
    gate_identity, gate_passed, publish_gate, wait_cohort, write_json,
)


class TestReplicaClusterStages(unittest.TestCase):
    """Missing feature evidence must never silently enable experimental performance."""

    def test_default_matrix_covers_all_features_and_separates_observation(self) -> None:
        cases = cluster_cases(','.join(PHASES))
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({case['label'] for case in cases}), len(cases))
        self.assertEqual(cases, cluster_cases(','.join(reversed(PHASES))))
        self.assertEqual(cases[0]['phase'], 'fsdp')
        self.assertEqual(cases[-3]['label'], 'gate-d2-pull')
        self.assertTrue(all(case['requires_d2_gate'] for case in cases[-2:]))
        for case in cases:
            if case['phase'] == 'timing':
                self.assertTrue(case['worker'].endswith('._replica_cost_worker'))
                self.assertNotIn('--outer-capture', case['arguments'])
                index = case['arguments'].index('--initial-capacity-factor')
                self.assertEqual(case['arguments'][index + 1], '1.25')
            if case['metadata']:
                self.assertTrue(case['requires_d2_gate'])
                self.assertEqual(case['phase'], 'timing')
            if case['phase'] == 'gate':
                self.assertIn('--feature', case['arguments'])
                self.assertIn('metadata', case['arguments'])

    def test_abba_adds_only_fresh_matching_old_fixed_controls(self) -> None:
        ab = cluster_cases('timing', 'AB')
        abba = cluster_cases('timing', 'ABBA')
        self.assertEqual(len(abba) - len(ab), 12)
        for variant in ('old-a', 'fixed-b', 'fixed-b2', 'old-a2'):
            self.assertEqual(sum(f'timing-{variant}-' in case['label'] for case in abba), 6)
        self.assertEqual(sum(case['host'] == 0 for case in abba), 2)
        self.assertEqual(sum(case['metadata'] == 1 for case in abba), 2)

    def test_causal_literal_and_dynamic_domains_are_distinct(self) -> None:
        cases = cluster_cases('causal')
        self.assertEqual(len(cases), 7)
        for case in cases:
            if '--dynamic-plan' in case['arguments']:
                self.assertEqual(case['cost'], 1)
                self.assertIn('pull', case['arguments'])
            else:
                self.assertEqual(case['cost'], 0)
                self.assertIn('push', case['arguments'])

    def test_invalid_stage_requests_fail_before_framework_launch(self) -> None:
        for phases, order in (('', 'AB'), ('gate,gate', 'AB'), ('unknown', 'AB'), ('timing', 'BA')):
            with self.subTest(phases=phases, order=order), self.assertRaises(ValueError):
                cluster_cases(phases, order)

    def test_gate_requires_all_ranks_device_facts_and_matching_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / 'bundle'
            bundle.mkdir()
            (bundle / 'SHA256SUMS').write_text('immutable manifest\n')
            self.assertFalse(gate_passed(root, bundle))
            required = ('nondefault_streams', 'same_counts_new_ids', 'registry_cleared',
                        'changed_counts_miss', 'changed_policy_miss', 'synthetic_cost_rejected_home')
            record = dict(all_tensors_identical=True, metadata_device_probe=dict.fromkeys(required, True),
                          variants=[{}, {'metadata': [dict(kind='lookup', hit=True)]}])
            for mode in ('push', 'pull'):
                for rank in range(2):
                    write_json(root / f'gate-d2-{mode}' / f'rank{rank}.json', record)
            publish_gate(root, bundle, 2)
            self.assertTrue(gate_passed(root, bundle))
            saved = json.loads((root / 'D2-GATE-PASS.json').read_text())
            self.assertEqual(saved['bundle_sha256'], gate_identity(bundle))
            (bundle / 'SHA256SUMS').write_text('changed manifest\n')
            self.assertFalse(gate_passed(root, bundle))
            record['metadata_device_probe']['changed_counts_miss'] = False
            write_json(root / 'gate-d2-pull/rank1.json', record)
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                publish_gate(root, bundle, 2)

    def test_peer_failure_and_table_mismatch_stop_the_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'nodes/node_0/task-stage.json', dict(status='passed', table_sha256='a'))
            write_json(root / 'nodes/node_1/task-stage.json', dict(status='failed', table_sha256='a'))
            with self.assertRaisesRegex(RuntimeError, 'peer failed'):
                wait_cohort(root, 'task', 2, time.monotonic() + 1)
            write_json(root / 'nodes/node_1/task-stage.json', dict(status='passed', table_sha256='b'))
            with self.assertRaisesRegex(RuntimeError, 'disagree'):
                wait_cohort(root, 'task', 2, time.monotonic() + 1)

    def test_fsdp_observer_preserves_saved_layer_and_heap_epoch(self) -> None:
        ledger = Ledger(True)
        ledger.active_layer = 6
        provider = SimpleNamespace(storage=torch.zeros(8, dtype=torch.uint8))
        weights = torch.zeros(6, 2, 2)
        workspace = SimpleNamespace(replica_provider=provider, heap_manager=SimpleNamespace(epoch=3))
        context = SimpleNamespace()

        def forward(ctx: object, *values: object) -> tuple:
            return 'forward', ledger.active_layer, len(values)

        def backward(ctx: object, gradient: object) -> tuple:
            return 'backward', ledger.active_layer, gradient

        with (patch.object(_MegaMoeFunction, 'forward', staticmethod(forward)),
              patch.object(_MegaMoeFunction, 'backward', staticmethod(backward)), ledger.observe()):
            self.assertEqual(_MegaMoeFunction.forward(
                context, None, weights, weights, None, None, workspace, None, None, False),
                             ('forward', 6, 9))
            ledger.active_layer = None
            self.assertEqual(_MegaMoeFunction.backward(context, 9), ('backward', 6, 9))
            self.assertIsNone(ledger.active_layer)
        self.assertEqual(ledger.rows[0]['heap_epoch'], 3)

    def test_device_probe_input_and_guard_flow_have_a_cpu_oracle(self) -> None:
        """Mock only stream creation; exercise real descriptor packing and remap logic."""
        backend = SimpleNamespace(Stream=lambda: object(), stream=lambda _stream: nullcontext(),
                                  synchronize=lambda: None)
        with patch.object(torch, 'npu', backend, create=True):
            facts = device_metadata_gate(torch.device('cpu'))
        self.assertTrue(facts['same_counts_new_ids'])
        self.assertTrue(facts['changed_counts_miss'])
        self.assertTrue(facts['changed_policy_miss'])
        self.assertTrue(facts['registry_cleared'])
        self.assertTrue(facts['synthetic_cost_rejected_home'])
        example = facts['token_order_counterexample']
        self.assertNotEqual(example['ids_forward'], example['ids_replay'])
        self.assertEqual(example['observed_forward'], example['expected_forward'])
        self.assertEqual(example['observed_replay'], example['expected_replay'])
        self.assertNotEqual(example['observed_forward'], example['observed_replay'])


if __name__ == '__main__':
    unittest.main()
