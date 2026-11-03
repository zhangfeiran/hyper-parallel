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
"""CPU gates for automatic replica model fitting, publication and runtime selection."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hyper_parallel.core.expert_parallel.hot_replica import policy
from hyper_parallel.core.expert_parallel.hot_replica.autocalibration import (
    calibration_ids, fit_directory, fit_nonnegative_services, load_manifest, plan_features,
)
from hyper_parallel.core.expert_parallel.hot_replica.calibration_catalog import external_profiles, runtime_contract
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica.planner import _build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.residency import _ReplicaWeightCache
from hyper_parallel.core.multicore.modules.mega_moe.inactive_tasks import UNMAPPED_GUEST_SUFFIX_SCHEDULE


class TestAutomaticReplicaCalibration(unittest.TestCase):
    """Keep candidate models private until disjoint model-in-loop validation succeeds."""

    @staticmethod
    def manifest(directory: Path) -> dict:
        """Create a small model-owned manifest with the production execution contract."""
        raw = {'hidden_size': 16, 'intermediate_size': 8, 'num_experts': 4, 'top_k': 1, 'ep_size': 2,
               'local_num_tokens': 128, 'checkpointed': True, 'dispatch_modes': ['push'], 'replica_caps': [1]}
        path = directory / 'model.json'
        path.write_text(json.dumps(raw))
        return load_manifest(path)

    def test_manifest_rejects_incompatible_execution(self) -> None:
        """A new model shape cannot silently change precision, replay or ownership contracts."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.manifest(root)
            for key, value in (('ep_size', 3), ('top_k', 5), ('hidden_size', 0), ('checkpointed', False),
                               ('dtype', 'float32'), ('replica_caps', [True]), ('frames', 3)):
                with self.subTest(key=key):
                    payload = manifest | {key: value}
                    (root / 'model.json').write_text(json.dumps(payload))
                    with self.assertRaises(ValueError):
                        load_manifest(root / 'model.json')

    def test_nonnegative_fit_recovers_identifiable_rank_max_services(self) -> None:
        """A known nonnegative max-linear service is recovered without a physical-service claim."""
        rng = np.random.default_rng(32)
        features = rng.random((150, 10))
        coefficients = rng.random(10) * 2
        measured = features @ coefficients
        actual, report = fit_nonnegative_services(features, measured)
        self.assertEqual(report['design_rank'], 10)
        self.assertLess(float(np.linalg.norm(actual - coefficients) / np.linalg.norm(coefficients)), .01)
        self.assertTrue(np.all(actual >= 0))
        with self.assertRaisesRegex(ValueError, 'Unidentifiable'):
            fit_nonnegative_services(np.ones((20, 10)), np.ones(20))
        features[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            fit_nonnegative_services(features, measured)

    def test_actual_ep64_collection_frames_identify_all_caps_and_backends(self) -> None:
        """Noiseless rank-local services recover known coefficients on the actual bounded frame design."""
        manifest = dict(hidden_size=5120, intermediate_size=1792, num_experts=384, top_k=8,
                        ep_size=64, local_num_tokens=4096)
        nodes = tuple(rank // 16 for rank in range(64))
        coefficients = np.array([.5, .001, .8, 1.8, .0001, .3])
        frames = []
        for frame in range(12):
            cached = {}
            counts = []
            for rank in range(64):
                tokens = 4096 // (2 if 4 <= frame < 8 and rank % 2 == 0 else 1)
                if frame % 4 == 3 and rank % 4 == 0:
                    tokens = 0
                if tokens not in cached:
                    ids = calibration_ids(tokens, manifest, frame, nodes)
                    cached[tokens] = tuple(torch.bincount(ids.flatten().long(), minlength=384).tolist())
                counts.append(cached[tokens])
            frames.append(tuple(counts))
        for mode in ('push', 'pull'):
            for cap in (1, 2, 6):
                features = []
                for frame, counts in enumerate(frames):
                    if frame % 3 == 2:
                        continue
                    owner = [sum(sum(row[start:start + 6]) for row in counts) for start in range(0, 384, 6)]
                    for budget in (0, cap):
                        plan = _build_expert_replica_plan(
                            counts, budget, adaptive=bool(budget), capacity_limit=max(owner),
                            target_load=40960 if mode == 'push' else None,
                            receive_capacity=40960 if mode == 'push' else None, minimum_replica_rows=1024,
                            _replica_transfer_row_credit=896, _rank_nodes=nodes)
                        raw = dict(config=asdict(plan.config), source_counts=plan.logical_counts,
                                   slots=plan.slot_to_logical, dispatch=plan.dispatch_counts, rank_nodes=nodes)
                        features.append(plan_features(raw)[0][0][:, [0, 1, 4, 5, 8, 9]])
                design = np.concatenate(features)
                with self.subTest(mode=mode, cap=cap):
                    actual, report = fit_nonnegative_services(design, design @ coefficients)
                    self.assertEqual(report['design_rank'], 6)
                    self.assertLess(float(np.linalg.norm(actual - coefficients) / np.linalg.norm(coefficients)), .02)

    @staticmethod
    def observations(root: Path, manifest: dict, *, harmful: bool = False) -> None:
        """Write complete rank/group evidence from lossless CPU plans and exact effective times."""
        root.mkdir()
        counts = ((100, 0, 0, 0), (0, 0, 0, 0))
        for rank in range(2):
            (root / f'status-rank{rank}.json').write_text(json.dumps({'rank': rank, 'completed': True}))
        for budget in (0, 1):
            plan = _build_expert_replica_plan(counts, budget, target_load=50, capacity_limit=100)
            peak = max(plan.destination_loads)
            records = []
            for frame in range(manifest['frames']):
                for sample in range(manifest['samples']):
                    records.append({'group_leader': 0, 'frame': frame, 'sample': sample,
                        'config': asdict(plan.config), 'source_counts': plan.logical_counts,
                        'slots': plan.slot_to_logical, 'dispatch': plan.dispatch_counts, 'rank_nodes': [0, 1],
                        'rank_phase_ms': [dict(forward=peak, recompute=peak, backward=peak)] * 2,
                        'wall_rank_max_ms': 400 if harmful and budget else peak * 3,
                        'profile_on': False, 'finite': True, 'cost_model_matched': bool(budget)})
            report = {'manifest': manifest, 'backend': 'push', 'budget': budget, 'completed': True,
                'world_size': 2, 'ep_members': [0, 1], 'soc_name': 'cpu-test',
                'execution_signature': 'a' * 64, 'schedule': UNMAPPED_GUEST_SUFFIX_SCHEDULE,
                'runtime_contract': runtime_contract(), 'records': records}
            (root / f'push-b{budget}-group0.json').write_text(json.dumps(report))

    def test_fit_verify_and_atomic_catalog_publication(self) -> None:
        """Only a successful untouched verification exports a runtime-readable catalog."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.manifest(root)
            self.observations(root / 'collect', manifest)
            self.observations(root / 'verify', manifest)
            beta = np.array([0., 3., 0., 0., 0., 0.])
            with patch('hyper_parallel.core.expert_parallel.hot_replica.autocalibration._fit_services',
                       return_value=(beta, {'train_absolute_error_p95_ms': 0.,
                                            'train_group_absolute_error_p95_ms': 0.})):
                candidate = fit_directory(manifest, root / 'collect', root / 'models')
            self.assertEqual(candidate['status'], 'candidate')
            self.assertFalse((root / 'models/catalog.json').exists())
            verified = fit_directory(manifest, root / 'verify', root / 'models', verify=True)
            self.assertEqual(verified['status'], 'validated')
            payloads = external_profiles(root / 'models')
            self.assertEqual(len(payloads), 1)
            self.assertEqual(payloads[0]['validation']['status'], 'passed')
            with self.assertRaisesRegex(ValueError, 'immutable'):
                fit_directory(manifest, root / 'verify', root / 'models', verify=True)

    def test_harmful_holdout_never_publishes(self) -> None:
        """An accurate kernel fit cannot hide a harmful measured component admission."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.manifest(root)
            self.observations(root / 'collect', manifest)
            self.observations(root / 'verify', manifest, harmful=True)
            beta = np.array([0., 3., 0., 0., 0., 0.])
            with patch('hyper_parallel.core.expert_parallel.hot_replica.autocalibration._fit_services',
                       return_value=(beta, {'train_absolute_error_p95_ms': 0.,
                                            'train_group_absolute_error_p95_ms': 0.})):
                fit_directory(manifest, root / 'collect', root / 'models')
            report = fit_directory(manifest, root / 'verify', root / 'models', verify=True)
            self.assertEqual(report['status'], 'failed')
            self.assertTrue(report['profiles'][0]['harmful_false_admissions'])
            self.assertFalse((root / 'models/catalog.json').exists())

    def test_catalog_checksum_environment_and_exact_signature_gates(self) -> None:
        """An external artifact cannot silently load for another execution or edited source."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
                intermediate_size=8, ep_size=2, transport='shmem_signal_kernel_gradient_bands',
                forward_ms=((0, 0.), (100, 1.)), backward_ms=((0, 0.), (100, 2.)),
                weight_ms=1., gradient_ms=1., schedule=UNMAPPED_GUEST_SUFFIX_SCHEDULE)
            payload = {'selector': dict(soc_name='cpu-test', num_experts=4, top_k=1,
                                       execution_signature='a' * 64), 'model': asdict(model),
                       'validation': {'status': 'passed'}}
            raw = json.dumps(payload).encode()
            (root / 'push-b1.json').write_bytes(raw)
            catalog = {'schema_version': 1, 'status': 'validated', 'runtime_contract': runtime_contract(),
                       'profiles': [{'file': 'push-b1.json', 'sha256': hashlib.sha256(raw).hexdigest()}]}
            (root / 'catalog.json').write_text(json.dumps(catalog))
            try:
                with patch.dict(os.environ, {'HYPER_PARALLEL_REPLICA_CALIBRATION_DIR': str(root)}):
                    policy._calibrations.cache_clear()
                    args = ('push', 16, 8, 2, model.transport, 4, 1)
                    self.assertTrue(policy._critical_path_available(*args))
                    self.assertIsNotNone(policy._resolve_cost_model(*args, 'cpu-test', True, 'a' * 64,
                                                                    UNMAPPED_GUEST_SUFFIX_SCHEDULE))
                    self.assertIsNone(policy._resolve_cost_model(*args, 'cpu-test', True, 'b' * 64,
                                                                 UNMAPPED_GUEST_SUFFIX_SCHEDULE))
                changed = deepcopy(catalog)
                changed['runtime_contract']['environment']['HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY'] = 'invalid'
                (root / 'catalog.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                    external_profiles(root)
                (root / 'catalog.json').write_text(json.dumps(catalog))
                (root / 'push-b1.json').write_bytes(raw + b' ')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    external_profiles(root)
            finally:
                for function in (policy._calibrations, policy._calibration_available, policy._critical_path_available,
                                 policy._critical_path_identity, policy._resolve_cost_model):
                    function.cache_clear()

    def test_residency_switch_preserves_default_and_forces_normal_misses(self) -> None:
        """Disabling the optional cross-forward cache does not remove its public lifetime contract."""
        cache = _ReplicaWeightCache()
        weights = (torch.ones(2, 3, 4), torch.ones(2, 4, 2))
        transfer = SimpleNamespace(target_rank=1, target_slot=2, owner_rank=0, owner_slot=0)
        route = SimpleNamespace(rank=0, plan=SimpleNamespace(config=SimpleNamespace(home_experts=2),
                                                            transfers=(transfer,)))
        with patch.dict(os.environ, {'HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY': '1'}):
            cache.commit(cache.prepare(weights, route))
            self.assertTrue(cache.prepare(weights, route).reused)
        with patch.dict(os.environ, {'HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY': '0'}):
            self.assertFalse(cache.prepare(weights, route).reused)
            cache.commit(cache.prepare(weights, route))
            self.assertFalse(cache.prepare(weights, route).reused)

    def test_cluster_dry_run_needs_no_job_id_and_uses_numa_wrapper(self) -> None:
        """Page submissions infer scheduler topology and perform no framework import in the coordinator."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.manifest(root)
            repository = Path(__file__).resolve().parents[4]
            environment = os.environ | {'NNODES': '8', 'NODE_RANK': '0', 'NUM_CARDS': '16', 'MASTER_ADDR': '127.0.0.1'}
            result = subprocess.run([sys.executable, str(repository / 'scripts/replica_autocalibration.py'),
                '--manifest', str(root / 'model.json'), '--output', str(root / 'run'), '--dry-run'],
                check=True, capture_output=True, text=True, env=environment)
            commands = json.loads(result.stdout)['commands']
            self.assertIn('tests.torch.expert_parallel._replica_capture_launcher', commands['collect'])
            self.assertIn('--placement-output', commands['verify'])
            self.assertFalse((root / 'run').exists())
