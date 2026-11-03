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
import runpy
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hyper_parallel.core.expert_parallel.hot_replica import policy, autocalibration
from hyper_parallel.core.expert_parallel.hot_replica.autocalibration import (
    MEASUREMENT_CONTRACT, calibration_ids, fit_directory, fit_nonnegative_services, load_manifest, plan_features,
)
from hyper_parallel.core.expert_parallel.hot_replica.calibration_catalog import external_profiles, runtime_contract
from hyper_parallel.core.expert_parallel.hot_replica.cost import (
    ExpertReplicaCostModel, _QuotaCostState, _effective_group_features,
)
from hyper_parallel.core.expert_parallel.hot_replica.planner import _build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.residency import _ReplicaWeightCache
from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state
from hyper_parallel.core.multicore.modules.mega_moe import module as mega
from hyper_parallel.core.multicore.modules.mega_moe.inactive_tasks import UNMAPPED_GUEST_SUFFIX_SCHEDULE
from tests.torch.expert_parallel._replica_autocalibration_worker import _checkpoint_operation


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
                               ('dtype', 'float32'), ('replica_caps', [True]), ('frames', 3),
                               ('checkpoint_contract', 'unknown'), ('measurement_contract', 'legacy')):
                with self.subTest(key=key):
                    payload = manifest | {key: value}
                    (root / 'model.json').write_text(json.dumps(payload))
                    with self.assertRaises(ValueError):
                        load_manifest(root / 'model.json')

    def test_checkpoint_contract_matches_grad_modes_rng_and_execution_signature(self) -> None:
        """Reentrant PanGu and nonreentrant Hyper replay retain distinct stable F/R signatures."""
        module = SimpleNamespace(dispatch_mode='push', hidden_size=16, intermediate_size=8, ep_size=2,
            num_experts=4, top_k=1, replica_transport='shmem_signal_kernel_gradient_bands', replica_slots_per_rank=1)
        resources = SimpleNamespace(rank_nodes=(0, 0), spec=SimpleNamespace(expert_weight_layout='EIO',
            swiglu_limit=None, dispatch_split=1, combine_split=1, num_cube_cores=24))
        signatures = {}
        for contract in ('nonreentrant', 'reentrant'):
            with self.subTest(contract=contract):
                phases = []
                hidden, probabilities = torch.tensor([.5, 1.], requires_grad=True), torch.ones(2, requires_grad=True)
                torch.manual_seed(7)
                baseline = (hidden * probabilities * torch.rand_like(hidden)).sin()
                expected = torch.autograd.grad(baseline.sum(), (hidden, probabilities))

                def operation(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
                    state = get_recompute_state()
                    mode = mega._checkpoint_contract()
                    with torch.no_grad():
                        signature = mega._critical_execution_signature(module, resources, execution_tokens=128)
                    phases.append((state.is_recomputing, torch.is_grad_enabled(), mode,
                                   state.invocation_id, signature))
                    return (values * weights * torch.rand_like(values)).sin()

                torch.manual_seed(7)
                with patch.object(mega, '_critical_native_fingerprint', return_value='a' * 64):
                    result = _checkpoint_operation(operation, hidden, probabilities, contract=contract)
                    result.sum().backward()
                self.assertEqual([(row[0], row[1], row[2]) for row in phases],
                    [(False, contract == 'nonreentrant', contract), (True, True, contract)])
                self.assertIs(phases[0][3], phases[1][3])
                self.assertEqual(phases[0][4], phases[1][4])
                self.assertTrue(torch.equal(result, baseline))
                self.assertTrue(torch.equal(hidden.grad, expected[0]))
                self.assertTrue(torch.equal(probabilities.grad, expected[1]))
                self.assertIsNone(get_recompute_state())
                signatures[contract] = phases[0][4]
        self.assertNotEqual(signatures['nonreentrant'], signatures['reentrant'])

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
                        'measurement_contract': MEASUREMENT_CONTRACT,
                        'config': asdict(plan.config), 'source_counts': plan.logical_counts,
                        'slots': plan.slot_to_logical, 'dispatch': plan.dispatch_counts, 'rank_nodes': [0, 1],
                        'rank_phase_ms': [dict(forward=peak, recompute=peak, backward=peak)] * 2,
                        'rank_component_ms': [400 if harmful and budget else peak * 3] * 2,
                        'wall_rank_max_ms': 400 if harmful and budget else peak * 3,
                        'profile_on': False, 'finite': True, 'cost_model_matched': bool(budget)})
            report = {'manifest': manifest, 'backend': 'push', 'budget': budget, 'completed': True,
                'rank_nodes': [0, 1],
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
            self.assertTrue(report['collection_complete'])
            self.assertEqual(report['calibration_status'], 'rejected')
            self.assertFalse(report['catalog_published'])
            self.assertTrue(report['profiles'][0]['harmful_false_admissions'])
            self.assertFalse((root / 'models/catalog.json').exists())

    def test_cli_distinguishes_quality_rejection_from_runtime_failure(self) -> None:
        """A completed capture rejected by quality gates has a dedicated launcher outcome."""
        with (patch.object(sys, 'argv', ['autocalibration', '--manifest', '/unused/model.json',
                '--input', '/unused/collect', '--output', '/unused/models']),
              patch.object(autocalibration, 'load_manifest', return_value={}),
              patch.object(autocalibration, 'fit_directory', return_value={'status': 'failed'}),
              patch('builtins.print')):
            with self.assertRaises(SystemExit) as error:
                autocalibration.main()
            self.assertEqual(error.exception.code, 20)

    def test_effective_group_score_reaches_planner_and_quota_trials(self) -> None:
        """A topology coefficient changes real admission, including the exact Python quota scorer."""
        counts = ((100, 0, 0, 0), (0, 0, 0, 0))
        options = dict(adaptive=True, target_load=50, receive_capacity=100, capacity_limit=100,
                       _rank_nodes=(0, 1))
        models = [ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
            intermediate_size=8, ep_size=2, transport='shmem_signal_kernel_gradient_bands',
            forward_ms=((0, 0.), (200, 200 / 3)), backward_ms=((0, 0.), (200, 200 / 3)),
            weight_ms=0., gradient_ms=0., forward_repeats=2,
            effective_group_ms=(0., 1., 0., penalty, 0., 0.), _rank_nodes=(0, 1)) for penalty in (10., 100.)]
        with patch('hyper_parallel.core.expert_parallel.hot_replica.cost._rank_compute',
                   side_effect=AssertionError('Effective group scoring cannot use the legacy native scorer')):
            cheap = _build_expert_replica_plan(counts, 1, cost_model=models[0], **options)
            expensive = _build_expert_replica_plan(counts, 1, cost_model=models[1], **options)
        cheap.validate()
        expensive.validate()
        self.assertEqual(len(cheap.transfers), 1)
        self.assertEqual(len(expensive.transfers), 0)
        self.assertLess(models[0].estimate_ms(cheap), models[0].estimate_ms(expensive))
        copies = [{expert: rows for expert, rows in zip(slots, rows_by_slot) if expert >= 0}
                  for slots, rows_by_slot in zip(cheap.slot_to_logical, cheap.destination_counts)]
        state = _QuotaCostState(models[0], copies, cheap.logical_counts, cheap.config)
        for expert in list(copies[1]):
            if expert // cheap.config.home_experts == 0:
                copies[0][expert] += copies[1].pop(expert)
        self.assertEqual(state.trial_score(0, 1, 0, (0., 0.)), models[0].estimate_ms(expensive))

    def test_group_features_match_extracted_plan_and_require_topology(self) -> None:
        """Offline and runtime features have identical task, row, link and token definitions."""
        counts = ((100, 5, 0, 0), (0, 0, 5, 0))
        plan = _build_expert_replica_plan(counts, 1, target_load=55, capacity_limit=105, _rank_nodes=(0, 1))
        raw = dict(config=asdict(plan.config), source_counts=plan.logical_counts,
                   slots=plan.slot_to_logical, dispatch=plan.dispatch_counts, rank_nodes=(0, 1))
        copies = [{expert: count for expert, count in zip(slots, values) if expert >= 0}
                  for slots, values in zip(plan.slot_to_logical, plan.destination_counts)]
        expected = plan_features(raw)[0][0][:, [0, 1, 4, 5, 8, 9]].max(axis=0)
        self.assertTrue(np.array_equal(_effective_group_features(copies, plan.logical_counts, 2, (0, 1)), expected))
        with self.assertRaisesRegex(ValueError, 'measured EP topology'):
            _effective_group_features(copies, plan.logical_counts, 2, ())

    def test_cumulative_measurement_contract_retains_phase_max_diagnostic(self) -> None:
        """Different critical ranks cannot turn the phase-max sum into an elapsed component duration."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.manifest(root)
            self.observations(root / 'collect', manifest)
            report = json.loads((root / 'collect/push-b1-group0.json').read_text())
            row = report['records'][0]
            row['rank_phase_ms'] = [dict(forward=100., recompute=10., backward=10.),
                                    dict(forward=10., recompute=100., backward=10.)]
            row['rank_component_ms'] = [121., 121.]
            sample = autocalibration._samples([report], 1)[0]
            self.assertEqual(sample['observed_ms'], 120.)
            self.assertEqual(sample['component_rank_max_ms'], 121.)
            self.assertEqual(sample['phase_rank_max_sum_ms'], 210.)
            del row['measurement_contract']
            with self.assertRaisesRegex(ValueError, 'fresh rank-cumulative'):
                autocalibration._samples([report], 1)

    def test_group_training_cv_identifies_services_without_holdout(self) -> None:
        """Complete route families, including all repeat samples, stay in one TRAIN-only fold."""
        rng = np.random.default_rng(17)
        beta = np.array([.3, .8, .6, 1.2, .1, .5])
        rows = []
        for frame in range(12):
            if frame % 3 == 2:
                continue
            for _ in range(3):
                features = rng.uniform(.1, 4., (2, 4, 10))
                observed = float(features[0][:, [0, 1, 4, 5, 8, 9]].max(axis=0) @ beta)
                rows.append(dict(frame=frame, features=features, observed_ms=observed, rank_nodes=(0, 0, 1, 1)))
        actual, report = autocalibration._fit_services(rows)
        self.assertLess(float(np.linalg.norm(actual - beta)), .001)
        self.assertLess(report['cross_validation']['relative_error_p95'], .001)
        self.assertEqual(report['measurement_contract'], MEASUREMENT_CONTRACT)
        with self.assertRaisesRegex(ValueError, 'TRAIN frames only'):
            autocalibration._fit_services(rows + [rows[0] | {'frame': 2}])

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
            fallback = environment | {'NNODES': '', 'NODE_RANK': '', 'MASTER_ADDR': '', 'MA_NUM_HOSTS': '',
                'RANK_AFTER_ACC': '', 'VC_TASK_INDEX': '0', 'MASTER_PORT_BASE': '', 'NUM_CARDS': '',
                'CALIBRATION_PHASE_TIMEOUT_SECONDS': '',
                'VC_WORKER_HOSTS': ','.join(f'127.0.0.{index}:8000' for index in range(1, 9)),
                'HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT': 'tcp://stale-host:1'}
            for active in (environment, fallback):
                with self.subTest(empty_scheduler_fields=not bool(active['NNODES'])):
                    result = subprocess.run([sys.executable, str(repository / 'scripts/replica_autocalibration.py'),
                        '--manifest', str(root / 'model.json'), '--output', str(root / 'run'), '--dry-run'],
                        check=True, capture_output=True, text=True, env=active)
                    payload = json.loads(result.stdout)
                    commands = payload['commands']
                    self.assertIn('tests.torch.expert_parallel._replica_capture_launcher', commands['collect'])
                    self.assertIn('--placement-output', commands['verify'])
                    self.assertEqual(payload['nodes'], 8)
                    self.assertEqual(payload['node_rank'], 0)
                    self.assertEqual(payload['cards_per_node'], 16)
                    for phase, port in (('collect', 29301), ('verify', 29303)):
                        self.assertEqual(payload['stage_environment'][phase]
                            ['HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT'], f'tcp://127.0.0.1:{port}')
            self.assertFalse((root / 'run').exists())

    def test_cluster_dry_run_bounds_long_scheduler_hostname_without_dns(self) -> None:
        """A long allocation identity must not become an overlong native SHMEM endpoint."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.manifest(root)
            repository = Path(__file__).resolve().parents[4]
            address = 'worker-0.' + 'a' * 60 + '.allocation.invalid'
            environment = os.environ | {'NNODES': '8', 'NODE_RANK': '0', 'NUM_CARDS': '16',
                                         'MASTER_ADDR': address, 'MASTER_PORT_BASE': '29300'}
            result = subprocess.run([sys.executable, str(repository / 'scripts/replica_autocalibration.py'),
                '--manifest', str(root / 'model.json'), '--output', str(root / 'run'), '--dry-run'],
                check=True, capture_output=True, text=True, env=environment)
            payload = json.loads(result.stdout)
            for phase in ('collect', 'verify'):
                endpoint = payload['stage_environment'][phase]['HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT']
                self.assertLessEqual(len(endpoint.encode()), 63)
                self.assertIn(f'--master-addr={address}', payload['commands'][phase])
            self.assertFalse((root / 'run').exists())

    def test_cluster_endpoint_resolves_only_the_shmem_host_and_checks_native_limit(self) -> None:
        """Both phase endpoints use the resolved IPv4 without changing the allocation hostname."""
        repository = Path(__file__).resolve().parents[4]
        coordinator = runpy.run_path(str(repository / 'scripts/replica_autocalibration.py'))
        address = 'worker-0.' + 'a' * 60 + '.allocation.invalid'
        with patch('socket.gethostbyname', return_value='192.0.2.100') as resolve:
            host = coordinator['_shmem_host'](address, dry_run=False)
            resolve.assert_called_once_with(address)
        for port in (29301, 29303):
            endpoint = coordinator['_shmem_endpoint'](host, port)
            self.assertEqual(endpoint, f'tcp://192.0.2.100:{port}')
        with patch('socket.gethostbyname', side_effect=AssertionError('Dry run must not perform DNS')):
            self.assertEqual(coordinator['_shmem_host'](address, dry_run=True), 'resolved-ipv4-at-launch')
            self.assertEqual(coordinator['_shmem_host']('192.0.2.100', dry_run=False), '192.0.2.100')
        with patch('socket.gethostbyname', side_effect=socket.gaierror('Unknown host')):
            with self.assertRaisesRegex(ValueError, 'Cannot resolve SHMEM bootstrap host to IPv4'):
                coordinator['_shmem_host'](address, dry_run=False)
        with self.assertRaisesRegex(ValueError, 'at most 63 bytes'):
            coordinator['_shmem_endpoint'](address, 29301)
        with self.assertRaisesRegex(ValueError, 'valid port'):
            coordinator['_shmem_endpoint'](host, 65536)
