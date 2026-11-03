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
"""Exercise held-out identities, rejection gates and exact dispatch provenance."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from hyper_parallel.core.expert_parallel.hot_replica.calibration_audit import (
    FitContract, SampleRejected, audit_cluster, decision_metrics, fit_provider_services,
    replay_decision, validate_fit_sample,
)
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica.planner import _build_expert_replica_plan


class TestCalibrationAudit(unittest.TestCase):
    """Use synthetic timing only to check rejection and statistics contracts."""

    @staticmethod
    def decision() -> dict:
        """Build an exact small owner-only plan through the real planner."""
        model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
            intermediate_size=8, ep_size=2, transport='shmem_signal_kernel_gradient_bands',
            forward_ms=((0, 0.), (32, 32.)), backward_ms=((0, 0.), (32, 32.)),
            weight_ms=1., gradient_ms=1.)
        options = dict(capacity_limit=32, receive_capacity=32, adaptive=True, _rank_nodes=(0, 1))
        plan = _build_expert_replica_plan(((4, 0), (0, 4)), 1, cost_model=model, **options)
        return dict(model=asdict(model), options=options, config=asdict(plan.config),
                    source_counts=plan.logical_counts, slots=plan.slot_to_logical, dispatch=plan.dispatch_counts,
                    calls=2, plan_reuses=1)

    @staticmethod
    def provider() -> dict:
        """Separate EP groups have deliberately different measured service."""
        config = dict(backend='push', replica_transport='shmem_signal_kernel_gradient_bands', ep_size=4,
                      hidden=16, intermediate=8, home_experts=1, reference_top_k=1, iterations=2, warmup=1)
        reports = []
        for group, duration in ((0, 1.), (4, 2.)):
            sample = dict(weight_ms=duration, physical_gradient_ms=duration * 2,
                          weight_wall_ms=duration * 3, gradient_wall_ms=duration * 4)
            pairs = [dict(kind=kind, owner=0, target=target, bf16_weight_bytes=768,
                          fp32_gradient_bytes=1536, samples=[sample, sample])
                     for kind, target in (('intra_node', 1), ('inter_node', 2))]
            reports.append(dict(global_leader=group, global_ep_members=list(range(group, group + 4)),
                                rank_nodes=[0, 0, 1, 1], soc_name='test', torch_npu='test',
                                native_adapter_sha256='a' * 64, pairs=pairs))
        return dict(configuration=config, reports=reports, measurement='rank-local stream events')

    def test_service_holdout_never_leaks_second_ep_group(self) -> None:
        """A held-out slowdown stays visible rather than changing the fitted median."""
        result = fit_provider_services(self.provider())
        values = result['domains']['inter_node']['services']['weight_ms']
        self.assertEqual(result['train_group'], 0)
        self.assertEqual(result['heldout_groups'], [4])
        self.assertEqual(values['candidate_ms'], 1.)
        self.assertEqual(values['heldout']['median_signed_error_ms'], -1.)
        self.assertEqual(values['heldout']['observations_ms'], [2., 2.])
        self.assertFalse(result['installable'])

    def test_link_identity_topology_and_matrix_bytes_fail_closed(self) -> None:
        """Shape, peer class and native identity cannot silently share fits."""
        for field in ('identity', 'topology', 'bytes', 'missing', 'overlap'):
            value = self.provider()
            if field == 'identity':
                value['reports'][1]['native_adapter_sha256'] = 'b' * 64
            elif field == 'topology':
                value['reports'][0]['pairs'][0]['kind'] = 'inter_node'
            elif field == 'bytes':
                value['reports'][0]['pairs'][0]['bf16_weight_bytes'] = 1
            elif field == 'missing':
                del value['reports'][0]['pairs'][0]['samples'][0]['weight_ms']
            else:
                value['reports'][1]['global_ep_members'] = [0, 1, 2, 3]
            with self.subTest(field=field), self.assertRaises(SampleRejected):
                fit_provider_services(value)

    def test_replay_checks_signature_and_dispatch_conservation(self) -> None:
        """A valid saved plan reproduces; a single dispatch-row edit is rejected."""
        record = self.decision()
        self.assertTrue(replay_decision(record, 'a' * 64, 'a' * 64)['eligible_for_fit'])
        with self.assertRaisesRegex(SampleRejected, 'execution_signature_mismatch'):
            replay_decision(record, 'b' * 64, 'a' * 64)
        record['dispatch'] = [list(row) for row in record['dispatch']]
        record['dispatch'][0][0] += 1
        with self.assertRaisesRegex(SampleRejected, 'invalid_plan'):
            replay_decision(record, 'a' * 64, 'a' * 64)

    def test_fitting_rejects_missing_identity_host_clock_and_schedule_evidence(self) -> None:
        """Even a numerically valid plan cannot compensate for missing causal inputs."""
        contract = FitContract('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64, 'static-bands-v2-test')
        sample = dict(execution=asdict(contract) | {'schedule_equivalence_verified': True},
                      decision=self.decision(), measurement=dict(forward_ms=1., recompute_ms=1., backward_ms=1.,
                      host_exposed_ms=.1, host_wall_ms=.5, profile_on=False, clock_domain='rank_local_stream_elapsed'))
        validate_fit_sample(sample, contract)
        changes = (('execution', 'host_signature', None), ('execution', 'schedule_equivalence_verified', False),
                   ('execution', 'observation_mode', 'causal_v1'),
                   ('execution', 'residency_signature', 'e' * 64),
                   ('measurement', 'host_exposed_ms', None), ('measurement', 'profile_on', True),
                   ('measurement', 'clock_domain', 'raw_cross_rank_counter'))
        for section, field, value in changes:
            changed = deepcopy(sample)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(SampleRejected):
                validate_fit_sample(changed, contract)

    def test_admission_report_keeps_negative_missed_and_unknown_samples(self) -> None:
        """A low average error cannot hide wrong-direction admission or missing coverage."""
        rows = [dict(id='wrong', predicted_delta_ms=-2., observed_delta_ms=1.),
                dict(id='missed', predicted_delta_ms=1., observed_delta_ms=-2.),
                dict(id='good', predicted_delta_ms=-1., observed_delta_ms=-1.),
                dict(id='unknown', predicted_delta_ms=None, observed_delta_ms=20.)]
        rows = [row | {'execution_signature': 'a' * 64} for row in rows]
        rows.append(dict(id='wrong-payload', execution_signature='b' * 64,
                         predicted_delta_ms=-1., observed_delta_ms=-1.))
        result = decision_metrics(rows, 'a' * 64)
        self.assertEqual((result['false_admissions'], result['missed_benefits']), (1, 1))
        self.assertEqual(result['false_admission_rate'], .5)
        self.assertEqual(result['missed_benefit_rate'], .5)
        self.assertEqual(result['unknown'], ['unknown'])
        self.assertEqual(result['rejected'][0]['id'], 'wrong-payload')

    def test_directory_existence_does_not_imply_training_or_completion(self) -> None:
        """An empty scheduler-created directory stays unknown in the node audit."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'nodes/node_0/started').mkdir(parents=True)
            report = audit_cluster(root, nodes=1, dies_per_node=1)
        self.assertFalse(report['terminal'])
        self.assertFalse(report['all_nodes_success'])
        self.assertFalse(any(stage['complete'] for stage in report['stages'].values()))
