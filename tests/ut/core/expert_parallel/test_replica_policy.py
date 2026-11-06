# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Public replica APIs need only a slot cap, with internal policy selection."""

import inspect
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.expert_parallel import ExpertParallel
from hyper_parallel.core.expert_parallel.hot_replica import build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica import policy
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe import module as mega_module
from hyper_parallel.core.multicore.modules.mega_moe.inactive_tasks import UNMAPPED_GUEST_SUFFIX_SCHEDULE


class TestReplicaPublicPolicy(unittest.TestCase):
    """Keep model injection and adaptive toggles out of business constructors."""

    def test_public_signatures_only_expose_the_resident_cap(self) -> None:
        """No public entry point requires a cost object or an adaptive opt-in."""
        for function in (ExpertParallel, MegaMoeExperts, build_expert_replica_plan):
            parameters = inspect.signature(function).parameters
            with self.subTest(function=function.__name__):
                self.assertNotIn('replica_cost_model', parameters)
                self.assertNotIn('replica_adaptive', parameters)
                self.assertNotIn('cost_model', parameters)
                self.assertNotIn('adaptive', parameters)
                self.assertIn('replica_slots_per_rank', parameters)

    def test_a_slot_cap_is_sufficient_without_calibration(self) -> None:
        """Unmeasured shapes retain valid load-driven adaptation rather than failing."""
        expert = MegaMoeExperts(hidden_size=16, intermediate_size=8,
                               num_experts=12, top_k=2, ep_size=4, replica_slots_per_rank=3)
        try:
            self.assertFalse(hasattr(expert, 'replica_cost_model'))
            self.assertFalse(hasattr(expert, 'replica_adaptive'))
        finally:
            expert.close()
        policy = ExpertParallel(replica_slots_per_rank=3)
        self.assertFalse(hasattr(policy, 'replica_cost_model'))
        self.assertFalse(hasattr(policy, 'replica_adaptive'))

    def test_zero_cap_remains_owner_only(self) -> None:
        """B0 requires no cost or hidden policy configuration."""
        plan = build_expert_replica_plan([[100] * 3 + [0] * 9] + [[0] * 12] * 3, 0)
        plan.validate()
        self.assertFalse(plan.transfers)

    def test_hidden_switch_defaults_on_and_is_captured_by_each_owner(self) -> None:
        """Disabling costs keeps adaptation available without calibration or a device probe."""
        with patch.dict('os.environ', {'HYPER_PARALLEL_REPLICA_COST_MODEL': '0'}):
            owner = ExpertParallel(replica_slots_per_rank=2)
            self.assertFalse(owner._cost_model_enabled)
            with patch.object(policy, '_soc_name', side_effect=AssertionError('Unexpected device query')):
                self.assertIsNone(policy._runtime_cost_model(False, 'push', 5120, 2304, 64,
                    'shmem_signal_kernel_gradient_bands', 384, 6, torch.device('cpu')))
        with patch.dict('os.environ', {'HYPER_PARALLEL_REPLICA_COST_MODEL': '1'}):
            self.assertTrue(ExpertParallel(replica_slots_per_rank=2)._cost_model_enabled)
        self.assertFalse(owner._cost_model_enabled)

    def test_invalid_switch_fails_before_device_or_collective_setup(self) -> None:
        """A mistyped diagnostic policy cannot silently select another routing mode."""
        with patch.dict('os.environ', {'HYPER_PARALLEL_REPLICA_COST_MODEL': 'invalid'}):
            with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                ExpertParallel(replica_slots_per_rank=1)
            self.assertIsNotNone(ExpertParallel())

    def test_internal_calibration_matches_hardware_shape_and_schedule(self) -> None:
        """Only measured EP64 configurations use the packaged advisory coefficients."""
        arguments = ('push', 5120, 2304, 64, 'shmem_signal_kernel_gradient_bands', 384, 6, 'Ascend910_9392')
        measured = policy._resolve_cost_model(*arguments, True)
        self.assertIsNotNone(measured)
        self.assertEqual(measured.forward_repeats, 2)
        self.assertEqual(measured.backward_weight_ms, 0)
        ordinary = policy._resolve_cost_model(*arguments, False)
        self.assertEqual(ordinary.forward_repeats, 1)
        self.assertEqual(ordinary.backward_weight_ms, ordinary.weight_ms)
        for index, value in ((0, 'native'), (1, 1024), (2, 1024), (3, 16), (4, 'p2p'),
                             (5, 96), (6, 8), (7, 'Ascend910_9381')):
            changed = list(arguments)
            changed[index] = value
            with self.subTest(index=index, value=value):
                self.assertIsNone(policy._resolve_cost_model(*changed, True))

    def test_cost_disabled_placement_still_adapts_within_the_cap(self) -> None:
        """Load-driven fallback chooses zero or multiple guests and honors hard receive capacity."""
        cold = [[12] * 24] + [[0] * 24] * 3
        hot = [[50] * 6 + [0] * 18] + [[0] * 24] * 3
        first = build_expert_replica_plan(cold, 6, capacity_limit=300)
        second = build_expert_replica_plan(hot, 6, capacity_limit=75)
        first.validate()
        second.validate()
        self.assertEqual(first.replica_slots_used, (0, 0, 0, 0))
        self.assertEqual(max(second.replica_slots_used), 2)
        self.assertEqual(max(second.destination_loads), 75)


class TestReplicaScheduleSelection(unittest.TestCase):
    """Keep changed runtime queues separate from legacy calibration measurements."""

    arguments = ('push', 5120, 2304, 64, 'shmem_signal_kernel_gradient_bands', 384, 6, 'Ascend910_9392')
    schedule = 'fixed_queue_global_guest_communication_suffix_v1'

    def setUp(self) -> None:
        """Retain real packaged coefficients only as synthetic selector fixtures."""
        policy._resolve_cost_model.cache_clear()
        self.addCleanup(policy._resolve_cost_model.cache_clear)
        policy._calibration_available.cache_clear()
        self.addCleanup(policy._calibration_available.cache_clear)
        self.legacy = policy._calibrations()[self.arguments]

    def test_new_schedule_rejects_legacy_profile_and_shared_soc_alias(self) -> None:
        """A runtime graph change cannot silently inherit the old eight-field profile."""
        for soc in ('Ascend910_9392', 'Ascend910_9382'):
            with self.subTest(soc=soc):
                arguments = self.arguments[:-1] + (soc,)
                self.assertIsNone(policy._resolve_cost_model(
                    *arguments, True, execution_schedule=self.schedule))
                self.assertTrue(policy._calibration_available(*arguments[:-1]))
                self.assertFalse(policy._calibration_available(
                    *arguments[:-1], soc_name=soc, execution_schedule=self.schedule))

    def test_communication_schedule_does_not_reuse_compute_only_coefficients(self) -> None:
        """Compute-only pruning and communication pruning have distinct execution costs."""
        compute_only = replace(self.legacy, schedule='fixed_queue_unmapped_guest_suffix_v1')
        with patch.object(policy, '_calibrations', return_value={self.arguments: compute_only}):
            self.assertIsNone(policy._resolve_cost_model(
                *self.arguments, True, execution_schedule=self.schedule))

    def test_schedule_participates_in_the_resolver_cache_key(self) -> None:
        """Interleaved old and new runtimes must not share a cached legacy match."""
        ordinary = policy._resolve_cost_model(*self.arguments, True)
        changed = policy._resolve_cost_model(*self.arguments, True, execution_schedule=self.schedule)
        self.assertEqual(ordinary.schedule, self.legacy.schedule)
        self.assertIsNone(changed)
        self.assertEqual(policy._resolve_cost_model(*self.arguments, True), ordinary)

    def test_matching_new_schedule_uses_only_its_declared_profile(self) -> None:
        """Synthetic profile identity tests do not constitute new runtime calibration."""
        measured = replace(self.legacy, schedule=self.schedule)
        with patch.object(policy, '_calibrations', return_value={self.arguments: measured}):
            matched = policy._resolve_cost_model(*self.arguments, True, execution_schedule=self.schedule)
            stale = policy._resolve_cost_model(*self.arguments, True)
            self.assertTrue(policy._calibration_available(
                *self.arguments[:-1], soc_name='Ascend910_9382', execution_schedule=self.schedule))
            self.assertFalse(policy._calibration_available(
                *self.arguments[:-1], soc_name='Ascend910_9381', execution_schedule=self.schedule))
        self.assertEqual(matched.schedule, self.schedule)
        self.assertIsNone(stale)

    def test_critical_signature_cannot_bypass_legacy_schedule_filter(self) -> None:
        """Even an exact signed key cannot authorize a different runtime schedule."""
        with patch.object(policy, '_calibrations', return_value={self.arguments: self.legacy,
                                                               self.arguments + ('a' * 64,): self.legacy}):
            self.assertIsNone(policy._resolve_cost_model(
                *self.arguments, True, 'a' * 64, execution_schedule=self.schedule))
            self.assertIsNone(policy._resolve_cost_model(
                *self.arguments, True, 'b' * 64, execution_schedule=self.schedule))

    def test_runtime_without_matching_schedule_retains_uncalibrated_adaptation(self) -> None:
        """A mismatch returns no cost model instead of disabling the resident budget."""
        with patch.object(policy, '_soc_name', return_value='Ascend910_9392'):
            self.assertIsNone(policy._runtime_cost_model(
                True, *self.arguments[:-1], torch.device('cpu'), execution_schedule=self.schedule))
        plan = build_expert_replica_plan([[50] * 6 + [0] * 18] + [[0] * 24] * 3,
                                         6, capacity_limit=75)
        plan.validate()
        self.assertTrue(plan.transfers)
        self.assertLessEqual(max(plan.destination_loads), 75)

    def test_disabled_policy_does_not_probe_hardware_for_new_schedule(self) -> None:
        """The existing hidden switch remains sufficient for all runtime schedules."""
        with patch.object(policy, '_soc_name', side_effect=AssertionError('Unexpected device query')):
            self.assertIsNone(policy._runtime_cost_model(
                False, *self.arguments[:-1], torch.device('cpu'), execution_schedule=self.schedule))

    def test_new_schedule_does_not_relax_other_transport_contracts(self) -> None:
        """Only the new kernel-gradient-bands graph can declare the new schedule."""
        for changes in ({'backend': 'native', 'transport': 'p2p'}, {'transport': 'p2p'},
                        {'transport': 'shmem_signal_kernel_gradient'}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'schedule'):
                replace(self.legacy, schedule=self.schedule, **changes)

    def test_actual_replica_module_selects_schedule_without_critical_profiles(self) -> None:
        """Legacy tables must be isolated even when no critical fingerprint is requested."""
        self.assertEqual(self.schedule, UNMAPPED_GUEST_SUFFIX_SCHEDULE)
        owner = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
                               ep_size=2, replica_slots_per_rank=1)
        self.addCleanup(owner.close)
        resources = SimpleNamespace(spec=SimpleNamespace(auto_tokens=True), home_spec=object(), rank_nodes=(0, 1),
                                    receive_capacity_limit=Mock())
        with (patch.object(mega_module, '_critical_path_available', return_value=False),
              patch.object(mega_module, '_runtime_cost_model', return_value=None) as resolve,
              patch.object(mega_module, 'prepare_replica_route', return_value=object()) as prepare):
            owner._prepare_replica_route(torch.zeros(1, 2, dtype=torch.int32), resources,
                                         torch.zeros(1, 16), 31, None)
        self.assertEqual(resolve.call_args.kwargs['execution_schedule'], self.schedule)
        self.assertIsNone(prepare.call_args.kwargs['_cost_model'])
        self.assertEqual(prepare.call_args.kwargs['minimum_load_reduction'], .4)

    def test_schedule_changes_replica_fingerprint_but_preserves_b0(self) -> None:
        """Only the changed guest graph invalidates the existing execution identity."""
        for budget in (0, 1):
            owner = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
                                   ep_size=2, replica_slots_per_rank=budget)
            self.addCleanup(owner.close)
            resources = SimpleNamespace(spec=SimpleNamespace(expert_weight_layout='EIO', swiglu_limit=10.,
                                                             swiglu_hf_bf16=False, local_num_tokens=128,
                                                             single_expert_tasks=False, gate_up_expert_padding=0,
                                                             route_task_bounds=False,
                                                             dispatch_split=128, combine_split=128),
                                        rank_nodes=(0, 1))
            with (self.subTest(budget=budget),
                  patch.object(mega_module, '_critical_native_fingerprint', return_value='a' * 64),
                  patch.object(mega_module, 'execution_fingerprint', return_value='fingerprint') as fingerprint):
                mega_module._critical_execution_signature(owner, resources)
                metadata = fingerprint.call_args.args[0]
                if budget:
                    self.assertEqual(metadata['runtime_schedule'], self.schedule)
                else:
                    self.assertNotIn('runtime_schedule', metadata)
