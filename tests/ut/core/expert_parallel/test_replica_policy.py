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
from unittest.mock import patch

import torch

from hyper_parallel.core.expert_parallel import ExpertParallel
from hyper_parallel.core.expert_parallel.hot_replica import build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica import policy
from hyper_parallel.core.multicore import MegaMoeExperts


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
