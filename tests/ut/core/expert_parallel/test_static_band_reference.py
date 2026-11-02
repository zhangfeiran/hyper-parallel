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
"""Hand-check static native queues and counterexamples to a more flexible heap."""

from dataclasses import replace
import unittest

from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.plan import ExpertExecutionPlan
from hyper_parallel.core.expert_parallel.hot_replica.planner import _materialize
from hyper_parallel.core.expert_parallel.hot_replica.static_band_reference import (
    BandObservation, gradient_band_layout, static_band_queue, static_band_reference,
)


class TestStaticBandReference(unittest.TestCase):
    """All durations are explicitly synthetic dependency examples, not calibration."""

    @staticmethod
    def plan() -> ExpertExecutionPlan:
        """Two distinct experts share one owner's ordered return queue."""
        config = ExpertReplicaConfig(6, 3, 1)
        counts = ((2, 2, 0, 0, 0, 0),) + ((0,) * 6,) * 2
        copies = [{0: 1, 1: 1}, {2: 0, 3: 0, 0: 1}, {4: 0, 5: 0, 1: 1}]
        return _materialize(counts, copies, config)

    @classmethod
    def observations(cls, bands: tuple[int, int] = (2, 3), workers: int = 2) -> dict[tuple, BandObservation]:
        """Each participating stripe takes one synthetic millisecond."""
        result = {}
        for projection, count in enumerate(bands):
            groups = min(4, count, workers)
            for edge in cls.plan().transfers:
                for band in range(count):
                    result[projection, edge.owner_rank, edge.target_rank, edge.target_slot, band] = BandObservation(
                        (0.,) * workers, (0.,) * workers,
                        tuple(1. if worker % groups == band % groups else None for worker in range(workers)), 1, 1)
        return result

    def test_per_projection_ack_and_final_guest_retention(self) -> None:
        """W2 ACK is early; final guest release still waits for its W13 ACK."""
        estimate = static_band_reference(self.plan(), (2, 3), 2, self.observations(), (0., 0., 0.), ack_service_ms=.5)
        self.assertTrue(estimate.covered)
        trace = {key: (start, end) for key, start, end in estimate.trace}
        self.assertEqual(trace['ack:0:0:1:2'], (2., 2.5))
        self.assertEqual(trace['ack:0:0:2:2'], (2.5, 3.))
        self.assertEqual(trace['ack:1:0:1:2'], (7., 7.5))
        self.assertEqual(trace['ack:1:0:2:2'], (7.5, 8.))
        self.assertEqual(estimate.owner_gradient_ready_ms, (7., 0., 0.))
        self.assertEqual(estimate.rank_completion_ms, (8., 7.5, 8.))
        self.assertFalse(estimate.production_calibrated)

    def test_nonzero_worker_enters_w13_before_other_worker_finishes_w2(self) -> None:
        """There is no global W2 barrier on the nonzero physical workers."""
        observations = self.observations()
        for key, value in tuple(observations.items()):
            if key[0] == 0 and key[-1] == 0:
                observations[key] = replace(value, owner_ready_ms=(10., 10.))
        estimate = static_band_reference(self.plan(), (2, 3), 2, observations, (0., 0., 0.), ack_service_ms=.5)
        trace = {key: (start, end) for key, start, end in estimate.trace}
        self.assertEqual(trace['return:1:0:1:1:1:2'], (2., 3.))
        self.assertEqual(trace['quorum:0:0'], (12., 12.))
        self.assertEqual(trace['return:1:0:0:0:1:2'][0], 13.)

    def test_fixed_band_edge_queue_cannot_skip_a_blocked_expert(self) -> None:
        """A later ready expert and a later ready band cannot bypass the native head."""
        observations = self.observations(bands=(2, 1), workers=1)
        key = (0, 0, 1, 2, 0)
        observations[key] = replace(observations[key], guest_ready_ms=(10.,))
        estimate = static_band_reference(self.plan(), (2, 1), 1, observations, (0., 0., 0.), ack_service_ms=0.)
        trace = {key: (start, end) for key, start, end in estimate.trace}
        self.assertEqual(trace['return:0:0:0:0:1:2'], (10., 11.))
        self.assertEqual(trace['return:0:0:0:0:2:2'], (11., 12.))
        self.assertEqual(trace['return:0:0:0:1:1:2'], (12., 13.))
        self.assertEqual(trace['return:0:0:0:1:2:2'], (13., 14.))

    def test_band_ordinals_restart_and_groups_are_not_physical_workers(self) -> None:
        """One W2 band does not offset W13 worker groups; all six workers participate."""
        estimate = static_band_reference(self.plan(), (1, 2), 6,
                                        self.observations(bands=(1, 2), workers=6), (0., 0., 0.), ack_service_ms=0.)
        returns = [task for task in estimate.tasks if task.key.startswith('return:') and task.rank == 0]
        self.assertEqual({task.worker for task in returns if task.projection == 0}, set(range(6)))
        self.assertTrue(any(task.key == 'return:1:0:0:0:1:2' for task in returns))
        self.assertTrue(any(task.key == 'return:1:0:1:1:1:2' for task in returns))
        self.assertEqual({task.engine for task in returns}, {'aiv_mte2_vector_mte3'})

    def test_owner_guest_readiness_and_epochs_remain_uncovered_when_missing(self) -> None:
        """Unobserved fences never silently become zero or an overlap certificate."""
        for changes in ({'owner_ready_ms': (None, 0.)}, {'guest_ready_ms': (0., None)},
                        {'owner_epoch': 2}, {'guest_epoch': 2}, {'owner_epoch': True},
                        {'clock_domain': 'raw_rank_counter'}):
            observations = self.observations()
            key = next(iter(observations))
            observations[key] = replace(observations[key], **changes)
            estimate = static_band_reference(self.plan(), (2, 3), 2, observations, (0., 0., 0.), ack_service_ms=0.)
            with self.subTest(changes=changes):
                self.assertFalse(estimate.covered)
                self.assertTrue(estimate.missing)
                self.assertIsNone(estimate.rank_completion_ms)

    def test_local_aic_completion_is_separate_from_return_worker_join(self) -> None:
        """A replica return quorum does not prove all local dW work is finished."""
        estimate = static_band_reference(self.plan(), (2, 3), 2, self.observations(), (20., 0., 0.), ack_service_ms=0.)
        self.assertEqual(estimate.owner_gradient_ready_ms[0], 20.)
        self.assertEqual(estimate.rank_completion_ms[0], 20.)

    def test_full_width_band_layout_keeps_ragged_last_band(self) -> None:
        """Physical W2/W13 band geometry matches the native whole-row partition."""
        w2, w13 = gradient_band_layout(2304, 5120), gradient_band_layout(5120, 4608)
        self.assertEqual((len(w2), len(w13)), (12, 23))
        self.assertEqual((w2[-1].rows, w13[-1].rows), (60, 126))
        self.assertEqual(sum(band.elements for band in w2), 2304 * 5120)
        self.assertEqual(sum(band.elements for band in w13), 5120 * 4608)

    def test_queue_preserves_guest_slot_order_on_the_same_peer(self) -> None:
        """Sorting logical experts would change the deployed owned-edge head order."""
        config = ExpertReplicaConfig(6, 3, 2)
        counts = ((2, 3, 0, 0, 0, 0),) + ((0,) * 6,) * 2
        copies = [{0: 1, 1: 1}, {2: 0, 3: 0, 1: 2, 0: 1}, {4: 0, 5: 0}]
        original = _materialize(counts, copies, config)
        slots = [list(row) for row in original.slot_to_logical]
        dispatch = [list(row) for row in original.dispatch_counts]
        slots[1][2], slots[1][3] = slots[1][3], slots[1][2]
        for row in dispatch:
            row[6], row[7] = row[7], row[6]
        plan = ExpertExecutionPlan(config, counts, slots, dispatch)
        plan.validate()
        queue = static_band_queue(plan, 0, 0, 0, 1, 2)
        self.assertEqual([item.logical_expert for item in queue], [1, 0])
        self.assertEqual([item.target_slot for item in queue], [2, 3])

    def test_multiple_owners_wait_for_incoming_ack_only_after_both_local_projections(self) -> None:
        """Cyclic replica edges do not introduce a false cross-rank projection barrier."""
        config = ExpertReplicaConfig(6, 3, 1)
        counts = ((2, 0, 0, 0, 0, 0), (0, 0, 2, 0, 0, 0), (0,) * 6)
        copies = [{0: 1, 1: 0, 2: 1}, {2: 1, 3: 0, 0: 1}, {4: 0, 5: 0}]
        plan = _materialize(counts, copies, config)
        observations = {(projection, edge.owner_rank, edge.target_rank, edge.target_slot, 0):
                        BandObservation((0., 0.), (0., 0.), (1., 1.), 1, 1)
                        for projection in range(2) for edge in plan.transfers}
        estimate = static_band_reference(plan, (1, 1), 2, observations, (0., 0., 0.), ack_service_ms=.1)
        self.assertTrue(estimate.covered)
        for value in estimate.rank_completion_ms[:2]:
            self.assertAlmostEqual(value, 2.2)
        self.assertEqual(estimate.rank_completion_ms[2], 0.)
        self.assertAlmostEqual(estimate.owner_gradient_ready_ms[0], 2.1)
        self.assertAlmostEqual(estimate.owner_gradient_ready_ms[1], 2.1)
