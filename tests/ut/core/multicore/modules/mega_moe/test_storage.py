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
"""Independent W13 producer fences preserve generic backward dependencies."""

from copy import deepcopy
import unittest

from hyper_parallel.core.multicore.modules.mega_moe.backward.storage import (
    can_reuse_backward_dispatch, prepare_w13_overlap_schedule, replica_w13_ready_events, replica_w2_ready_events,
)
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import TaskType, TaskSplitValue, validate_runtime_config
from hyper_parallel.core.multicore.scheduler.runtime import serialize_runtime_config


class TestW13ProducerFence(unittest.TestCase):
    """Reject partial completion and preserve ordinary counters and dX priority."""

    @staticmethod
    def _config(ep=2, logical=8, budget=2, cores=20, mode="push"):
        """Generate physical queues including spare guest slots."""
        physical = logical + ep * budget
        spec = MegaMoeSpec(128, 128, 128, physical, 2, 1.25, 512, ep, None, ep - 1, cores,
                           dispatch_mode=mode, replica_slots_per_rank=budget, logical_num_experts=logical,
                           replica_transport="shmem_signal_kernel_gradient")
        return spec, _build_runtime_artifacts(spec, overlap_w13=False)[3]

    def test_fences_preserve_queue_and_existing_dependencies_across_topologies(self):
        """Core count, EP topology and guest budget do not alter the producer contract."""
        for mode in ("push", "pull"):
            for ep, logical, budget, cores in ((1, 4, 1, 1), (2, 8, 1, 8), (2, 8, 2, 24),
                                              (16, 96, 4, 20), (1, 256, 1, 2)):
                with self.subTest(mode=mode, ep=ep, budget=budget, cores=cores):
                    spec, cfg = self._config(ep, logical, budget, cores, mode)
                    original = deepcopy(cfg)
                    self.assertTrue(prepare_w13_overlap_schedule(cfg, spec.local_experts, cores))
                    events = replica_w13_ready_events(cfg, spec.local_experts, cores)
                    self.assertEqual(len(set(events)), spec.local_experts)
                    self.assertEqual(list(cfg.cube_task_indices), list(original.cube_task_indices))
                    self.assertEqual(list(cfg.vector_task_indices), list(original.vector_task_indices))
                    self.assertEqual(replica_w2_ready_events(cfg, spec.local_experts, cores),
                                     replica_w2_ready_events(original, spec.local_experts, cores))
                    self.assertTrue(can_reuse_backward_dispatch(cfg, spec.local_experts, cores))
                    for index in range(cfg.task_num):
                        task, before = cfg.all_tasks[index], original.all_tasks[index]
                        self.assertEqual(task.trigger_event, before.trigger_event)
                        self.assertEqual(task.dependent_event, before.dependent_event)
                        if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.outputs[0].input_position == 18:
                            self.assertEqual(task.extra_value_1, events[task.task_index // cores])
                        else:
                            self.assertEqual(bytes(task), bytes(before))
                    for index, count in enumerate(original.all_event_num_triggers):
                        if count:
                            self.assertEqual(cfg.all_event_num_triggers[index], count)
                    values = TaskSplitValue(tp=1, ep=ep, seq_size=128, all_expert_num=spec.num_experts, top_k=2,
                                            dispatch_mode=mode)
                    validate_runtime_config(cfg, values, cores)
                    self.assertTrue(serialize_runtime_config(cfg))

    def test_guest_priority_preserves_dx_and_worker_ownership(self):
        """Publish guest matrices sooner without changing any task or ordinary event."""
        for mode in ("push", "pull"):
            for ep, logical, budget, cores in ((1, 4, 1, 1), (2, 8, 1, 8), (2, 8, 2, 24),
                                              (16, 96, 4, 20), (1, 256, 1, 2)):
                with self.subTest(mode=mode, ep=ep, budget=budget, cores=cores):
                    spec, cfg = self._config(ep, logical, budget, cores, mode)
                    reference = deepcopy(cfg)
                    self.assertTrue(prepare_w13_overlap_schedule(reference, spec.local_experts, cores))
                    self.assertTrue(prepare_w13_overlap_schedule(cfg, spec.local_experts, cores,
                                                                replica_slots=budget))
                    count = cfg.task_index_num[0]
                    prefix = count - spec.local_experts * cores
                    self.assertEqual(list(cfg.cube_task_indices[:prefix]),
                                     list(reference.cube_task_indices[:prefix]))
                    self.assertEqual(list(cfg.vector_task_indices), list(reference.vector_task_indices))
                    self.assertEqual(list(cfg.all_event_num_triggers), list(reference.all_event_num_triggers))
                    for index in range(cfg.task_num):
                        self.assertEqual(bytes(cfg.all_tasks[index]), bytes(reference.all_tasks[index]))
                    home = spec.local_experts - budget
                    experts = (*range(home, spec.local_experts), *range(home))
                    for offset, index in enumerate(cfg.cube_task_indices[prefix:count]):
                        self.assertEqual(cfg.all_tasks[index].task_index,
                                         experts[offset // cores] * cores + offset % cores)
                    self.assertEqual(replica_w13_ready_events(cfg, spec.local_experts, cores),
                                     replica_w13_ready_events(reference, spec.local_experts, cores))
                    self.assertTrue(can_reuse_backward_dispatch(cfg, spec.local_experts, cores))
                    values = TaskSplitValue(tp=1, ep=ep, seq_size=128, all_expert_num=spec.num_experts, top_k=2,
                                            dispatch_mode=mode)
                    validate_runtime_config(cfg, values, cores)
                    self.assertTrue(serialize_runtime_config(cfg))

    def test_runtime_builder_passes_the_physical_guest_budget(self):
        """The production builder enables guest priority only for the replica gradient graph."""
        spec, reference = self._config()
        cfg = _build_runtime_artifacts(spec)[3]
        cores = spec.num_cube_cores
        prefix = cfg.task_index_num[0] - spec.local_experts * cores
        self.assertEqual(list(cfg.cube_task_indices[:prefix]), list(reference.cube_task_indices[:prefix]))
        first = cfg.all_tasks[cfg.cube_task_indices[prefix]]
        self.assertEqual(first.task_index // cores, spec.local_experts - spec.replica_slots_per_rank)
        self.assertEqual(len(replica_w13_ready_events(cfg, spec.local_experts, cores)), spec.local_experts)

    def test_invalid_guest_priority_does_not_modify_any_descriptor(self):
        """A budget outside the proven local topology falls back before allocating events."""
        spec, original = self._config()
        for budget in (-1, spec.local_experts, spec.local_experts + 1):
            cfg = deepcopy(original)
            before = bytes(cfg)
            self.assertFalse(prepare_w13_overlap_schedule(cfg, spec.local_experts, spec.num_cube_cores,
                                                         replica_slots=budget))
            self.assertEqual(bytes(cfg), before)

    def test_incomplete_or_colliding_fences_disable_kernel_return(self):
        """One early writer, a duplicate producer, or a stale ready word cannot publish completion."""
        spec, original = self._config()
        self.assertTrue(prepare_w13_overlap_schedule(original, spec.local_experts, 20))
        for mutation in ("threshold", "partial", "collision", "duplicate", "bounds"):
            with self.subTest(mutation=mutation):
                cfg = deepcopy(original)
                event = replica_w13_ready_events(cfg, spec.local_experts, 20)[0]
                weights = [task for task in cfg.all_tasks[:cfg.task_num]
                           if task.outputs[0].input_position == 18]
                if mutation == "threshold":
                    cfg.all_event_num_triggers[event] = 19
                elif mutation == "partial":
                    weights[0].extra_value_1 = 0
                elif mutation == "collision":
                    weights[0].extra_value_1 = weights[20].extra_value_1
                elif mutation == "duplicate":
                    task = next(task for task in cfg.all_tasks[:cfg.task_num]
                                if task.outputs[0].input_position == 6)
                    task.extra_value_1 = event
                else:
                    weights[0].extra_value_1 = cfg.event_capacity - 1
                self.assertEqual(replica_w13_ready_events(cfg, spec.local_experts, 20), ())

    def test_side_events_cannot_count_as_another_readers_completion(self):
        """An unrelated GMM cannot release ActGrad readers through its optional event."""
        spec, cfg = self._config()
        weight = next(task for task in cfg.all_tasks[:cfg.task_num] if task.outputs[0].input_position == 6)
        activation = next(task for task in cfg.all_tasks[:cfg.task_num] if task.outputs[0].input_position == 8)
        weight.extra_value_1 = activation.trigger_event
        self.assertFalse(can_reuse_backward_dispatch(cfg, spec.local_experts, 20))
        self.assertEqual(replica_w2_ready_events(cfg, spec.local_experts, 20), ())
        self.assertEqual(replica_w13_ready_events(cfg, spec.local_experts, 20), ())

    def test_unknown_schedule_or_insufficient_storage_is_unchanged(self):
        """Failed preparation leaves the serialized queue untouched for late return."""
        spec, original = self._config()
        for mutation in ("reader", "order", "counter", "other_counter", "capacity"):
            with self.subTest(mutation=mutation):
                cfg = deepcopy(original)
                weight = next(task for task in cfg.all_tasks[:cfg.task_num]
                              if task.outputs[0].input_position == 18)
                if mutation == "reader":
                    weight.inputs[0].input_position = 0
                elif mutation == "order":
                    index = next(i for i, t in enumerate(cfg.cube_task_indices[:cfg.task_index_num[0]])
                                 if cfg.all_tasks[t].outputs[0].input_position == 18)
                    cfg.cube_task_indices[index], cfg.cube_task_indices[index + 1] = (
                        cfg.cube_task_indices[index + 1], cfg.cube_task_indices[index])
                elif mutation == "counter":
                    weight.extra_value_1 = 1
                elif mutation == "other_counter":
                    task = next(task for task in cfg.all_tasks[:cfg.task_num]
                                if task.outputs[0].input_position == 6)
                    task.extra_value_1 = 1
                else:
                    cfg.event_capacity = 32
                before = bytes(cfg)
                self.assertFalse(prepare_w13_overlap_schedule(cfg, spec.local_experts, 20))
                self.assertEqual(bytes(cfg), before)
