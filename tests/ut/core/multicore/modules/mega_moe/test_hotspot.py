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
"""CPU checks for the load-selective backward W2 phase join."""

import unittest
from itertools import product
from types import SimpleNamespace

from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import build_config_for_rank
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import build_backward_graph
from hyper_parallel.core.multicore.modules.mega_moe.backward.hotspot import (
    build_hotspot_config,
    supports_hotspot_schedule,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from hyper_parallel.core.multicore.modules.mega_moe.plan import MegaMoePlan
from hyper_parallel.core.multicore.scheduler.config import EVENT_INVALID_ID, TaskSplitValue, validate_runtime_config
from hyper_parallel.core.multicore.scheduler.runtime import serialize_runtime_config


class TestHotspotSchedule(unittest.TestCase):
    """Keep original dependencies and transport state while joining all W2 workers."""

    def test_join_preserves_original_descriptors_events_and_receive_reuse(self) -> None:
        """Exercise push/pull, variable topology, core count, and communication tiling."""
        for mode, ep, cores, split in product(('push', 'pull'), (2, 16, 32), (12, 24), (128, 512)):
            with self.subTest(mode=mode, ep=ep, cores=cores, split=split):
                topology = TaskSplitValue(tp=1, ep=ep, seq_size=512, all_expert_num=ep * 6,
                                         top_k=8, dispatch_mode=mode)
                graph = build_backward_graph(topology, hidden_size=5120, intermediate_size=1792,
                                             num_cube_cores=cores, dispatch_sv=split, combine_sv=split)
                graph.propagate_splits(topology)
                original = build_config_for_rank(graph, topology, ep - 1, cores)
                before = serialize_runtime_config(original)
                changed = build_hotspot_config(original, topology, cores)
                self.assertEqual(serialize_runtime_config(original), before)
                self.assertEqual(bytes(changed.all_tasks)[:len(bytes(original.all_tasks))], bytes(original.all_tasks))
                for name in ('vector_task_indices', 'mix_task_indices', 'dynamic_data', 'all_events'):
                    self.assertEqual(bytes(getattr(changed, name))[:len(bytes(getattr(original, name)))],
                                     bytes(getattr(original, name)))
                for name in ('ready_event', 'completion_event', 'protocol_version'):
                    self.assertEqual(getattr(changed, name), getattr(original, name))
                queued = changed.cube_task_indices[:changed.task_index_num[0]]
                old_count = 4 * 6 * cores
                self.assertEqual(len(queued), old_count + 2 * cores)
                stages = [changed.all_tasks[index].outputs[0].input_position
                          for index in queued if index < original.task_capacity]
                self.assertEqual(stages, [stage for stage in (6, 8, 12, 18) for _ in range(6 * cores)])
                arrivals = [changed.all_tasks[index] for index in queued[6 * cores:7 * cores]]
                waits = [changed.all_tasks[index] for index in queued[7 * cores:8 * cores]]
                event = arrivals[0].trigger_event
                self.assertEqual(changed.all_event_num_triggers[event], cores)
                self.assertTrue(all(task.dependent_event == EVENT_INVALID_ID for task in arrivals))
                self.assertTrue(all(task.trigger_event == event for task in arrivals))
                self.assertTrue(all(task.dependent_event == event for task in waits))
                self.assertTrue(can_reuse_backward_dispatch(changed, 6, cores))
                validate_runtime_config(changed, topology, cores)

    def test_shape_guard_keeps_unmeasured_shapes_on_original_schedule(self) -> None:
        """Limit the automatic policy to the currently validated expert shape."""
        shape = {"hidden_size": 5120, "intermediate_size": 1792, "local_experts": 6, "num_cube_cores": 24}
        self.assertTrue(supports_hotspot_schedule(SimpleNamespace(**shape)))
        for field in shape:
            changed = dict(shape)
            changed[field] += 1
            self.assertFalse(supports_hotspot_schedule(SimpleNamespace(**changed)))

    def test_selection_uses_each_saved_route_size_not_grown_heap_capacity(self) -> None:
        """Return to stock for balanced/empty microbatches after a hotspot backward."""
        stock, hotspot = object(), object()
        plan = SimpleNamespace(bwd_runtime=stock, hotspot_bwd_runtime=hotspot,
                               spec=SimpleNamespace(routed_slots=32768))
        for rows, expected in ((1, stock), (32768, stock), (65536, stock), (131071, stock),
                               (131072, hotspot), (262144, hotspot), (32768, stock)):
            self.assertIs(MegaMoePlan.backward_runtime(plan, rows), expected)
        plan.spec.routed_slots = 65536
        self.assertIs(MegaMoePlan.backward_runtime(plan, 131072), stock)
        plan.hotspot_bwd_runtime = None
        self.assertIs(MegaMoePlan.backward_runtime(plan, 262144), stock)
