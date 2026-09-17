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
"""Shared branch scheduling must preserve routed events and avoid worker deadlocks."""

import unittest

from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import build_forward_graph
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import build_backward_graph
from hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank as forward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as backward_config,
)
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue, TaskType


class TestSharedSchedule(unittest.TestCase):
    """Validate actual generated queues using a per-worker event simulator."""

    def test_shared_queues_complete_with_exact_task_coverage(self) -> None:
        """Every worker can advance, including minimal and production expert counts."""
        for backward in (False, True):
            for placement in ("dispatch", "tail", "overlap", "idle"):
                for experts, ep in ((4, 2), (384, 16)):
                    with self.subTest(backward=backward, placement=placement, experts=experts):
                        tsv = TaskSplitValue(tp=1, ep=ep, seq_size=128, all_expert_num=experts, top_k=2)
                        builder = build_backward_graph if backward else build_forward_graph
                        graph = builder(tsv, hidden_size=512, intermediate_size=128,
                                        shared_intermediate_size=256, num_cube_cores=24)
                        graph.propagate_splits(tsv)
                        config_fn = backward_config if backward else forward_config
                        cfg = config_fn(graph, tsv, 0, 24, shared_placement=placement)
                        queues = [list(cfg.cube_task_indices[i:cfg.task_index_num[0]:24]) for i in range(24)]
                        vector = list(cfg.vector_task_indices[:cfg.task_index_num[1]])
                        if placement == "idle":
                            shared_start = len(vector) - 1 - tsv.per_expert_seq_to_other // 128
                            routed_queues = [vector[i:shared_start:24] for i in range(24)]
                            routed_queues[(len(vector) - 1) % 24].append(vector[-1])
                            queues += routed_queues
                            queues += [vector[shared_start + i:-1:24] for i in range(24)]
                        else:
                            queues += [vector[i::24] for i in range(24)]
                        scheduled = [task for queue in queues for task in queue]
                        self.assertEqual(len(scheduled), len(set(scheduled)))
                        expected_count = 1 + sum(op.task_num for op in graph.topological_sort())
                        self.assertEqual(set(scheduled), set(range(expected_count)))
                        # Remote producers are already ready. Local compute events are simulated exactly.
                        events = [0] * cfg.event_capacity
                        for task in cfg.all_tasks:
                            if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.inputs[0].dynamic_shape:
                                if task.outputs[0].input_position in (7, 6, 8):
                                    events[task.dependent_event] = cfg.all_event_num_triggers[task.dependent_event]
                        while any(queues):
                            advanced = False
                            for queue in queues:
                                if not queue:
                                    continue
                                task = cfg.all_tasks[queue[0]]
                                dep = task.dependent_event
                                if dep != 0xFFFFFFFF and events[dep] < cfg.all_event_num_triggers[dep]:
                                    continue
                                queue.pop(0)
                                events[task.trigger_event] += 1
                                advanced = True
                            self.assertTrue(advanced, "Generated queues contain a cross-worker dependency deadlock")
