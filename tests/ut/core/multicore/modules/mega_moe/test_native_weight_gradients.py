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
"""CPU contracts for native-order operand lifetimes, events and retained-state isolation."""

import unittest
from dataclasses import replace

import torch

from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import build_config_for_rank
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import build_backward_graph
from hyper_parallel.core.multicore.modules.mega_moe.native_weight_gradients import (
    allocate_ordered_workspace, build_native_backward, ordered_workspace_views, supports_native_weight_gradients,
)
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue, TaskType


class TestNativeWeightGradients(unittest.TestCase):
    """Keep both native dW operands independent of the transport-order dX path."""

    @staticmethod
    def _spec(mode: str = "push", rank: int = 0, tokens: int = 128) -> MegaMoeSpec:
        return MegaMoeSpec(local_num_tokens=tokens, hidden_size=5120, intermediate_size=2304,
                           num_experts=96, top_k=6, initial_capacity_factor=None, receive_capacity=1024,
                           ep_size=16, ep_group=None, rank_id=rank, num_cube_cores=24,
                           dispatch_mode=mode, swiglu_limit=10.0, route_policy="torch")

    def test_unvalidated_geometry_uses_the_existing_exact_fallback(self) -> None:
        spec = self._spec()
        self.assertTrue(supports_native_weight_gradients(spec))
        for changes in ({"route_policy": "cann"}, {"expert_weight_layout": "EOI"}, {"ep_size": 8},
                        {"num_cube_cores": 20}, {"hidden_size": 8192}, {"intermediate_size": 2048}):
            with self.subTest(changes=changes):
                self.assertFalse(supports_native_weight_gradients(replace(spec, **changes)))

    def test_workspace_places_fresh_int64_map_after_aligned_bf16_scratch(self) -> None:
        for rows in (0, 1, 17, 129):
            with self.subTest(rows=rows):
                order = torch.arange(rows - 1, -1, -1, dtype=torch.int64)
                original = order.clone()
                workspace = allocate_ordered_workspace(order, 2304)
                offset = 0
                for width in (4608, 2304, 5120):
                    offset = (offset + rows * width * 2 + 127) // 128 * 128
                views = ordered_workspace_views(workspace, rows, 2304, 5120)
                self.assertEqual([v.shape[1] for v in views], [4608, 2304, 5120])
                for index, value in enumerate(views):
                    value.fill_(index + 1)
                actual = workspace.view(torch.int64)[offset // 8:offset // 8 + rows]
                self.assertTrue(torch.equal(actual, original), f"Expected map {original}, got {actual}")
                self.assertEqual(workspace.dtype, torch.uint8)
                self.assertGreaterEqual(workspace.numel(), max(4096, offset + rows * 8))
                self.assertTrue(torch.equal(order, original), f"Expected input {original}, got {order}")

    def test_pull_input_buffer_does_not_overwrite_the_fresh_row_map(self) -> None:
        order = torch.arange(16, -1, -1, dtype=torch.int64)
        workspace = allocate_ordered_workspace(order, 2304, include_input=True)
        views = ordered_workspace_views(workspace, 17, 2304, 5120, include_input=True)
        self.assertEqual([v.shape[1] for v in views], [4608, 2304, 5120, 5120])
        for index, value in enumerate(views):
            value.fill_(index + 1)
        offset = 17 * (4608 + 2304 + 5120 + 5120) * 2
        self.assertTrue(torch.equal(workspace.view(torch.int64)[offset // 8:offset // 8 + 17], order))

    def test_independent_operands_and_pipeline_queues_preserve_completion_joins(self) -> None:
        for mode in ("push", "pull"):
            for rank in (0, 7, 15):
                with self.subTest(mode=mode, rank=rank):
                    spec = self._spec(mode, rank)
                    values = TaskSplitValue(tp=1, ep=16, seq_size=128, all_expert_num=96,
                                            top_k=6, dispatch_mode=mode)
                    graph = build_backward_graph(values, hidden_size=5120, intermediate_size=2304,
                                                 swiglu_limit=10.0)
                    graph.propagate_splits(values)
                    original = build_config_for_rank(graph, values, rank)
                    original_bytes = bytes(original)
                    new_graph, actual = build_native_backward(graph, original, values, spec)
                    self.assertEqual(bytes(original), original_bytes)
                    stages = {}
                    cursor = 0
                    for node in new_graph.topological_sort():
                        stages[node.name] = list(actual.all_tasks[cursor:cursor + node.task_num])
                        cursor += node.task_num
                    terminal = actual.all_tasks[cursor]
                    self.assertEqual(terminal.task_type, TaskType.TASK_TERMINATE)
                    expected = len(stages["w1_grad"]) + len(stages["w2_grad"]) + len(stages["combine"])
                    self.assertEqual(actual.all_event_num_triggers[terminal.dependent_event], expected)
                    copies = len(stages["native_order_w2"]) // 6
                    self.assertEqual(actual.native_order_intermediate, 2304)
                    queue = [actual.all_tasks[i].tiling_data_position for i in actual.cube_task_indices[:576]]
                    self.assertEqual(queue, [20] * 144 + [23] * 144 + [21] * 144 + [22] * 144)
                    for expert in range(6):
                        w2 = stages["w2_grad"][expert * 24:(expert + 1) * 24]
                        act = stages["act_grad"][expert * 24:(expert + 1) * 24]
                        first_copy = stages["native_order_w2"][expert * copies:(expert + 1) * copies]
                        second_copy = stages["native_order_w1"][expert * copies:(expert + 1) * copies]
                        w1 = stages["w1_grad"][expert * 24:(expert + 1) * 24]
                        self.assertTrue(all(t.trigger_event == w2[0].dependent_event for t in first_copy))
                        self.assertEqual(actual.all_event_num_triggers[w2[0].dependent_event], copies)
                        self.assertNotEqual(w2[0].dependent_event, act[0].dependent_event)
                        self.assertEqual(w2[0].trigger_event, terminal.dependent_event)
                        self.assertEqual(w2[0].inputs[0].input_position, 33)
                        self.assertEqual(w2[0].inputs[1].input_position, 34)
                        self.assertEqual(act[0].inputs[0].input_position, 0)
                        self.assertTrue(all(t.trigger_event == w1[0].dependent_event for t in second_copy))
                        self.assertEqual(actual.all_event_num_triggers[w1[0].dependent_event], copies)
                        self.assertEqual(w1[0].inputs[0].input_position, 35 if mode == "pull" else 17)
                        self.assertEqual(w1[0].inputs[1].input_position, 26)
                        self.assertTrue(all(t.task_type == TaskType.TASK_GROUPED_MATMUL for t in w1 + w2))


if __name__ == "__main__":
    unittest.main()
