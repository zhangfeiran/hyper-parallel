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
"""Check local fused FFN dependencies and native-layout derivative semantics on CPU."""

from collections import Counter
import unittest

import torch
from torch.nn import functional

from hyper_parallel.core.multicore.modules.routed_ffn import _local_backward_config, routed_ffn
from hyper_parallel.core.multicore.scheduler.config import EVENT_INVALID_ID, TaskType
from tests.common.mark_utils import arg_mark


class TestRoutedFfn(unittest.TestCase):
    """Validate the generated descriptors independently of the device kernel."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_native_layout_derivatives(self):
        """Interpret native-layout GMM descriptors against an autograd oracle.

        Feature: Native local FFN gradients.
        Description: Execute serialized projection operands against a Torch CPU oracle.
        Expectation: Input and both native-layout weight gradients agree.
        """
        torch.manual_seed(13)
        hidden, intermediate, experts, cores = 64, 32, 4, 2
        counts = (0, 3, 7, 0)
        inputs = torch.randn(sum(counts), hidden, dtype=torch.float64, requires_grad=True)
        weight1 = torch.randn(experts, 2 * intermediate, hidden, dtype=torch.float64, requires_grad=True)
        weight2 = torch.randn(experts, hidden, intermediate, dtype=torch.float64, requires_grad=True)
        upstream = torch.randn_like(inputs)
        first = torch.cat([chunk @ weight1[expert].T
                           for expert, chunk in enumerate(inputs.split(counts))])
        gate, up = first.chunk(2, dim=-1)
        activation = functional.silu(gate) * up
        output = torch.cat([chunk @ weight2[expert].T
                            for expert, chunk in enumerate(activation.split(counts))])
        expected = torch.autograd.grad(output, (inputs, weight1, weight2), upstream)
        config = _local_backward_config(128, hidden, intermediate, experts, cores)
        values = {0: upstream, 5: activation.detach(), 7: weight2.detach(), 9: first.detach(),
                  11: weight1.detach(), 17: inputs.detach()}

        def _read(descriptor, expert):
            tensor = values[descriptor.input_position]
            if descriptor.dynamic_shape:
                tensor = tensor.split(counts)[expert]
            elif tensor.ndim == 3:
                tensor = tensor[expert]
            return tensor.T if descriptor.transpose_flag else tensor

        completed = set()
        for index in config.cube_task_indices[:config.task_index_num[0]]:
            task = config.all_tasks[index]
            position = task.outputs[0].input_position
            if position in completed:
                continue
            if position in (12, 18) and 10 not in values:
                saved_first = values[9].requires_grad_()
                gate, up = saved_first.chunk(2, dim=-1)
                values[10] = torch.autograd.grad(functional.silu(gate) * up, saved_first, values[8])[0]
            products = [_read(task.inputs[0], expert) @ _read(task.inputs[1], expert)
                        for expert in range(experts)]
            values[position] = (torch.cat(products) if task.outputs[0].dynamic_shape else torch.stack(products))
            completed.add(position)
        for actual, reference in zip((values[12], values[18], values[6]), expected):
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dependency_coverage(self):
        """Check complete compute producer sets.

        Feature: Local task dependencies.
        Description: Generate dynamic row schedules without transport handshakes.
        Expectation: Every consumed event has its complete producer set.
        """
        for tokens in (128, 1024, 8192, 16384):
            with self.subTest(tokens=tokens):
                config = _local_backward_config(tokens, 2048, 1536, 64, 24)
                tasks = [config.all_tasks[index]
                         for queue, kind in ((config.cube_task_indices, 0), (config.vector_task_indices, 1))
                         for index in queue[:config.task_index_num[kind]]]
                producers = Counter(task.trigger_event for task in tasks)
                self.assertEqual({task.task_type for task in tasks},
                                 {TaskType.TASK_GROUPED_MATMUL, TaskType.TASK_SWI_GLU_GRAD,
                                  TaskType.TASK_ZERO_EMPTY_EXPERTS})
                self.assertEqual(config.ready_event, 0)
                self.assertEqual(config.completion_event, 0)
                for task in tasks:
                    event = task.dependent_event
                    if event != EVENT_INVALID_ID:
                        self.assertEqual(producers[event], config.all_event_num_triggers[event])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cpu_rejected(self):
        """Reject unsupported hardware before native activation.

        Feature: Device boundary validation.
        Description: Supply CPU tensors to the NPU-only operator.
        Expectation: A clear ValueError precedes any native initialization.
        """
        with self.assertRaisesRegex(ValueError, "Ascend NPU"):
            routed_ffn(torch.empty(0, 64), torch.empty(4, 64, 64), torch.empty(4, 64, 32),
                       torch.zeros(4, dtype=torch.int64))
