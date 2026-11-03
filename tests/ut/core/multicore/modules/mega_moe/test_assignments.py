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
"""Generic variable-assignment execution with an independent synthetic executor."""

import unittest
from unittest.mock import patch

import torch

from tests.common.mark_utils import arg_mark

from hyper_parallel.core.multicore.modules.mega_moe.assignments import masked_expert_forward


class TestMaskedAssignments(unittest.TestCase):
    """Exercise the generic contract with a second, synthetic expert executor."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_masked_slots_do_not_execute_or_lose_router_gradients(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: Only valid rows execute; each active slot returns the right weight derivative.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        class Executor:
            num_experts = 2

            def __call__(self, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                         *, expert_weights: tuple) -> torch.Tensor:
                """Execute the independent affine expert reference."""
                self.ids = ids
                return hidden * 2 + ids

        executor = Executor()
        x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        p = torch.ones(2, 2, requires_grad=True)
        ids = torch.tensor([[0, 99], [99, 1]])
        output = masked_expert_forward(executor, x, ids, p, ids != 99, expert_weights=(None, None))
        torch.testing.assert_close(output, torch.tensor([[2., 4.], [7., 9.]]))
        output.sum().backward()
        torch.testing.assert_close(p.grad, torch.tensor([[6., 0.], [0., 16.]]))
        self.assertEqual(executor.ids.shape, (2, 1))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_empty_source_still_participates_when_peer_has_real_work(self) -> None:
        """
        Feature: LongCat Flash model integration.
        Description: An empty source calls its executor so that it can serve remote assignments.
        Expectation: Outputs, gradients, and state obey the asserted contract.
        """
        class Executor:
            num_experts = 2

            def __call__(self, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                         *, expert_weights: tuple) -> torch.Tensor:
                """Execute the independent affine expert reference."""
                self.called = True
                self.gradient_shapes = []
                output = hidden * 2
                output.register_hook(lambda gradient: self.gradient_shapes.append(tuple(gradient.shape)))
                return output

        executor = Executor()
        x = torch.ones(1, 2, requires_grad=True)
        ids = torch.zeros(1, 2, dtype=torch.long)
        original_copy = torch.Tensor.index_copy

        def _copy_nonempty(tensor: torch.Tensor, dimension: int, index: torch.Tensor,
                           source: torch.Tensor) -> torch.Tensor:
            """Model a backend that cannot safely copy an empty index set."""
            self.assertGreater(index.numel(), 0)
            return original_copy(tensor, dimension, index, source)

        with patch.object(torch.Tensor, "index_copy", new=_copy_nonempty), patch(
                "hyper_parallel.core.multicore.modules.mega_moe.assignments.dist.all_reduce",
                   side_effect=lambda value, **kwargs: value.copy_(torch.tensor([3, 0]))):
            result = masked_expert_forward(executor, x, ids, torch.ones(1, 2), ids.bool(),
                                           expert_weights=(None, None), ep_group=object())
        self.assertTrue(executor.called)
        torch.testing.assert_close(result, torch.zeros_like(x))
        result.sum().backward()
        self.assertEqual(executor.gradient_shapes, [(0, 2)])
        torch.testing.assert_close(x.grad, torch.zeros_like(x))
