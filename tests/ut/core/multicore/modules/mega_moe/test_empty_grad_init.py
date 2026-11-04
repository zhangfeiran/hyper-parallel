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
"""Check exact empty-expert metadata and gradient initialization through the autograd bridge."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import function as function_module
from hyper_parallel.core.multicore.modules.mega_moe import route as route_module
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from tests.common.mark_utils import arg_mark


class TestEmptyGradInit(unittest.TestCase):
    """Retain exact skipped gradients without another route synchronization."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_host_route_preserves_exact_empty_experts_without_another_readback(self) -> None:
        """Feature: Existing host count reuse.
        Description: Pack valid all-active, skipped, empty-source and empty-rank routes.
        Expectation: Exact local empty indices are retained without a tensor-to-list conversion.
        """
        patterns = ([[1, 1, 1, 1], [1, 1, 1, 1]], [[2, 0, 2, 0], [2, 0, 2, 0]],
                    [[2, 0, 0, 2], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]])
        for counts in patterns:
            raw = [[0, sum(row) // 2, 15, *row] for row in counts]
            for rank in range(2):
                with self.subTest(counts=counts, rank=rank):
                    spec = MegaMoeSpec(128, 4, 3, 4, 2, None, 512, 2, None, rank, 32)
                    expected = tuple(local for local in range(2)
                                     if sum(source[rank * 2 + local] for source in counts) == 0)
                    with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("Unexpected readback")):
                        metadata = route_module._host_route_metadata(raw, spec, 512, "cpu")
                    self.assertEqual(metadata.empty_experts, expected,
                                     f"expected_empty={expected}, actual={metadata.empty_experts}")
                    group = torch.tensor([sum(source[rank * 2 + local] for source in counts)
                                          for local in range(2)]).cumsum(0)
                    torch.testing.assert_close(metadata.group_list, group)

    @staticmethod
    def specification(cubes: int = 32, replicas: int = 0) -> SimpleNamespace:
        """Build the exact subset of a real plan consumed by this bridge test."""
        return SimpleNamespace(dispatch_mode="push", forward_ingress_mode="legacy", backward_ingress_mode="legacy",
                               num_cube_cores=cubes, replica_slots_per_rank=replicas, expert_weight_layout="EIO",
                               hidden_size=4, intermediate_size=3, top_k=2, auto_tokens=False)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_retains_the_invocation_empty_indices(self) -> None:
        """Feature: Per-invocation gradient initialization.
        Description: Run the actual forward bridge with an exact empty-expert tuple.
        Expectation: Backward selection retains this invocation's tuple rather than workspace state.
        """
        context = Mock()
        workspace = Mock(in_use=True, replica_provider=None)
        plan = SimpleNamespace(spec=self.specification(), ingress_diagnostics=None)
        metadata = SimpleNamespace(expert_capacity=4, replica_route=None, empty_experts=(1,))
        rows = torch.ones(4, 4)
        execution = SimpleNamespace(source=rows, intermediates=SimpleNamespace(up_proj=rows, activation=rows))
        with (
            patch.object(function_module, "_prepare_forward_execution", return_value=execution),
            patch.object(function_module, "_launch_forward_kernel"),
            patch.object(function_module, "_restore_forward_output", return_value=(rows.clone(), rows, ())),
            patch.object(function_module, "_save_forward_state"),
        ):
            result = function_module._MegaMoeFunction.forward(
                context, rows, torch.ones(3, 4, 6), torch.ones(3, 3, 4), metadata,
                plan, workspace, None, None, True, None, None, True)
        self.assertEqual(context.empty_experts, (1,), f"expected_empty=(1,), actual={context.empty_experts}")
        torch.testing.assert_close(result, rows)

    def backward_case(self, empty: tuple[int, ...] | None, cubes: int = 32, replicas: int = 0,
                      dtype: torch.dtype = torch.bfloat16) -> int:
        """Run real allocation and backward selection with poisoned uninitialized matrices."""
        first, second = torch.ones(3, 4, 6, dtype=dtype), torch.ones(3, 3, 4, dtype=dtype)
        saved = (torch.ones(4, 4), torch.ones(4, 6), torch.ones(4, 3), first, second,
                 torch.tensor([2, 2, 4]), *(torch.zeros(1, dtype=torch.int64) for _ in range(6)))
        workspace = Mock(replica_provider=None, expert_buffer=torch.empty(4, 4), routed_buffer=torch.empty(4, 4),
                         gmm_workspace=torch.empty(1), swiglu_grad_workspace=torch.empty(1))
        plan = SimpleNamespace(spec=self.specification(cubes, replicas), bwd_diagnostics=Mock(),
                               reuse_backward_dispatch=False, bwd_diagnostics_no_replica=None, bwd_task_prefixes=())
        context = Mock(plan=plan, workspace=workspace, saved_tensors=saved, replica_route=None,
                       replica_weight_generation=None, shared_backward=False, empty_experts=empty,
                       needs_input_grad=(False,) * 12)
        zero_like = torch.zeros_like
        actual_empty = (1,) if empty is None else empty

        def overwrite(_plan: object, _saved: object, _source: object, execution: object, *_args: object) -> None:
            """Model the native complete-K stores only on experts that have input rows."""
            for expert in range(3):
                if expert not in actual_empty:
                    execution.intermediates.grad_weight1[expert].fill_(expert + 1)
                    execution.intermediates.grad_weight2[expert].fill_(expert + 4)

        with (
            patch.object(function_module.torch, "zeros_like", wraps=zero_like) as initialize,
            patch.object(function_module.torch, "empty_like",
                         side_effect=lambda value, **kwargs: torch.full_like(value, float("nan"), **kwargs)),
            patch.object(function_module, "_launch_backward_kernel", side_effect=overwrite),
            patch.object(function_module, "_release_completed_graph"),
        ):
            result = function_module._MegaMoeFunction.backward(context, torch.ones(4, 4))
        for value in result[1:3]:
            self.assertTrue(torch.isfinite(value).all(), "A skipped or written expert retained poisoned storage")
            for expert in actual_empty:
                torch.testing.assert_close(value[expert], zero_like(value[expert]))
        workspace.release.assert_called_once_with()
        return initialize.call_count

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a5_backward_initializes_only_skipped_experts(self) -> None:
        """Feature: Non-replica A5 gradient initialization.
        Description: Exercise actual backward selection with complete, partial and empty expert sets.
        Expectation: Whole-matrix zero kernels disappear only when exact selective initialization is possible.
        """
        for empty, expected in (((), 0), ((1,), 0), ((0, 1, 2), 2), (None, 2)):
            with self.subTest(empty=empty):
                actual = self.backward_case(empty)
                self.assertEqual(actual, expected, f"expected_full_zero_calls={expected}, actual={actual}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_other_architectures_and_replica_fallback_keep_full_initialization(self) -> None:
        """Feature: Conservative gradient initialization fallback.
        Description: Present known empty indices on A3 and on an inactive replica plan.
        Expectation: These paths retain their original full initialization.
        """
        for cubes, replicas, dtype in ((24, 0, torch.bfloat16), (32, 1, torch.bfloat16),
                                       (32, 0, torch.float32), (32, 0, torch.float16)):
            with self.subTest(cubes=cubes, replicas=replicas, dtype=dtype):
                actual = self.backward_case((1,), cubes, replicas, dtype)
                self.assertEqual(actual, 2, f"expected_full_zero_calls=2, actual={actual}")
