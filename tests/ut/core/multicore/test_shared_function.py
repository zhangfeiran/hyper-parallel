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
"""Exercise real shared autograd contexts while emulating only device math on CPU."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as functional

from hyper_parallel.core.multicore.modules.mega_moe import function as moe_function


def shared_math(
    source: torch.Tensor, first: torch.Tensor, second: torch.Tensor, limit: float | None = None,
) -> torch.Tensor:
    """Compute an independent dense SwiGLU reference."""
    gate, up = functional.linear(source, first).chunk(2, dim=-1)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return functional.linear(functional.silu(gate) * up, second)


class TestSharedFunction(unittest.TestCase):
    """Check input/parameter gradient bindings after multiple deferred forwards."""

    # The private launch boundary is the lifecycle contract under test.
    # pylint: disable=protected-access

    def test_deferred_shared_outputs_and_all_parameter_gradients(self) -> None:
        """Release poisoning must not invalidate shared saved tensors or owned outputs."""
        for input_grad, limit in ((False, None), (True, None), (True, 1.25)):
            with self.subTest(input_grad=input_grad, limit=limit):
                self._check_deferred(input_grad, limit)

    def _check_deferred(self, input_grad: bool, limit: float | None) -> None:
        """Emulate routed zeros and compare the integrated shared gradients to Torch."""
        torch.manual_seed(73)
        spec = SimpleNamespace(hidden_size=4, intermediate_size=2, rank_id=0, ep_size=2,
                               num_experts=4, top_k=2, dispatch_mode="push", shared_intermediate_size=2,
                               forward_ingress_mode="legacy", swiglu_limit=limit)
        plan = SimpleNamespace(spec=spec, reuse_backward_dispatch=False, ingress_diagnostics=None)
        workspace = Mock(expert_buffer=torch.zeros(8, 4), routed_buffer=torch.zeros(4, 4),
                         gmm_workspace=torch.zeros(512, dtype=torch.uint8), swiglu_grad_workspace=torch.zeros(1))
        workspace.release.side_effect = lambda: (workspace.expert_buffer.fill_(float("nan")),
                                                 workspace.routed_buffer.fill_(float("nan")))
        w1 = torch.randn(4, 4, requires_grad=True)
        w2 = torch.randn(4, 2, requires_grad=True)
        if limit is not None:
            with torch.no_grad():
                w1[:, 0].copy_(torch.tensor([limit, -limit, limit, -limit]))
        ref1, ref2 = w1.detach().clone().requires_grad_(), w2.detach().clone().requires_grad_()
        routed1, routed2 = torch.ones(2, 4, 4, requires_grad=True), torch.ones(2, 2, 4, requires_grad=True)

        def forward_kernel(_plan, _route, _source, _first, _second, execution) -> None:
            """Fill independent outputs through the actual indirect tensor table."""
            source, first, second, up, activation, output, _scratch = execution.shared_ptrs
            up.copy_(functional.linear(source, first))
            gate, branch = up.chunk(2, dim=-1)
            if limit is not None:
                gate, branch = gate.clamp(max=limit), branch.clamp(-limit, limit)
            activation.copy_(functional.silu(gate) * branch)
            output.copy_(functional.linear(activation, second))
            execution.dispatch.zero_()
            execution.combine.zero_()

        def backward_kernel(_plan, _saved, _gradient, execution) -> None:
            """Keep routed derivatives zero while shared uses its independent native node."""
            execution.grad_x.zero_()

        def activation_backward(gradient: torch.Tensor, up: torch.Tensor, _dim: int) -> torch.Tensor:
            """Emulate only the native activation derivative on CPU."""
            with torch.enable_grad():
                value = up.detach().clone().requires_grad_()
                gate, branch = value.chunk(2, dim=-1)
                output = functional.silu(gate) * branch
                return torch.autograd.grad(output, value, gradient)[0]

        calls = []
        with (
            patch.object(moe_function, "pointer_table", side_effect=lambda tensors, **_: tensors),
            patch.object(moe_function.torch.ops.npu, "npu_swiglu_backward", side_effect=activation_backward),
            patch.object(moe_function, "_launch_forward_kernel", side_effect=forward_kernel),
            patch.object(moe_function, "_launch_backward_kernel", side_effect=backward_kernel),
            patch.object(moe_function.multicore_ops, "moe_token_permute_grad",
                         side_effect=lambda grad, _mapping, rows, top_k: grad.reshape(rows, top_k, 4).sum(1)),
        ):
            for _ in range(2):
                source = (torch.randn(2, 4) if limit is None else torch.eye(4)[:2]).requires_grad_(input_grad)
                reference = source.detach().clone().requires_grad_(input_grad)
                route = SimpleNamespace(expert_capacity=3, group_list=torch.tensor([0, 3]))
                for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size",
                             "combine_src_off", "combine_target_off", "combine_size"):
                    setattr(route, name, torch.zeros(4, dtype=torch.int64))
                prepared = SimpleNamespace(metadata=route, routed_tokens=source.detach().repeat_interleave(2, dim=0),
                                           unpermute_mapping=torch.arange(4, dtype=torch.int32))
                routed_output, shared_output = moe_function.execute_mega_moe_with_permutation(
                    source, torch.zeros(2, 2, dtype=torch.int32), routed1, routed2, prepared, plan, workspace,
                    shared_weight1=w1, shared_weight2=w2)
                output = shared_output + routed_output.reshape(2, 2, 4).sum(1)
                expected = shared_math(reference, ref1, ref2, limit)
                torch.testing.assert_close(output, expected)
                calls.append((source, reference, output, expected, torch.randn_like(output)))
            for source, reference, output, expected, dy in reversed(calls):
                output.backward(dy)
                expected.backward(dy)
                if input_grad:
                    torch.testing.assert_close(source.grad, reference.grad)
                else:
                    self.assertIsNone(source.grad)
            torch.testing.assert_close(w1.grad, ref1.grad)
            torch.testing.assert_close(w2.grad, ref2.grad)
        self.assertEqual(workspace.claim.call_count, 4)
        self.assertEqual(workspace.release.call_count, 4)


if __name__ == "__main__":
    unittest.main()
