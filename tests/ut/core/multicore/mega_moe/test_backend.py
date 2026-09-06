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
"""Unit tests for the Torch MegaMoe backend."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from hyper_parallel.core.multicore.modules.mega_moe import (
    module as backend_module,
)
from hyper_parallel.core.multicore.modules.mega_moe.module import (
    MegaMoeExperts,
)


class TestMegaMoeExperts(unittest.TestCase):
    """Validate backend selection without distributed hardware."""

    @patch.object(MegaMoeExperts, "_validate_forward_inputs")
    @patch.object(backend_module, "execute_mega_moe")
    @patch.object(backend_module, "prepare_topk_route")
    def test_receive_overflow_uses_kernel_multi_wave(
        self,
        mock_prepare_route,
        mock_execute_mega_moe,
        mock_validate_tensors,
    ):
        """
        Feature: kernel-internal multi-wave backend selection
        Description: prepare a route whose metadata requests managed execution
        Expectation: execute MegaMoe directly without selecting native fallback
        """
        metadata = Mock(name="metadata")
        execution_runtime = Mock(name="execution_runtime")
        route = SimpleNamespace(
            plan=execution_runtime,
            routed_tokens=Mock(name="routed_tokens"),
            physical_topk_ids=Mock(name="physical_topk_ids"),
            unpermute_mapping=Mock(name="unpermute_mapping"),
            tokens_per_expert=Mock(name="tokens_per_expert"),
            received_counts=Mock(name="received_counts"),
            receive_overflow=False,
            metadata=metadata,
            placement=None,
        )
        mock_prepare_route.return_value = route
        mega_output = Mock(name="mega_output")
        mock_execute_mega_moe.return_value = mega_output
        backend = MegaMoeExperts(
            local_num_tokens=128, hidden_size=4, intermediate_size=3,
            num_experts=4, top_k=2, ep_size=2,
        )
        self.addCleanup(backend.close)
        backend.spec = Mock(name="spec")
        backend.plan = Mock(name="plan")
        backend.workspace = Mock(name="workspace")
        backend.replica_manager = None

        hidden_states = Mock(name="hidden_states")
        topk_ids = Mock(name="topk_ids")
        topk_weights = Mock(name="topk_weights")
        route_ids = Mock(name="route_ids")
        route_weights = Mock(name="route_weights")
        topk_ids.to.return_value.contiguous.return_value = route_ids
        topk_weights.float.return_value.contiguous.return_value = route_weights
        weight1 = backend.gate_up_weight
        weight2 = backend.down_weight
        backend._get_execution_resources = Mock(return_value=backend)
        backend._validate_forward_inputs.return_value = hidden_states
        mock_execute_mega_moe.return_value.reshape.return_value = mock_execute_mega_moe.return_value
        result = backend.forward(
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert=route.tokens_per_expert,
        )

        self.assertIs(result, mega_output)
        mock_validate_tensors.assert_called_once_with(
            hidden_states, topk_ids, topk_weights, route.tokens_per_expert,
        )
        mock_execute_mega_moe.assert_called_once_with(
            hidden_states,
            route.physical_topk_ids,
            route_weights,
            weight1,
            weight2,
            route,
            execution_runtime,
            backend.workspace,
        )

    @patch.object(MegaMoeExperts, "_validate_forward_inputs")
    @patch.object(backend_module, "execute_mega_moe")
    @patch.object(backend_module, "prepare_topk_route")
    def test_replica_route_prefetches_before_entering_logical_weight_autograd(
        self,
        mock_prepare_route: Mock,
        mock_execute_mega_moe: Mock,
        _mock_validate_tensors: Mock,
    ) -> None:
        """
        Feature: physical expert backend integration
        Description: prepare a route carrying a hot-expert placement
        Expectation: defer materialization to MoE autograd while passing logical home weights
        """
        placement = Mock(name="placement")
        execution_runtime = Mock(name="execution_runtime")
        route = SimpleNamespace(
            plan=execution_runtime,
            physical_topk_ids=Mock(name="physical_topk_ids"),
            placement=placement,
            placement_payload=None,
        )
        mock_execute_mega_moe.return_value = Mock(name="output")
        backend = MegaMoeExperts(
            local_num_tokens=128, hidden_size=4, intermediate_size=3,
            num_experts=4, top_k=2, ep_size=2,
        )
        self.addCleanup(backend.close)
        backend.spec = Mock(name="spec")
        backend.plan = Mock(name="plan")
        backend.home_runtime = Mock(name="home_runtime")
        backend.workspace = Mock(name="workspace")
        backend.replica_manager = Mock(name="replica_manager")
        pending = backend.replica_manager.start_prefetch.return_value

        def prepare_with_prefetch(*_args: Any, **kwargs: Any) -> Any:
            """Launch the backend callback at the route/weight overlap point."""
            backend.replica_manager.prepare_for_forward.assert_called_once_with(
                weight1,
                weight2,
            )
            route.placement_payload = kwargs["placement_callback"](placement)
            return route

        mock_prepare_route.side_effect = prepare_with_prefetch
        hidden_states = Mock(name="hidden_states")
        topk_ids = Mock(name="topk_ids")
        topk_weights = Mock(name="topk_weights")
        route_ids = topk_ids.to.return_value.contiguous.return_value
        route_weights = topk_weights.float.return_value.contiguous.return_value
        weight1 = backend.gate_up_weight
        weight2 = backend.down_weight

        backend._get_execution_resources = Mock(return_value=backend)
        backend.plan = backend.plan
        backend._validate_forward_inputs.return_value = hidden_states
        mock_execute_mega_moe.return_value.reshape.return_value = mock_execute_mega_moe.return_value
        result = backend.forward(
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert=None,
        )

        self.assertIs(result, mock_execute_mega_moe.return_value)
        backend.replica_manager.prepare_for_forward.assert_called_once_with(
            weight1,
            weight2,
        )
        backend.replica_manager.start_prefetch.assert_called_once_with(
            weight1,
            weight2,
            placement,
        )
        pending.wait.assert_not_called()
        prepare_call = mock_prepare_route.call_args
        self.assertEqual(
            prepare_call.args,
            (hidden_states, route_ids, route_weights, backend.plan, None),
        )
        self.assertIs(prepare_call.kwargs["workspace"], backend.workspace)
        self.assertTrue(callable(prepare_call.kwargs["placement_callback"]))
        self.assertIs(
            prepare_call.kwargs["home_runtime"],
            backend.home_runtime,
        )
        backend.workspace.ensure.assert_called_once_with(
            backend.spec,
            hidden_states.dtype,
            hidden_states.device,
        )
        mock_execute_mega_moe.assert_called_once_with(
            hidden_states,
            route.physical_topk_ids,
            route_weights,
            weight1,
            weight2,
            route,
            execution_runtime,
            backend.workspace,
        )

    @patch.object(MegaMoeExperts, "_validate_forward_inputs")
    @patch.object(backend_module, "execute_mega_moe")
    @patch.object(backend_module, "prepare_topk_route")
    def test_home_route_tracks_lazy_compact_graph_before_execution(
        self,
        mock_prepare_route,
        mock_execute_mega_moe,
        _mock_validate_tensors,
    ):
        """Track a home-only graph that can overlap first compact placement.

        Feature: lazy compact-layout graph safety.
        Description: Prepare a replica-free route with a replica manager.
        Expectation: Eager preparation precedes routing and leased home weights execute.
        """
        execution_runtime = Mock(name="execution_runtime")
        route = SimpleNamespace(
            plan=execution_runtime,
            physical_topk_ids=Mock(name="physical_topk_ids"),
            placement=None,
        )
        mock_prepare_route.return_value = route
        mock_execute_mega_moe.return_value = Mock(name="output")
        backend = MegaMoeExperts(
            local_num_tokens=128, hidden_size=4, intermediate_size=3,
            num_experts=4, top_k=2, ep_size=2,
        )
        self.addCleanup(backend.close)
        backend.spec = Mock(name="spec")
        backend.plan = Mock(name="plan")
        backend.home_runtime = Mock(name="home_runtime")
        backend.workspace = Mock(name="workspace")
        backend.replica_manager = Mock(name="replica_manager")
        leased_weight1 = Mock(name="leased_weight1")
        leased_weight2 = Mock(name="leased_weight2")
        backend.replica_manager.materialize_home_weights.return_value = (
            leased_weight1,
            leased_weight2,
        )
        hidden_states = Mock(name="hidden_states")
        topk_ids = Mock(name="topk_ids")
        topk_weights = Mock(name="topk_weights")
        route_weights = topk_weights.float.return_value.contiguous.return_value
        weight1 = backend.gate_up_weight
        weight2 = backend.down_weight

        backend._get_execution_resources = Mock(return_value=backend)
        backend._validate_forward_inputs.return_value = hidden_states
        mock_execute_mega_moe.return_value.reshape.return_value = mock_execute_mega_moe.return_value
        result = backend.forward(
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert=None,
        )

        self.assertIs(result, mock_execute_mega_moe.return_value)
        backend.replica_manager.prepare_for_forward.assert_called_once_with(
            weight1,
            weight2,
        )
        backend.replica_manager.materialize_home_weights.assert_called_once_with(
            weight1,
            weight2,
        )
        mock_execute_mega_moe.assert_called_once_with(
            hidden_states,
            route.physical_topk_ids,
            route_weights,
            leased_weight1,
            leased_weight2,
            route,
            execution_runtime,
            backend.workspace,
        )


if __name__ == "__main__":
    unittest.main()
