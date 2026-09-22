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
"""Unit tests for the model-facing MegaMoe module."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.core.multicore.modules import module as module_api
from hyper_parallel.core.multicore.modules.mega_moe.workspace import MegaMoeWorkspace
from hyper_parallel.core.multicore.modules.mega_moe import module as mega_moe_module
from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts
from tests.common.mark_utils import arg_mark


class TestMegaMoeExperts(unittest.TestCase):
    """Validate the public API and execution-resource lifecycle."""

    # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_defaults_to_growing_push(self, mock_create_parameters: Mock) -> None:
        """Feature: Push capacity defaults.
        Description: Construct a layer without explicit capacity factors.
        Expectation: Resolve both factors to 1.25 and create the expected local weights.
        """
        mock_create_parameters.return_value = (object(), object())
        experts = MegaMoeExperts(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                                 num_experts=4, top_k=2, ep_size=2)
        self.addCleanup(experts.close)
        self.assertEqual((experts.dispatch_mode, experts.initial_capacity_factor, experts.capacity_growth_factor),
                         ("push", 1.25, 1.25))
        mock_create_parameters.assert_called_once_with(2, 16, 8)

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_rejects_invalid_static_values_before_allocation(
        self,
        mock_create_parameters: Mock,
    ) -> None:
        """Reject invalid capacity and token split during construction."""
        for overrides, message in (
            ({"local_num_tokens": 129}, "divisible"),
            ({"initial_capacity_factor": 0.999}, "initial_capacity_factor"),
            ({"num_experts": 35}, "divisible"),
            ({"dispatch_mode": "invalid"}, "dispatch_mode"),
            ({"capacity_growth_factor": 0.999}, "capacity_growth_factor"),
            ({"initial_capacity_factor": float("nan")}, "initial_capacity_factor"),
            ({"capacity_growth_factor": float("inf")}, "capacity_growth_factor"),
            ({"initial_capacity_factor": True}, "initial_capacity_factor"),
            ({"capacity_growth_factor": 1.5, "dispatch_mode": "pull"}, "push"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                MegaMoeExperts(
                    local_num_tokens=overrides.get("local_num_tokens", 128),
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=overrides.get("num_experts", 4),
                    dispatch_mode=overrides.get("dispatch_mode", "push"),
                    capacity_growth_factor=overrides.get("capacity_growth_factor"),
                    top_k=2,
                    initial_capacity_factor=overrides.get("initial_capacity_factor"),
                    ep_size=2,
                )

        mock_create_parameters.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_passes_router_inputs_and_restores_shape(self) -> None:
        """Feature: forward passes router inputs and restores shape.
        Description: Execute forward with router metadata and caller-shaped inputs.
        Expectation: Preserve Router inputs, expert parameters and the caller's shape.
        """
        experts = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            ep_size=2,
        ).to(dtype=torch.bfloat16)
        self.addCleanup(experts.close)
        hidden_states = torch.arange(2048, dtype=torch.bfloat16).reshape(2, 64, 16)
        topk_ids = torch.zeros((128, 2), dtype=torch.int32)
        topk_weights = torch.full((128, 2), 0.5)
        tokens_per_expert = torch.tensor([256, 0, 0, 0], dtype=torch.int32)
        expected = hidden_states.reshape(128, 16) + 1
        resources = SimpleNamespace(spec=object(), plan=object(), workspace=object(), ensure_capacity=Mock())
        route = SimpleNamespace(
            routed_tokens=object(), metadata=object(), unpermute_mapping=object(), maximum_received_slots=512
        )
        expert_output = object()

        with (
            patch.object(
                torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True
            ),
            patch.object(torch.npu, "is_current_stream_capturing", return_value=False),
            patch.object(experts, "_get_execution_resources", return_value=resources),
            patch.object(
                mega_moe_module, "prepare_topk_route", return_value=route
            ) as mock_prepare,
            patch.object(
                mega_moe_module, "execute_mega_moe_with_permutation", return_value=expert_output
            ) as mock_execute,
            patch.object(
                mega_moe_module, "restore_topk_output", return_value=expected
            ) as mock_restore,
        ):
            actual = experts(
                hidden_states,
                topk_ids,
                topk_weights,
                tokens_per_expert=tokens_per_expert,
            )

        torch.testing.assert_close(actual, expected.reshape(hidden_states.shape))
        hidden_flat = mock_prepare.call_args.args[0]
        torch.testing.assert_close(hidden_flat, hidden_states.reshape(128, 16))
        mock_prepare.assert_called_once_with(
            hidden_flat, topk_ids, topk_weights, resources.spec, tokens_per_expert,
            workspace=resources.workspace, replica_route=None,
        )
        resources.ensure_capacity.assert_called_once_with(512)
        mock_execute.assert_called_once_with(
            hidden_flat,
            topk_ids,
            experts.gate_up_weight,
            experts.down_weight,
            route,
            resources.plan,
            resources.workspace,
            topk_weights=None,
            workspace_claimed=False,
        )
        mock_restore.assert_called_once_with(
            expert_output, route.unpermute_mapping, topk_weights
        )

    def test_forward_rejects_invalid_weights_before_resource_creation(self) -> None:
        """Reject invalid expert weights before initializing native resources."""
        experts = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            ep_size=2,
        ).to(dtype=torch.bfloat16)
        self.addCleanup(experts.close)
        experts.down_weight = torch.nn.Parameter(
            torch.zeros((2, 16, 8), dtype=torch.bfloat16)
        )

        with (
            patch.object(
                torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True
            ),
            patch.object(
                experts, "_create_execution_resources"
            ) as mock_create_resources,
            self.assertRaisesRegex(ValueError, "down_weight must have shape"),
        ):
            experts(
                torch.zeros((128, 16), dtype=torch.bfloat16),
                torch.zeros((128, 2), dtype=torch.int32),
                torch.full((128, 2), 0.5),
            )
        experts.gate_up_weight = experts.down_weight = None
        with (patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True),
              patch.object(experts, "_create_execution_resources", mock_create_resources)):
            for weights, error in ((None, ValueError),
                                   ((torch.zeros(2, 16, 16), torch.zeros(2, 8, 16)), TypeError)):
                with self.assertRaises(error):
                    experts(torch.zeros(128, 16, dtype=torch.bfloat16), torch.zeros(128, 2),
                            torch.ones(128, 2), expert_weights=weights)
        mock_create_resources.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    @patch.object(MegaMoeExperts, "_retain_runtime", return_value=None)
    def test_shared_layers_create_once_and_close_after_last_owner(
        self,
        mock_retain_runtime: Mock,
        mock_create_parameters: Mock,
    ) -> None:
        """Keep parameters independent while one serial resource group is shared."""
        mock_create_parameters.return_value = (object(), object())
        layers = [
            MegaMoeExperts(
                local_num_tokens=128,
                hidden_size=16,
                intermediate_size=8,
                num_experts=4,
                top_k=2,
                ep_size=2,
            )
            for _ in range(4)
        ]
        resources = Mock()
        input_tensor = SimpleNamespace(device="npu:0", dtype="bfloat16")

        other = MegaMoeExperts(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                               num_experts=4, top_k=2, ep_size=2, dispatch_mode="pull")
        self.addCleanup(other.close)
        with self.assertRaises(ValueError):
            MegaMoeExperts.share_execution_resources([layers[0], other])
        other.close()
        MegaMoeExperts.share_execution_resources(layers)
        shared_group = layers[0]._resource_group
        with patch.object(
            MegaMoeExperts,
            "_create_execution_resources",
            return_value=resources,
        ) as mock_create_resources:
            resolved = [
                layer._get_execution_resources(input_tensor) for layer in layers
            ]
        mock_retain_runtime.assert_called_once()

        self.assertTrue(shared_group.shared)
        self.assertEqual(len(shared_group.members), len(layers))
        self.assertTrue(all(layer._resource_group is shared_group for layer in layers))
        self.assertTrue(all(value is resources for value in resolved))
        mock_create_resources.assert_called_once()
        self.assertTrue(mock_create_resources.call_args.kwargs["shared"])
        self.assertEqual(
            len(mock_create_resources.call_args.kwargs["active_specifications"]),
            1,
        )

        for layer in layers[:-1]:
            layer.close()
        resources.close.assert_not_called()
        layers[-1].close()
        layers[-1].close()
        resources.close.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_execution_resource_pairs_shmem_acquire_and_release(self) -> None:
        """Feature: execution resource pairs shmem acquire and release.
        Description: Create one execution resource and close it twice.
        Expectation: Pair one SHMEM reference with one execution-resource lifetime.
        """
        root_group = object()
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128, dispatch_mode="push")
        workspace = Mock()

        with (
            patch.object(
                mega_moe_module,
                "bind_mega_moe_spec",
                return_value=bound_spec,
            ),
            patch.object(mega_moe_module.shmem, "get_heap_manager",
                         return_value=Mock(heap_bytes=1024, access=nullcontext, growth_records=[])),
            patch.object(mega_moe_module.shmem, "acquire") as mock_acquire,
            patch.object(mega_moe_module.shmem, "release") as mock_release,
            patch.object(
                mega_moe_module, "build_mega_moe_plan", return_value=object()
            ) as mock_build_plan,
            patch.object(mega_moe_module, "MegaMoeWorkspace", return_value=workspace),
        ):
            resources = mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                {},
                SimpleNamespace(device="npu:0", dtype=torch.bfloat16, element_size=lambda: 2),
                shared=False,
                active_specifications=(),
            )
            mock_acquire.assert_called_once_with(root_group, heap_size_bytes=1024)
            mock_build_plan.assert_called_once_with(bound_spec, "npu:0")
            mock_release.assert_not_called()
            resources.close()
            resources.close()

        workspace.close.assert_called_once_with()
        mock_release.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_execution_resource_construction_failure_releases_shmem(self) -> None:
        """Feature: execution resource construction failure releases shmem.
        Description: Fail plan creation after acquiring the native reference.
        Expectation: Release the acquired SHMEM reference when resource construction fails.
        """
        root_group = object()
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128, dispatch_mode="push")

        with (
            patch.object(
                mega_moe_module,
                "bind_mega_moe_spec",
                return_value=bound_spec,
            ),
            patch.object(mega_moe_module.shmem, "get_heap_manager",
                         return_value=Mock(heap_bytes=1024, access=nullcontext, growth_records=[])),
            patch.object(mega_moe_module.shmem, "acquire") as mock_acquire,
            patch.object(mega_moe_module.shmem, "release") as mock_release,
            patch.object(
                mega_moe_module,
                "build_mega_moe_plan",
                side_effect=RuntimeError("plan failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "plan failed"),
        ):
            mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                {},
                SimpleNamespace(device="npu:0", dtype=torch.bfloat16, element_size=lambda: 2),
                shared=False,
                active_specifications=(),
            )

        mock_acquire.assert_called_once_with(root_group, heap_size_bytes=1024)
        mock_release.assert_called_once_with()

    def test_workspace_close_failure_keeps_shmem_user(self) -> None:
        """Do not leave SHMEM when a workspace cannot release its resources."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(  # pylint: disable=protected-access
            mega_moe_module._MegaMoeExecutionResources  # pylint: disable=protected-access
        )
        resources.workspace = Mock()
        resources.heap_manager = Mock(access=nullcontext)
        resources.workspace.close.side_effect = RuntimeError("workspace busy")
        resources._workspace_closed = False  # pylint: disable=protected-access
        resources._closed = False  # pylint: disable=protected-access

        with (
            patch.object(mega_moe_module.shmem, "release") as mock_release,
            self.assertRaisesRegex(RuntimeError, "workspace busy"),
        ):
            resources.close()

        mock_release.assert_not_called()
        self.assertFalse(resources._closed)  # pylint: disable=protected-access

    def test_shmem_release_failure_keeps_execution_resource_open(self) -> None:
        """Keep the resource open when its SHMEM reference cannot be released."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(  # pylint: disable=protected-access
            mega_moe_module._MegaMoeExecutionResources  # pylint: disable=protected-access
        )
        resources.workspace = Mock()
        resources._workspace_closed = False  # pylint: disable=protected-access
        resources.heap_manager = Mock(access=nullcontext)
        resources._closed = False  # pylint: disable=protected-access

        with (
            patch.object(
                mega_moe_module.shmem,
                "release",
                side_effect=RuntimeError("release failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "release failed"),
        ):
            resources.close()

        resources.workspace.close.assert_called_once_with()
        self.assertFalse(resources._closed)  # pylint: disable=protected-access

        with patch.object(mega_moe_module.shmem, "release") as release:
            resources.close()
        resources.workspace.close.assert_called_once_with()
        release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
