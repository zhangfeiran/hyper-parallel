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
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
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
            ({"capacity_growth_factor": True}, "capacity_growth_factor"),
            ({"initial_capacity_factor": "1.0"}, "initial_capacity_factor"),
            ({"capacity_growth_factor": 10**400}, "capacity_growth_factor"),
            ({"initial_capacity_factor": 1.0, "dispatch_mode": "pull"}, "push"),
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

    def test_transport_factors_and_sharing_compatibility(self) -> None:
        """Keep pull independent of capacity knobs and reject sharing different push growth factors."""
        arguments = {"local_num_tokens": 128, "hidden_size": 16, "intermediate_size": 8,
                     "num_experts": 4, "top_k": 2, "ep_size": 2, "create_parameters": False}
        layers = [MegaMoeExperts(**arguments, dispatch_mode="pull"),
                  MegaMoeExperts(**arguments, initial_capacity_factor=2.0, capacity_growth_factor=1.0),
                  MegaMoeExperts(**arguments, initial_capacity_factor=2.0, capacity_growth_factor=2.0)]
        try:
            self.assertIsNone(layers[0].initial_capacity_factor)
            self.assertIsNone(layers[0].capacity_growth_factor)
            self.assertEqual(layers[1].initial_capacity_factor, 2.0)
            self.assertEqual(layers[1].capacity_growth_factor, 1.0)
            with self.assertRaises(ValueError):
                MegaMoeExperts.share_execution_resources(layers[1:])
        finally:
            for layer in layers:
                layer.close()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_validates_and_records_swiglu_limit(
        self,
        mock_create_parameters: Mock,
    ) -> None:
        """Feature: validate the model-facing SwiGLU clamp option.

        Description: Construct experts with one valid limit and several invalid
            or non-float32-representable values.
        Expectation: The valid limit is retained and invalid limits fail before
            parameter allocation.
        """
        mock_create_parameters.return_value = (object(), object())
        experts = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            swiglu_limit=10,
            ep_size=2,
        )
        try:
            self.assertEqual(experts.swiglu_limit, 10.0)
            self.assertEqual(
                experts._resource_group.specification["swiglu_limit"],
                10.0,
            )
        finally:
            experts.close()

        for invalid_limit in (
            0,
            -1,
            1e-50,
            1e39,
            float("nan"),
            float("inf"),
            True,
            "10",
        ):
            with (
                self.subTest(swiglu_limit=invalid_limit),
                self.assertRaisesRegex(ValueError, "swiglu_limit"),
            ):
                MegaMoeExperts(
                    local_num_tokens=128,
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=4,
                    top_k=2,
                    swiglu_limit=invalid_limit,
                    ep_size=2,
                )

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

    def test_replica_target_load_rejects_invalid_values_before_allocation(self) -> None:
        """Reject invalid independent load targets before creating parameters."""
        with patch.object(mega_moe_module, "_create_mega_moe_parameters") as create:
            for target in (0, -1, True, 1.5, "256"):
                with self.subTest(target=target), self.assertRaisesRegex(ValueError, "replica_target_load"):
                    MegaMoeExperts(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                                   num_experts=4, top_k=2, ep_size=2, replica_target_load=target)
            create.assert_not_called()

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_replica_target_load_is_independent_of_capacity_in_push_and_pull(self) -> None:
        """
        description: Preserve default routing targets and pass explicit targets without resizing buffers.
        expectation: Explicit targets and invocation graph switches preserve workspace ownership.
        feature: Expert hot replica resource selection.
        """
        hidden = torch.ones(128, 16, dtype=torch.bfloat16)
        ids = torch.zeros(128, 2, dtype=torch.int32)
        probabilities = torch.full((128, 2), 0.5)
        for mode, target, expected in (("push", None, 384), ("push", 256, 256),
                                       ("pull", None, None), ("pull", 256, 256)):
            with self.subTest(mode=mode, target=target):
                experts = MegaMoeExperts(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                                         num_experts=4, top_k=2, ep_size=2, dispatch_mode=mode,
                                         replica_slots_per_rank=1, replica_target_load=target).bfloat16()
                resources = SimpleNamespace(spec=SimpleNamespace(rank_id=0, replica_slots_per_rank=1),
                                            plan=SimpleNamespace(spec=object()),
                                            home_plan=None,
                                            workspace=Mock(capacity_floor=384), heap_manager=Mock(), ensure_capacity=Mock())
                replica = SimpleNamespace(physical_ids=ids, counts_by_source=torch.zeros(2, 6), home_only=False)
                route = SimpleNamespace(routed_tokens=object(), metadata=object(), unpermute_mapping=object(),
                                        maximum_received_slots=256)
                try:
                    with (patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True),
                          patch.object(torch.npu, "is_current_stream_capturing", return_value=False),
                          patch.object(experts, "_get_execution_resources", return_value=resources),
                          patch.object(mega_moe_module, "prepare_replica_route", return_value=replica) as prepare,
                          patch.object(mega_moe_module, "prepare_topk_route", return_value=route),
                          patch.object(mega_moe_module, "execute_mega_moe_with_permutation", return_value=hidden),
                          patch.object(mega_moe_module, "restore_topk_output", return_value=hidden)):
                        torch.testing.assert_close(experts(hidden, ids, probabilities), hidden)
                    self.assertEqual(prepare.call_args.kwargs["target_load"], expected)
                    self.assertEqual(prepare.call_args.kwargs["minimum_replica_rows"], 1024)
                    self.assertEqual(resources.workspace.capacity_floor, 384)
                    if mode == "push":
                        resources.ensure_capacity.assert_called_once_with(256)
                finally:
                    experts.close()

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_home_only_execution_can_switch_back_to_replica_graph(self) -> None:
        """
        description: Alternate globally empty and active guest plans while retaining physical storage.
        expectation: Explicit targets and invocation graph switches preserve workspace ownership.
        feature: Expert hot replica resource selection.
        """
        hidden = torch.ones(128, 16, dtype=torch.bfloat16)
        ids, probabilities = torch.zeros(128, 2, dtype=torch.int32), torch.full((128, 2), 0.5)
        for mode, budget in (("push", 1), ("push", 2), ("pull", 1), ("pull", 2)):
            with self.subTest(mode=mode, budget=budget):
                experts = MegaMoeExperts(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                                         num_experts=4, top_k=2, ep_size=2, dispatch_mode=mode,
                                         replica_slots_per_rank=budget).bfloat16()
                spec = SimpleNamespace(rank_id=0, replica_slots_per_rank=budget)
                resources = SimpleNamespace(spec=spec, plan=SimpleNamespace(spec=spec),
                                            home_plan=SimpleNamespace(spec=SimpleNamespace(rank_id=0)),
                                            workspace=Mock(capacity_floor=384, in_use=True), heap_manager=Mock(), ensure_capacity=Mock())
                route = SimpleNamespace(unpermute_mapping=object(), maximum_received_slots=256)
                try:
                    with (patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True),
                          patch.object(torch.npu, "is_current_stream_capturing", return_value=False),
                          patch.object(experts, "_get_execution_resources", return_value=resources),
                          patch.object(mega_moe_module, "prepare_replica_route") as prepare,
                          patch.object(mega_moe_module, "prepare_topk_route", return_value=route) as topk,
                          patch.object(mega_moe_module, "execute_mega_moe_with_permutation",
                                       return_value=hidden) as execute,
                          patch.object(mega_moe_module, "restore_topk_output", return_value=hidden)):
                        for home_only in (True, False, True, False):
                            prepare.return_value = SimpleNamespace(physical_ids=ids, home_only=home_only,
                                                                   counts_by_source=torch.zeros(2, 4))
                            torch.testing.assert_close(experts(hidden, ids, probabilities), hidden)
                            selected = resources.home_plan if home_only else resources.plan
                            self.assertIs(topk.call_args.args[3], selected.spec)
                            self.assertIs(execute.call_args.args[5], selected)
                            self.assertIs(resources.last_execution_plan, selected)
                            self.assertTrue(prepare.call_args.kwargs["compact_home"])
                    self.assertTrue(all(call.args[0] is spec for call in resources.workspace.ensure.call_args_list))
                    self.assertEqual(resources.workspace.ensure.call_count, 2 if mode == "push" else 6)
                finally:
                    experts.close()

    def test_home_plan_requires_matching_persistent_ready_offsets(self) -> None:
        """Reject workspace sharing when guest slots change the ready-generation offset."""
        for logical, expected_plans in ((96, 2), (800, 1)):
            spec = MegaMoeSpec(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                               num_experts=logical + 16 * 4, logical_num_experts=logical,
                               replica_slots_per_rank=4, top_k=2, initial_capacity_factor=1.5,
                               receive_capacity=384, ep_size=16, ep_group=object(), rank_id=0, num_cube_cores=24)
            with (self.subTest(logical=logical),
                  patch.object(mega_moe_module, "bind_mega_moe_spec", return_value=spec),
                  patch.object(mega_moe_module.shmem, "get_heap_manager",
                               return_value=Mock(heap_bytes=1024, growth_records=[])),
                  patch.object(mega_moe_module.shmem, "acquire"),
                  patch.object(mega_moe_module, "build_mega_moe_plan") as build,
                  patch.object(mega_moe_module, "MegaMoeWorkspace")):
                resources = mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                    {}, torch.empty(0), shared=False, active_specifications=())
                self.assertEqual(build.call_count, expected_plans)
                if expected_plans == 2:
                    home_spec = build.call_args.args[0]
                    self.assertEqual(home_spec.num_experts, logical)
                    self.assertEqual(home_spec.replica_slots_per_rank, 0)
                    self.assertEqual(home_spec.receive_capacity, spec.receive_capacity)
                else:
                    self.assertIsNone(resources.home_plan)

    def test_pull_forward_releases_route_lease_on_success_and_failure(self) -> None:
        """Keep preparation and execution within one lease, including every failure boundary."""
        experts = MegaMoeExperts(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                                 num_experts=4, top_k=2, ep_size=2, dispatch_mode="pull").bfloat16()
        self.addCleanup(experts.close)
        hidden = torch.ones(2, 64, 16, dtype=torch.bfloat16)
        ids, probs = torch.zeros(128, 2, dtype=torch.int32), torch.full((128, 2), 0.5)
        calls = Mock()
        resources = SimpleNamespace(spec=object(), plan=object(), workspace=calls.workspace)
        stages = (calls.workspace.ensure, calls.workspace.claim, calls.prepare, calls.execute)
        names = ["workspace.ensure", "workspace.claim", "prepare", "execute"]

        def _prepare(*_args, **_kwargs):
            self.assertFalse(torch.is_grad_enabled())
            return object()

        with (patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True),
              patch.object(experts, "_get_execution_resources", return_value=resources),
              patch.object(mega_moe_module, "prepare_topk_route", calls.prepare),
              patch.object(mega_moe_module, "execute_mega_moe_with_permutation", calls.execute),
              patch.object(mega_moe_module, "restore_topk_output") as restore):
            for failure in (None, 0, 1, 2, 3):
                calls.reset_mock(side_effect=True)
                calls.prepare.side_effect = _prepare
                calls.execute.return_value = hidden + 1
                if failure is None:
                    torch.testing.assert_close(experts(hidden, ids, probs), hidden + 1)
                    self.assertTrue(calls.execute.call_args.kwargs["workspace_claimed"])
                    self.assertIs(calls.execute.call_args.kwargs["topk_weights"], probs)
                else:
                    stages[failure].side_effect = RuntimeError("injected failure")
                    with self.assertRaisesRegex(RuntimeError, "injected failure"):
                        experts(hidden, ids, probs)
                expected = names if failure is None else names[:failure + 1]
                if failure not in (0, 1):
                    expected = expected + ["workspace.release"]
                self.assertEqual([entry[0] for entry in calls.mock_calls], expected)
            restore.assert_not_called()

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
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128,
                                     dispatch_mode="push", replica_slots_per_rank=0)
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
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128,
                                     dispatch_mode="push", replica_slots_per_rank=0)

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

    def test_replica_bound_precedes_existing_capacity_fast_path(self) -> None:
        """Reject impossible receive counts even when the old heap happens to be larger."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(mega_moe_module._MegaMoeExecutionResources)
        resources.spec = SimpleNamespace(maximum_receive_capacity=24576)
        resources.workspace = SimpleNamespace(capacity_floor=49152)
        resources.heap_manager = Mock()
        with self.assertRaisesRegex(ValueError, "lossless token bound"):
            resources.ensure_capacity(24577)
        resources.heap_manager.replace.assert_not_called()
