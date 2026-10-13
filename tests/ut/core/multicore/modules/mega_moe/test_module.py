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

import gc
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.core.multicore.modules import module as module_api
from hyper_parallel.core.multicore.shmem import _heap_manager as heap_module
from hyper_parallel.core.multicore.modules.mega_moe.workspace import MegaMoeWorkspace
from hyper_parallel.core.multicore.modules.mega_moe import module as mega_moe_module
from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.spec import bind_mega_moe_spec
from hyper_parallel.core.multicore.shmem import _lifecycle
from tests.common.mark_utils import arg_mark


class TestMegaMoeExperts(unittest.TestCase):
    """Validate the public API and execution-resource lifecycle."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_noncontiguous_external_weights_pass_metadata_validation(self) -> None:
        """Feature: External expert-weight layout compatibility.

        Description: Supply transposed and sliced weights with valid shapes.
        Expectation: Shape-compatible noncontiguous weights pass input validation.
        """
        layer = MegaMoeExperts(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                               num_experts=4, top_k=2, create_parameters=False)
        self.addCleanup(layer.close)
        hidden = torch.ones(128, 16, dtype=torch.bfloat16)
        ids, probs = torch.zeros(128, 2, dtype=torch.int32), torch.full((128, 2), 0.5)
        weight1 = torch.ones(4, 16, 16, dtype=torch.bfloat16)
        weight2 = torch.ones(4, 8, 16, dtype=torch.bfloat16)
        cases = (
            (weight1.transpose(-1, -2), weight2),
            (weight1, weight2.transpose(-1, -2).contiguous().transpose(-1, -2)),
            (torch.ones(4, 16, 32, dtype=torch.bfloat16)[:, :, ::2], weight2),
        )
        with patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True):
            layer._validate_tensors(hidden, (weight1, weight2))
            for index, weights in enumerate(cases):
                with self.subTest(case=index):
                    actual = layer._validate_forward_inputs(hidden, ids, probs, None, weights)
                    self.assertEqual(actual.data_ptr(), hidden.data_ptr())
                    torch.testing.assert_close(actual, hidden)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_larger_model_binds_after_old_model_gc(self) -> None:
        """Feature: Automatic heap ownership across model replacement.

        Description: Collect the only old model and bind a larger push or pull model.
        Expectation: The actual root pin survives eviction and the larger heap is rebuilt before acquire.
        """
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                self._check_gc_heap_replacement(mode)

    def _check_gc_heap_replacement(self, mode: str) -> None:
        """Exercise the real resource pool and lifecycle with mocked hardware calls."""
        manager = module_api._MulticoreResourceManager()
        state = {"config": {"heap_size_bytes": 0}, "active_allocations": []}
        native = Mock()

        def initialize(_rank: int, _size: int, _unique_id: bytes, heap_bytes: int) -> None:
            """Publish the physical size selected by actual initialization or rebuild."""
            state["config"]["heap_size_bytes"] = heap_bytes

        def debug_state() -> dict:
            """Use the actual acquire/release reference count in the ownership preflight."""
            return {**state, "reference_count": _lifecycle._reference_count()}

        native._initialize.side_effect = initialize
        native._debug_state.side_effect = debug_state
        with (
            patch.object(module_api, "_RESOURCE_MANAGER", manager),
            patch.object(heap_module, "_MANAGER", None),
            patch.object(_lifecycle, "_users", 0),
            patch.object(_lifecycle, "_retained_users", 0),
            patch.object(_lifecycle, "_shutdown_failed", False),
            patch.object(_lifecycle, "_root_uses_distributed", None),
            patch.object(_lifecycle, "_root_size", None),
            patch.object(_lifecycle, "_root_group", None),
            patch.object(_lifecycle, "_root_ranks", None),
            patch.object(_lifecycle, "_load_native", return_value=native),
            patch.object(mega_moe_module.dist, "is_initialized", return_value=False),
            patch.object(module_api._automatic, "register"),
            patch.object(torch.npu, "get_device_limit", return_value={"cube_core_num": 24}),
            patch.object(torch.npu, "is_current_stream_capturing", return_value=False),
            patch.object(torch.npu, "synchronize"),
            patch.object(mega_moe_module, "build_mega_moe_plan", return_value=Mock()),
            patch.object(heap_module, "debug_state", side_effect=debug_state),
            patch.dict(heap_module.os.environ, {}, clear=True),
        ):
            tensor = SimpleNamespace(dtype=torch.bfloat16, device=torch.device("npu:0"), element_size=lambda: 2)
            small = MegaMoeExperts(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                                   num_experts=4, top_k=1, create_parameters=False, dispatch_mode=mode)
            old = small._get_execution_resources(tensor)
            self.assertEqual(old.heap_manager.heap_bytes, 2 * 1024**2)
            del small
            gc.collect()
            large = MegaMoeExperts(local_num_tokens=512, hidden_size=1024, intermediate_size=8,
                                   num_experts=4, top_k=1, create_parameters=False, dispatch_mode=mode)
            try:
                resources = large._get_execution_resources(tensor)
                self.assertIs(resources.heap_manager, old.heap_manager)
                self.assertEqual(resources.heap_manager.heap_bytes, 4 * 1024**2)
                self.assertEqual(resources.heap_manager.epoch, 1)
                self.assertEqual(_lifecycle._reference_count(), 2)
                self.assertEqual(_lifecycle._retained_users, 1)
                self.assertEqual(native._initialize.call_count, 2)
                native._shutdown.assert_called_once_with()
            finally:
                large.close()
            self.assertEqual(_lifecycle._reference_count(), 0)
            self.assertEqual(native._shutdown.call_count, 2)

    def test_collective_signature_ignores_process_local_group_addresses(self) -> None:
        """Feature: Collective resource compatibility.

        Description: Compare equal configurations with distinct process-local group handles.
        Expectation: Local compatibility keys differ, while collective signatures match.
        """
        arguments = {"local_num_tokens": 512, "hidden_size": 16, "intermediate_size": 8,
                     "num_experts": 4, "top_k": 2, "ep_size": 2, "create_parameters": False}
        tensor = SimpleNamespace(dtype=torch.bfloat16)
        with patch.object(mega_moe_module.dist, "is_initialized", return_value=False):
            layers = [MegaMoeExperts(**arguments, ep_group=object()) for _ in range(2)]
        for layer in layers:
            self.addCleanup(layer.close)
        self.assertNotEqual(layers[0]._resource_group.compatibility_key, layers[1]._resource_group.compatibility_key)
        self.assertEqual(layers[0]._resource_signature(tensor), layers[1]._resource_signature(tensor))

    def test_collective_signature_checks_every_static_configuration_field(self) -> None:
        """Feature: Collective configuration validation.

        Description: Change each static configuration field, including both communication splits.
        Expectation: Every configuration change produces a different collective signature.
        """
        arguments = {"local_num_tokens": 512, "hidden_size": 16, "intermediate_size": 8,
                     "num_experts": 4, "top_k": 2, "ep_size": 2, "create_parameters": False}
        tensor = SimpleNamespace(dtype=torch.bfloat16)
        with patch.object(mega_moe_module.dist, "is_initialized", return_value=False):
            first = MegaMoeExperts(**arguments)
            self.addCleanup(first.close)
            for name, value in (("local_num_tokens", 1024), ("hidden_size", 32), ("intermediate_size", 16),
                                ("num_experts", 8), ("top_k", 1), ("ep_size", 1),
                                ("initial_capacity_factor", 2.0), ("capacity_growth_factor", 2.0),
                                ("dispatch_split", 256), ("combine_split", 256), ("dispatch_mode", "pull")):
                with self.subTest(field=name):
                    changed = MegaMoeExperts(**{**arguments, name: value})
                    self.addCleanup(changed.close)
                    self.assertNotEqual(first._resource_signature(tensor), changed._resource_signature(tensor))

    def test_communication_splits_reach_both_graphs_and_isolate_sharing(self) -> None:
        """Propagate independent public split values without sharing incompatible plans."""
        arguments = {"local_num_tokens": 512, "hidden_size": 16, "intermediate_size": 8,
                     "num_experts": 4, "top_k": 2, "create_parameters": False}
        for mode in ("push", "pull"):
            for dispatch_split, combine_split in ((128, 128), (512, 512), (128, 512), (512, 128)):
                with self.subTest(mode=mode, dispatch=dispatch_split, combine=combine_split):
                    layer = MegaMoeExperts(**arguments, dispatch_mode=mode,
                                           dispatch_split=dispatch_split, combine_split=combine_split)
                    self.addCleanup(layer.close)
                    with (patch.object(mega_moe_module.dist, "is_initialized", return_value=False),
                          patch.object(torch.npu, "get_device_limit", return_value={"cube_core_num": 24})):
                        spec = bind_mega_moe_spec(
                            layer._resource_group.specification,
                            SimpleNamespace(device="npu:0", dtype=torch.bfloat16),
                        )
                    self.assertEqual((spec.dispatch_split, spec.combine_split), (dispatch_split, combine_split))
                    forward, _, backward, _ = _build_runtime_artifacts(spec)
                    for graph in (forward, backward):
                        self.assertEqual(graph.get_op("dispatch").split_value, dispatch_split)
                        self.assertEqual(graph.get_op("combine").split_value, combine_split)
                    same = MegaMoeExperts(**arguments, dispatch_mode=mode,
                                          dispatch_split=dispatch_split, combine_split=combine_split)
                    self.addCleanup(same.close)
                    for key in ("dispatch_split", "combine_split"):
                        splits = {"dispatch_split": dispatch_split, "combine_split": combine_split}
                        splits[key] = 512 if splits[key] == 128 else 128
                        other = MegaMoeExperts(**arguments, dispatch_mode=mode, **splits)
                        self.addCleanup(other.close)
                        with self.assertRaises(ValueError):
                            MegaMoeExperts.share_execution_resources([same, other])
                    MegaMoeExperts.share_execution_resources([layer, same])
                    self.assertIs(layer._resource_group, same._resource_group)

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_invalid_communication_splits_fail_before_allocation(self, allocate: Mock) -> None:
        """Reject invalid types, alignment, and incomplete communication tiles."""
        for key in ("dispatch_split", "combine_split"):
            for value in (0, -128, True, 128.0, "512", 64, 129, 384, 1024):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    MegaMoeExperts(local_num_tokens=512, hidden_size=16, intermediate_size=8,
                                   num_experts=4, top_k=2, **{key: value})
        allocate.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_defaults_to_growing_push(
        self,
        mock_create_parameters: Mock,
    ) -> None:
        """Resolve the push-only defaults before registering resource compatibility."""
        mock_create_parameters.return_value = (object(), object())

        experts = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            ep_size=2,
        )
        try:
            self.assertEqual(experts.local_num_tokens, 128)
            self.assertEqual(experts.initial_capacity_factor, 1.25)
            self.assertEqual(
                experts._resource_group.specification,
                {
                    "local_num_tokens": 128,
                    "hidden_size": 16,
                    "intermediate_size": 8,
                    "num_experts": 4,
                    "top_k": 2,
                    "initial_capacity_factor": 1.25,
                    "ep_size": 2,
                    "ep_group": None,
                    "dispatch_mode": "push",
                    "capacity_growth_factor": 1.25,
                    "dispatch_split": 128,
                    "combine_split": 128,
                },
            )
            mock_create_parameters.assert_called_once_with(2, 16, 8)
        finally:
            experts.close()

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
            workspace=resources.workspace,
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


class TestGrowingMegaMoeResources(unittest.TestCase):
    """Exercise capacity planning, ownership checks and live workspace rebinding."""

    def setUp(self) -> None:
        """Bind a CPU-only root with mocked native and NPU lifecycle operations."""
        self.specification = {"local_num_tokens": 4096, "top_k": 8, "hidden_size": 5120,
                              "intermediate_size": 1792, "num_experts": 48, "ep_size": 8,
                              "initial_capacity_factor": 1.0, "dispatch_mode": "push", "capacity_growth_factor": 1.5}
        self.spec = SimpleNamespace(**self.specification, ep_group=None, rank_id=0, routed_slots=32768,
                                    receive_capacity=32768)
        self.addCleanup(patch.stopall)
        patch.object(_lifecycle, "_shutdown_failed", False).start()
        patch.object(_lifecycle, "_retained_users", 0).start()
        patch.dict(heap_module.os.environ, {}, clear=True).start()
        patch.object(mega_moe_module.dist, "is_initialized", return_value=False).start()
        self.manager = heap_module.ShmemHeapManager(None, torch.empty(0, dtype=torch.bfloat16),
                                                   ((self.specification, self._layout(32768)),))
        self.resource = mega_moe_module._MegaMoeExecutionResources.__new__(mega_moe_module._MegaMoeExecutionResources)
        self.resource.spec, self.resource.specification = self.spec, self.specification
        self.resource.dtype, self.resource.device, self.resource.element_size = torch.bfloat16, torch.device("cpu"), 2
        self.resource.heap_manager = self.manager
        self.resource.workspace = MegaMoeWorkspace(shared=True)
        self.state = {"reference_count": 1, "config": {"heap_size_bytes": self.manager.heap_bytes},
                      "active_allocations": []}
        patch.object(heap_module, "debug_state", side_effect=lambda: self.state).start()
        self.resource.workspace.capacity_floor = self.manager.bind(self.resource, self.specification)
        self.resource.workspace.heap_manager = self.manager
        self.free = patch.object(mega_moe_module.shmem, "free").start()
        patch.object(mega_moe_module.shmem, "host_barrier").start()
        patch.object(torch.npu, "is_current_stream_capturing", return_value=False).start()
        self.sync = patch.object(torch.npu, "synchronize").start()
        self.reinitialize = patch.object(heap_module._lifecycle, "_reinitialize", return_value={}).start()

    def _layout(self, capacity: int) -> tuple:
        """Build the layout through the actual MegaMoe byte calculation."""
        return mega_moe_module._heap_layout(self.specification, 2, capacity)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_monotonic_growth_and_fast_path(self) -> None:
        """Feature: monotonic growth and fast path.
        Description: Submit in-capacity demand and jumps beyond geometric headroom.
        Expectation: Do no control work within capacity and grow enough to cover jumps beyond the geometric margin.
        """
        with patch.object(self.manager, "_rebuild") as rebuild:
            self.resource.ensure_capacity(32768)
            rebuild.assert_not_called()
            for received, capacity, heap_mib in ((32769, 49152, 802), (71920, 71936, 1024), (262144, 262144, 2882)):
                with self.subTest(received=received):
                    self.resource.ensure_capacity(received)
                    rebuild.assert_called_with([self._layout(capacity)], heap_mib * 1024**2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_orphan_replacement_grows_the_retained_physical_heap(self) -> None:
        """Feature: Larger model replacement after the last owner is retired.

        Description: Bind push and pull layouts while only the automatic root reference remains.
        Expectation: Reuse the existing manager and rebuild the actual smaller heap before binding.
        """
        self.manager.remove(self.resource)
        tensor = torch.empty(0, dtype=torch.bfloat16)
        for mode in ("push", "pull"):
            specification = dict(self.specification, hidden_size=6144, dispatch_mode=mode,
                                 initial_capacity_factor=1.0 if mode == "push" else None,
                                 capacity_growth_factor=1.5 if mode == "push" else None)
            expected = 770 * 1024**2
            with (
                self.subTest(mode=mode),
                patch.object(heap_module, "_MANAGER", self.manager),
                patch.object(_lifecycle, "_retained_users", 1),
            ):
                selected = heap_module.get_heap_manager(
                    None, tensor, ((specification, mega_moe_module._heap_layout(specification, 2)),))
            self.assertIs(selected, self.manager)
            self.reinitialize.assert_called_once_with(expected)
            self.assertEqual(selected.heap_bytes, expected)
            self.assertEqual(selected.epoch, 1)
            self.assertEqual(len(selected.entries), 1)
            self.manager.entries.clear()
            self.manager.heap_bytes = self.state["config"]["heap_size_bytes"]
            self.manager.epoch = 0
            self.reinitialize.reset_mock()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_closed_runtime_replaces_the_empty_manager(self) -> None:
        """Feature: Fresh root lifecycle after complete shutdown.

        Description: Request a smaller layout once the automatic root reference is released.
        Expectation: Start with a new manager and initial budget without rebuilding the closed runtime.
        """
        self.manager.remove(self.resource)
        specification = dict(self.specification, local_num_tokens=128, top_k=1, hidden_size=16)
        with (
            patch.object(heap_module, "_MANAGER", self.manager),
            patch.object(_lifecycle, "_retained_users", 0),
        ):
            selected = heap_module.get_heap_manager(
                None, torch.empty(0, dtype=torch.bfloat16),
                ((specification, mega_moe_module._heap_layout(specification, 2)),))
        self.assertIsNot(selected, self.manager)
        self.assertEqual(selected.heap_bytes, 2 * 1024**2)
        self.assertEqual(selected.epoch, 0)
        self.reinitialize.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_configurable_growth_multiplier(self) -> None:
        """Feature: configurable growth multiplier.
        Description: Use minimal, geometric and extremely large growth factors.
        Expectation: Honor minimal growth, custom headroom and the lossless bound even for huge factors.
        """
        with patch.object(self.manager, "_rebuild") as rebuild:
            for factor, capacity in ((1.0, 32896), (1.25, 40960), (2.0, 65536), (1e308, 262144)):
                with self.subTest(factor=factor):
                    self.spec.capacity_growth_factor = factor
                    self.resource.ensure_capacity(32769)
                    self.assertEqual(rebuild.call_args.args[0], [self._layout(capacity)])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_explicit_budget_uses_minimum_capacity_before_rejecting(self) -> None:
        """Feature: explicit budget uses minimum capacity before rejecting.
        Description: Try routes that fit only without headroom and routes exceeding the limit.
        Expectation: Fit a route without growth headroom and preserve the heap when even the minimum will not fit.
        """
        self.manager.heap_limit = 660 * 1024**2
        with patch.object(self.manager, "_rebuild") as rebuild:
            self.resource.ensure_capacity(32769)
            rebuild.assert_called_once_with([self._layout(32896)], self.manager.heap_limit)
            rebuild.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "explicit heap budget"):
                self.resource.ensure_capacity(40000)
            rebuild.assert_not_called()
            self.assertEqual(self.manager.epoch, 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unmanaged_users_allocations_and_active_leases_keep_old_heap(self) -> None:
        """Feature: unmanaged users allocations and active leases keep old heap.
        Description: Inject an external owner, allocation, active lease or capture.
        Expectation: Reject unsafe ownership before any destructive operation.
        """
        for reason in ("owner", "allocation", "lease", "capture"):
            with self.subTest(reason=reason):
                self.state["reference_count"] = 2 if reason == "owner" else 1
                self.state["active_allocations"] = [{"allocation_base": 1234}] if reason == "allocation" else []
                self.resource.workspace.in_use = reason == "lease"
                with patch.object(torch.npu, "is_current_stream_capturing", return_value=reason == "capture"):
                    with self.assertRaisesRegex(RuntimeError, "reconfiguration failed"):
                        self.resource.ensure_capacity(40000)
                self.free.assert_not_called()
                self.reinitialize.assert_not_called()
                self.assertEqual(self.manager.state, "ready")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_root_reference_is_managed_but_external_owners_are_rejected(self) -> None:
        """Feature: Managed heap ownership.

        Description: Grow with an automatic root pin, then introduce another external owner.
        Expectation: The managed pin permits growth, while the extra owner preserves the old heap.
        """
        with patch.object(_lifecycle, "_retained_users", 1):
            self.state["reference_count"] = 2
            self.resource.ensure_capacity(40000)
            self.assertEqual(self.manager.epoch, 1)
            self.reinitialize.assert_called_once()
            self.state["reference_count"] = 3
            with self.assertRaisesRegex(RuntimeError, "unmanaged SHMEM owner"):
                self.resource.ensure_capacity(60000)
            self.assertEqual(self.manager.epoch, 1)
            self.reinitialize.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_peer_manifest_mismatch_precedes_teardown(self) -> None:
        """Feature: peer manifest mismatch precedes teardown.
        Description: Supply an incompatible peer epoch during preflight.
        Expectation: A rank disagreement must not free any still-valid symmetric tensor.
        """


        def disagree(peers: list, payload: tuple, **_kwargs: object) -> None:
            """Simulate a peer entering the collective with an incompatible manifest."""
            peers[:] = [payload, (("different epoch",), None)]

        with (patch.object(mega_moe_module.dist, "is_initialized", return_value=True),
              patch.object(mega_moe_module.dist, "get_world_size", return_value=2),
              patch.object(mega_moe_module.dist, "all_gather_object", side_effect=disagree)):
            with self.assertRaisesRegex(RuntimeError, "differs across EP ranks"):
                self.resource.ensure_capacity(40000)
        self.reinitialize.assert_not_called()
        self.assertEqual(self.manager.state, "ready")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sync_failure_preserves_heap_but_init_failure_is_terminal(self) -> None:
        """Feature: sync failure preserves heap but init failure is terminal.
        Description: Fail first during synchronization and then after heap destruction.
        Expectation: Differentiate failures before destruction from failures after the old heap is gone.
        """
        self.sync.side_effect = RuntimeError("sync failed")
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            self.resource.ensure_capacity(40000)
        self.assertEqual(self.manager.state, "ready")
        self.reinitialize.assert_not_called()
        self.sync.side_effect = None
        self.reinitialize.side_effect = RuntimeError("init failed")
        with self.assertRaisesRegex(RuntimeError, "init failed"):
            self.resource.ensure_capacity(40000)
        self.assertEqual(self.manager.state, "failed")
        self.assertEqual(self.manager.epoch, 0)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.resource.workspace.claim()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_growth_preserves_lazy_reservations_and_logical_workspace(self) -> None:
        """Feature: growth preserves lazy reservations and logical workspace.
        Description: Grow one bound owner while another layout remains unbound.
        Expectation: Keep the object held by old autograd contexts while allocating its next generation lazily.
        """
        other = dict(self.specification, hidden_size=1024, dispatch_mode="pull",
                     initial_capacity_factor=None, capacity_growth_factor=None)
        self.manager._reserve(((other, mega_moe_module._heap_layout(other, 2)),))
        workspace = self.resource.workspace
        expected = self.manager.required_bytes([self._layout(49152), mega_moe_module._heap_layout(other, 2)])
        self.resource.ensure_capacity(40000)
        self.assertIs(self.resource.workspace, workspace)
        self.assertEqual(workspace.capacity_floor, 49152)
        self.assertIsNone(workspace.expert_buffer)
        self.assertEqual(self.manager.entries[1].layout[2], 32768)
        self.assertEqual(self.manager.heap_bytes, expected)
        self.assertEqual(self.manager.epoch, 1)
        self.resource.ensure_capacity(1)
        self.reinitialize.assert_called_once_with(expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rebuild_preserves_graph_leases_and_reopens_workspace(self) -> None:
        """Feature: rebuild preserves graph leases and reopens workspace.
        Description: Replace physical buffers with an old autograd context still registered.
        Expectation: Replace physical buffers while old graphs retain the logical workspace.
        """
        workspace = self.resource.workspace
        graph = Mock()
        workspace.track_graph(graph)
        workspace.expert_buffer = torch.empty(8, 4)
        workspace.routed_buffer = torch.empty(8, 4)
        workspace.completion_event = Mock()
        workspace.dtype, workspace.device = torch.bfloat16, torch.device("cpu")
        self.state["active_allocations"] = [
            {"allocation_base": tensor.data_ptr()}
            for tensor in (workspace.expert_buffer, workspace.routed_buffer)
        ]

        def reallocate(*_args: object) -> None:
            """Require the workspace to accept its replacement allocation."""
            self.assertFalse(workspace.closing)
            self.assertIsNone(workspace.expert_buffer)
            self.assertIsNone(workspace.routed_buffer)

        with patch.object(workspace, "ensure", side_effect=reallocate) as ensure:
            self.resource.ensure_capacity(40000)
        ensure.assert_called_once_with(self.spec, torch.bfloat16, torch.device("cpu"))
        self.assertIn(graph, workspace.graphs)
        self.assertFalse(workspace.closing)
        self.assertEqual(self.manager.epoch, 1)
        with self.assertRaisesRegex(RuntimeError, "backward graphs"):
            workspace.close()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_partial_allocation_failure_never_publishes_new_epoch(self) -> None:
        """Feature: partial allocation failure never publishes new epoch.
        Description: Raise OOM while restoring replacement buffers.
        Expectation: Prevent further launches after replacement allocation fails on a participating rank.
        """
        workspace = self.resource.workspace
        workspace.completion_event = Mock()
        workspace.dtype, workspace.device = torch.bfloat16, torch.device("cpu")
        with patch.object(workspace, "close"), patch.object(workspace, "ensure", side_effect=RuntimeError("OOM")):
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                self.resource.ensure_capacity(40000)
        self.assertEqual(self.manager.state, "failed")
        self.assertEqual(self.manager.epoch, 0)
        self.assertFalse(self.manager.growth_records)


if __name__ == "__main__":
    unittest.main()
