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

import json
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe import module as mega_moe_module


class TestMegaMoeExperts(unittest.TestCase):
    """Validate the public API and execution-resource lifecycle."""

    def test_managed_implementation_loads_only_after_symbol_access(self) -> None:
        """Verify lazy loading in an uncontaminated interpreter."""
        script = """
import json
import os
import sys

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import hyper_parallel.core.multicore as multicore

module_name = "hyper_parallel.core.multicore.modules.mega_moe.module"
binding_name = "hyper_parallel.platform.torch.symmetric_memory.symmetric_memory"
before = module_name in sys.modules
before_binding = binding_name in sys.modules
import hyper_parallel.core.multicore.modules.mega_moe.forward.graph
after_graph_import = module_name in sys.modules
after_graph_binding = binding_name in sys.modules
managed_class = multicore.MegaMoeExperts
after = module_name in sys.modules
after_binding = binding_name in sys.modules
print(json.dumps({
    "before": before,
    "before_binding": before_binding,
    "after_graph_import": after_graph_import,
    "after_graph_binding": after_graph_binding,
    "after": after,
    "after_binding": after_binding,
    "name": managed_class.__name__,
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertFalse(result["before"])
        self.assertFalse(result["before_binding"])
        self.assertFalse(result["after_graph_import"])
        self.assertFalse(result["after_graph_binding"])
        self.assertTrue(result["after"])
        self.assertFalse(result["after_binding"])
        self.assertEqual(result["name"], "MegaMoeExperts")

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_defaults_to_lossless_capacity(
        self,
        mock_create_parameters: Mock,
    ) -> None:
        """Expose local-token topology with a lossless default capacity."""
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
            self.assertIsNone(experts.expert_capacity_factor)
            self.assertEqual(
                experts._resource_group.specification,
                {
                    "local_num_tokens": 128,
                    "hidden_size": 16,
                    "intermediate_size": 8,
                    "num_experts": 4,
                    "top_k": 2,
                    "expert_capacity_factor": None,
                    "replica_slots_per_rank": 0,
                    "replica_grad_receive_slots": 8,
                    "compact_physical_weights": False,
                    "shared_physical_weights": False,
                    "compact_layout_strategy": "eager_on_first_forward",
                    "ep_size": 2,
                    "ep_group": None,
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
            ({"expert_capacity_factor": 0.999}, "expert_capacity_factor"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                MegaMoeExperts(
                    local_num_tokens=overrides.get("local_num_tokens", 128),
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=4,
                    top_k=2,
                    expert_capacity_factor=overrides.get("expert_capacity_factor"),
                    ep_size=2,
                )

        mock_create_parameters.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_gradient_window_validates_and_participates_in_resource_compatibility(
        self, mock_create_parameters: Mock,
    ) -> None:
        """Keep communication batching identical within an execution group."""
        shape = {"local_num_tokens": 128, "hidden_size": 16, "intermediate_size": 8,
                 "num_experts": 4, "top_k": 2, "ep_size": 2, "replica_slots_per_rank": 1}
        for slots in (0, -1, True, 1.5, "4"):
            with self.subTest(slots=slots), self.assertRaisesRegex(ValueError, "replica_grad_receive_slots"):
                MegaMoeExperts(**shape, replica_grad_receive_slots=slots)
        mock_create_parameters.assert_not_called()
        mock_create_parameters.return_value = (object(), object())
        first = MegaMoeExperts(**shape, replica_grad_receive_slots=2)
        second = MegaMoeExperts(**shape, replica_grad_receive_slots=4)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        with self.assertRaisesRegex(ValueError, "identical configuration"):
            MegaMoeExperts.share_execution_resources((first, second))

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_shared_physical_weights_validate_storage_contract(self, mock_create_parameters):
        """Reject ambiguous flags and incompatible storage modes before allocation.

        Feature: replayable shared-weight configuration
        Description: validate mode types, replica budget and compact exclusion
        Expectation: invalid modes fail and storage modes cannot share a backend
        """
        mock_create_parameters.return_value = (object(), object())
        shape = {"local_num_tokens": 128, "hidden_size": 16, "intermediate_size": 8,
                 "num_experts": 4, "top_k": 2, "ep_size": 2}
        for flag in (0, 1, "true", None):
            with self.subTest(flag=flag), self.assertRaisesRegex(TypeError, "shared_physical_weights"):
                MegaMoeExperts(**shape, replica_slots_per_rank=1, shared_physical_weights=flag)
        for slots, compact in ((0, False), (1, True)):
            with self.subTest(slots=slots, compact=compact), self.assertRaisesRegex(ValueError, "requires"):
                MegaMoeExperts(**shape, replica_slots_per_rank=slots,
                               compact_physical_weights=compact, shared_physical_weights=True)
        ordinary = MegaMoeExperts(**shape, replica_slots_per_rank=1)
        shared = MegaMoeExperts(**shape, replica_slots_per_rank=1, shared_physical_weights=True)
        with self.assertRaisesRegex(ValueError, "identical configuration"):
            MegaMoeExperts.share_execution_resources((ordinary, shared))


    def test_forward_passes_router_inputs_and_restores_shape(self) -> None:
        """Preserve Router inputs, expert parameters and the caller's shape."""
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
        resources = SimpleNamespace(spec=object(), plan=object(), workspace=object(), replica_manager=None)
        route = SimpleNamespace(
            routed_tokens=object(), metadata=object(), unpermute_mapping=object(),
            physical_topk_ids=topk_ids, placement=None, plan=resources.plan,
        )

        with (
            patch.object(
                torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True
            ),
            patch.object(experts, "_get_execution_resources", return_value=resources),
            patch.object(
                mega_moe_module, "prepare_topk_route", return_value=route
            ) as mock_prepare,
            patch.object(
                mega_moe_module, "execute_mega_moe", return_value=expected
            ) as mock_execute,
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
            hidden_flat, topk_ids, topk_weights, resources.plan, tokens_per_expert,
            workspace=resources.workspace,
        )
        mock_execute.assert_called_once_with(
            hidden_flat,
            topk_ids,
            topk_weights,
            experts.gate_up_weight,
            experts.down_weight,
            route,
            resources.plan,
            resources.workspace,
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
        mock_create_resources.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_shared_layers_create_once_and_close_after_last_owner(
        self,
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
        resources.close.assert_called_once_with()


    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_validates_replica_slot_budget(self, mock_create_parameters):
        """Reject negative and non-integer replica slot budgets."""
        mock_create_parameters.return_value = (object(), object())

        for replica_slots_per_rank in (-1, True, 1.5, "1"):
            with (
                self.subTest(replica_slots_per_rank=replica_slots_per_rank),
                self.assertRaisesRegex(ValueError, "replica_slots_per_rank"),
            ):
                MegaMoeExperts(
                    local_num_tokens=128,
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=4,
                    top_k=2,
                    replica_slots_per_rank=replica_slots_per_rank,
                    ep_size=2,
                )

        mock_create_parameters.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_replica_budget_participates_in_resource_compatibility(
        self,
        mock_create_parameters,
    ):
        """Prevent modules with different physical group counts from sharing."""
        mock_create_parameters.return_value = (object(), object())
        static_layer = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            ep_size=2,
        )
        replicated_layer = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            replica_slots_per_rank=1,
            ep_size=2,
        )

        with self.assertRaisesRegex(ValueError, "identical configuration"):
            MegaMoeExperts.share_execution_resources(
                (static_layer, replicated_layer)
            )


    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_validates_compact_physical_weights(self, mock_create_parameters):
        """Reject non-Boolean compact physical-weight selectors.

        Feature: compact physical-weight configuration.
        Description: Try integer, string, and null compact-mode values.
        Expectation: Every invalid selector fails before parameter allocation.
        """
        mock_create_parameters.return_value = (object(), object())

        for compact_physical_weights in (0, 1, "true", None):
            with (
                self.subTest(compact_physical_weights=compact_physical_weights),
                self.assertRaisesRegex(ValueError, "compact_physical_weights"),
            ):
                MegaMoeExperts(
                    local_num_tokens=128,
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=4,
                    top_k=2,
                    replica_slots_per_rank=1,
                    compact_physical_weights=compact_physical_weights,
                    ep_size=2,
                )

        mock_create_parameters.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_validates_compact_layout_strategy(self, mock_create_parameters):
        """Reject unknown compact-layout timing policies.

        Feature: compact physical-weight layout timing.
        Description: Try unsupported string and non-string strategy values.
        Expectation: Every invalid selector fails before parameter allocation.
        """
        mock_create_parameters.return_value = (object(), object())

        for compact_layout_strategy in ("first_use", 1, None):
            with (
                self.subTest(compact_layout_strategy=compact_layout_strategy),
                self.assertRaisesRegex(ValueError, "compact_layout_strategy"),
            ):
                MegaMoeExperts(
                    local_num_tokens=128,
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=4,
                    top_k=2,
                    replica_slots_per_rank=1,
                    compact_physical_weights=True,
                    compact_layout_strategy=compact_layout_strategy,
                    ep_size=2,
                )

        mock_create_parameters.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_compact_mode_participates_in_resource_compatibility(
        self,
        mock_create_parameters,
    ):
        """Keep snapshot and compact storage backends in separate groups.

        Feature: compact physical-weight resource compatibility.
        Description: Combine replicated modules whose storage modes differ.
        Expectation: Resource sharing rejects the incompatible storage contract.
        """
        mock_create_parameters.return_value = (object(), object())
        snapshot_layer = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            replica_slots_per_rank=1,
            ep_size=2,
        )
        compact_layer = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            replica_slots_per_rank=1,
            compact_physical_weights=True,
            ep_size=2,
        )

        with self.assertRaisesRegex(ValueError, "identical configuration"):
            MegaMoeExperts.share_execution_resources((snapshot_layer, compact_layer))

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_compact_layout_strategy_participates_in_resource_compatibility(
        self,
        mock_create_parameters,
    ):
        """Keep incompatible compact-layout timing in separate resource groups.

        Feature: compact physical-weight resource compatibility.
        Description: Combine eager and lazy compact modules.
        Expectation: Resource sharing rejects the incompatible storage timing.
        """
        mock_create_parameters.return_value = (object(), object())
        eager_layer = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            replica_slots_per_rank=1,
            compact_physical_weights=True,
            ep_size=2,
        )
        lazy_layer = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            replica_slots_per_rank=1,
            compact_physical_weights=True,
            compact_layout_strategy="lazy_on_first_replica",
            ep_size=2,
        )

        with self.assertRaisesRegex(ValueError, "identical configuration"):
            MegaMoeExperts.share_execution_resources((eager_layer, lazy_layer))


if __name__ == "__main__":
    unittest.main()
