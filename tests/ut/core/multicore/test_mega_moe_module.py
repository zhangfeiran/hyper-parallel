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

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hyper_parallel.core import multicore
from hyper_parallel.core.multicore.modules.mega_moe import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe import module as mega_moe_module
from hyper_parallel.core.multicore.platform import torch as torch_multicore


class TestMegaMoeExperts(unittest.TestCase):
    """Validate the local-token public construction contract."""

    def test_only_managed_apis_are_public(self):
        """Keep native adapter calls out of the managed public namespace."""
        self.assertEqual(multicore.__all__, ["MegaMoeExperts", "MulticoreModule"])
        self.assertFalse(hasattr(multicore, "mega_moe"))
        self.assertFalse(hasattr(multicore, "mega_moe_grad"))
        self.assertFalse(hasattr(multicore, "_ops"))
        self.assertEqual(torch_multicore.__all__, [])
        self.assertTrue(callable(torch_multicore.run_mega_moe))
        self.assertTrue(callable(torch_multicore.run_mega_moe_grad))

    @patch.object(mega_moe_module, "create_mega_moe_parameters")
    def test_constructor_uses_local_tokens_without_tp_topology(
        self, mock_create_parameters
    ):
        """Store local token rows directly and omit TP from resource compatibility."""
        mock_create_parameters.return_value = (object(), object())

        experts = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            ep_size=2,
        )

        self.assertEqual(experts.local_num_tokens, 128)
        self.assertNotIn("tp_size", inspect.signature(MegaMoeExperts).parameters)
        self.assertNotIn(
            "global_num_tokens", inspect.signature(MegaMoeExperts).parameters
        )
        self.assertEqual(
            experts._resource_group.specification,
            {
                "local_num_tokens": 128,
                "hidden_size": 16,
                "intermediate_size": 8,
                "num_experts": 4,
                "top_k": 2,
                "expert_capacity_factor": 1.5,
                "ep_size": 2,
                "ep_group": None,
            },
        )
        self.assertEqual(experts.expert_capacity_factor, 1.5)
        self.assertNotIn(
            "receive_capacity", inspect.signature(MegaMoeExperts).parameters
        )
        mock_create_parameters.assert_called_once_with(2, 16, 8)

    @patch.object(mega_moe_module, "create_mega_moe_parameters")
    def test_constructor_rejects_removed_tp_size(self, mock_create_parameters):
        """Reject the removed TP argument at the public API boundary."""
        mock_create_parameters.return_value = (object(), object())

        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'tp_size'"):
            MegaMoeExperts(
                local_num_tokens=128,
                hidden_size=16,
                intermediate_size=8,
                num_experts=4,
                top_k=2,
                tp_size=2,
                ep_size=2,
            )

        mock_create_parameters.assert_not_called()

    @patch.object(mega_moe_module, "create_mega_moe_parameters")
    def test_constructor_validates_expert_capacity_factor(self, mock_create_parameters):
        """Require a finite receive-capacity multiplier of at least one."""
        mock_create_parameters.return_value = (object(), object())

        invalid_factors = (0, -1, 0.999, True, None, "1.25", float("inf"), float("nan"))
        for expert_capacity_factor in invalid_factors:
            with (
                self.subTest(expert_capacity_factor=expert_capacity_factor),
                self.assertRaisesRegex(ValueError, "expert_capacity_factor"),
            ):
                MegaMoeExperts(
                    local_num_tokens=128,
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=4,
                    top_k=2,
                    expert_capacity_factor=expert_capacity_factor,
                    ep_size=2,
                )

        mock_create_parameters.assert_not_called()

    @patch.object(mega_moe_module, "create_mega_moe_parameters")
    def test_constructor_validates_local_num_tokens(self, mock_create_parameters):
        """Reject non-positive and non-integer local token counts."""
        mock_create_parameters.return_value = (object(), object())

        for local_num_tokens in (0, -1, True, 1.5):
            with (
                self.subTest(local_num_tokens=local_num_tokens),
                self.assertRaisesRegex(
                    ValueError,
                    "local_num_tokens must be a positive integer",
                ),
            ):
                MegaMoeExperts(
                    local_num_tokens=local_num_tokens,
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=4,
                    top_k=2,
                    ep_size=2,
                )

        mock_create_parameters.assert_not_called()

    @patch.object(mega_moe_module, "create_mega_moe_parameters")
    def test_multiple_layers_share_one_execution_resource_group(
        self, mock_create_parameters
    ):
        """Route compatible serial layers to one future workspace."""
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
            for _ in range(8)
        ]

        MegaMoeExperts.share_execution_resources(layers)

        shared_group = layers[0]._resource_group
        self.assertTrue(shared_group.shared)
        self.assertEqual(len(shared_group.members), len(layers))
        self.assertTrue(all(layer._resource_group is shared_group for layer in layers))

        shared_backend = object()
        input_tensor = SimpleNamespace(device="npu:0", dtype="bfloat16")
        try:
            with patch.object(
                mega_moe_module,
                "TorchMegaMoeBackend",
                return_value=shared_backend,
            ) as mock_create_backend:
                backends = [
                    layer._get_execution_backend(input_tensor) for layer in layers
                ]

            self.assertTrue(all(backend is shared_backend for backend in backends))
            mock_create_backend.assert_called_once()
            create_kwargs = mock_create_backend.call_args.kwargs
            self.assertTrue(create_kwargs["shared"])
            self.assertEqual(len(create_kwargs["active_specifications"]), 1)
        finally:
            shared_group.backend = None
            shared_group.binding = None
