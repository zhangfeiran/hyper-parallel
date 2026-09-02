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
from unittest.mock import Mock, patch

from hyper_parallel.core.multicore.platform.torch.mega_moe import (
    backend as backend_module,
)
from hyper_parallel.core.multicore.platform.torch.mega_moe.backend import (
    TorchMegaMoeBackend,
)


class TestTorchMegaMoeBackend(unittest.TestCase):
    """Validate backend selection without distributed hardware."""

    @patch.object(TorchMegaMoeBackend, "_validate_tensors")
    @patch.object(backend_module, "restore_topk_output")
    @patch.object(backend_module, "execute_mega_moe")
    @patch.object(backend_module, "execute_native_from_permuted_route")
    @patch.object(backend_module, "prepare_topk_route")
    def test_receive_overflow_uses_native_fallback(
        self,
        mock_prepare_route,
        mock_native_route,
        mock_execute_mega_moe,
        mock_restore_output,
        mock_validate_tensors,
    ):
        """Route every rank to native execution when receive capacity overflows."""
        route = SimpleNamespace(
            routed_tokens=Mock(name="routed_tokens"),
            unpermute_mapping=Mock(name="unpermute_mapping"),
            tokens_per_expert=Mock(name="tokens_per_expert"),
            received_counts=Mock(name="received_counts"),
            receive_overflow=True,
            metadata=None,
        )
        mock_prepare_route.return_value = route
        native_output = Mock(name="native_output")
        mock_native_route.return_value = native_output
        backend = object.__new__(TorchMegaMoeBackend)
        backend.config = Mock(name="config")
        backend.runtime = Mock(name="runtime")
        backend.workspace = Mock(name="workspace")

        hidden_states = Mock(name="hidden_states")
        topk_ids = Mock(name="topk_ids")
        topk_weights = Mock(name="topk_weights")
        weight1 = Mock(name="weight1")
        weight2 = Mock(name="weight2")
        result = backend.forward(
            hidden_states,
            topk_ids,
            topk_weights,
            weight1,
            weight2,
            tokens_per_expert=route.tokens_per_expert,
        )

        self.assertIs(result, native_output)
        mock_validate_tensors.assert_called_once_with(hidden_states, weight1, weight2)
        mock_native_route.assert_called_once_with(
            route.routed_tokens,
            route.unpermute_mapping,
            topk_weights,
            weight1,
            weight2,
            route.tokens_per_expert,
            route.received_counts,
            backend.config,
        )
        mock_execute_mega_moe.assert_not_called()
        mock_restore_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
