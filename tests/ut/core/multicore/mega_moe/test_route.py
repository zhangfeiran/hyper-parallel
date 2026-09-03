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
"""Unit tests for MegaMoe route preparation."""

import sys
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.platform.torch.mega_moe import route as route_module
from hyper_parallel.core.multicore.platform.torch.mega_moe.config import MegaMoeConfig
from hyper_parallel.core.multicore.platform.torch.mega_moe.route import (
    _build_route_indices,
    _permute_topk_input,
    _resolve_counts,
    _validate_topk_inputs,
    prepare_topk_route,
    restore_topk_output,
)


class TestMegaMoeRoute(unittest.TestCase):
    """Validate trusted router-count handling without distributed hardware."""

    @staticmethod
    def _config(*, ep_size: int = 1, rank_id: int = 0) -> MegaMoeConfig:
        """Build a small CPU-only route configuration."""
        return MegaMoeConfig(
            local_num_tokens=2,
            hidden_size=4,
            intermediate_size=2,
            num_experts=4,
            top_k=2,
            receive_capacity=128,
            ep_size=ep_size,
            ep_group=None,
            rank_id=rank_id,
            num_cube_cores=24,
            forward_fusion_mode="legacy",
            backward_fusion_mode="legacy",
        )

    def test_supplied_counts_skip_bincount_validation(self):
        """Use supplied counts directly without rebuilding their histogram."""
        config = self._config()
        flat_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        supplied_counts = torch.tensor([4, 0, 0, 0], dtype=torch.int32)

        with patch("torch.bincount") as mock_bincount:
            resolved = _resolve_counts(flat_ids, supplied_counts, config)

        mock_bincount.assert_not_called()
        self.assertTrue(torch.equal(resolved, supplied_counts))

    def test_topk_validation_allows_repeated_experts(self):
        """Leave repeated-expert policy to the trusted upstream router."""
        hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[1, 1], [2, 2]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)

        flat_ids = _validate_topk_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
        )

        self.assertTrue(torch.equal(flat_ids, topk_ids.reshape(-1).to(torch.int64)))

    def test_native_route_endpoints_use_large_npu_operators(self):
        """Delegate payload permutation and weighted restoration to torch_npu."""
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.bfloat16)
        permuted = hidden_states.repeat_interleave(2, dim=0)
        mapping = torch.tensor([3, 1, 2, 0], dtype=torch.int64)
        restored = torch.full_like(hidden_states, 7)
        mock_permute = Mock(return_value=(permuted, mapping))
        mock_unpermute = Mock(return_value=restored)
        fake_torch_npu = SimpleNamespace(
            npu_moe_token_permute=mock_permute,
            npu_moe_token_unpermute=mock_unpermute,
        )

        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            routed_tokens, unpermute_mapping = _permute_topk_input(
                hidden_states,
                topk_ids,
            )
            output = restore_topk_output(
                routed_tokens,
                unpermute_mapping,
                topk_weights,
            )

        self.assertIs(routed_tokens, permuted)
        self.assertTrue(torch.equal(unpermute_mapping, mapping.to(torch.int32)))
        self.assertIs(output, restored)
        permute_call = mock_permute.call_args
        self.assertIs(permute_call.args[0], hidden_states)
        self.assertIs(permute_call.args[1], topk_ids)
        unpermute_call = mock_unpermute.call_args
        self.assertIs(unpermute_call.args[0], permuted)
        self.assertIs(unpermute_call.args[1], unpermute_mapping)
        self.assertEqual(unpermute_call.kwargs["probs"].dtype, torch.float32)
        self.assertTrue(unpermute_call.kwargs["probs"].is_contiguous())

    def test_fused_route_indices_preserve_expert_major_route_slots(self):
        """Keep route-slot identity so kernels can recover tokens and weights."""
        route_slots = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
        topk_ids = torch.tensor([[2, 0], [1, 2], [0, 1]], dtype=torch.int32)
        expert_order = torch.tensor([1, 4, 2, 5, 0, 3], dtype=torch.float32)
        inverse_order = torch.tensor([4, 0, 2, 5, 1, 3], dtype=torch.int32)
        mock_permute = Mock(return_value=(expert_order.reshape(-1, 1), inverse_order))
        fake_torch_npu = SimpleNamespace(npu_moe_token_permute=mock_permute)

        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            route_index, unpermute_mapping = _build_route_indices(
                route_slots,
                topk_ids,
            )

        self.assertTrue(torch.equal(route_index, expert_order.to(torch.int32)))
        self.assertTrue(
            torch.equal(unpermute_mapping, inverse_order.reshape_as(topk_ids))
        )
        permute_call = mock_permute.call_args
        self.assertIs(permute_call.args[0], route_slots)
        self.assertEqual(tuple(permute_call.args[1].shape), (6, 1))
        self.assertTrue(
            torch.equal(permute_call.args[1].reshape_as(topk_ids), topk_ids)
        )

    def test_async_count_gather_overlaps_permute_and_builds_metadata_locally(self):
        """Wait after permutation and derive every legacy offset from gathered counts."""
        config = self._config(ep_size=2, rank_id=1)
        runtime = SimpleNamespace(config=config)
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        supplied_counts = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
        global_counts = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            dtype=torch.int32,
        )
        routed_tokens = hidden_states.repeat_interleave(2, dim=0)
        unpermute_mapping = torch.arange(4, dtype=torch.int32)
        events = []
        work = Mock()
        work.wait.side_effect = lambda: events.append("wait")

        def gather_counts(output: Any, input_tensor: Any, **kwargs: Any) -> Any:
            """Populate the mocked rank-major gather output."""
            self.assertIs(input_tensor, supplied_counts)
            self.assertTrue(kwargs["async_op"])
            events.append("gather")
            output.copy_(global_counts.reshape(-1))
            return work

        def permute_input(
            input_hidden_states: Any, input_topk_ids: Any
        ) -> tuple[Any, Any]:
            """Record the mocked payload permutation launch."""
            self.assertIs(input_hidden_states, hidden_states)
            self.assertIs(input_topk_ids, topk_ids)
            events.append("permute")
            return routed_tokens, unpermute_mapping

        with (
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=gather_counts,
            ) as mock_gather,
            patch.object(
                route_module,
                "_permute_topk_input",
                side_effect=permute_input,
            ),
        ):
            route = prepare_topk_route(
                hidden_states,
                topk_ids,
                topk_weights,
                runtime,
                supplied_counts,
            )

        self.assertEqual(events, ["gather", "permute", "wait"])
        self.assertTrue(mock_gather.call_args.kwargs["async_op"])
        self.assertIs(route.routed_tokens, routed_tokens)
        self.assertIs(route.unpermute_mapping, unpermute_mapping)
        self.assertFalse(route.receive_overflow)
        self.assertTrue(
            torch.equal(
                route.received_counts,
                torch.tensor([[3, 4], [7, 8]], dtype=torch.int32),
            )
        )
        metadata = route.metadata
        self.assertIsNotNone(metadata)
        self.assertTrue(
            torch.equal(metadata.dispatch_src_off, torch.tensor([0, 5, 11, 18]))
        )
        self.assertTrue(
            torch.equal(metadata.dispatch_target_off, torch.tensor([1, 8, 3, 14]))
        )
        self.assertTrue(torch.equal(metadata.dispatch_size, supplied_counts))
        self.assertTrue(
            torch.equal(metadata.combine_src_off, torch.tensor([0, 10, 3, 14]))
        )
        self.assertTrue(
            torch.equal(metadata.combine_target_off, torch.tensor([3, 6, 11, 18]))
        )
        self.assertTrue(torch.equal(metadata.combine_size, torch.tensor([3, 4, 7, 8])))
        self.assertTrue(torch.equal(metadata.group_list, torch.tensor([10, 22])))
        self.assertEqual(metadata.expert_capacity, 128)


if __name__ == "__main__":
    unittest.main()
