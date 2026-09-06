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

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe.placement import (
    build_constrained_expert_placement,
    remap_logical_routes,
)
from hyper_parallel.core.multicore.modules.mega_moe import route as route_module
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.modules.mega_moe.route import (
    _build_route_indices,
    _compute_route_metadata,
    _permute_topk_input,
    _remap_topk_ids_to_physical,
    _resolve_counts,
    _finish_static_count_gather,
    _validate_topk_inputs,
    prepare_topk_route,
    restore_topk_output,
)


class TestMegaMoeRoute(unittest.TestCase):
    """Validate trusted router-count handling without distributed hardware."""

    @staticmethod
    def _config(*, ep_size: int = 1, rank_id: int = 0) -> MegaMoeSpec:
        """Build a small CPU-only route configuration."""
        return MegaMoeSpec(
            expert_capacity_factor=1.5,
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

    def test_capacity_checks_only_explicit_bounded_mode(self) -> None:
        """Skip default scalar sync and detect explicit bounded overflow."""
        spec = replace(self._config(ep_size=2), expert_capacity_factor=None)
        counts = torch.ones((2, 4), dtype=torch.int32)

        with patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected scalar readback")):
            _, maximum = _finish_static_count_gather(counts, None, spec)
        self.assertEqual(maximum, spec.receive_capacity)
        bounded_spec = replace(self._config(ep_size=2), expert_capacity_factor=1.0, receive_capacity=4)
        counts = torch.tensor([[4, 0, 0, 0], [4, 0, 0, 0]], dtype=torch.int32)
        _, maximum = _finish_static_count_gather(counts, None, bounded_spec)
        self.assertEqual(maximum, 8)
        self.assertGreater(maximum, bounded_spec.receive_capacity)


    def test_supplied_counts_skip_bincount_validation(self):
        """
        Feature: trusted router count reuse
        Description: provide a precomputed expert-count vector
        Expectation: use it without launching a second bincount
        """
        spec = self._config()
        flat_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        supplied_counts = torch.tensor([4, 0, 0, 0], dtype=torch.int32)

        with patch("torch.bincount") as mock_bincount:
            resolved = _resolve_counts(flat_ids, supplied_counts, spec)

        mock_bincount.assert_not_called()
        self.assertTrue(torch.equal(resolved, supplied_counts))

    def test_topk_validation_allows_repeated_experts(self):
        """
        Feature: trusted repeated-expert routing
        Description: validate TopK rows containing repeated expert ids
        Expectation: preserve the ids and leave policy enforcement upstream
        """
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
        """
        Feature: native large route operators
        Description: mock Torch NPU permutation and weighted restoration
        Expectation: route payloads through both large-operator endpoints
        """
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

        with patch.object(route_module, "torch_npu", fake_torch_npu):
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
        """
        Feature: scatter-free fused route indices
        Description: permute route-slot identities into expert-major order
        Expectation: let kernels recover token rows and token-major weights
        """
        route_slots = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
        topk_ids = torch.tensor([[2, 0], [1, 2], [0, 1]], dtype=torch.int32)
        expert_order = torch.tensor([1, 4, 2, 5, 0, 3], dtype=torch.float32)
        inverse_order = torch.tensor([4, 0, 2, 5, 1, 3], dtype=torch.int32)
        mock_permute = Mock(return_value=(expert_order.reshape(-1, 1), inverse_order))
        fake_torch_npu = SimpleNamespace(npu_moe_token_permute=mock_permute)

        with patch.object(route_module, "torch_npu", fake_torch_npu):
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
        """
        Feature: overlapped count exchange and payload permutation
        Description: gather rank counts asynchronously while permuting tokens
        Expectation: wait afterward and derive all route metadata locally
        """
        spec = self._config(ep_size=2, rank_id=1)
        plan = SimpleNamespace(spec=spec)
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
                plan,
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
        self.assertEqual(metadata.retained_capacity, 128)
        self.assertEqual(metadata.receive_capacity, 128)
        self.assertEqual(metadata.wave_count, 1)

    def test_overflow_metadata_uses_fixed_rolling_capacity(self):
        """
        Feature: fixed-capacity multi-wave route metadata
        Description: build metadata for a route above one receive window
        Expectation: retain exact GMM1 rows while bounding rolling buffers
        """
        spec = MegaMoeSpec(
            expert_capacity_factor=1.5,
            local_num_tokens=128,
            hidden_size=4,
            intermediate_size=2,
            num_experts=4,
            top_k=2,
            receive_capacity=256,
            ep_size=2,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
        )
        plan = SimpleNamespace(spec=spec)
        counts_by_source = torch.tensor(
            [[96, 96, 32, 32], [96, 96, 32, 32]],
            dtype=torch.int32,
        )
        counts = counts_by_source[spec.rank_id]
        received_counts = counts_by_source[:, : spec.local_experts]

        metadata = _compute_route_metadata(
            counts,
            counts_by_source,
            received_counts,
            plan,
            maximum_received_slots=384,
        )

        self.assertEqual(metadata.expert_capacity, spec.receive_capacity)
        self.assertEqual(metadata.retained_capacity, 384)
        self.assertEqual(metadata.receive_capacity, spec.receive_capacity)
        self.assertEqual(metadata.wave_count, 2)

    def test_replica_route_remaps_logical_ids_into_physical_groups(self):
        """
        Feature: MoonEP-style physical route construction
        Description: place rank one's hot logical expert into its replica suffix
        Expectation: produce physical IDs, exact counts and three local GMM groups
        """
        spec = MegaMoeSpec(
            expert_capacity_factor=1.5,
            local_num_tokens=2,
            hidden_size=4,
            intermediate_size=2,
            num_experts=4,
            top_k=2,
            receive_capacity=4,
            ep_size=2,
            ep_group=None,
            rank_id=1,
            num_cube_cores=24,
            replica_slots_per_rank=1,
            forward_fusion_mode="legacy",
            backward_fusion_mode="legacy",
        )
        plan = SimpleNamespace(spec=spec)
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        topk_ids = torch.zeros((2, 2), dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        local_counts = torch.tensor([4, 0, 0, 0], dtype=torch.int32)
        global_counts = torch.tensor(
            [[4, 0, 0, 0], [4, 0, 0, 0]],
            dtype=torch.int32,
        )
        routed_tokens = hidden_states.repeat_interleave(2, dim=0)
        unpermute_mapping = torch.arange(4, dtype=torch.int32)
        work = Mock()
        events = []
        work.wait.side_effect = lambda: events.append("wait")
        placement_payload = Mock(name="placement_payload")

        def gather_counts(output: Any, _input_tensor: Any, **_kwargs: Any) -> Any:
            """Populate the mocked logical count gather."""
            output.copy_(global_counts.reshape(-1))
            return work

        def logical_route_order(route_slots: Any, _logical_ids: Any) -> tuple[Any, Any]:
            """Return stable logical-expert order for the remap helper."""
            events.append("remap")
            return (
                route_slots.reshape(-1).to(torch.int32),
                torch.arange(4, dtype=torch.int32).reshape(2, 2),
            )

        def start_prefetch(_plan: Any) -> Any:
            """Record the point where remote weight transfer can overlap route work."""
            events.append("prefetch")
            return placement_payload

        def permute_physical_route(*_args: Any) -> tuple[Any, Any]:
            """Record physical permutation after weight prefetch has launched."""
            events.append("permute")
            return routed_tokens, unpermute_mapping

        with (
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=gather_counts,
            ),
            patch.object(
                route_module,
                "_build_route_indices",
                side_effect=logical_route_order,
            ),
            patch.object(
                route_module,
                "_permute_topk_input",
                side_effect=permute_physical_route,
            ) as mock_permute,
        ):
            route = prepare_topk_route(
                hidden_states,
                topk_ids,
                topk_weights,
                plan,
                local_counts,
                placement_callback=start_prefetch,
            )

        work.wait.assert_called_once_with()
        self.assertEqual(events, ["wait", "prefetch", "remap", "permute"])
        self.assertEqual(spec.physical_local_experts, 3)
        self.assertEqual(spec.physical_num_experts, 6)
        self.assertTrue(
            torch.equal(route.physical_topk_ids, torch.full((2, 2), 5, dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(route.tokens_per_expert, torch.tensor([0, 0, 0, 0, 0, 4]))
        )
        self.assertTrue(
            torch.equal(
                route.received_counts,
                torch.tensor([[0, 0, 0], [0, 0, 4]], dtype=torch.int32),
            )
        )
        self.assertEqual(route.placement.replica_expert_ids, ((-1,), (0,)))
        self.assertEqual(route.placement.destination_loads, (4, 4))
        self.assertIs(route.placement_payload, placement_payload)
        self.assertTrue(torch.equal(route.metadata.group_list, torch.tensor([0, 0, 4])))
        self.assertIs(mock_permute.call_args.args[1], route.physical_topk_ids)

    def test_one_wave_route_does_not_create_replica_traffic(self):
        """Stop replica planning when static placement already fits one wave.

        Feature: adaptive replica planning threshold
        Description: plan an imbalanced route whose slowest rank still fits capacity
        Expectation: leave both replica suffixes empty and preserve home loads
        """
        spec = MegaMoeSpec(
            expert_capacity_factor=1.5,
            local_num_tokens=2,
            hidden_size=4,
            intermediate_size=2,
            num_experts=4,
            top_k=2,
            receive_capacity=8,
            ep_size=2,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
            replica_slots_per_rank=1,
        )
        counts_by_source = torch.tensor(
            [[2, 2, 0, 0], [2, 2, 0, 0]],
            dtype=torch.int32,
        )

        plan = route_module._build_replica_plan(  # pylint: disable=W0212
            counts_by_source,
            spec,
        )

        self.assertEqual(plan.baseline_destination_loads, (8, 0))
        self.assertEqual(plan.destination_loads, (8, 0))
        self.assertEqual(plan.replica_expert_ids, ((-1,), (-1,)))
        self.assertEqual(plan.replica_transfers, ())

    def test_one_wave_route_selects_replica_free_runtime(self):
        """Execute an in-capacity route without the physical replica adapter.

        Feature: replica-free plan fast path
        Description: provide a second home plan for a one-wave logical route
        Expectation: skip placement/prefetch and retain logical IDs and counts
        """
        spec = MegaMoeSpec(
            expert_capacity_factor=1.5,
            local_num_tokens=2,
            hidden_size=4,
            intermediate_size=2,
            num_experts=4,
            top_k=2,
            receive_capacity=8,
            ep_size=2,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
            replica_slots_per_rank=1,
            forward_fusion_mode="legacy",
            backward_fusion_mode="legacy",
        )
        plan = SimpleNamespace(spec=spec)
        home_runtime = SimpleNamespace(
            spec=replace(spec, replica_slots_per_rank=0),
        )
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        local_counts = torch.tensor([2, 2, 0, 0], dtype=torch.int32)
        global_counts = torch.tensor(
            [[2, 2, 0, 0], [2, 2, 0, 0]],
            dtype=torch.int32,
        )
        routed_tokens = hidden_states.repeat_interleave(2, dim=0)
        unpermute_mapping = torch.arange(4, dtype=torch.int32)
        prefetch = Mock(name="prefetch")

        def gather_counts(output: Any, _input_tensor: Any, **_kwargs: Any) -> Any:
            """Populate one completed mocked logical count gather."""
            output.copy_(global_counts.reshape(-1))
            return Mock(wait=Mock())

        with (
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=gather_counts,
            ),
            patch.object(
                route_module,
                "_permute_topk_input",
                return_value=(routed_tokens, unpermute_mapping),
            ),
        ):
            route = prepare_topk_route(
                hidden_states,
                topk_ids,
                topk_weights,
                plan,
                local_counts,
                placement_callback=prefetch,
                home_runtime=home_runtime,
            )

        prefetch.assert_not_called()
        self.assertIs(route.plan, home_runtime)
        self.assertIs(route.physical_topk_ids, topk_ids)
        self.assertIsNone(route.placement)
        self.assertTrue(torch.equal(route.tokens_per_expert, local_counts))
        self.assertTrue(torch.equal(route.received_counts, global_counts[:, :2]))

    def test_device_remap_matches_host_quotas_when_one_source_is_split(self):
        """
        Feature: per-source physical copy quotas
        Description: split one source's repeated logical expert across two copies
        Expectation: device remap preserves the host plan's deterministic assignment
        """
        plan = build_constrained_expert_placement(
            ((3, 0, 1, 0), (3, 0, 1, 0)),
            1,
            receive_capacity=4,
        )
        logical_ids = torch.tensor([[0, 2], [0, 0]], dtype=torch.int32)
        stable_slots = torch.tensor([0, 2, 3, 1], dtype=torch.int32)

        with patch.object(
            route_module,
            "_build_route_indices",
            return_value=(stable_slots, torch.arange(4, dtype=torch.int32)),
        ):
            physical_ids = _remap_topk_ids_to_physical(
                logical_ids,
                plan,
                source_rank=1,
            )

        expected = torch.tensor(
            remap_logical_routes(tuple(logical_ids.reshape(-1).tolist()), 1, plan),
            dtype=torch.int32,
        ).reshape_as(logical_ids)
        self.assertTrue(torch.equal(physical_ids, expected))
        self.assertTrue(torch.equal(physical_ids, torch.tensor([[0, 3], [5, 5]])))

    def test_repeated_replica_counts_reuse_cached_host_plan(self):
        """
        Feature: bounded host placement cache
        Description: plan the same immutable count matrix twice
        Expectation: run the placement heuristic once and reuse its result
        """
        logical_counts = ((3, 0, 1, 0), (3, 0, 1, 0))
        route_module._cached_replica_plan.cache_clear()  # pylint: disable=W0212
        with patch.object(
            route_module,
            "build_constrained_expert_placement",
            wraps=build_constrained_expert_placement,
        ) as mock_build:
            first = route_module._cached_replica_plan(  # pylint: disable=W0212
                logical_counts,
                1,
                4,
            )
            second = route_module._cached_replica_plan(  # pylint: disable=W0212
                logical_counts,
                1,
                4,
            )

        self.assertIs(first, second)
        mock_build.assert_called_once_with(
            logical_counts,
            1,
            receive_capacity=4,
        )


if __name__ == "__main__":
    unittest.main()
