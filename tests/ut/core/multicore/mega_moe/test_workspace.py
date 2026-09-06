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
"""Unit tests for MegaMoe workspace sizing and stream ordering."""

import os
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import workspace as workspace_module
from hyper_parallel.core.multicore.modules.mega_moe.spec import (
    _resolve_receive_capacity,
)
from hyper_parallel.core.multicore.modules.mega_moe.workspace import (
    _EVENT_COUNTER_BYTES,
    _CYCLE_PROFILE_CATEGORIES,
    _CYCLE_PROFILE_STORAGE_BYTES,
    _GMM_WORKSPACE_BYTES,
    _event_workspace_bytes,
    _plan_weight_grad_scratch,
    _ROUTE_ARRIVAL_STRIDE_BYTES,
    _ROUTE_SIGNAL_ARENAS,
    _SWIGLU_GRAD_WORKSPACE_BYTES,
    _WORKSPACE_ALIGNMENT,
    MegaMoeWorkspace,
    _spec_workspace_bytes,
)
from hyper_parallel.core.multicore.scheduler.config import READY_CACHE_LINE_BYTES as _READY_CACHE_LINE_BYTES


class TestMegaMoeWorkspaceSizing(unittest.TestCase):
    """Validate SHMEM planning without allocating accelerator memory."""

    @staticmethod
    def _specification(expert_capacity_factor):
        """Build a fixed capacity-planning specification."""
        return {
            "local_num_tokens": 128,
            "hidden_size": 16,
            "num_experts": 8,
            "top_k": 2,
            "expert_capacity_factor": expert_capacity_factor,
            "ep_size": 8,
        }

    def test_capacity_planning_covers_lossless_and_bounded_modes(self) -> None:
        """Reserve the EP maximum by default and align explicit factors."""
        specification = self._specification(None)
        routed_slots = 256
        expected_capacity = 2048
        element_size = 2
        expected_bytes = (
            (expected_capacity + routed_slots)
            * specification["hidden_size"]
            * element_size
            + 2 * _EVENT_COUNTER_BYTES
            + 2 * (specification["ep_size"] + 1) * _READY_CACHE_LINE_BYTES
            + 2
            * _ROUTE_SIGNAL_ARENAS
            * specification["num_experts"]
            * _ROUTE_ARRIVAL_STRIDE_BYTES
            + 4 * (_WORKSPACE_ALIGNMENT - 1)
        )

        actual_capacity = _resolve_receive_capacity(
            specification["expert_capacity_factor"],
            routed_slots,
            specification["ep_size"],
        )
        actual_bytes = _spec_workspace_bytes(specification, element_size)

        self.assertEqual(actual_capacity, expected_capacity)
        self.assertEqual(actual_bytes, expected_bytes)
        self.assertEqual(
            _resolve_receive_capacity(
                1.5,
                routed_slots,
                specification["ep_size"],
            ),
            384,
        )

    def test_reuse_waits_with_event_without_host_synchronize(self) -> None:
        """Order serial reuse across streams without blocking the CPU."""
        symmetric_memory = Mock()
        workspace = MegaMoeWorkspace(symmetric_memory=symmetric_memory, shared=True)
        completion_event = Mock()
        current_stream = Mock()
        workspace.completion_event = completion_event
        workspace.used = True

        with (
            patch.object(
                workspace_module.torch.npu,
                "current_stream",
                return_value=current_stream,
            ),
            patch.object(workspace_module.torch.npu, "synchronize") as mock_synchronize,
        ):
            workspace.claim()
            workspace.release()

        current_stream.wait_event.assert_called_once_with(completion_event)
        completion_event.record.assert_called_once_with(current_stream)
        mock_synchronize.assert_not_called()

    def test_route_events_reserve_arrival_and_copy_ack_arenas(self):
        """
        Feature: tile-level hidden/dY buffer reuse
        Description: size cache-line-isolated route signal storage
        Expectation: reserve one arrival and one copy-ack arena
        """
        ep_size = 8
        local_num_tokens = 4096
        num_experts = 48
        dispatch_split = 128
        tasks_per_expert = local_num_tokens // dispatch_split
        route_arena_bytes = (
            num_experts
            * tasks_per_expert
            * _ROUTE_ARRIVAL_STRIDE_BYTES
        )

        actual_bytes = _event_workspace_bytes(
            ep_size=ep_size,
            local_num_tokens=local_num_tokens,
            num_experts=num_experts,
            dispatch_split=dispatch_split,
            fusion_mode="both",
        )
        expected_bytes = (
            _EVENT_COUNTER_BYTES
            + (ep_size + 1) * _READY_CACHE_LINE_BYTES
            + 2 * route_arena_bytes
        )

        self.assertEqual(actual_bytes, expected_bytes)

    def test_replica_slots_reserve_physical_route_event_arenas(self):
        """
        Feature: physical expert route events
        Description: add one replica group on each of two EP ranks
        Expectation: size forward and backward arenas for both physical IDs
        """
        specification = {
            "local_num_tokens": 128,
            "hidden_size": 16,
            "num_experts": 8,
            "top_k": 2,
            "expert_capacity_factor": 1.5,
            "replica_slots_per_rank": 1,
            "ep_size": 2,
        }
        static_specification = dict(specification)
        static_specification["replica_slots_per_rank"] = 0
        added_physical_experts = specification["ep_size"]
        tasks_per_expert = 1
        expected_added_bytes = (
            2
            * _ROUTE_SIGNAL_ARENAS
            * added_physical_experts
            * tasks_per_expert
            * _ROUTE_ARRIVAL_STRIDE_BYTES
        )

        actual_added_bytes = _spec_workspace_bytes(
            specification,
            2,
        ) - _spec_workspace_bytes(static_specification, 2)

        self.assertEqual(actual_added_bytes, expected_added_bytes)

    @patch.dict(
        os.environ,
        {
            "HYPER_PARALLEL_MEGA_MOE_FORWARD_FUSION_MODE": "permute",
            "HYPER_PARALLEL_MEGA_MOE_BACKWARD_FUSION_MODE": "legacy",
        },
    )
    def test_forward_only_permute_reserves_backward_hidden_route_events(self):
        """
        Feature: directional multi-wave route workspace
        Description: fuse forward permute while keeping backward dY legacy
        Expectation: reserve route arenas for backward hidden redispatch too
        """
        specification = {
            "local_num_tokens": 128,
            "hidden_size": 16,
            "num_experts": 8,
            "top_k": 2,
            "expert_capacity_factor": 1.5,
            "ep_size": 8,
        }
        event_bytes = _event_workspace_bytes(
            ep_size=specification["ep_size"],
            local_num_tokens=specification["local_num_tokens"],
            num_experts=specification["num_experts"],
            dispatch_split=128,
            fusion_mode="permute",
        )
        capacity = _resolve_receive_capacity(1.5, 256, specification["ep_size"])
        expected = (
            (capacity + 256) * specification["hidden_size"] * 2
            + 2 * event_bytes
            + 4 * (_WORKSPACE_ALIGNMENT - 1)
        )

        self.assertEqual(_spec_workspace_bytes(specification, 2), expected)

    def test_one_wave_preserves_outstanding_multi_wave_scratch(self):
        """
        Feature: GMM workspace high-watermark capacity
        Description: enqueue W2 then W1 while the W2 backward is outstanding
        Expectation: keep the W2 scratch without reallocating or synchronizing
        """
        workspace = MegaMoeWorkspace(
            symmetric_memory=Mock(),
            shared=True,
            device=torch.device("cpu"),
            gmm_workspace=torch.empty((_GMM_WORKSPACE_BYTES,), dtype=torch.uint8),
        )
        weight1 = torch.empty((2, 4, 6), dtype=torch.bfloat16)
        weight2 = torch.empty((2, 3, 4), dtype=torch.bfloat16)
        scratch_bytes = (weight1.numel() + weight2.numel()) * weight1.element_size()
        original_workspace = workspace.gmm_workspace

        with patch.object(torch.npu, "synchronize") as mock_synchronize:
            workspace.prepare_gmm_workspace(
                weight1,
                weight2,
                wave_count=1,
                receive_capacity=0,
                reuse_retained_scratch=False,
                atomic_dw_accumulation=False,
            )
            self.assertIs(workspace.gmm_workspace, original_workspace)
            self.assertEqual(workspace.gmm_workspace.numel(), _GMM_WORKSPACE_BYTES)
            mock_synchronize.assert_not_called()

            workspace.prepare_gmm_workspace(
                weight1,
                weight2,
                wave_count=2,
                receive_capacity=0,
                reuse_retained_scratch=False,
                atomic_dw_accumulation=False,
            )
            self.assertEqual(
                workspace.gmm_workspace.numel(),
                _GMM_WORKSPACE_BYTES + scratch_bytes,
            )
            mock_synchronize.assert_called_once_with(workspace.device)

            grown_workspace = workspace.gmm_workspace
            for wave_count in (1, 2, 1, 4):
                workspace.prepare_gmm_workspace(
                    weight1,
                    weight2,
                    wave_count=wave_count,
                    receive_capacity=0,
                    reuse_retained_scratch=False,
                    atomic_dw_accumulation=False,
                )
                self.assertIs(workspace.gmm_workspace, grown_workspace)
                self.assertEqual(
                    workspace.gmm_workspace.numel(),
                    _GMM_WORKSPACE_BYTES + scratch_bytes,
                )
            mock_synchronize.assert_called_once_with(workspace.device)

    def test_weight_gradient_scratch_only_grows_for_larger_weights(self):
        """
        Feature: shared GMM scratch capacity
        Description: vary matrix sizes while earlier backward graphs remain live
        Expectation: grow for larger matrices and retain the maximum for smaller ones
        """
        workspace = MegaMoeWorkspace(
            symmetric_memory=Mock(),
            shared=True,
            device=torch.device("cpu"),
            gmm_workspace=torch.empty((_GMM_WORKSPACE_BYTES,), dtype=torch.uint8),
        )
        maximum_bytes = _GMM_WORKSPACE_BYTES
        with patch.object(torch.npu, "synchronize") as mock_synchronize:
            for hidden_size in (4, 8, 4):
                weight1 = torch.empty((2, hidden_size, 6), dtype=torch.bfloat16)
                weight2 = torch.empty((2, 3, hidden_size), dtype=torch.bfloat16)
                maximum_bytes = max(
                    maximum_bytes,
                    _GMM_WORKSPACE_BYTES
                    + (weight1.numel() + weight2.numel()) * weight1.element_size(),
                )
                workspace.prepare_gmm_workspace(
                    weight1,
                    weight2,
                    wave_count=2,
                    receive_capacity=0,
                    reuse_retained_scratch=False,
                    atomic_dw_accumulation=False,
                )
                self.assertEqual(workspace.gmm_workspace.numel(), maximum_bytes)
        self.assertEqual(mock_synchronize.call_count, 2)

    def test_split_weights_keep_physical_multi_wave_scratch_capacity(self):
        """Size non-atomic dW scratch for N+B even with N-only operator inputs.

        Feature: split-weight multi-wave scratch
        Description: pass two home groups while three physical groups execute
        Expectation: each physical matrix has scratch with either retained-prefix policy
        """
        weight1 = torch.empty((2, 4, 6), dtype=torch.bfloat16)
        weight2 = torch.empty((2, 3, 4), dtype=torch.bfloat16)
        for retained in (False, True):
            with self.subTest(retained=retained), patch.object(torch.npu, "synchronize"):
                workspace = MegaMoeWorkspace(
            symmetric_memory=Mock(),
                    shared=True,
                    device=torch.device("cpu"),
                    gmm_workspace=torch.empty((_GMM_WORKSPACE_BYTES,), dtype=torch.uint8),
                )
                workspace.prepare_gmm_workspace(
                    weight1, weight2, 2, receive_capacity=1, reuse_retained_scratch=retained,
                    atomic_dw_accumulation=False, physical_experts=3,
                )
                # One retained row cannot fit any of these whole matrices.
                self.assertEqual(workspace.gmm_workspace.numel(), _GMM_WORKSPACE_BYTES + 3 * (24 + 12) * 2)

    def test_retained_prefix_packs_whole_weight_gradient_matrices(self):
        """
        Feature: retained up-projection scratch reuse
        Description: plan production and small gated-MLP matrix layouts
        Expectation: keep only unpacked whole matrices in the GMM workspace tail
        """
        production = _plan_weight_grad_scratch(
            local_experts=6,
            hidden_size=7168,
            intermediate_size=2048,
            element_size=2,
            receive_capacity=49152,
        )
        self.assertEqual(production.retained_w1_experts, 6)
        self.assertEqual(production.retained_w2_experts, 1)
        self.assertEqual(production.tail_bytes, 140 * 1024 * 1024)

        partial = _plan_weight_grad_scratch(
            local_experts=2,
            hidden_size=4,
            intermediate_size=3,
            element_size=2,
            receive_capacity=10,
        )
        self.assertEqual(partial.retained_w1_experts, 2)
        self.assertEqual(partial.retained_w2_experts, 1)
        self.assertEqual(partial.tail_bytes, 24)

    def test_retained_prefix_reduces_multi_wave_workspace_tail(self):
        """
        Feature: retained up-projection scratch reuse
        Description: size a W2 workspace with the internal reuse selector
        Expectation: allocate only the matrices that do not fit in the prefix
        """
        workspace = MegaMoeWorkspace(
            symmetric_memory=Mock(),
            shared=False,
            device=torch.device("cpu"),
            gmm_workspace=torch.empty((_GMM_WORKSPACE_BYTES,), dtype=torch.uint8),
        )
        weight1 = torch.empty((2, 4, 6), dtype=torch.bfloat16)
        weight2 = torch.empty((2, 3, 4), dtype=torch.bfloat16)

        with patch.object(torch.npu, "synchronize") as mock_synchronize:
            workspace.prepare_gmm_workspace(
                weight1,
                weight2,
                wave_count=2,
                receive_capacity=10,
                reuse_retained_scratch=True,
                atomic_dw_accumulation=False,
            )

        self.assertEqual(
            workspace.gmm_workspace.numel(),
            _GMM_WORKSPACE_BYTES + 24,
        )
        mock_synchronize.assert_called_once_with(workspace.device)

    def test_atomic_weight_grad_accumulation_omits_matrix_scratch(self):
        """
        Feature: atomic multi-wave weight-gradient accumulation
        Description: size a W2 workspace for direct Matmul output accumulation
        Expectation: retain only the fixed GMM workspace
        """
        workspace = MegaMoeWorkspace(
            symmetric_memory=Mock(),
            shared=False,
            device=torch.device("cpu"),
            gmm_workspace=torch.empty((_GMM_WORKSPACE_BYTES,), dtype=torch.uint8),
        )
        weight1 = torch.empty((2, 4, 6), dtype=torch.bfloat16)
        weight2 = torch.empty((2, 3, 4), dtype=torch.bfloat16)

        with patch.object(torch.npu, "synchronize") as mock_synchronize:
            workspace.prepare_gmm_workspace(
                weight1,
                weight2,
                wave_count=4,
                receive_capacity=10,
                reuse_retained_scratch=False,
                atomic_dw_accumulation=True,
            )

        self.assertEqual(workspace.gmm_workspace.numel(), _GMM_WORKSPACE_BYTES)
        mock_synchronize.assert_not_called()

    def test_routed_buffer_generation_requires_workspace_lease(self):
        """
        Feature: routed SHMEM replay tracking
        Description: record forward and backward writes to the shared buffer
        Expectation: advance only while the workspace has an active lease
        """
        workspace = MegaMoeWorkspace(symmetric_memory=Mock(), shared=False)

        with self.assertRaisesRegex(RuntimeError, "active workspace lease"):
            workspace.mark_routed_buffer_write()

        workspace.in_use = True
        self.assertEqual(workspace.mark_routed_buffer_write(), 1)
        self.assertEqual(workspace.mark_routed_buffer_write(), 2)
        workspace.in_use = False

        with self.assertRaisesRegex(RuntimeError, "active workspace lease"):
            workspace.mark_routed_buffer_write()

    def test_cycle_profile_uses_fixed_workspace_tail(self):
        """
        Feature: opt-in kernel cycle profiling
        Description: clear and decode the fixed SwiGLU workspace tail
        Expectation: keep disabled data untouched and report per-core counters
        """
        workspace = MegaMoeWorkspace(
            symmetric_memory=Mock(),
            shared=False,
            swiglu_grad_workspace=torch.full(
                (
                    _SWIGLU_GRAD_WORKSPACE_BYTES
                    + _CYCLE_PROFILE_STORAGE_BYTES,
                ),
                0xFF,
                dtype=torch.uint8,
            ),
        )
        profile_tail = workspace.swiglu_grad_workspace.narrow(
            0,
            _SWIGLU_GRAD_WORKSPACE_BYTES,
            _CYCLE_PROFILE_STORAGE_BYTES,
        )

        workspace.prepare_cycle_profile(enabled=False)
        self.assertTrue(torch.all(profile_tail == 0xFF))
        workspace.prepare_cycle_profile(enabled=True)
        self.assertEqual(torch.count_nonzero(profile_tail), 0)

        counters = profile_tail.view(torch.int64)
        counters[0] = 11
        counters[len(_CYCLE_PROFILE_CATEGORIES)] = 13
        vector_offset = 24 * len(_CYCLE_PROFILE_CATEGORIES)
        counters[vector_offset] = 17
        counters[vector_offset + len(_CYCLE_PROFILE_CATEGORIES)] = 19

        result = workspace.collect_cycle_profile(num_cube_cores=2)

        self.assertEqual(result["cube"]["per_core"][0][0], 11)
        self.assertEqual(result["cube"]["sum"]["descriptor_decode"], 24)
        self.assertEqual(result["vector"]["per_core"][0][0], 17)
        self.assertEqual(result["vector"]["max"]["descriptor_decode"], 19)


if __name__ == "__main__":
    unittest.main()
