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
"""Unit tests for MegaMoe plan construction."""

import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import (
    spec as config_module,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import (
    build_multi_wave_backward_graph,
    build_route_backward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank as build_forward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_route_forward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.function import (
    _mark_retained_scratch_modified,
    _prepare_backward_route,
    _preserve_forward_input,
)
from hyper_parallel.core.multicore.modules.mega_moe.plan import (
    _build_route_slots,
    _build_task_values,
    _use_token_a2a_offsets,
)
from hyper_parallel.core.multicore.modules.mega_moe.spec import (
    _MAX_EXACT_FLOAT32_INTEGER,
    MegaMoeSpec,
    _resolve_fusion_modes,
    bind_mega_moe_spec,
)
from hyper_parallel.core.multicore.scheduler.config import (
    ATOMIC_DW_ACCUMULATION_FLAG,
    NUM_WORKERS_CUBE,
    NUM_WORKERS_VECTOR,
    READY_HANDSHAKE_COPY_ACK_VERSION,
    READY_HANDSHAKE_DESTINATION_FENCE_VERSION,
    READY_HANDSHAKE_VERSION,
    RETAINED_DW_SCRATCH_FLAG,
    ROUTE_ARRIVAL_WAIT_FLAG,
    ROUTE_COMPUTE_ACK_FLAG,
    ROUTE_COPY_ACK_FLAG,
    ROUTE_EXPERT_ACK_FLAG,
    RuntimeConfigC,
    TaskSplitValue,
    TaskType,
)
from hyper_parallel.core.multicore.tasks.gmm import GmmFillConfig


class TestMegaMoePlan(unittest.TestCase):
    """Validate translation from the public local-token contract."""

    @patch.dict(os.environ, {}, clear=True)
    def test_route_permute_is_the_supported_shape_default(self):
        """
        Feature: optimized route-fusion defaults
        Description: resolve modes for a supported Top-K and route size
        Expectation: fuse forward permute and both backward endpoints
        """
        self.assertEqual(_resolve_fusion_modes(8, 32_768), ("permute", "both"))

    @patch.dict(os.environ, {}, clear=True)
    def test_unsupported_default_route_fusion_uses_legacy(self):
        """
        Feature: route-fusion compatibility fallback
        Description: resolve modes outside Top-K or exact-index limits
        Expectation: select legacy endpoints for both directions
        """
        unsupported_shapes = (
            (17, 32_768),
            (8, _MAX_EXACT_FLOAT32_INTEGER + 1),
        )
        for top_k, routed_slots in unsupported_shapes:
            with self.subTest(top_k=top_k, routed_slots=routed_slots):
                self.assertEqual(
                    _resolve_fusion_modes(top_k, routed_slots),
                    ("legacy", "legacy"),
                )

    @patch.dict(os.environ, {}, clear=True)
    def test_replica_remap_rejects_inexact_float_route_slots(self):
        """
        Feature: exact physical route remapping
        Description: enable replicas above the exact float32 route-slot range
        Expectation: reject the configuration even when fusion falls back to legacy
        """
        fake_torch = SimpleNamespace(
            bfloat16=torch.bfloat16,
            device=lambda _device: SimpleNamespace(type="npu", index=0),
            npu=SimpleNamespace(
                current_device=lambda: 0,
                get_device_limit=lambda _device: {"cube_core_num": 24},
            ),
        )
        fake_dist = SimpleNamespace(is_initialized=lambda: False)
        specification = {
            "local_num_tokens": _MAX_EXACT_FLOAT32_INTEGER + 128,
            "hidden_size": 8,
            "intermediate_size": 4,
            "num_experts": 1,
            "top_k": 1,
            "expert_capacity_factor": 1.0,
            "replica_slots_per_rank": 1,
            "ep_size": 1,
            "ep_group": None,
        }
        tensor = SimpleNamespace(device="npu:0", dtype=torch.bfloat16)

        with (
            patch.object(config_module, "torch", new=fake_torch),
            patch.object(
                config_module,
                "dist",
                new=fake_dist,
            ),
            self.assertRaisesRegex(ValueError, "route-indexed MegaMoe"),
        ):
            bind_mega_moe_spec(specification, tensor)

    def test_local_tokens_use_one_legacy_scheduler_partition(self):
        """
        Feature: TP-free MegaMoe public configuration
        Description: lower local-token dimensions into the legacy scheduler
        Expectation: use one internal partition without exposing TP to callers
        """
        spec = MegaMoeSpec(
            expert_capacity_factor=1.5,
            local_num_tokens=4096,
            hidden_size=5120,
            intermediate_size=1792,
            num_experts=32,
            top_k=8,
            receive_capacity=40960,
            ep_size=4,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
        )

        task_values = _build_task_values(spec)

        self.assertEqual(task_values.tp, 1)
        self.assertEqual(task_values.seq_size, spec.local_num_tokens)
        self.assertEqual(
            task_values.per_rank_seq, spec.local_num_tokens * spec.top_k
        )
        self.assertEqual(task_values.per_expert_seq_to_other, spec.local_num_tokens)

    def test_replica_slots_expand_only_the_physical_scheduler_groups(self):
        """
        Feature: fixed physical expert groups
        Description: configure two replica slots beside eight home experts per rank
        Expectation: keep 32 logical experts while scheduling 40 physical groups
        """
        spec = MegaMoeSpec(
            expert_capacity_factor=1.5,
            local_num_tokens=4096,
            hidden_size=5120,
            intermediate_size=1792,
            num_experts=32,
            top_k=8,
            receive_capacity=40960,
            ep_size=4,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
            replica_slots_per_rank=2,
        )

        task_values = _build_task_values(spec)

        self.assertEqual(spec.local_experts, 8)
        self.assertEqual(spec.physical_local_experts, 10)
        self.assertEqual(spec.physical_num_experts, 40)
        self.assertEqual(task_values.all_expert_num, 40)
        self.assertEqual(task_values.single_rank_expert_num, 10)

    def test_permute_fusion_caches_each_route_slot(self):
        """
        Feature: scatter-free route-slot cache
        Description: build static route identities for a fused plan
        Expectation: represent every flattened Top-K slot exactly once
        """
        spec = MegaMoeSpec(
            expert_capacity_factor=1.5,
            local_num_tokens=4,
            hidden_size=8,
            intermediate_size=4,
            num_experts=4,
            top_k=2,
            receive_capacity=128,
            ep_size=1,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
            forward_fusion_mode="permute",
        )

        route_slots = _build_route_slots(spec, "cpu")

        self.assertEqual(tuple(route_slots.shape), (spec.routed_slots, 1))
        self.assertEqual(route_slots.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                route_slots.reshape(-1),
                torch.arange(spec.routed_slots, dtype=torch.float32),
            )
        )

    def test_one_wave_preserves_dispatch_in_dead_down_projection_storage(self):
        """
        Feature: fixed-HBM one-wave forward retention
        Description: preserve dispatched hidden input after forward Combine
        Expectation: recycle dead down-projection storage instead of cloning
        """
        dispatch = Mock()
        down_proj = Mock()
        kernel_input = Mock()
        down_proj.copy_.return_value = down_proj

        preserved = _preserve_forward_input(
            SimpleNamespace(wave_count=1),
            dispatch,
            down_proj,
            kernel_input,
            redispatch_w1_input=False,
        )

        down_proj.copy_.assert_called_once_with(dispatch)
        self.assertIs(preserved, down_proj)

    def test_multi_wave_preserves_original_kernel_input(self):
        """
        Feature: bounded multi-wave forward retention
        Description: retain token-order input for backward route replay
        Expectation: avoid touching rolling down-projection storage
        """
        dispatch = Mock()
        down_proj = Mock()
        kernel_input = Mock()

        preserved = _preserve_forward_input(
            SimpleNamespace(wave_count=2),
            dispatch,
            down_proj,
            kernel_input,
            redispatch_w1_input=False,
        )

        down_proj.copy_.assert_not_called()
        self.assertIs(preserved, kernel_input)

    def test_one_wave_redispatch_preserves_original_kernel_input(self):
        """
        Feature: fixed-HBM one-wave backward redispatch
        Description: retain token-order hidden input for the late route phase
        Expectation: avoid a capacity-sized forward-input tensor
        """
        dispatch = Mock()
        down_proj = Mock()
        kernel_input = Mock()

        preserved = _preserve_forward_input(
            SimpleNamespace(wave_count=1),
            dispatch,
            down_proj,
            kernel_input,
            redispatch_w1_input=True,
        )

        down_proj.copy_.assert_not_called()
        self.assertIs(preserved, kernel_input)

    def test_retained_scratch_marks_saved_tensor_modified(self):
        """
        Feature: retained up-projection scratch lifecycle
        Description: finish a multi-wave backward that overwrites saved storage
        Expectation: increment its version while leaving disabled and W1 paths untouched
        """
        increment_version = Mock()
        torch_module = SimpleNamespace(
            autograd=SimpleNamespace(
                graph=SimpleNamespace(increment_version=increment_version)
            )
        )
        saved = SimpleNamespace(up_proj=object())

        with patch(
            "hyper_parallel.core.multicore.modules.mega_moe.function.torch",
            new=torch_module,
        ):
            _mark_retained_scratch_modified(
                SimpleNamespace(
                    wave_count=2,
                    plan=SimpleNamespace(reuse_retained_dw_scratch=True),
                ),
                saved,
            )
            _mark_retained_scratch_modified(
                SimpleNamespace(
                    wave_count=1,
                    plan=SimpleNamespace(reuse_retained_dw_scratch=True),
                ),
                saved,
            )
            _mark_retained_scratch_modified(
                SimpleNamespace(
                    wave_count=2,
                    plan=SimpleNamespace(reuse_retained_dw_scratch=False),
                ),
                saved,
            )

        increment_version.assert_called_once_with(saved.up_proj)

    def test_retained_scratch_rejects_second_backward_on_same_graph(self):
        """
        Feature: retained up-projection scratch lifecycle
        Description: retain an autograd graph after its saved tensor is reused
        Expectation: a second backward fails before consuming overwritten storage
        """

        class ReuseSavedTensor(torch.autograd.Function):
            """Model the version transition made after the device launch."""

            @staticmethod
            def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
                """Save a derived tensor that backward later invalidates."""
                saved = value.square()
                ctx.save_for_backward(saved)
                return saved.sum()

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                """Invalidate the saved tensor after its first consumption."""
                (saved,) = ctx.saved_tensors
                _mark_retained_scratch_modified(
                    SimpleNamespace(
                        wave_count=2,
                        plan=SimpleNamespace(
                            reuse_retained_dw_scratch=True
                        ),
                    ),
                    SimpleNamespace(up_proj=saved),
                )
                return grad_output.expand_as(saved)

        value = torch.ones(4, requires_grad=True)
        result = ReuseSavedTensor.apply(value)
        result.backward(retain_graph=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "modified by an inplace operation",
        ):
            result.backward()

    def test_route_replay_restores_expert_output_and_uses_rebuilt_mapping(self):
        """
        Feature: retained route-fused autograd graph
        Description: prepare backward after its shared route buffer was overwritten
        Expectation: restore expert output and select the rebuilt route mapping
        """
        expert_output = torch.full((4, 3), 7, dtype=torch.bfloat16)
        routed_buffer = torch.full((4, 3), -1, dtype=torch.bfloat16)
        original_mapping = torch.arange(8, dtype=torch.int32)
        replay_mapping = original_mapping.flip(0).contiguous()
        workspace = SimpleNamespace(
            expert_buffer=torch.empty((4, 3), dtype=torch.bfloat16),
            routed_buffer=routed_buffer,
            routed_buffer_generation=2,
        )
        ctx = SimpleNamespace(
            plan=SimpleNamespace(
                spec=SimpleNamespace(
                    local_num_tokens=2,
                    hidden_size=3,
                    forward_permute_fused=True,
                    backward_permute_fused=True,
                ),
            ),
            workspace=workspace,
            permute_fused=True,
            backward_unpermute_fused=False,
            expert_capacity=4,
            wave_count=2,
            routed_buffer_generation=1,
            replay_reduce_route_index=replay_mapping,
        )
        saved = SimpleNamespace(
            permute_src_index=torch.arange(8, dtype=torch.int32),
            topk_weights=torch.ones(8),
            expert_output=expert_output,
            reduce_route_index=original_mapping,
        )

        route_state = _prepare_backward_route(
            ctx,
            torch.empty((2, 3), dtype=torch.bfloat16),
            saved,
        )

        self.assertIs(route_state.grad_x, routed_buffer)
        self.assertTrue(torch.equal(route_state.grad_x, expert_output))
        self.assertIs(route_state.reduce_route_index, replay_mapping)

    def test_route_first_backward_uses_existing_shmem_without_copy(self):
        """
        Feature: common route-fused backward path
        Description: prepare backward directly after its matching forward
        Expectation: consume the current routed SHMEM generation in place
        """
        routed_buffer = torch.full((4, 3), 5, dtype=torch.bfloat16)
        workspace = SimpleNamespace(
            expert_buffer=torch.empty((4, 3), dtype=torch.bfloat16),
            routed_buffer=routed_buffer,
            routed_buffer_generation=3,
        )
        ctx = SimpleNamespace(
            plan=SimpleNamespace(
                spec=SimpleNamespace(
                    local_num_tokens=2,
                    hidden_size=3,
                    forward_permute_fused=True,
                    backward_permute_fused=True,
                ),
            ),
            workspace=workspace,
            permute_fused=True,
            backward_unpermute_fused=False,
            expert_capacity=4,
            wave_count=2,
            routed_buffer_generation=3,
            replay_reduce_route_index=None,
        )
        original_mapping = torch.arange(8, dtype=torch.int32)
        saved = SimpleNamespace(
            permute_src_index=torch.arange(8, dtype=torch.int32),
            topk_weights=torch.ones(8),
            expert_output=torch.full((4, 3), 9, dtype=torch.bfloat16),
            reduce_route_index=original_mapping,
        )

        route_state = _prepare_backward_route(
            ctx,
            torch.empty((2, 3), dtype=torch.bfloat16),
            saved,
        )

        self.assertIs(route_state.grad_x, routed_buffer)
        self.assertTrue(torch.all(route_state.grad_x == 5))
        self.assertIs(route_state.reduce_route_index, original_mapping)

    @patch(
        "hyper_parallel.core.multicore.modules.mega_moe.function._unpermute_grad"
    )
    def test_multi_wave_hidden_route_survives_directional_fusion(
        self,
        unpermute_grad: Mock,
    ):
        """
        Feature: independently fused multi-wave endpoints
        Description: fuse forward permute while backward dY dispatch is legacy
        Expectation: retain the expert-major route slots for hidden redispatch
        """
        kernel_grad = torch.ones((4, 3), dtype=torch.bfloat16)
        grad_topk = torch.ones((2, 2), dtype=torch.float32)
        unpermute_grad.return_value = (kernel_grad, grad_topk)
        permute_src_index = torch.arange(4, dtype=torch.int32)
        dispatch_size = torch.ones(4, dtype=torch.int32)
        routed_buffer = torch.empty((4, 3), dtype=torch.bfloat16)
        ctx = SimpleNamespace(
            plan=SimpleNamespace(
                spec=SimpleNamespace(
                    local_num_tokens=2,
                    hidden_size=3,
                    forward_permute_fused=True,
                    backward_permute_fused=False,
                ),
            ),
            workspace=SimpleNamespace(
                expert_buffer=torch.empty((4, 3), dtype=torch.bfloat16),
                routed_buffer=routed_buffer,
                routed_buffer_generation=1,
            ),
            permute_fused=False,
            backward_unpermute_fused=False,
            expert_capacity=4,
            wave_count=2,
            routed_buffer_generation=1,
            replay_reduce_route_index=None,
        )
        saved = SimpleNamespace(
            expert_output=torch.empty((4, 3), dtype=torch.bfloat16),
            reduce_route_index=torch.arange(4, dtype=torch.int32),
            topk_weights=torch.ones((2, 2), dtype=torch.float32),
            dispatch_size=dispatch_size,
            permute_src_index=permute_src_index,
        )

        route_state = _prepare_backward_route(
            ctx,
            torch.empty((2, 3), dtype=torch.bfloat16),
            saved,
        )

        self.assertIs(route_state.permute_src_index, permute_src_index)
        self.assertIs(route_state.grad_x, routed_buffer)

    def test_multi_wave_hidden_route_can_differ_from_dy_route(self):
        """
        Feature: independently fused multi-wave graph
        Description: route token-major hidden input with expert-major dY
        Expectation: only hidden dispatch carries route slots and Top-K
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )
        graph = build_multi_wave_backward_graph(
            task_values,
            integrated_permute=False,
            integrated_hidden_permute=True,
        )
        graph.propagate_splits(task_values)
        runtime_config = build_backward_config(
            graph,
            task_values,
            rank_id=0,
            kernel_multi_wave=True,
        )
        tasks = runtime_config.all_tasks[: runtime_config.task_num]
        dy_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            and task.extra_value_3 & 0xFF == 1
        ]
        hidden_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            and task.extra_value_3 & 0xFF == 3
        ]

        self.assertTrue(dy_tasks)
        self.assertTrue(hidden_tasks)
        self.assertTrue(all(task.extra_value_0 == 0xFFFFFFFF for task in dy_tasks))
        self.assertTrue(all(task.extra_value_4 >> 8 == 0 for task in dy_tasks))
        self.assertTrue(all(task.extra_value_0 != 0xFFFFFFFF for task in hidden_tasks))
        self.assertTrue(
            all(task.extra_value_4 >> 8 == task_values.top_k
                for task in hidden_tasks)
        )
        self.assertEqual(
            runtime_config.ready_handshake_meta.version,
            READY_HANDSHAKE_VERSION,
        )

    def test_weight_grad_completion_events_do_not_alias_activation_grad(self):
        """
        Feature: multi-wave weight-gradient event allocation
        Description: serialize an overlapped dW GMM with an event offset
        Expectation: keep dW and activation-gradient completion generations disjoint
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )
        task_values.pre_pre_event_num = 8
        task_values.pre_event_num = 12
        tensor = SimpleNamespace(
            dtype_size=2,
            tensor_type=1,
            is_dynamic=True,
            transpose=False,
            shape=[256, 128],
        )
        weight = SimpleNamespace(
            dtype_size=2,
            tensor_type=1,
            is_dynamic=False,
            transpose=False,
            shape=[2, 128, 512],
        )
        group_list = SimpleNamespace(
            dtype_size=8,
            tensor_type=0,
            is_dynamic=False,
            transpose=False,
            shape=[2],
        )
        output = SimpleNamespace(
            dtype_size=2,
            tensor_type=1,
            is_dynamic=True,
            shape=[256, 512],
        )
        operator = SimpleNamespace(
            task_num=48,
            param_positions=[0, 7, 19, 8],
            inputs=[tensor, weight, group_list],
            outputs=[output],
            split_value=4096,
            tiling_position=20,
        )
        runtime_config = RuntimeConfigC()
        fill_config = GmmFillConfig(
            completion_event_offset=task_values.single_rank_expert_num,
            num_cube_cores=24,
        )

        fill_config.fill(runtime_config, operator, task_values)

        trigger_events = {
            runtime_config.all_tasks[index].trigger_event
            for index in range(operator.task_num)
        }
        self.assertEqual(trigger_events, {15, 16})
        self.assertEqual(fill_config.completion_event_base, 14)
        self.assertEqual(runtime_config.all_event_num_triggers[15], 24)
        self.assertEqual(runtime_config.all_event_num_triggers[16], 24)

    def test_multi_wave_topk_grad_reuses_route_index_before_combine(self):
        """
        Feature: fused multi-wave TopK gradient ABI
        Description: inspect the backward graph around dTopK and Combine
        Expectation: reuse route-index storage and stay within 32 input slots
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )

        graph = build_multi_wave_backward_graph(
            task_values,
            integrated_permute=True,
        )
        topk_weight_grad = graph.get_op("topk_weight_grad")
        combine = graph.get_op("combine")

        self.assertEqual(topk_weight_grad.param_positions[0], 13)
        self.assertEqual(topk_weight_grad.inputs[0].name, "expert_output")
        self.assertEqual(topk_weight_grad.param_positions[2:], [31, 31])
        self.assertTrue(topk_weight_grad.fill_config.first_wave_only)
        self.assertEqual(combine.param_positions[-1], 13)
        self.assertIn(topk_weight_grad, combine.predecessors)
        self.assertEqual(
            combine.fill_config.additional_dependent_event,
            task_values.all_event_num - 2,
        )
        self.assertLessEqual(max(topk_weight_grad.param_positions), 31)

    def test_forward_route_dispatch_packs_topk_with_offset_units(self):
        """
        Feature: route-dispatch descriptor ABI
        Description: lower a forward route graph using token-unit offsets
        Expectation: preserve both the offset flag and Top-K route divisor
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )
        graph = build_route_forward_graph(task_values)
        _use_token_a2a_offsets(graph)
        graph.propagate_splits(task_values)

        runtime_config = build_forward_config(
            graph,
            task_values,
            rank_id=0,
        )
        dispatch_tasks = [
            task
            for task in runtime_config.all_tasks[: runtime_config.task_num]
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            and task.extra_value_3 & 0xFF == 1
        ]

        self.assertTrue(dispatch_tasks)
        self.assertTrue(
            all(task.extra_value_4 & 0x1 == 1 for task in dispatch_tasks)
        )
        self.assertTrue(
            all(task.extra_value_4 >> 8 == task_values.top_k
                for task in dispatch_tasks)
        )

    def test_one_wave_inplace_gate_dx_orders_weight_and_route_consumers(self):
        """
        Feature: fixed-HBM one-wave backward
        Description: reuse dispatched dY storage for gate dX and routed output
        Expectation: wait for dW2 and dTopK before overwriting either input
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )

        graph = build_route_backward_graph(
            task_values,
            integrated_permute=True,
            integrated_unpermute=False,
            inplace_gate_dx=True,
        )
        gate_grad = graph.get_op("gate_grad")
        w2_grad = graph.get_op("w2_grad")
        combine = graph.get_op("combine")
        topk_weight_grad = graph.get_op("topk_weight_grad")

        self.assertEqual(gate_grad.param_positions[-1], 0)
        self.assertIn(w2_grad, gate_grad.predecessors)
        self.assertEqual(combine.param_positions[1], 0)
        self.assertEqual(combine.param_positions[-1], 13)
        self.assertIn(topk_weight_grad, combine.predecessors)
        self.assertTrue(topk_weight_grad.fill_config.first_wave_only)
        self.assertEqual(
            combine.fill_config.additional_dependent_event,
            task_values.all_event_num - 2,
        )
        self.assertTrue(graph.has_op("external_output_acquire"))

        graph.propagate_splits(task_values)
        for rank_id in range(task_values.ep):
            runtime_config = build_backward_config(
                graph,
                task_values,
                rank_id=rank_id,
            )
            weight_tasks = [
                runtime_config.all_tasks[task_id]
                for task_id in range(runtime_config.task_num)
                if (
                    runtime_config.all_tasks[task_id].task_type
                    == TaskType.TASK_GROUPED_MATMUL
                    and runtime_config.all_tasks[task_id].tiling_data_position
                    == 22
                )
            ]
            first_dependency = weight_tasks[0].dependent_event
            self.assertEqual(
                (first_dependency - 1) % task_values.single_rank_expert_num,
                0,
            )
            self.assertLess(
                max(task.dependent_event for task in weight_tasks),
                task_values.all_event_num - 2,
            )
            dispatch_tasks = [
                task
                for task in runtime_config.all_tasks[: runtime_config.task_num]
                if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
                and task.extra_value_3 & 0xFF == 1
            ]
            self.assertTrue(dispatch_tasks)
            self.assertTrue(
                all(task.extra_value_4 >> 8 == task_values.top_k
                    for task in dispatch_tasks)
            )

    def test_one_wave_inplace_route_reorder_starts_at_combine_segment(self):
        """
        Feature: fixed-HBM one-wave backward vector scheduling
        Description: apply rank-aware Combine reordering after an early dTopK
        Expectation: no SwiGLU producer is moved behind a waiting Combine task
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=8,
            seq_size=4096,
            all_expert_num=48,
            top_k=8,
        )
        graph = build_route_backward_graph(
            task_values,
            integrated_permute=True,
            integrated_unpermute=False,
            inplace_gate_dx=True,
        )
        graph.propagate_splits(task_values)

        runtime_config = build_backward_config(
            graph,
            task_values,
            rank_id=0,
        )
        dispatch = graph.get_op("dispatch")
        topk_weight_grad = graph.get_op("topk_weight_grad")
        swiglu_grad = graph.get_op("swiglu_grad")
        combine = graph.get_op("combine")
        combine_start = (
            dispatch.task_num
            + topk_weight_grad.task_num
            + swiglu_grad.task_num
        )
        reordered_task_ids = runtime_config.vector_task_indices[
            combine_start:combine_start + combine.task_num
        ]

        self.assertEqual(len(reordered_task_ids), combine.task_num)
        self.assertTrue(
            all(
                runtime_config.all_tasks[task_id].task_type
                == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
                and runtime_config.all_tasks[task_id].extra_value_3 & 0xFF == 2
                for task_id in reordered_task_ids
            )
        )

    def test_route_fused_multi_wave_pairs_copy_and_dy_tiles(self):
        """
        Feature: tile-level hidden/dY buffer reuse
        Description: serialize source-aligned copy acknowledgements
        Expectation: copy waits one arrival and dY waits the matching ack
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )
        graph = build_multi_wave_backward_graph(
            task_values,
            integrated_permute=True,
        )
        graph.propagate_splits(task_values)
        runtime_config = build_backward_config(
            graph,
            task_values,
            rank_id=0,
            kernel_multi_wave=True,
        )
        tasks = runtime_config.all_tasks[: runtime_config.task_num]
        copy_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_LOCAL_COPY
        ]
        dy_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            and task.extra_value_3 & 0xFF == 1
        ]
        hidden_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            and task.extra_value_3 & 0xFF == 3
        ]
        self.assertEqual(len(copy_tasks), 4)
        self.assertEqual(
            {task.dependent_event for task in copy_tasks},
            {0xFFFFFFFF},
        )
        self.assertEqual(
            {task.extra_value_0 for task in copy_tasks},
            {ROUTE_COPY_ACK_FLAG},
        )
        self.assertEqual(
            [
                copy_tasks[0].inputs[index].input_position
                for index in range(3)
            ],
            [0, 15, 16],
        )
        self.assertTrue(all(task.num_inputs == 3 for task in copy_tasks))
        self.assertTrue(dy_tasks)
        self.assertTrue(
            all(task.extra_value_4 >> 8 == task_values.top_k
                for task in dy_tasks)
        )
        self.assertTrue(
            all(task.extra_value_3 & ROUTE_COPY_ACK_FLAG for task in dy_tasks)
        )
        self.assertTrue(hidden_tasks)
        self.assertTrue(
            all(task.extra_value_4 >> 8 == task_values.top_k
                for task in hidden_tasks)
        )
        self.assertTrue(
            all(
                not task.extra_value_3 & ROUTE_COPY_ACK_FLAG
                for task in hidden_tasks
            )
        )
        self.assertEqual(
            runtime_config.ready_handshake_meta.version,
            READY_HANDSHAKE_COPY_ACK_VERSION,
        )

    def test_inplace_multi_wave_reuses_expert_buffer_by_phase(self):
        """
        Feature: in-place multi-wave backward staging
        Description: serialize dY, hidden and dX phases on one symmetric buffer
        Expectation: wait for hidden arrivals without copying to position 12
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )
        graph = build_multi_wave_backward_graph(
            task_values,
            integrated_permute=True,
            inplace_wave_backward=True,
        )
        graph.propagate_splits(task_values)
        runtime_config = build_backward_config(
            graph,
            task_values,
            rank_id=0,
            kernel_multi_wave=True,
        )
        tasks = runtime_config.all_tasks[: runtime_config.task_num]
        wait_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_LOCAL_COPY
        ]
        dy_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            and task.extra_value_3 & 0xFF == 1
        ]
        hidden_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            and task.extra_value_3 & 0xFF == 3
        ]
        phase_task_ids = [
            index
            for index, task in enumerate(tasks)
            if task.task_type == TaskType.TASK_PHASE_BARRIER
        ]

        self.assertFalse(graph.has_op("copy_hidden"))
        self.assertTrue(graph.has_op("wait_hidden_arrivals"))
        self.assertEqual(len(wait_tasks), 4)
        self.assertTrue(
            all(task.extra_value_0 == ROUTE_ARRIVAL_WAIT_FLAG for task in wait_tasks)
        )
        self.assertTrue(all(task.num_inputs == 1 for task in wait_tasks))
        self.assertTrue(all(task.num_outputs == 1 for task in wait_tasks))
        self.assertTrue(
            all(not task.extra_value_3 & ROUTE_COPY_ACK_FLAG for task in dy_tasks)
        )
        self.assertTrue(
            all((task.extra_value_3 >> 8) & 0xFF == 0 for task in dy_tasks)
        )
        self.assertTrue(
            all((task.extra_value_3 >> 8) & 0xFF == 1 for task in hidden_tasks)
        )
        self.assertEqual(len(phase_task_ids), 1)
        phase_task_id = phase_task_ids[0]
        cube_task_ids = runtime_config.cube_task_indices[
            : runtime_config.task_index_num[0]
        ]
        vector_task_ids = runtime_config.vector_task_indices[
            : runtime_config.task_index_num[1]
        ]
        self.assertEqual(
            list(cube_task_ids).count(phase_task_id),
            NUM_WORKERS_CUBE,
        )
        self.assertEqual(
            list(vector_task_ids).count(phase_task_id),
            NUM_WORKERS_VECTOR // 2,
        )

        w1_grad = graph.get_op("w1_grad")
        gate_grad = graph.get_op("gate_grad")
        combine = graph.get_op("combine")
        self.assertEqual(w1_grad.param_positions[0], 0)
        self.assertEqual(gate_grad.param_positions[-1], 0)
        self.assertEqual(combine.param_positions[1], 0)
        hidden_ready_events = {
            task.trigger_event
            for task in wait_tasks
        }
        self.assertEqual(len(hidden_ready_events), 1)
        self.assertIn(w1_grad.fill_config.dependent_event, hidden_ready_events)
        self.assertEqual(
            runtime_config.ready_handshake_meta.version,
            READY_HANDSHAKE_VERSION,
        )

    def test_tile_phase_ack_releases_inplace_buffer_without_global_barrier(self):
        """
        Feature: tile-granular in-place multi-wave phase transition
        Description: acknowledge local dY consumption before hidden overwrites
        Expectation: every expert/source pair publishes before phase-two waits
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )
        graph = build_multi_wave_backward_graph(
            task_values,
            integrated_permute=True,
            inplace_wave_backward=True,
            tile_phase_ack=True,
        )
        graph.propagate_splits(task_values)
        runtime_config = build_backward_config(
            graph,
            task_values,
            rank_id=0,
            kernel_multi_wave=True,
        )
        tasks = runtime_config.all_tasks[: runtime_config.task_num]
        ack_task_ids = [
            index
            for index, task in enumerate(tasks)
            if task.task_type == TaskType.TASK_LOCAL_COPY
            and task.extra_value_0 == ROUTE_COMPUTE_ACK_FLAG
        ]
        wait_task_ids = [
            index
            for index, task in enumerate(tasks)
            if task.task_type == TaskType.TASK_LOCAL_COPY
            and task.extra_value_0 == ROUTE_ARRIVAL_WAIT_FLAG
        ]
        hidden_task_ids = [
            index
            for index, task in enumerate(tasks)
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            and task.extra_value_3 & 0xFF == 3
        ]

        self.assertTrue(graph.has_op("release_dy_tiles"))
        self.assertFalse(graph.has_op("reuse_phase_barrier"))
        self.assertEqual(
            len(ack_task_ids),
            task_values.single_rank_expert_num * task_values.ep,
        )
        self.assertEqual(len(wait_task_ids), 4)
        self.assertTrue(
            all(tasks[index].num_inputs == 0 for index in ack_task_ids)
        )
        self.assertTrue(
            all(tasks[index].num_outputs == 0 for index in ack_task_ids)
        )
        self.assertTrue(
            all(tasks[index].extra_value_2 == 1 for index in ack_task_ids)
        )

        act_base = graph.get_op("act_grad").fill_config.completion_event_base
        w2_base = graph.get_op("w2_grad").fill_config.completion_event_base
        for task_id in ack_task_ids:
            task = tasks[task_id]
            expert = task.task_index // task_values.ep
            self.assertEqual(task.dependent_event, act_base + expert + 1)
            self.assertEqual(task.extra_value_1, w2_base + expert + 1)

        vector_task_ids = list(
            runtime_config.vector_task_indices[
                : runtime_config.task_index_num[1]
            ]
        )
        ack_positions = [
            index
            for index, task_id in enumerate(vector_task_ids)
            if task_id in ack_task_ids
        ]
        hidden_positions = [
            index
            for index, task_id in enumerate(vector_task_ids)
            if task_id in hidden_task_ids
        ]
        self.assertLess(max(ack_positions), min(hidden_positions))
        self.assertTrue(
            all(tasks[index].extra_value_3 & ROUTE_COPY_ACK_FLAG
                for index in hidden_task_ids)
        )
        self.assertTrue(
            all(tasks[index].extra_value_3 & ROUTE_EXPERT_ACK_FLAG
                for index in hidden_task_ids)
        )
        self.assertEqual(
            runtime_config.ready_handshake_meta.version,
            READY_HANDSHAKE_COPY_ACK_VERSION,
        )

    def test_destination_wave_fence_reuses_ready_state_without_new_tasks(self):
        """
        Feature: per-destination inter-wave buffer ownership fence
        Description: select handshake v3 without serializing extra descriptors
        Expectation: dispatch carries the host selector and task counts stay fixed
        """
        invalid_task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )
        with self.assertRaisesRegex(ValueError, "tile phase acknowledgement"):
            build_multi_wave_backward_graph(
                invalid_task_values,
                integrated_permute=True,
                inplace_wave_backward=True,
                destination_wave_fence=True,
            )

        def _build(destination_wave_fence):
            task_values = TaskSplitValue(
                tp=1,
                ep=2,
                seq_size=128,
                all_expert_num=4,
                top_k=2,
            )
            graph = build_multi_wave_backward_graph(
                task_values,
                integrated_permute=True,
                inplace_wave_backward=True,
                tile_phase_ack=True,
                destination_wave_fence=destination_wave_fence,
            )
            graph.propagate_splits(task_values)
            runtime_config = build_backward_config(
                graph,
                task_values,
                rank_id=0,
                kernel_multi_wave=True,
            )
            return graph, runtime_config

        control_graph, control_config = _build(False)
        candidate_graph, candidate_config = _build(True)

        self.assertFalse(
            control_graph.get_op("dispatch").fill_config.destination_wave_fence
        )
        self.assertTrue(
            candidate_graph.get_op("dispatch").fill_config.destination_wave_fence
        )
        self.assertEqual(
            control_config.ready_handshake_meta.version,
            READY_HANDSHAKE_COPY_ACK_VERSION,
        )
        self.assertEqual(
            candidate_config.ready_handshake_meta.version,
            READY_HANDSHAKE_DESTINATION_FENCE_VERSION,
        )
        self.assertEqual(candidate_config.task_num, control_config.task_num)
        self.assertEqual(
            list(candidate_config.task_index_num),
            list(control_config.task_index_num),
        )
        dispatch = candidate_graph.get_op("dispatch")
        candidate_tasks = candidate_config.all_tasks[: candidate_config.task_num]
        candidate_dispatch_ids = [
            task_id
            for task_id in candidate_config.vector_task_indices[
                : candidate_config.task_index_num[1]
            ]
            if (
                candidate_tasks[task_id].task_type
                == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
                and candidate_tasks[task_id].extra_value_3 & 0xFF == 1
            )
        ]
        tasks_per_destination = dispatch.task_num // 2
        destination_order = [
            candidate_tasks[task_id].task_index // tasks_per_destination
            for task_id in candidate_dispatch_ids
        ]
        self.assertEqual(destination_order, [0, 1] * (dispatch.task_num // 2))
        dispatch_hidden = candidate_graph.get_op("dispatch_hidden")
        candidate_hidden_ids = [
            task_id
            for task_id in candidate_config.vector_task_indices[
                : candidate_config.task_index_num[1]
            ]
            if (
                candidate_tasks[task_id].task_type
                == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
                and candidate_tasks[task_id].extra_value_3 & 0xFF == 3
            )
        ]
        hidden_tasks_per_destination = dispatch_hidden.task_num // 2
        hidden_destination_order = [
            candidate_tasks[task_id].task_index // hidden_tasks_per_destination
            for task_id in candidate_hidden_ids
        ]
        self.assertEqual(
            hidden_destination_order,
            [0, 1] * (dispatch_hidden.task_num // 2),
        )
        combine = candidate_graph.get_op("combine")
        candidate_combine_ids = [
            task_id
            for task_id in candidate_config.vector_task_indices[
                : candidate_config.task_index_num[1]
            ]
            if (
                candidate_tasks[task_id].task_type
                == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
                and candidate_tasks[task_id].extra_value_3 & 0xFF == 2
            )
        ]
        combine_tasks_per_destination = combine.task_num // 2
        combine_destination_order = [
            candidate_tasks[task_id].task_index // combine_tasks_per_destination
            for task_id in candidate_combine_ids
        ]
        self.assertEqual(
            combine_destination_order,
            [0, 1] * (combine.task_num // 2),
        )

    def test_retained_weight_grad_scratch_marks_producer_and_accumulator(self):
        """
        Feature: retained up-projection scratch reuse
        Description: serialize the opt-in address selector on both task types
        Expectation: only dW GMMs and their accumulators carry the internal flag
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )
        graph = build_multi_wave_backward_graph(
            task_values,
            integrated_permute=True,
            inplace_wave_backward=True,
            tile_phase_ack=True,
            reuse_retained_dw_scratch=True,
        )
        graph.propagate_splits(task_values)
        runtime_config = build_backward_config(
            graph,
            task_values,
            rank_id=0,
            kernel_multi_wave=True,
        )
        tasks = runtime_config.all_tasks[: runtime_config.task_num]
        weight_grad_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_GROUPED_MATMUL
            and task.outputs[0].input_position in {6, 18}
        ]
        activation_gmm_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_GROUPED_MATMUL
            and task.outputs[0].input_position not in {6, 18}
        ]
        accumulate_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_LOCAL_ACCUMULATE
        ]

        self.assertTrue(weight_grad_tasks)
        self.assertTrue(activation_gmm_tasks)
        self.assertTrue(accumulate_tasks)
        self.assertTrue(
            all(
                task.extra_value_0 == RETAINED_DW_SCRATCH_FLAG
                for task in weight_grad_tasks + accumulate_tasks
            )
        )
        self.assertTrue(
            all(task.extra_value_0 == 0 for task in activation_gmm_tasks)
        )

    def test_atomic_weight_grad_marks_producer_and_accumulator(self):
        """
        Feature: atomic multi-wave weight-gradient accumulation
        Description: serialize direct-add mode on dW GMMs
        Expectation: mark matching producers and their completion relays
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )
        graph = build_multi_wave_backward_graph(
            task_values,
            integrated_permute=True,
            inplace_wave_backward=True,
            tile_phase_ack=True,
            atomic_dw_accumulation=True,
        )
        graph.propagate_splits(task_values)
        runtime_config = build_backward_config(
            graph,
            task_values,
            rank_id=0,
            kernel_multi_wave=True,
        )
        tasks = runtime_config.all_tasks[: runtime_config.task_num]
        weight_grad_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_GROUPED_MATMUL
            and task.outputs[0].input_position in {6, 18}
        ]
        activation_gmm_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_GROUPED_MATMUL
            and task.outputs[0].input_position not in {6, 18}
        ]
        accumulate_tasks = [
            task
            for task in tasks
            if task.task_type == TaskType.TASK_LOCAL_ACCUMULATE
        ]

        self.assertTrue(weight_grad_tasks)
        self.assertTrue(activation_gmm_tasks)
        self.assertTrue(accumulate_tasks)
        self.assertTrue(
            all(
                task.extra_value_0 == ATOMIC_DW_ACCUMULATION_FLAG
                for task in weight_grad_tasks
            )
        )
        self.assertTrue(
            all(
                task.extra_value_0 == ATOMIC_DW_ACCUMULATION_FLAG
                for task in accumulate_tasks
            )
        )
        self.assertTrue(
            all(task.extra_value_0 == 0 for task in activation_gmm_tasks)
        )

    def test_weight_grad_accumulation_modes_are_mutually_exclusive(self):
        """
        Feature: multi-wave weight-gradient accumulation selection
        Description: request retained scratch and direct atomic add together
        Expectation: reject the ambiguous device address contract
        """
        task_values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_multi_wave_backward_graph(
                task_values,
                integrated_permute=True,
                inplace_wave_backward=True,
                tile_phase_ack=True,
                reuse_retained_dw_scratch=True,
                atomic_dw_accumulation=True,
            )
