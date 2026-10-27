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
"""Route fusion contracts independent of device availability."""

import itertools
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import fusion, route
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts, _prepare_diagnostics
from hyper_parallel.core.multicore.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec, _resolve_fusion_modes
from hyper_parallel.core.multicore.scheduler.config import TaskType
from tests.common.mark_utils import arg_mark


class TestMegaMoeFusion(unittest.TestCase):
    """Check mapping storage, graph events, and owned gradients."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mixed_endpoints_share_the_route_diagnostic_abi(self):
        """Feature: MegaMoe route diagnostics.
        Description: Keep one legacy endpoint when the other direction uses fusion.
        Expectation: Both directions bind the route ABI for the shared autograd bridge.
        """
        for forward, backward in itertools.product(("legacy", "both"), repeat=2):
            with self.subTest(forward=forward, backward=backward):
                spec = self._spec(forward_fusion_mode=forward, backward_fusion_mode=backward)
                graphs = _build_runtime_artifacts(spec)
                with patch("hyper_parallel.core.multicore.modules.mega_moe.plan.prepare_mega_kernel_diagnostics") as bind:
                    _prepare_diagnostics(spec, *graphs, "cpu")
                counts = [call.kwargs["parameter_count"] for call in bind.call_args_list]
                self.assertEqual(counts, [16, 20] if forward == backward == "legacy" else [27, 33])

    @staticmethod
    def _spec(**options):
        values = {"local_num_tokens": 128, "hidden_size": 512, "intermediate_size": 128,
                  "num_experts": 8, "top_k": 2, "initial_capacity_factor": 1.25, "receive_capacity": 384,
                  "ep_size": 2, "ep_group": None, "rank_id": 0, "num_cube_cores": 24}
        return MegaMoeSpec(**{**values, **options})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_endpoint_defaults_and_explicit_unsupported_modes(self):
        """Feature: MegaMoe.
        Description: Resolve switches before allocation and reject unsupported explicit fusion.
        Expectation: endpoint defaults and explicit unsupported modes.
        """
        specification = {"local_num_tokens": 128, "top_k": 2}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_resolve_fusion_modes(specification), ("permute", "legacy"))
            for unsupported in ({"dispatch_mode": "pull"}, {"max_local_num_tokens": 128}, {"top_k": 17}):
                configured = {**specification, **unsupported}
                self.assertEqual(_resolve_fusion_modes(configured), ("legacy", "legacy"))
                with self.assertRaisesRegex(ValueError, "static push"):
                    _resolve_fusion_modes({**configured, "forward_fusion_mode": "permute"})
            with patch.dict(os.environ, {"HYPER_PARALLEL_MEGA_MOE_FUSION_MODE": "legacy"}):
                self.assertEqual(_resolve_fusion_modes(specification), ("legacy", "legacy"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_all_endpoint_combinations_preserve_wire_and_dependencies(self):
        """Feature: MegaMoe.
        Description: Serialize all sixteen graph combinations without task-ID or event collisions.
        Expectation: all endpoint combinations preserve wire and dependencies.
        """
        self.assertEqual(len(set(TaskType)), len(TaskType.__members__))
        modes = ("legacy", "permute", "unpermute", "both")
        for forward, backward in itertools.product(modes, repeat=2):
            with self.subTest(forward=forward, backward=backward):
                spec = self._spec(forward_fusion_mode=forward, backward_fusion_mode=backward)
                fg, fc, bg, bc = _build_runtime_artifacts(spec)
                if spec.forward_permute_fused:
                    self.assertEqual(fg.get_op("dispatch").inputs[1].shape, [128, 512])
                for graph, config, integrated in ((fg, fc, spec.forward_permute_fused),
                                                  (bg, bc, spec.backward_permute_fused)):
                    dispatch = graph.get_op("dispatch")
                    descriptor = next(task for task in config.all_tasks[:config.task_num]
                                      if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL)
                    self.assertEqual(descriptor.inputs[1].dim[3], spec.top_k if integrated else 0)
                    for index in range(config.task_num):
                        task = config.all_tasks[index]
                        self.assertLess(task.trigger_event, config.event_capacity)
                    self.assertGreater(dispatch.task_num, 0)
                self.assertTrue(can_reuse_backward_dispatch(bc, spec.local_experts, spec.num_cube_cores))
                if spec.backward_permute_fused or spec.backward_unpermute_fused:
                    self.assertEqual(bg.get_op("combine").param_positions[-1], 32)
                if spec.backward_unpermute_fused:
                    self.assertTrue(bg.has_op("unpermute_grad"))
                    self.assertEqual(bg.get_op("unpermute_grad").param_positions[0], 32)
                if spec.forward_unpermute_fused:
                    self.assertTrue(fg.has_op("weighted_unpermute"))
                    event = fg.get_op("combine").fill_config.route_completion_event
                    self.assertIsNotNone(event)
                    self.assertEqual(fc.all_event_num_triggers[event], fg.get_op("combine").task_num)
                    reductions = [task for task in fc.all_tasks[:fc.task_num]
                                  if task.task_type == TaskType.TASK_WEIGHTED_UNPERMUTE]
                    self.assertTrue(reductions)
                    self.assertTrue(all(task.inputs[3].dim[3] == event for task in reductions))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_prepare_permute_sorts_only_scalar_metadata(self):
        """Feature: MegaMoe.
        Description: Prohibit passing a token-width tensor to the stand-alone permute operator.
        Expectation: prepare permute sorts only scalar metadata.
        """
        spec = self._spec(ep_size=1, forward_fusion_mode="permute")
        hidden = torch.randn(128, 512, dtype=torch.bfloat16)
        ids = torch.arange(256, dtype=torch.int32).reshape(128, 2) % 8
        weights = torch.full((128, 2), 0.5)
        slots = torch.arange(256, dtype=torch.float32).reshape(-1, 1)

        def _scalar_permute(values: torch.Tensor, experts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Model the scalar permutation without hardware."""
            self.assertEqual(values.shape, (256, 1))
            order = torch.argsort(experts.reshape(-1), stable=True)
            inverse = torch.argsort(order)
            return values[order], inverse.to(torch.int32)

        with patch.object(route.torch_npu, "npu_moe_token_permute", side_effect=_scalar_permute) as permutation:
            prepared = route.prepare_topk_route(hidden, ids, weights, spec, None, route_slots=slots)
        self.assertIsNone(prepared.routed_tokens)
        self.assertEqual(permutation.call_count, 1)
        self.assertEqual(prepared.permute_src_index.numel(), ids.numel())
        expected = hidden[prepared.permute_src_index.long() // spec.top_k]
        restored = expected[prepared.unpermute_mapping.reshape(-1).long()].reshape(128, 2, 512)
        torch.testing.assert_close(restored[:, 0], hidden)
        torch.testing.assert_close(restored[:, 1], hidden)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backward_router_weights_are_owned(self):
        """Feature: MegaMoe.
        Description: Native dTopK output must never overwrite saved forward probabilities.
        Expectation: backward router weights are owned.
        """
        spec = self._spec(forward_fusion_mode="permute", backward_fusion_mode="both")
        probabilities = torch.rand(128, 2, requires_grad=True)
        expected = probabilities.detach().clone()
        saved = SimpleNamespace(permute_src_index=torch.arange(256, dtype=torch.int32),
                                topk_weights=probabilities)
        _, gradient = fusion._prepare_fused_backward_route(SimpleNamespace(spec=spec), saved)
        gradient.zero_()
        torch.testing.assert_close(probabilities, expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_endpoint_does_not_shift_backward_events(self):
        """Feature: MegaMoe route fusion.
        Description: Switch forward fusion while keeping backward legacy.
        Expectation: Backward runtime bytes remain identical across forward modes.
        """
        configs = []
        for forward in ("legacy", "permute", "unpermute", "both"):
            spec = self._spec(forward_fusion_mode=forward, backward_fusion_mode="legacy")
            configs.append(bytes(_build_runtime_artifacts(spec)[3]))
        self.assertTrue(all(config == configs[0] for config in configs))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backward_combine_covers_source_rows_on_a_light_receiver(self):
        """Feature: MegaMoe route fusion.
        Description: A rank receives fewer rows than it sends in Top-K dispatch.
        Expectation: Backward Combine retains the complete source-row extent.
        """
        spec = self._spec(forward_fusion_mode="permute", backward_fusion_mode="both")
        workspace = SimpleNamespace(expert_buffer=torch.empty(384, 512), routed_buffer=torch.empty(256, 512))
        saved = SimpleNamespace(dispatch=torch.empty(240, 512), expert_output=torch.empty(256, 512),
                                permute_src_index=torch.arange(256, dtype=torch.int32),
                                topk_weights=torch.ones(128, 2))
        ctx = SimpleNamespace(plan=SimpleNamespace(spec=spec), workspace=workspace)
        with patch.object(torch._C._autograd, "_get_current_graph_task_keep_graph", return_value=True):
            state = fusion._prepare_backward_route(ctx, torch.empty(128, 512), saved)
        self.assertEqual(state.dispatch.shape, (256, 512))
        self.assertNotEqual(state.grad_x.data_ptr(), saved.expert_output.data_ptr())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fused_input_gradient_has_independent_storage(self):
        """Feature: MegaMoe.
        Description: A following layer may reuse communication memory without changing dX.
        Expectation: fused input gradient has independent storage.
        """
        spec = self._spec(backward_fusion_mode="both")
        shared = torch.randn(256, 512, dtype=torch.bfloat16)
        route_state = SimpleNamespace(route_weight=torch.rand(128, 2), grad_topk_weights=None, grad_x=shared)
        buffers = SimpleNamespace(grad_weight1=torch.ones(1), grad_weight2=torch.ones(1))
        saved = SimpleNamespace(topk_weights=torch.ones(128, 2))
        ctx = SimpleNamespace(plan=SimpleNamespace(spec=spec), topk_dtype=torch.float32)
        gradients = fusion._finish_backward(ctx, torch.ones(128, 512), saved, route_state, buffers)
        expected = shared[:128].clone()
        shared.zero_()
        torch.testing.assert_close(gradients[0], expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fused_dispatch_reads_contiguous_original_tokens(self):
        """Feature: MegaMoe route fusion.
        Description: Pass a strided activation view to the fused dispatch launch.
        Expectation: Both native source ports share the correctly packed token tensor.
        """
        spec = self._spec(forward_fusion_mode="permute")
        hidden = torch.randn(128, 1024, dtype=torch.bfloat16)[:, ::2]
        placeholder = torch.ones(1)
        plan = SimpleNamespace(spec=spec, fwd_runtime_config=placeholder, fwd_diagnostics=None,
                               up_proj_tiling=placeholder, swiglu_tiling=placeholder, down_proj_tiling=placeholder)
        metadata = SimpleNamespace(**{name: placeholder for name in (
            "dispatch_target_off", "dispatch_src_off", "dispatch_size", "group_list",
            "combine_target_off", "combine_src_off", "combine_size")})
        buffers = SimpleNamespace(**{name: placeholder for name in (
            "dispatch", "up_proj", "activation", "down_proj", "combine", "gmm_workspace", "events",
            "permute_src_index", "final_out")})
        prepared = SimpleNamespace(unpermute_mapping=placeholder)
        with patch.object(fusion, "record_plan_stream"), patch.object(
                fusion.multicore_ops, "launch_route_fused") as launch:
            fusion._launch_forward(plan, metadata, hidden, hidden, placeholder, placeholder,
                                   prepared, placeholder, buffers)
        native = launch.call_args.args[0]
        self.assertFalse(hidden.is_contiguous())
        self.assertTrue(native[22].is_contiguous())
        self.assertEqual(native[22].data_ptr(), native[2].data_ptr())
        torch.testing.assert_close(native[22], hidden)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backward_honors_schedule_approved_receive_reuse(self):
        """Feature: MegaMoe route fusion.
        Description: Select scratch storage after the planner checks all receive readers.
        Expectation: Only an approved schedule aliases gate dX with the received gradient.
        """
        spec = self._spec(forward_fusion_mode="permute", backward_fusion_mode="legacy")
        saved = SimpleNamespace(dispatch=torch.empty(128, 512),
                                weight1=torch.ones(4, 512, 256), weight2=torch.ones(4, 128, 512))
        received = torch.empty(384, 512)
        state = SimpleNamespace(dispatch=received, kernel_grad_output=torch.empty(256, 512))
        workspace = SimpleNamespace(gmm_workspace=torch.empty(1), swiglu_grad_workspace=torch.empty(1),
                                    prepare_event_counters=Mock(return_value=torch.zeros(1)))
        for approved in (False, True):
            with self.subTest(approved=approved):
                plan = SimpleNamespace(spec=spec, reuse_backward_dispatch=approved)
                buffers = fusion._prepare_backward_buffers(plan, workspace, saved, state)
                self.assertEqual(buffers.gate_dx.shape, saved.dispatch.shape)
                self.assertEqual(buffers.gate_dx.data_ptr() == received.data_ptr(), approved)
                self.assertEqual(torch.count_nonzero(buffers.grad_weight1).item(), 0)
                self.assertEqual(torch.count_nonzero(buffers.grad_weight2).item(), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_combine_return_port_uses_registered_symmetric_storage(self):
        """Feature: MegaMoe route fusion.
        Description: Map the independent return buffer through the bounded native ABI.
        Expectation: All route operands survive without exceeding 32 native inputs.
        """
        parameters = tuple(torch.empty(1) for _ in range(32)) + (torch.empty(256, 512, dtype=torch.bfloat16),)
        native = fusion.multicore_ops._route_native_parameters(parameters, forward=False)
        self.assertEqual(len(native), 31)
        self.assertEqual(native[25].dtype, torch.uint8)
        self.assertEqual(native[25].data_ptr(), parameters[32].data_ptr())
        self.assertIs(native[26], parameters[27])
        self.assertIs(native[27], parameters[28])
        self.assertIs(native[30], parameters[31])
