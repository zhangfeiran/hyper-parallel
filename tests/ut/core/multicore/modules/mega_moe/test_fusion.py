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
from hyper_parallel.core.multicore.modules.mega_moe import workspace as workspace_module
from hyper_parallel.core.multicore.modules.mega_moe.workspace import _pull_workspace_options, _workspace_allocation_specs
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
            self.assertEqual(_resolve_fusion_modes({**specification, "dispatch_mode": "pull"}), ("legacy", "legacy"))
            self.assertEqual(_resolve_fusion_modes({**specification, "dispatch_mode": "pull",
                                                   "forward_fusion_mode": "both"}), ("both", "legacy"))
            for unsupported in ({"max_local_num_tokens": 128}, {"top_k": 17}):
                configured = {**specification, **unsupported}
                self.assertEqual(_resolve_fusion_modes(configured), ("legacy", "legacy"))
                with self.assertRaisesRegex(ValueError, "static tokens"):
                    _resolve_fusion_modes({**configured, "forward_fusion_mode": "permute"})
            with patch.dict(os.environ, {"HYPER_PARALLEL_MEGA_MOE_FUSION_MODE": "legacy"}):
                self.assertEqual(_resolve_fusion_modes(specification), ("legacy", "legacy"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pull_publication_keeps_saved_indices_owned(self):
        """Feature: Indexed pull route publication.
        Description: Publish sorted indices and FP32 weight bits, then reuse the publication buffer.
        Expectation: Metadata matches the route and saved tensors remain independent.
        """
        indices = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
        weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        workspace = SimpleNamespace(route_metadata=torch.empty((4, 2), dtype=torch.int32))
        published = fusion._publish_pull_route(workspace, indices, weights)
        self.assertEqual(published[:, 0].tolist(), indices.tolist())
        self.assertTrue(torch.equal(published[:, 1].contiguous().view(torch.float32),
                                    weights.flatten()[indices]))
        published.zero_()
        self.assertEqual(indices.tolist(), [2, 0, 3, 1])
        self.assertTrue(torch.equal(weights, torch.tensor([[0.1, 0.2], [0.3, 0.4]])))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_indexed_pull_metadata_survives_restart_and_is_freed(self) -> None:
        """Feature: Indexed pull workspace lifecycle.
        Description: Allocate, validate and replace every symmetric tensor, including routing metadata.
        Expectation: The original-row source and metadata use the same restart and free protocol.
        """
        spec = SimpleNamespace(receive_capacity=256, routed_slots=256, hidden_size=16,
                               ep_size=2, num_experts=4, dispatch_mode="pull", local_num_tokens=128,
                               forward_fusion_mode="permute", backward_fusion_mode="both")
        workspace = workspace_module.MegaMoeWorkspace(shared=True)
        with (
            patch.object(workspace_module.shmem, "empty", side_effect=lambda shape, **kw: torch.empty(
                shape, dtype=kw["dtype"],
            )) as allocate,
            patch.object(workspace_module.torch.npu, "Event"),
        ):
            workspace.ensure(spec, torch.bfloat16, torch.device("cpu"))
        self.assertEqual(allocate.call_count, 5)
        self.assertEqual(workspace.source_buffer.shape, (128, 16))
        self.assertEqual(workspace.route_metadata.shape, (256, 2))
        replacements = workspace.describe_restart_allocations(spec, torch.bfloat16, 256)
        self.assertEqual(len(replacements), 5)
        self.assertTrue(workspace.validate_restart_allocations(spec, torch.bfloat16, 256, replacements))
        tensors = tuple(torch.empty_like(replacement.current) for replacement in replacements)
        workspace.validate_restarted_allocations(spec, torch.bfloat16, 256, tensors)
        workspace.bind_restarted_allocations(spec, torch.bfloat16, torch.device("cpu"), 256, tensors)
        self.assertIs(workspace.route_metadata, tensors[-1])
        with patch.object(workspace_module.shmem, "free") as free:
            workspace._free_symmetric_tensors()  # pylint: disable=protected-access
        self.assertEqual(free.call_count, 5)
        self.assertIsNone(workspace.route_metadata)
        self.assertIsNone(workspace.source_buffer)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pull_reserves_original_rows_only_when_both_dispatches_are_fused(self):
        """Feature: Indexed pull workspace sizing.
        Description: Check the source capacity for all forward and backward mode combinations.
        Expectation: Both fused dispatches reserve T rows; either legacy dispatch reserves T*K.
        """
        for forward, backward in itertools.product(("legacy", "permute", "unpermute", "both"), repeat=2):
            spec = self._spec(dispatch_mode="pull", forward_fusion_mode=forward, backward_fusion_mode=backward)
            allocations = dict(_workspace_allocation_specs(
                dispatch_mode="pull", routed_slots=spec.routed_slots, receive_capacity=spec.receive_capacity,
                hidden_size=spec.hidden_size, ep_size=spec.ep_size, num_experts=spec.num_experts,
                dtype=torch.bfloat16, **_pull_workspace_options(spec)))
            rows = spec.local_num_tokens if spec.forward_permute_fused and spec.backward_permute_fused else spec.routed_slots
            self.assertEqual(allocations["source_buffer"].shape, (rows, spec.hidden_size))
            self.assertEqual(allocations["routed_buffer"].shape, (spec.routed_slots, spec.hidden_size))
            self.assertEqual("route_metadata" in allocations, spec.forward_permute_fused or spec.backward_permute_fused)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_all_endpoint_combinations_preserve_wire_and_dependencies(self):
        """Feature: MegaMoe.
        Description: Serialize all sixteen graph combinations without task-ID or event collisions.
        Expectation: all endpoint combinations preserve wire and dependencies.
        """
        self.assertEqual(len(set(TaskType)), len(TaskType.__members__))
        modes = ("legacy", "permute", "unpermute", "both")
        for transport, layout, forward, backward in itertools.product(("push", "pull"), ("EIO", "EOI"), modes, modes):
            with self.subTest(transport=transport, layout=layout, forward=forward, backward=backward):
                spec = self._spec(dispatch_mode=transport, expert_weight_layout=layout,
                                  forward_fusion_mode=forward, backward_fusion_mode=backward)
                fg, fc, bg, bc = _build_runtime_artifacts(spec)
                if spec.forward_permute_fused:
                    self.assertEqual(fg.get_op("dispatch").inputs[1].shape, [128, 512])
                for graph, config, integrated in ((fg, fc, spec.forward_permute_fused),
                                                  (bg, bc, spec.backward_permute_fused)):
                    dispatch = graph.get_op("dispatch")
                    task_type = {"pull": TaskType.TASK_SHMEM_GET_MEM,
                                 "push": TaskType.TASK_SHMEM_PUT_MEM_SIGNAL}[transport]
                    descriptor = next(task for task in config.all_tasks[:config.task_num]
                                      if task.task_type == task_type)
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

        permutation = Mock(side_effect=_scalar_permute)
        with patch.object(route, "load_torch_npu", return_value=SimpleNamespace(npu_moe_token_permute=permutation)):
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
    def test_expert_snapshot_does_not_pin_surplus_receive_storage(self):
        """Feature: Expert-output snapshot lifetime.
        Description: Retain a routed output when a receiver owns more rows than the sender routes.
        Expectation: Fused backward saves exactly the routed rows; no-grad does not copy them.
        """
        cases = (
            (True, True, True, 512, True),
            (True, True, True, 256, False),
            (True, True, False, 512, False),
            (True, False, True, 512, False),
            (False, True, True, 512, True),
            (False, False, True, 256, True),
            (False, True, False, 512, False),
        )
        for fused_forward, fused_backward, save, capacity, owned in cases:
            with self.subTest(forward=fused_forward, backward=fused_backward, save=save, capacity=capacity):
                spec = SimpleNamespace(forward_unpermute_fused=fused_forward,
                                       backward_permute_fused=fused_backward, routed_slots=256)
                buffers = SimpleNamespace(down_proj=torch.randn(capacity, 16), combine=torch.randn(256, 16))
                output = fusion._forward_expert_output(spec, buffers, save)
                source = buffers.down_proj[:256] if fused_forward else buffers.combine
                torch.testing.assert_close(output, source)
                if owned:
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())
                    self.assertEqual(output.untyped_storage().nbytes(), output.numel() * output.element_size())
                    source.zero_()
                    self.assertGreater(output.abs().sum().item(), 0)
                else:
                    self.assertEqual(output.data_ptr(), source.data_ptr())

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
