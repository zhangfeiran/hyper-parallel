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
"""Check indexed ingress row identity, backward isolation, and allocation ownership."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import route as route_module
from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_ingress_config, _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.modules.mega_moe.workspace import _workspace_allocation_specs
from hyper_parallel.core.multicore.scheduler.config import TaskType


class TestIndexedIngress(unittest.TestCase):
    """Keep the prototype confined to forward transport and scalar route state."""

    def test_route_uses_scalar_sort_and_preserves_owned_mapping(self) -> None:
        """Skew and empty experts retain exact sorted slots without any token permutation."""
        ids = torch.tensor([[3, 0], [0, 2], [3, 2]], dtype=torch.int32)
        hidden = torch.randn(3, 16)
        counts = torch.tensor([2, 0, 2, 2], dtype=torch.int32)
        spec = MegaMoeSpec(3, 16, 8, 4, 2, None, 128, 1, None, 0, 32,
                           dispatch_mode="pull", forward_ingress_mode="indexed")
        workspace = SimpleNamespace(in_use=True, source_buffer=torch.empty(6, 16))
        order = torch.tensor([1, 2, 3, 5, 0, 4], dtype=torch.int32)
        inverse = torch.tensor([4, 0, 1, 2, 5, 3], dtype=torch.int32)

        def scalar_permute(values: torch.Tensor, indices: torch.Tensor) -> tuple:
            """Emulate only the scalar-sort contract and reject expanded payloads."""
            self.assertEqual(tuple(values.shape), (6, 1), f"Expanded payload shape: {values.shape}")
            self.assertEqual(indices.flatten().tolist(), ids.flatten().tolist())
            return values[order.long()], inverse

        with (patch.object(route_module.torch_npu, "npu_moe_token_permute", side_effect=scalar_permute),
              patch.object(route_module, "_permute_topk_input_out") as expanded):
            route = route_module.prepare_topk_route(hidden, ids, torch.ones(3, 2), spec, counts, workspace)
        expanded.assert_not_called()
        self.assertIsNone(route.routed_tokens, "Indexed ingress must not materialize token vectors")
        self.assertIs(route.tokens_per_expert, counts, "Final router counts should remain reusable")
        torch.testing.assert_close(route.permute_src_index, order)
        torch.testing.assert_close(route.unpermute_mapping, inverse)
        self.assertEqual(route.metadata.dispatch_size.tolist(), [2, 0, 2, 2])

    def test_forward_patch_preserves_backward_bytes_and_ready_dependencies(self) -> None:
        """The backward schedule and every forward event dependency remain unchanged."""
        legacy = MegaMoeSpec(512, 5120, 1792, 32, 8, None, 4096, 4, None, 0, 32, dispatch_mode="pull")
        old = _build_runtime_artifacts(legacy)
        new = _build_runtime_artifacts(replace(legacy, forward_ingress_mode="indexed"))
        self.assertEqual(bytes(old[3]), bytes(new[3]), "Indexed mode changed backward task bytes")
        changed_gets = 0
        for before, after in zip(old[1].all_tasks[:old[1].task_num], new[1].all_tasks[:new[1].task_num]):
            self.assertEqual((before.dependent_event, before.trigger_event, before.extra_value_2),
                             (after.dependent_event, after.trigger_event, after.extra_value_2))
            if after.task_type == TaskType.TASK_SHMEM_GET_MEM:
                changed_gets += 1
                self.assertEqual((after.extra_value_0, after.extra_value_1, after.inputs[1].dim[3]), (16, 1, 8))
            elif after.task_type == TaskType.TASK_GROUPED_MATMUL:
                self.assertEqual(after.extra_value_1, 2, "Port 16 requires Blaze F2")
        self.assertGreater(changed_gets, 0, "Indexed ingress did not reach any transport task")

    def test_only_indexed_mode_reserves_symmetric_metadata(self) -> None:
        """Retain legacy heap sizing and give indexed state a tracked restart allocation."""
        kwargs = dict(dispatch_mode="pull", routed_slots=4096, receive_capacity=4096,
                      hidden_size=5120, ep_size=4, num_experts=32, dtype=torch.bfloat16)
        old = _workspace_allocation_specs(**kwargs)
        new = _workspace_allocation_specs(**kwargs, forward_ingress_mode="indexed")
        self.assertEqual(new[:-1], old, "Legacy symmetric allocation order changed")
        self.assertEqual((new[-1][0], new[-1][1].shape, new[-1][1].dtype),
                         ("route_metadata", (12288, 4), torch.int32))

    def test_unsupported_modes_fail_before_parameter_allocation(self) -> None:
        """Reject push, variable tokens, and inexact scalar indices without NPU work."""
        for kwargs in (dict(dispatch_mode="push", local_num_tokens=128),
                       dict(dispatch_mode="pull", max_local_num_tokens=128),
                       dict(dispatch_mode="pull", local_num_tokens=2**24)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
                               forward_ingress_mode="indexed", **kwargs)

    def test_peer_prepass_queues_only_independent_reads_and_keeps_outer_fences(self) -> None:
        """No per-expert dependency or unvisited terminate can block an owner fanout."""
        spec = MegaMoeSpec(512, 5120, 1792, 32, 8, None, 4096, 4, None, 0, 32,
                           dispatch_mode="pull", forward_ingress_mode="peer_dedup")
        artifacts = _build_runtime_artifacts(spec)
        full = artifacts[1]
        prepass = _build_ingress_config(full)
        indexed = _build_runtime_artifacts(replace(spec, forward_ingress_mode="indexed"))
        self.assertEqual(bytes(artifacts[3]), bytes(indexed[3]), "Peer dedup changed backward tasks")
        self.assertEqual(prepass.task_index_num[0], 0, "Prepass must not queue GMM")
        queued = [int(prepass.vector_task_indices[i]) for i in range(prepass.task_index_num[1])]
        expected = [i for i, task in enumerate(full.all_tasks[:full.task_num])
                    if task.task_type == TaskType.TASK_SHMEM_GET_MEM]
        self.assertEqual(set(queued), set(expected), "Prepass must queue every dispatch exactly once")
        self.assertEqual(len(queued), len(set(queued)), "Duplicate dispatch task")
        for index in queued:
            task = prepass.all_tasks[index]
            self.assertEqual((task.dependent_event, task.trigger_event, task.extra_value_1), (0xFFFFFFFF, 0, 2))
            self.assertEqual(full.all_tasks[index].extra_value_1, 3, "Main GET should be a no-op")
        self.assertEqual((prepass.ready_event, prepass.completion_event, prepass.protocol_version),
                         (full.ready_event, full.completion_event, full.protocol_version))
        self.assertGreater(prepass.ready_event, 8, "Dummy credit must not overlap source readiness")
        self.assertNotIn(full.task_num, queued, "Prepass must not wait on the old Combine terminate")

    def test_peer_ordinals_reconstruct_exact_expert_rows_with_one_owner(self) -> None:
        """Exercise token groups selecting one and two destination ranks and an empty expert."""
        ids = torch.tensor([[3, 0, 2, 1], [6, 7, 0, 4], [2, 0, 7, 1]], dtype=torch.int32)
        slots = torch.argsort(ids.flatten(), stable=True).int()
        counts = torch.bincount(ids.flatten().long(), minlength=8).int()
        ordinals = route_module._peer_row_ordinals(ids, slots, counts)
        values = torch.arange(3 * 16).reshape(3, 16)
        received = [torch.empty((int(count), 16), dtype=values.dtype) for count in counts]
        writes = [torch.zeros(int(count), dtype=torch.int32) for count in counts]
        reads = []
        for slot in slots.tolist():
            token, selected = divmod(slot, 4)
            peer = int(ids[token, selected]) // 4
            selected_slots = [j for j in range(4) if int(ids[token, j]) // 4 == peer]
            if selected != selected_slots[0]:
                continue
            reads.append((token, peer))
            for j in selected_slots:
                expert = int(ids[token, j])
                row = int(ordinals[token * 4 + j])
                received[expert][row].copy_(values[token])
                writes[expert][row] += 1
        expected_reads = {(token, int(expert) // 4) for token in range(3) for expert in ids[token]}
        self.assertEqual(set(reads), expected_reads)
        self.assertEqual(len(reads), len(expected_reads), "Token-peer was fetched more than once")
        for expert in range(8):
            expected_slots = [slot for slot in slots.tolist() if int(ids.flatten()[slot]) == expert]
            torch.testing.assert_close(received[expert], values[torch.tensor(expected_slots).long() // 4])
            torch.testing.assert_close(writes[expert], torch.ones_like(writes[expert]))
