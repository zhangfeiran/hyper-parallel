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
"""Exercise the A5 scheduler and dynamic physical expert-replica contract together."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe.inactive_tasks import build_guest_task_prefixes
from hyper_parallel.core.multicore.modules.mega_moe import route as routing
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.modules.mega_moe.workspace import _workspace_allocation_specs
from hyper_parallel.core.multicore.scheduler.config import A5_GMM_KERNEL_TAG, TaskType


class TestA5ReplicaIntegration(unittest.TestCase):
    """Keep producer events, shared tasks and final physical routes independent."""

    @staticmethod
    def _spec() -> MegaMoeSpec:
        """Use the requested EP16 physical shape without device allocation."""
        return MegaMoeSpec(
            local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
            num_experts=192, logical_num_experts=96, replica_slots_per_rank=6,
            top_k=8, initial_capacity_factor=None, receive_capacity=32768,
            ep_size=16, ep_group=None, rank_id=0, num_cube_cores=32, dispatch_mode="pull",
            forward_ingress_mode="indexed", backward_ingress_mode="indexed",
            shared_intermediate_size=1792, forward_shared_expert=True, backward_shared_expert=True,
            dispatch_split=512, combine_split=512, auto_tokens=True,
        )

    def test_a5_selectors_preserve_fp32_producer_events_and_shared_work(self) -> None:
        """Changing the GEMM implementation must not overwrite gradient-return events."""
        spec = self._spec()
        artifacts = _build_runtime_artifacts(spec)
        for direction, config in enumerate((artifacts[1], artifacts[3])):
            tasks = config.all_tasks[:config.task_num]
            shared = [task for task in tasks if task.inputs[0].tensor_type == 2]
            self.assertTrue(shared)
            routed = [task for task in tasks if task.task_type == TaskType.TASK_GROUPED_MATMUL
                      and task.inputs[0].tensor_type != 2]
            self.assertTrue(routed)
            self.assertTrue(all(task.extra_value_2 == A5_GMM_KERNEL_TAG | (2 if direction == 0 else 3)
                                for task in routed))
            if direction == 1:
                producers = [task for task in routed if task.extra_value_1]
                self.assertTrue(producers)
                self.assertTrue(all(task.extra_value_1 > 3 for task in producers))
                self.assertTrue(all(task.outputs[0].data_type == 4 for task in producers))
            shared_ids = {index for index, task in enumerate(tasks) if task.inputs[0].tensor_type == 2}
            prefixes = build_guest_task_prefixes(config, home_experts=6, ep_size=16, dispatch_mode="pull")
            self.assertTrue(prefixes)
            for prefix in prefixes:
                for selected in (prefix, *prefix.communication_variants):
                    raw = bytearray(bytes(config))
                    for offset, payload in selected.patches:
                        raw[offset:offset+len(payload)] = payload
                    changed = type(config).from_buffer_copy(raw)
                    retained = set(changed.cube_task_indices[:changed.task_index_num[0]])
                    retained.update(changed.vector_task_indices[:changed.task_index_num[1]])
                    self.assertTrue(shared_ids.issubset(retained))

    def test_compact_indexed_source_retains_both_replica_and_route_storage(self) -> None:
        """The source-token buffer and all symmetric protocol buffers must be accounted together."""
        spec = self._spec()
        entries = dict(_workspace_allocation_specs(
            dispatch_mode="pull", routed_slots=spec.routed_slots, receive_capacity=spec.receive_capacity,
            hidden_size=spec.hidden_size, ep_size=spec.ep_size, num_experts=spec.num_experts,
            dtype=torch.bfloat16, replica_inbox_bytes=123456,
            forward_ingress_mode="indexed", backward_ingress_mode="indexed", top_k=spec.top_k,
            shared_expert=True,
        ))
        self.assertEqual(entries["source_buffer"].shape, (4096, 5120))
        self.assertEqual(entries["replica_inbox"].shape, (123456,))
        self.assertIn("route_metadata", entries)
        self.assertIn("backward_route_metadata", entries)

    def test_dynamic_indexed_path_uses_final_physical_ids_without_expanded_tokens(self) -> None:
        """A guest destination must reach scalar sorting and the exact physical metadata layout."""
        spec = replace(self._spec(), local_num_tokens=128, hidden_size=16, intermediate_size=8,
                       num_experts=6, logical_num_experts=4, replica_slots_per_rank=1, top_k=2,
                       ep_size=2, shared_intermediate_size=0, receive_capacity=256,
                       dispatch_split=128, combine_split=128)
        ids = torch.tensor([[5, 1]], dtype=torch.int32).expand(128, -1).contiguous()
        counts = torch.tensor([0, 128, 0, 0, 0, 128], dtype=torch.int32)
        replica = SimpleNamespace(
            dynamic_summary=[[0, 128, 31, 256, 0, 128, 256]] * 2,
            counts_by_source=torch.stack((counts, counts)), _host_counts=[counts.tolist()] * 2,
            home_only=False,
        )
        workspace = SimpleNamespace(in_use=True, source_buffer=torch.zeros(128, 16))
        order = torch.argsort(ids.reshape(-1), stable=True).to(torch.int32)
        inverse = torch.argsort(order).to(torch.int32)
        prepare = Mock()
        with (patch.object(routing, "_build_route_indices", return_value=(order, inverse)) as scalar,
              patch.object(routing, "_permute_dynamic_pull") as expanded):
            actual = routing.prepare_topk_route(
                torch.zeros(128, 16), ids, torch.ones(128, 2), spec, counts,
                workspace=workspace, replica_route=replica, autograd_mask=31, prepare_workspace=prepare,
            )
        scalar.assert_called_once()
        self.assertIs(scalar.call_args.args[0], ids)
        expanded.assert_not_called()
        self.assertIsNone(actual.routed_tokens)
        self.assertIs(actual.permute_src_index, order)
        self.assertIs(actual.tokens_per_expert, counts)
        self.assertIs(actual.metadata.replica_route, replica)
        self.assertEqual(actual.metadata.group_list.tolist(), [0, 256, 256])
        prepare.assert_called_once_with(128, 256)
