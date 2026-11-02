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
"""Check real queue ownership before reducing the process-wide UDMA QP bank."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.modules.mega_moe import udma_order
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import TaskType
from hyper_parallel.core.multicore.scheduler.runtime import serialize_runtime_config
from hyper_parallel.core.multicore.shmem import _runtime


class TestUdmaQpOwnership(unittest.TestCase):
    """Use hardware-independent descriptors and the actual static queue builders."""

    def test_configuration_defaults_and_explicit_values(self) -> None:
        """Keep the existing bank size unless the hybrid consumer opts in."""
        for count in (16, 32):
            with self.subTest(count=count), patch.dict(os.environ, {
                'HYPER_PARALLEL_SHMEM_DATA_ENGINE': 'mte_udma',
                'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': str(count),
            }):
                self.assertEqual(_runtime.configured_udma_qp_count(), count)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_runtime.configured_udma_qp_count(), 32)

    def test_invalid_values_are_rejected(self) -> None:
        """Refuse malformed or unqualified pools before native initialization."""
        for value in ('', '0', '1', '8', '17', ' 16', '16 ', '+16', '16.0'):
            with self.subTest(value=value), patch.dict(os.environ, {
                'HYPER_PARALLEL_SHMEM_DATA_ENGINE': 'mte_udma',
                'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': value,
            }):
                with self.assertRaisesRegex(ValueError, '16 or 32'):
                    _runtime.configured_udma_qp_count()

    def test_reduced_pool_requires_hybrid_transport(self) -> None:
        """Avoid silently applying a UDMA resource choice to MTE-only execution."""
        with patch.dict(os.environ, {'HYPER_PARALLEL_SHMEM_DATA_ENGINE': 'mte',
                                     'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': '16'}):
            with self.assertRaisesRegex(ValueError, 'mte_udma'):
                _runtime.configured_udma_qp_count()

    def test_distinct_peers_can_reuse_a_qp_number(self) -> None:
        """Keep adjacent odd AIVs on distinct peer connections in the smaller bank."""
        tasks = [SimpleNamespace(task_type=TaskType.TASK_SHMEM_PUT_MEM_SIGNAL,
                                 task_index=peer, task_split_num=4) for peer in (1, 2)]
        config = SimpleNamespace(num_workers=64, task_index_num=[0, 2, 0],
                                 vector_task_indices=[0, 1], all_tasks=tasks)
        owners = udma_order.validate_udma_qp_owners(config, ep_size=4, num_experts=4, rank_id=0, qp_count=16)
        self.assertEqual(owners, {(1, 0): 1, (2, 0): 3})

    def test_same_peer_qp_with_two_workers_is_rejected(self) -> None:
        """Reject the hazardous modulo-style alias even when all tasks are otherwise valid."""
        tasks = [SimpleNamespace(task_type=TaskType.TASK_SHMEM_PUT_MEM_SIGNAL,
                                 task_index=1, task_split_num=4)] * 2
        config = SimpleNamespace(num_workers=64, task_index_num=[0, 2, 0],
                                 vector_task_indices=[0, 1], all_tasks=tasks)
        with self.assertRaisesRegex(ValueError, 'multiple.*owner'):
            udma_order.validate_udma_qp_owners(config, ep_size=4, num_experts=4, rank_id=0, qp_count=16)

    def test_self_and_non_put_tasks_do_not_claim_a_qp(self) -> None:
        """Only nonempty-potential remote Put tasks consume the UDMA connection bank."""
        tasks = [SimpleNamespace(task_type=TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, task_index=0, task_split_num=4),
                 SimpleNamespace(task_type=TaskType.TASK_TERMINATE)]
        config = SimpleNamespace(num_workers=64, task_index_num=[0, 2, 0],
                                 vector_task_indices=[0, 1], all_tasks=tasks)
        owners = udma_order.validate_udma_qp_owners(config, ep_size=4, num_experts=4, rank_id=0, qp_count=16)
        self.assertEqual(owners, {})

    def test_real_forward_backward_queues_are_preserved(self) -> None:
        """Cover EP2/4/8/64 and both dispatch modes without changing a descriptor byte."""
        for ep in (2, 4, 8, 64):
            for rank in (0, ep-1):
                for dispatch in ('push', 'pull'):
                    with self.subTest(ep=ep, rank=rank, dispatch=dispatch):
                        specification = MegaMoeSpec(
                            local_num_tokens=128, hidden_size=5120, intermediate_size=1792,
                            num_experts=ep*6, top_k=min(8, ep*6), ep_size=ep,
                            rank_id=rank, num_cube_cores=32, dispatch_mode=dispatch,
                            initial_capacity_factor=None if dispatch == 'pull' else 1.25,
                            receive_capacity=2048, ep_group=None,
                        )
                        with patch.dict(os.environ, {'HYPER_PARALLEL_SHMEM_DATA_ENGINE': 'mte_udma',
                                                     'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': '32'}):
                            _, forward, _, backward = _build_runtime_artifacts(specification)
                        with patch.dict(os.environ, {'HYPER_PARALLEL_SHMEM_DATA_ENGINE': 'mte_udma',
                                                     'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': '16'}):
                            _, reduced_forward, _, reduced_backward = _build_runtime_artifacts(specification)
                        self.assertEqual(serialize_runtime_config(forward), serialize_runtime_config(reduced_forward))
                        self.assertEqual(serialize_runtime_config(backward), serialize_runtime_config(reduced_backward))
                        for config in (forward, backward):
                            before = serialize_runtime_config(config)
                            owners = udma_order.validate_udma_qp_owners(
                                config, ep_size=ep, num_experts=ep*6, rank_id=rank, qp_count=16)
                            self.assertTrue(owners)
                            self.assertEqual(serialize_runtime_config(config), before)
                            self.assertTrue(all(0 <= qp < 16 for _, qp in owners))

    def test_actual_indexed_pull_plan_has_unique_owners(self) -> None:
        """Exercise the current TP1/seq4096/E32 ingress rather than only legacy plans."""
        specification = MegaMoeSpec(
            local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
            num_experts=32, top_k=8, initial_capacity_factor=None,
            receive_capacity=32768, ep_size=4, ep_group=None, rank_id=0,
            num_cube_cores=32, dispatch_mode='pull', forward_ingress_mode='indexed',
        )
        with patch.dict(os.environ, {'HYPER_PARALLEL_SHMEM_DATA_ENGINE': 'mte_udma',
                                     'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': '16'}):
            _, forward, _, backward = _build_runtime_artifacts(specification)
        for config in (forward, backward):
            owners = udma_order.validate_udma_qp_owners(
                config, ep_size=4, num_experts=32, rank_id=0, qp_count=16)
            self.assertEqual(len(owners), 24)

    def test_odd_ep_collision_is_rejected_by_actual_builder(self) -> None:
        """Reject unqualified queue geometries before any device task can submit."""
        specification = MegaMoeSpec(
            local_num_tokens=128, hidden_size=5120, intermediate_size=1792,
            num_experts=18, top_k=8, initial_capacity_factor=1.25,
            receive_capacity=2048, ep_size=3, ep_group=None, rank_id=0,
            num_cube_cores=32,
        )
        with patch.dict(os.environ, {'HYPER_PARALLEL_SHMEM_DATA_ENGINE': 'mte_udma',
                                     'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': '16'}):
            with self.assertRaisesRegex(ValueError, 'multiple.*owner'):
                _build_runtime_artifacts(specification)


if __name__ == '__main__':
    unittest.main()
