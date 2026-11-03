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
"""CPU contracts for shared tasks alongside the current Ascend950 routed plan."""

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.modules.mega_moe.shared_schedule import bind_shared_diagnostic_graph
from hyper_parallel.core.multicore.diagnostics.common.graph import validate_mega_kernel_diagnostic_contract
from hyper_parallel.core.multicore.scheduler.config import TaskType


class TestA5SharedFusion(unittest.TestCase):
    """Keep routed scheduling and local shared ownership independently verifiable."""

    def test_shared_width_2048_keeps_current_routed_task_contracts(self) -> None:
        """Strip local shared tasks and recover every original routed descriptor's semantics."""
        spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                           num_experts=32, top_k=8, initial_capacity_factor=None,
                           receive_capacity=32768, ep_size=4, ep_group=None, rank_id=0,
                           num_cube_cores=32, dispatch_mode="pull", forward_ingress_mode="indexed")
        with patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
                                     "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "32"}):
            baseline = _build_runtime_artifacts(spec)
            for placement in ("idle", "front"):
                with self.subTest(placement=placement):
                    fused = _build_runtime_artifacts(replace(spec, shared_intermediate_size=2048,
                                                           shared_placement=placement))
                    for graph, config, ports in ((fused[0], fused[1], 28), (fused[2], fused[3], 35)):
                        diagnostic = bind_shared_diagnostic_graph(graph, config)
                        validate_mega_kernel_diagnostic_contract(diagnostic, config, ports)
                    for graph, config, original in ((fused[0], fused[1], baseline[1]),
                                                     (fused[2], fused[3], baseline[3])):
                        shared = [task for task in config.all_tasks[:config.task_num]
                                  if task.inputs[0].tensor_type == 2]
                        self.assertTrue(shared)
                        self.assertTrue(any(op.name.startswith("shared_") for op in graph.topological_sort()))
                        for queue_name, index in (("cube_task_indices", 0), ("vector_task_indices", 1)):
                            actual = [config.all_tasks[i] for i in getattr(config, queue_name)[:config.task_index_num[index]]
                                      if config.all_tasks[i].inputs[0].tensor_type != 2]
                            expected = [original.all_tasks[i] for i in getattr(original, queue_name)[:original.task_index_num[index]]]
                            signature = lambda task: (task.task_type, task.task_index, task.task_split_num,
                                                      task.task_split_value, task.dependent_event, task.trigger_event,
                                                      task.extra_value_0, task.extra_value_1, task.extra_value_2)
                            self.assertEqual([signature(task) for task in actual], [signature(task) for task in expected])
                        routed_gmm = [task for task in config.all_tasks[:config.task_num]
                                      if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.inputs[0].tensor_type != 2]
                        self.assertTrue(all(task.extra_value_1 == 2 for task in routed_gmm)
                                        if graph is fused[0] else bool(routed_gmm))


if __name__ == "__main__":
    unittest.main()
