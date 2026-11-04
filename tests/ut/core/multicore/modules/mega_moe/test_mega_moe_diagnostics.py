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
"""MegaMoe-owned diagnostic names and expert resolution tests."""

import unittest
from dataclasses import replace

from hyper_parallel.core.multicore.diagnostics.common.graph import DiagnosticSpec, build_graph_metadata
from hyper_parallel.core.multicore.modules.mega_moe.diagnostics import _mega_moe_owner_options
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import build_forward_graph
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts, _build_task_values
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import TaskDescC, TaskSplitValue, TaskType
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config
from tests.common.mark_utils import arg_mark


class TestMegaMoeDiagnostics(unittest.TestCase):
    """Keep business-specific diagnostic rules outside generic tool tests."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mega_moe_uses_graph_names_and_expert_owners(self):
        """Feature: MegaKernel diagnostics.
        Description: Keep MegaMoe display semantics without a second task-matching table.
        Expectation: Mega moe uses graph names and expert owners.
        """
        topology = TaskSplitValue()
        topology.rank_id = 1
        graph = build_forward_graph(topology)
        for operator in graph.topological_sort():
            operator.task_num = 1
        runtime_config = allocate_runtime_config(16)
        runtime_config.task_num = 5

        dispatch = TaskDescC()
        dispatch.task_type = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
        dispatch.task_split_num = 64
        dispatch.task_index = 4
        runtime_config.all_tasks[0] = dispatch

        gmm1 = TaskDescC()
        gmm1.task_type = TaskType.TASK_GROUPED_MATMUL
        gmm1.task_index = 48
        runtime_config.all_tasks[1] = gmm1

        swiglu = TaskDescC()
        swiglu.task_type = TaskType.TASK_SWI_GLU
        swiglu.task_split_value = 128
        swiglu.task_index = 128
        runtime_config.all_tasks[2] = swiglu

        gmm2 = TaskDescC()
        gmm2.task_type = TaskType.TASK_GROUPED_MATMUL
        gmm2.task_index = 72
        runtime_config.all_tasks[3] = gmm2

        combine = TaskDescC()
        combine.task_type = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
        combine.task_split_num = 64
        combine.task_index = 4
        runtime_config.all_tasks[4] = combine

        options = _mega_moe_owner_options(topology, num_cube_cores=24)
        original = bytes(runtime_config)
        metadata = build_graph_metadata(graph, runtime_config, DiagnosticSpec("mega_moe", **options))
        self.assertEqual(bytes(runtime_config), original)
        self.assertEqual(
            metadata.task_stage_names,
            {0: "Dispatch", 1: "GMM1", 2: "SwiGLU", 3: "GMM2", 4: "Combine"},
        )
        self.assertEqual(metadata.task_owner_ids[0], 2)
        self.assertEqual(metadata.task_owner_ids[1], 10)
        self.assertEqual(metadata.task_owner_ids[2], 10)
        self.assertEqual(metadata.task_owner_ids[3], 11)
        self.assertEqual(metadata.task_owner_ids[4], 10)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_indexed_backward_metadata_keeps_token_gradients_without_expert_owner(self) -> None:
        """Feature: Indexed backward profiling metadata.
        Description: Build real backward metadata with and without shared experts.
        Expectation: Source-token route-gradient jobs have stage names without a single expert owner.
        """
        base = MegaMoeSpec(512, 5120, 1792, 32, 8, None, 4096, 4, None, 0, 32,
                           dispatch_mode="pull", forward_ingress_mode="indexed", backward_ingress_mode="indexed")
        for shared in (False, True):
            with self.subTest(shared=shared):
                spec = (replace(base, shared_intermediate_size=2048, forward_shared_expert=True,
                                backward_shared_expert=True, shared_placement="idle") if shared else base)
                _, _, graph, runtime = _build_runtime_artifacts(spec)
                original = bytes(runtime)
                options = _mega_moe_owner_options(_build_task_values(spec), num_cube_cores=32)

                metadata = build_graph_metadata(graph, runtime, DiagnosticSpec("mega_moe", **options))

                self.assertEqual(bytes(runtime), original)
                route_tasks = [index for index in range(runtime.task_num)
                               if runtime.all_tasks[index].task_type == TaskType.TASK_MOE_ROUTE_GRAD]
                self.assertTrue(route_tasks, msg=f"Expected source-token route jobs, actual={route_tasks}")
                for index in route_tasks:
                    self.assertIsNone(metadata.task_owner_ids.get(index))
                    self.assertIn("Route", metadata.task_stage_names[index])
