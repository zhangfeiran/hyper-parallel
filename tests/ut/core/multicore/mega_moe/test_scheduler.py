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
"""CPU contracts for rank-aware communication task queues."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hyper_parallel.core.multicore.modules.mega_moe.forward import (
    gen_runtime_data as forward_data,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_forward_graph,
    build_route_forward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.placement import (
    build_constrained_expert_placement,
)
from hyper_parallel.core.multicore.scheduler.config import (
    RuntimeConfigC,
    TaskSplitValue,
    serialize_runtime_config,
)
from hyper_parallel.core.multicore.scheduler.scheduler import (
    revise_rank_aware_alltoall_segment,
    revise_task_queue,
)


class TestRankAwareTaskQueue(unittest.TestCase):
    """Guard expert readiness order, segment boundaries and signal-task coverage."""

    def test_segment_permutation_preserves_tile_expert_and_unrelated_work(self) -> None:
        """
        Feature: rank-aware communication queue segments
        Description: reorder unequal dispatch/combine segments across EP and worker counts
        Expectation: preserve every task, expert/tile order and intervening work
        """
        for ep, workers in ((1, 24), (2, 24), (8, 24), (3, 8), (8, 4)):
            for rank in range(ep):
                with self.subTest(ep=ep, workers=workers, rank=rank):
                    values = TaskSplitValue(ep=ep, all_expert_num=ep * 3)
                    values.rank_id = rank
                    dispatch = values.all_expert_num * 5
                    combine = values.all_expert_num * 7
                    combine_start = dispatch + 13
                    original = list(range(combine_start + combine + 11))
                    config = SimpleNamespace(num_workers=2 * workers, vector_task_indices=original.copy())

                    revise_task_queue(config, values, dispatch, 9, combine, combine_start)

                    queue = config.vector_task_indices
                    self.assertEqual(queue[dispatch:combine_start], original[dispatch:combine_start])
                    self.assertEqual(queue[combine_start + combine:], original[combine_start + combine:])
                    for start, tasks, tiles in ((0, dispatch, 5), (combine_start, combine, 7)):
                        segment = queue[start:start + tasks]
                        self.assertCountEqual(segment, original[start:start + tasks])
                        for group in range(3 * tiles):
                            ids = [task - start for task in segment[group * ep:(group + 1) * ep]]
                            self.assertEqual([task % (3 * tiles) for task in ids], [group] * ep)
                            peers = [task // (3 * tiles) for task in ids]
                            self.assertEqual(peers, [(lane + rank) % ep for lane in range(ep)])

    def test_explicit_alltoall_segment_matches_forward_rank_order(self) -> None:
        """
        Feature: explicit communication-segment reordering
        Description: use the backward helper on the same nonaligned segments as forward
        Expectation: both entry points preserve the same rank-aware task and boundary order
        """
        values = TaskSplitValue(ep=3, all_expert_num=9)
        values.rank_id = 2
        dispatch, combine, start = 45, 63, 58
        original = list(range(start + combine + 11))
        forward = SimpleNamespace(vector_task_indices=original.copy())
        explicit = SimpleNamespace(vector_task_indices=original.copy())

        revise_task_queue(forward, values, dispatch, 9, combine, start)
        revise_rank_aware_alltoall_segment(explicit, values, start=0, task_num=dispatch)
        revise_rank_aware_alltoall_segment(explicit, values, start=start, task_num=combine)

        self.assertEqual(explicit.vector_task_indices, forward.vector_task_indices)

    def test_invalid_segment_count_fails_before_mutating_queue(self) -> None:
        """
        Feature: explicit communication-segment validation
        Description: pass empty, negative and non-expert-aligned task counts
        Expectation: reject invalid counts without touching the original queue
        """
        values = TaskSplitValue(ep=2, all_expert_num=4)
        for tasks in (0, -4, 3, 5):
            with self.subTest(tasks=tasks):
                config = SimpleNamespace(vector_task_indices=list(range(16)))
                with self.assertRaisesRegex(ValueError, "positive multiple of the expert"):
                    revise_rank_aware_alltoall_segment(config, values, start=2, task_num=tasks)
                self.assertEqual(config.vector_task_indices, list(range(16)))

    def test_forward_reordering_changes_only_vector_indices(self) -> None:
        """
        Feature: rank-aware queue isolation across endpoint modes
        Description: build legacy, route-dispatch and weighted-unpermute descriptors with and without RATR
        Expectation: all task payloads, event counts, handshakes and storage sizes stay identical
        """
        for builder, unpermute in ((build_forward_graph, False), (build_route_forward_graph, False),
                                   (build_route_forward_graph, True)):
            with self.subTest(builder=builder.__name__, unpermute=unpermute):
                values = TaskSplitValue(tp=1, ep=2, seq_size=512, all_expert_num=6, top_k=2)
                graph = builder(values, num_cube_cores=4, integrated_unpermute=unpermute)
                graph.propagate_splits(values)
                with patch.object(forward_data, "revise_task_queue"):
                    original = forward_data.build_config_for_rank(graph, values, 1, num_cube_cores=4)
                reordered = forward_data.build_config_for_rank(graph, values, 1, num_cube_cores=4)
                original_wire = serialize_runtime_config(original)
                reordered_wire = serialize_runtime_config(reordered)

                self.assertNotEqual(original_wire, reordered_wire)
                self.assertEqual(len(original_wire), len(reordered_wire))
                self.assertCountEqual(list(original.vector_task_indices), list(reordered.vector_task_indices))
                restored = RuntimeConfigC.from_buffer_copy(bytes(reordered))
                restored.vector_task_indices = original.vector_task_indices
                self.assertEqual(serialize_runtime_config(restored), original_wire)

    def test_balanced_replica_receivers_can_have_uneven_worker_payloads(self) -> None:
        """
        Feature: skewed replica communication load and empty-tile coverage
        Description: model EP8 B1 alpha=0.1 seed=42 using recorded logical counts and real planner quotas
        Expectation: conserve payload and every tile while distinguishing receiver from worker balance
        """
        # Logical routes match across sources; replica copy quotas need not.
        counts = (4096, 6, 0, 1277, 0, 1354, 0, 245, 4096, 2138, 399, 1,
                  24, 877, 0, 1232, 3, 1, 40, 0, 0, 0, 493, 1,
                  2107, 0, 0, 591, 0, 3747, 6, 216, 2, 14, 0, 4096,
                  0, 1, 857, 0, 0, 0, 153, 0, 2594, 2100, 1, 0)
        plan = build_constrained_expert_placement([counts] * 8, 1, receive_capacity=32768)
        self.assertEqual(plan.destination_loads, (32768,) * 8)
        values = TaskSplitValue(ep=8, all_expert_num=56)
        tasks = 56 * 32
        dispatch_loads, combine_loads = [], []
        for rank in range(8):
            values.rank_id = rank
            config = SimpleNamespace(num_workers=48, vector_task_indices=list(range(2 * tasks)))
            revise_task_queue(config, values, tasks, 0)
            dispatch_load, combine_load = [0] * 24, [0] * 24
            empty_dispatch_tiles = 0
            for index, task in enumerate(config.vector_task_indices[:tasks]):
                physical_expert, tile = divmod(task, 32)
                peer, local_expert = divmod(physical_expert, 7)
                dispatch_rows = plan.dispatch_counts[rank][physical_expert]
                combine_rows = plan.dispatch_counts[peer][rank * 7 + local_expert]
                dispatch_load[index % 24] += max(0, min(128, dispatch_rows - tile * 128))
                combine_load[index % 24] += max(0, min(128, combine_rows - tile * 128))
                empty_dispatch_tiles += int(dispatch_rows <= tile * 128)
            self.assertCountEqual(config.vector_task_indices[:tasks], range(tasks))
            self.assertGreater(empty_dispatch_tiles, 0)
            self.assertLess(empty_dispatch_tiles, tasks)
            self.assertEqual(sum(dispatch_load), 32768)
            self.assertEqual(sum(combine_load), 32768)
            dispatch_loads.extend(dispatch_load)
            combine_loads.extend(combine_load)
        self.assertEqual((max(dispatch_loads), max(combine_loads)), (3189, 3189))
        self.assertLess(min(dispatch_loads), 32768 // 24)
