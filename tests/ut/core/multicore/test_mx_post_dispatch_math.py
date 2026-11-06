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
"""Protect communication ownership when borrowing post-dispatch vector workers."""
import os
import unittest
from unittest import mock

from tests.common.mark_utils import arg_mark
from hyper_parallel.core.multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_parallel.core.multicore.modules.mega_moe.mxfp8_module import _mx_home_spec
from hyper_parallel.core.multicore.scheduler.runtime import serialize_runtime_config
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import TaskType


class TestMXPostDispatchMath(unittest.TestCase):
    """Verify actual serialized task ownership without native or distributed initialization."""

    def setUp(self) -> None:
        """Isolate experimental switches and the qualified CPU graph environment."""
        patch = mock.patch.dict(os.environ, {
            'PANGU_MX_GMM_TILE': 't6', 'HYPER_PARALLEL_SHMEM_DATA_ENGINE': 'mte_udma',
            'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': '16', 'PANGU_MX_REUSE_DISPATCH_WORKERS': '0',
        })
        patch.start()
        self.addCleanup(patch.stop)

    @staticmethod
    def queues(config: object) -> list[list[int]]:
        """Decode the actual CSR vector ownership from the runtime image."""
        return [list(config.vector_task_indices[config.mix_task_indices[worker]:config.mix_task_indices[worker+1]])
                for worker in range(64)]

    def check_event_progress(self, runtime: object, ep: int) -> None:
        """Replay a symmetric-peer event model; this does not prove network progress."""
        config = runtime.config
        cubes = list(config.cube_task_indices[:config.task_index_num[0]])
        queues = [cubes[worker::32] for worker in range(32)] + self.queues(config)
        heads, completed = [0] * len(queues), [set() for _ in queues]
        counters = [0] * len(config.all_event_num_triggers)
        total = sum(map(len, queues))
        while total:
            progressed = 0
            for worker, queue in enumerate(queues):
                head = heads[worker]
                if head == len(queue):
                    continue
                flags = config.all_tasks[queue[0]].extra_value_1
                depth = 8 if flags & 0x10000 else 1
                for index in range(head, min(head + depth, len(queue))):
                    task = config.all_tasks[queue[index]]
                    dependency = task.dependent_event
                    if index in completed[worker]:
                        continue
                    if dependency != 0xFFFFFFFF and counters[dependency] < config.all_event_num_triggers[dependency]:
                        continue
                    completed[worker].add(index)
                    if task.trigger_event != 0xFFFFFFFF:
                        # SHMEM signals arrive from every peer; local math triggers once.
                        counters[task.trigger_event] += ep if task.task_type == 105 else 1
                    while heads[worker] in completed[worker]:
                        heads[worker] += 1
                    progressed += 1
                    break
            self.assertGreater(progressed, 0, msg=f'Local queue/event replay stalled with {total} tasks pending')
            total -= progressed

    def check_wire_ownership(self, before: object, after: object,
                             old: list[list[int]], new: list[list[int]]) -> None:
        """Require exact task coverage and the unchanged communication subsequence on each worker."""
        self.assertEqual(sorted(task for queue in old for task in queue),
                         sorted(task for queue in new for task in queue))
        for worker in range(64):
            self.assertEqual(
                [task for task in old[worker] if before.config.all_tasks[task].task_type in (0, 105)],
                [task for task in new[worker] if after.config.all_tasks[task].task_type in (0, 105)],
            )

    def check_borrowed_workers(self, after: object, new: list[list[int]], mode: int) -> None:
        """Require borrowed quantization to follow every dispatch task."""
        for worker in range(1, 64, 2):
            quant = [index for index, task in enumerate(new[worker])
                                     if after.config.all_tasks[task].task_type in (161, 162, 163, 164, 165)]
            self.assertTrue(quant, msg=f'Borrowed worker {worker} has no quantization tasks')
            if mode == 2:
                self.assertTrue(all(after.config.all_tasks[new[worker][index]].inputs[0].input_position
                                    == 0 for index in quant))
            dispatch = after.graph.get_op('dispatch')
            start = next(index for index, task in enumerate(after.config.all_tasks)
                         if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL)
            dispatch_ids = set(range(start, start + dispatch.task_num))
            self.assertTrue(all(task in dispatch_ids for task in new[worker][:quant[0]]))

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_borrowed_math_preserves_wire_owners_and_task_bytes(self) -> None:
        """Feature: Post-dispatch vector role reuse.
        Description: Compare actual queues across EP sizes, QP banks, phases and guest budgets.
        Expectation: Communication owners/order and every task byte remain fixed; odd workers gain math.
        """
        for ep, qps in ((4, 16), (16, 16), (16, 32), (64, 1)):
            os.environ['HYPER_PARALLEL_SHMEM_UDMA_QP_NUM'] = str(qps)
            for budget in (0, 2, 6):
                spec = MegaMoeSpec(4096, 5120, 1792, ep*(6+budget), 8, 1.25, 40960, ep, None, ep-1, 32,
                    mx_vector_policy='window2', dispatch_split=512, combine_split=512, compute_stages=6,
                    logical_num_experts=ep*6 if budget else 0, replica_slots_per_rank=budget,
                    replica_transport='shmem_signal_kernel_gradient')
                for forward, mode in ((False, 1), (True, 1), (False, 2), (True, 2)):
                    with self.subTest(ep=ep, qps=qps, budget=budget, forward=forward, mode=mode):
                        os.environ['PANGU_MX_REUSE_DISPATCH_WORKERS'] = '0'
                        before = build_mxfp8_runtime(spec, forward=forward, ready_depth=8,
                                                    weight_prefetch=bool(budget))
                        os.environ['PANGU_MX_REUSE_DISPATCH_WORKERS'] = str(mode)
                        after = build_mxfp8_runtime(spec, forward=forward, ready_depth=8,
                                                   weight_prefetch=bool(budget))
                        old, new = self.queues(before.config), self.queues(after.config)
                        self.assertEqual(before.gmm_descriptors, after.gmm_descriptors)
                        self.assertEqual(before.quant_descriptors, after.quant_descriptors)
                        self.assertEqual(list(before.config.all_event_num_triggers),
                                         list(after.config.all_event_num_triggers))
                        self.assertEqual(list(before.config.cube_task_indices), list(after.config.cube_task_indices))
                        for a, b in zip(before.config.all_tasks, after.config.all_tasks):
                            self.assertEqual(bytes(a), bytes(b))
                        self.check_wire_ownership(before, after, old, new)
                        self.check_borrowed_workers(after, new, mode)
                        self.check_event_progress(after, ep)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_active_policy_selects_exact_previously_qualified_queues(self) -> None:
        """Feature: Actual-replica policy selection.
        Description: Compare active and home-only specs in both directions and budgets.
        Expectation: Mode3 is byte-identical to mode1 for active specs and mode0 for home specs.
        """
        for budget in (0, 1, 2, 6):
            active = MegaMoeSpec(4096, 5120, 1792, 16*(6+budget), 8, 1.25, 40960, 16, None, 15, 32,
                mx_vector_policy='window2', dispatch_split=512, combine_split=512,
                logical_num_experts=96 if budget else 0, replica_slots_per_rank=budget,
                replica_transport='shmem_signal_kernel_gradient')
            variants = (active, _mx_home_spec(active)) if budget else (active,)
            for spec in variants:
                for forward in (False, True):
                    with self.subTest(budget=budget, slots=spec.replica_slots_per_rank, forward=forward):
                        os.environ['PANGU_MX_REUSE_DISPATCH_WORKERS'] = str(int(bool(spec.replica_slots_per_rank)))
                        expected = build_mxfp8_runtime(spec, forward=forward, ready_depth=8)
                        os.environ['PANGU_MX_REUSE_DISPATCH_WORKERS'] = '3'
                        actual = build_mxfp8_runtime(spec, forward=forward, ready_depth=8)
                        self.assertEqual(serialize_runtime_config(actual.config), serialize_runtime_config(expected.config))
                        self.assertEqual(actual.gmm_descriptors, expected.gmm_descriptors)
                        self.assertEqual(actual.quant_descriptors, expected.quant_descriptors)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_invalid_flag_fails_before_installing_a_queue(self) -> None:
        """Feature: Experimental switch boundary.
        Description: Supply an unsupported value before building a runtime image.
        Expectation: A descriptive ValueError rejects the value.
        """
        os.environ['PANGU_MX_REUSE_DISPATCH_WORKERS'] = 'yes'
        spec = MegaMoeSpec(4096, 5120, 1792, 96, 8, 1.25, 40960, 16, None, 15, 32,
                          mx_vector_policy='window2', dispatch_split=512, combine_split=512)
        with self.assertRaisesRegex(ValueError, 'REUSE_DISPATCH_WORKERS'):
            build_mxfp8_runtime(spec, forward=True)
