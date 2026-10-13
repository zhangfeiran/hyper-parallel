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
"""MHC 静态事件调度的进展、所有权与原生队列一致性验证。"""

import unittest
from dataclasses import replace

from hyper_parallel.core.multicore.backends.events import decode_event_wire, lower_mhc_worker_queues
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.modules.mega_mhc.module import HyperMegaMhc
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
from tests.common.mark_utils import arg_mark


def _owners(image):
    return {(resource, worker): tasks[worker::count]
            for resource, tasks, count in zip(("cube", "vector", "mix"), image.queues, image.workers)
            for worker in range(count)}


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
class TestEventSchedule(unittest.TestCase):
    """受限队列变换不会改变数值任务、计数和 scratch 的 worker 归属。"""

    def test_tail_tiles_and_ownership_remain_valid(self):
        """覆盖不同拓扑、整波与非整波、尾部，以及多种 token tile。"""
        for cores, tokens, tile in ((5, 10, 32), (20, 40, 32), (20, 41, 32), (24, 129, 64),
                                    (20, 2560, 32), (20, 2593, 32), (20, 4097, 64), (20, 12000, 96)):
            with self.subTest(cores=cores, tokens=tokens, tile=tile):
                original = mhc_boundary.plan(MhcSpec(tokens, 128, cores, tile, tile))
                candidate = mhc_boundary.plan(original.spec, event_schedule="mhc_backward_overlap")
                self.assertEqual(candidate.forward, original.forward,
                                 f"actual={candidate.forward.stages}, expected={original.forward.stages}")
                old, new = _owners(original.backward), _owners(candidate.backward)
                for owner, tasks in old.items():
                    self.assertEqual(set(new[owner]), set(tasks), f"actual={new[owner]}, expected={tasks}")
                self.assertEqual(candidate.backward.tasks, original.backward.tasks,
                                 f"actual={candidate.backward.tasks}, expected={original.backward.tasks}")
                self.assertEqual(candidate.backward.thresholds, original.backward.thresholds,
                                 f"actual={candidate.backward.thresholds}, expected={original.backward.thresholds}")
                for data, baseline in ((candidate.backward.normal, original.backward.normal),
                                       (candidate.backward.profiled, original.backward.profiled)):
                    view = decode_event_wire("mhc", baseline)
                    begin = view.queue_offset + 16 + view.task_capacity * 4
                    end = begin + len(view.queues[1]) * 4
                    actual, expected = data[:begin] + data[end:], baseline[:begin] + baseline[end:]
                    self.assertEqual(actual, expected, f"actual_bytes={len(actual)}, expected_bytes={len(expected)}")
                actual = candidate.event_graphs[1].simulate()["counters"]
                expected = original.event_graphs[1].simulate()["counters"]
                self.assertEqual(actual, expected, f"actual={actual}, expected={expected}")

    def test_final_can_start_before_all_prepare_tiles(self):
        """依赖仿真应显示更早的 Final，而不把逻辑进展当作性能结果。"""
        original = mhc_boundary.plan(MhcSpec(12000, 128, 20, 64, 64))
        candidate = mhc_boundary.plan(original.spec, event_schedule="mhc_backward_overlap")
        counts = []
        for graph in (original.event_graphs[1], candidate.event_graphs[1]):
            order = graph.simulate()["visited"]
            first = next(index for index, (number, _) in enumerate(order)
                         if graph.grids[number].name == "mhc_grad_prev_x_and_post")
            counts.append(sum(number == 2 for number, _ in order[:first]))
        self.assertLess(counts[1], counts[0], f"actual={counts[1]}, expected_less_than={counts[0]}")

    def test_all_worker_phase_offsets_make_progress(self):
        """覆盖 tile 数对 40 个 worker 的所有余数，防止交错引入尾波死锁。"""
        for tiles in range(40, 81):
            with self.subTest(tiles=tiles):
                plan = mhc_boundary.plan(MhcSpec(tiles * 64 - 1, 128, 20, 64, 64),
                                         event_schedule="mhc_backward_overlap")
                graph = plan.event_graphs[1]
                actual = len(graph.simulate()["visited"])
                expected = sum(len(grid.native_tasks) for grid in graph.grids)
                self.assertEqual(actual, expected, f"actual={actual}, expected={expected}")

    def test_emission_seals_the_selected_schedule(self):
        """事件 JSON、设备二进制和计划键都需要对应实际选中的队列。"""
        spec = MhcSpec(2593, 128, 20)
        old = mhc_boundary.compile(spec)
        new = mhc_boundary.compile(spec, event_schedule="mhc_backward_overlap")
        self.assertNotEqual(new.plan_key, old.plan_key, f"actual={new.plan_key}, expected_not={old.plan_key}")
        files = {item.path: item.content for item in new.files}
        for suffix, expected in (("bin", new.plan.backward.normal),
                                 ("profiled.bin", new.plan.backward.profiled)):
            actual = files[f"schedule/backward.{suffix}"]
            self.assertEqual(actual, expected, f"actual_bytes={len(actual)}, expected_bytes={len(expected)}")

    def test_unsupported_family_and_policy_are_rejected(self):
        """显式实验策略不能静默应用到 Gate、无反向计划或未知策略。"""
        with self.assertRaisesRegex(ValueError, "requires an MHC"):
            _route.plan({"T": 49, "E": 16}, k=3, scale=2.5, event_schedule="mhc_backward_overlap")
        with self.assertRaisesRegex(ValueError, "with backward"):
            mhc_boundary.plan(MhcSpec(1, 128, 20, need_backward=False), event_schedule="mhc_backward_overlap")
        with self.assertRaisesRegex(ValueError, "Unknown event schedule"):
            mhc_boundary.plan(MhcSpec(40, 128, 20), event_schedule="unknown")

    def test_module_strategy_is_fixed_before_descriptors_are_cached(self):
        """CPU 构造应校验策略，并防止缓存建立后通过公共属性改换策略。"""
        model = HyperMegaMhc(128, event_schedule="mhc_backward_overlap")
        self.assertEqual(model.event_schedule, "mhc_backward_overlap",
                         f"actual={model.event_schedule}, expected=mhc_backward_overlap")
        with self.assertRaisesRegex(AttributeError, "property|setter"):
            setattr(model, "event_schedule", "compatibility")
        model.close()
        with self.assertRaisesRegex(ValueError, "Unknown event schedule"):
            HyperMegaMhc(128, event_schedule="unknown")

    def test_cross_worker_reassignment_and_cube_reorder_are_rejected(self):
        """即使任务集合相同，也不能改变 worker 或 Cube 的 scratch 使用顺序。"""
        plan = mhc_boundary.plan(MhcSpec(12000, 128, 20, 64, 64))
        graph = plan.event_graphs[1]
        queues = list(graph.queues)
        first, second = [index for index, queue in enumerate(queues) if queue.resource == "vector"][:2]
        left, right = queues[first], queues[second]
        queues[first] = replace(left, tasks=(right.tasks[0], *left.tasks[1:]))
        queues[second] = replace(right, tasks=(left.tasks[0], *right.tasks[1:]))
        changed = replace(graph, queues=tuple(queues))
        with self.assertRaisesRegex(ValueError, "per-worker task ownership"):
            lower_mhc_worker_queues(changed, plan.backward.normal)
        cube = graph.queues[0]
        changed = replace(graph, queues=(replace(cube, tasks=tuple(reversed(cube.tasks))), *graph.queues[1:]))
        with self.assertRaisesRegex(ValueError, "Cube or Mix"):
            lower_mhc_worker_queues(changed, plan.backward.normal)

    def test_deadlock_and_threshold_changes_are_rejected(self):
        """足够的通知总数不代表固定 FIFO 可推进，阈值也不能随队列实验变化。"""
        plan = mhc_boundary.plan(MhcSpec(2593, 128, 20))
        graph = plan.event_graphs[1]
        queues = list(graph.queues)
        index = next(index for index, queue in enumerate(queues) if queue.resource == "vector")
        queue = queues[index]
        queues[index] = replace(queue, tasks=queue.tasks[1:] + queue.tasks[:1])
        changed = replace(graph, queues=tuple(queues))
        with self.assertRaisesRegex(ValueError, "unsatisfied dependency"):
            lower_mhc_worker_queues(changed, plan.backward.normal)
        tensor = graph.event_tensor
        changed = replace(graph, event_tensor=replace(tensor, wait_counts=(*tensor.wait_counts[:-1], 1)))
        with self.assertRaisesRegex(ValueError, "event thresholds"):
            lower_mhc_worker_queues(changed, plan.backward.normal)
