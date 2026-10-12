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
"""三类前端事件 IR 的原始镜像、物理 FIFO 和原生边界验收。"""

import itertools
import json
import struct
import unittest
from dataclasses import replace

from hyper_parallel.core.multicore.backends.events import decode_event_wire, lower_event_fields
from hyper_parallel.core.multicore.compiler.events import attach_event_graphs
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region
from hyper_parallel.core.multicore.ir.events import EventIndexMap, EventTensorIR
from hyper_parallel.core.multicore.ir.schedule import HardwareSpec
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
from tests.common.mark_utils import arg_mark


def _moe_spec(mode, rank, replicas=0):
    return MegaMoeSpec(local_num_tokens=128, hidden_size=512, intermediate_size=128,
                       num_experts=(2 + replicas) * 2, top_k=2,
                       initial_capacity_factor=1.25 if mode == "push" else None,
                       receive_capacity=384, ep_size=2, ep_group=None, rank_id=rank,
                       num_cube_cores=20, dispatch_mode=mode, replica_slots_per_rank=replicas,
                       logical_num_experts=4, replica_transport="shmem_signal_kernel_gradient")


def _native_visited(graph):
    return tuple(graph.grids[number].native_tasks[index] for number, index in graph.simulate()["visited"])


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
class TestEventBindings(unittest.TestCase):
    """静态事件图在普通与性能记录镜像上保持相同的原生执行契约。"""

    def test_gate_has_worker_order_without_global_events(self):
        """覆盖 k=1/k>1、单行、空尾部以及反向外部后处理边界。"""
        for tokens, top_k in itertools.product((1, 41, 49), (1, 3)):
            with self.subTest(tokens=tokens, top_k=top_k):
                plan = _route.plan({"T": tokens, "E": 16}, HardwareSpec(40), k=top_k, scale=2.5)
                for graph in plan.event_graphs:
                    image = getattr(plan, graph.direction)
                    self.assertIsNone(graph.event_tensor, f"actual={graph.event_tensor}, expected=None")
                    workers = len(plan.schedule.partitions) if graph.direction == "forward" else sum(
                        bool(part.row_count) for part in plan.schedule.partitions)
                    actual, expected = len(graph.simulate()["visited"]), workers * len(image.stages)
                    self.assertEqual(actual, expected, f"actual={actual}, expected={expected}")
                    for data in (image.normal, image.profiled):
                        actual = lower_event_fields(graph, data)
                        self.assertEqual(actual, data, f"actual_bytes={len(actual)}, expected_bytes={len(data)}")
                calls = plan.event_graphs[1].boundaries[0].calls
                self.assertEqual(calls, plan.external_backward_calls,
                                 f"actual={calls}, expected={plan.external_backward_calls}")

    def test_mhc_ring_and_macro_joins_preserve_original_fifo(self):
        """覆盖跨 ring 复用、不同 token tile 和反向宏块汇合。"""
        for tokens, tile in ((48, 32), (129, 64), (2593, 32), (12000, 96)):
            with self.subTest(tokens=tokens, tile=tile):
                plan = mhc_boundary.plan(MhcSpec(tokens, 128, 20, tile, tile))
                for graph in plan.event_graphs:
                    image = getattr(plan, graph.direction)
                    actual, expected = _native_visited(graph), tuple(image.simulate()["visited"])
                    self.assertEqual(actual, expected, f"actual={actual}, expected={expected}")
                    counters = graph.simulate()["counters"]
                    expected = tuple(image.simulate()["counters"])
                    self.assertEqual(counters, expected, f"actual={counters}, expected={expected}")
                norm = next(grid for grid in plan.event_graphs[0].grids if grid.name == "mhc_norm_cast")
                values = norm.waits.values()
                if len(values) > plan.ring_slots:
                    actual = values[plan.ring_slots]
                    projection = next(grid for grid in plan.event_graphs[0].grids if grid.name == "mhc_projection")
                    expected = projection.notifies.values()[0]
                    self.assertEqual(actual, expected, f"actual={actual}, expected={expected}")

    def test_moe_transport_rank_and_replica_variants_are_explicit(self):
        """通信发布、入退场握手和副本回退均保留在各方向的 IR 中。"""
        for mode, rank, replicas in itertools.product(("push", "pull"), (0, 1), (0, 1)):
            with self.subTest(mode=mode, rank=rank, replicas=replicas):
                plan = moe_region.plan(_moe_spec(mode, rank, replicas))
                for graph in plan.event_graphs:
                    image = getattr(plan, graph.direction)
                    for data in (image.normal, image.profiled):
                        actual = lower_event_fields(graph, data)
                        self.assertEqual(actual, data, f"actual_bytes={len(actual)}, expected_bytes={len(data)}")
                    names = tuple(boundary.name for boundary in graph.boundaries)
                    self.assertIn(f"native_{mode}_transport", names, f"actual={names}, expected_mode={mode}")
                    self.assertIn("ready_event", names, f"actual={names}, expected=ready_event")
                    self.assertEqual("completion_event" in names, mode == "pull",
                                     f"actual={names}, expected_completion={mode == 'pull'}")
                    with self.assertRaisesRegex(ValueError, "external event"):
                        graph.simulate()

    def test_emission_contains_ir_and_matching_schedule_images(self):
        """公共生成路径封存事件图，并保留两个镜像的独立身份。"""
        emissions = (_route.compile({"T": 49, "E": 16}, k=3, scale=2.5),
                     mhc_boundary.compile(MhcSpec(129, 128, 20)),
                     moe_region.compile(_moe_spec("pull", 0)))
        for emission in emissions:
            files = {item.path: item.content for item in emission.files}
            for graph in emission.plan.event_graphs:
                data = json.loads(files[f"events/{graph.direction}.json"])
                actual, expected = data["family"], graph.family
                self.assertEqual(actual, expected, f"actual={actual}, expected={expected}")
                image = getattr(emission.plan, graph.direction)
                actual = files[f"schedule/{graph.direction}.bin"]
                self.assertEqual(actual, image.normal,
                                 f"actual_bytes={len(actual)}, expected_bytes={len(image.normal)}")

    def test_corrupt_capacity_and_dependency_are_rejected(self):
        """不可信镜像的数组范围和本地生产者数必须在绑定前校验。"""
        plan = mhc_boundary.plan(MhcSpec(129, 128, 20))
        data = bytearray(plan.forward.normal)
        struct.pack_into("<I", data, 12, 0xFFFFFFF0)
        with self.assertRaisesRegex(ValueError, "truncated"):
            decode_event_wire("mhc", bytes(data))
        data = bytearray(plan.forward.normal)
        struct.pack_into("<i", data, 64, 100000)
        corrupted = replace(plan, forward=replace(plan.forward, normal=bytes(data)))
        with self.assertRaisesRegex(ValueError, "fewer local producers"):
            attach_event_graphs(corrupted)

    def test_pipeline_reordering_and_per_worker_events_need_new_lowering(self):
        """现有广播 ABI 不能表达不同 worker 的事件，也不能静默忽略队列变换。"""
        plan = _route.plan({"T": 41, "E": 16}, k=3, scale=2.5)
        graph = plan.event_graphs[0]
        queue = graph.queues[0]
        reordered = replace(graph, queues=(replace(queue, tasks=tuple(reversed(queue.tasks))), *graph.queues[1:]))
        with self.assertRaisesRegex(ValueError, "FIFO order"):
            lower_event_fields(reordered, plan.forward.normal)
        grid = graph.grids[0]
        wrong = replace(grid, native_tasks=(999,) * len(grid.native_tasks))
        malformed = replace(graph, grids=(wrong, *graph.grids[1:]))
        with self.assertRaisesRegex(ValueError, "placement|descriptor"):
            lower_event_fields(malformed, plan.forward.normal)
        waits = (None, 0) + (None,) * (len(grid.native_tasks) - 2)
        mapped = replace(grid, waits=EventIndexMap.from_values(grid.shape, waits))
        different = replace(graph, event_tensor=EventTensorIR("per_worker", (1,), (0,)),
                            grids=(mapped, *graph.grids[1:]))
        with self.assertRaisesRegex(ValueError, "Broadcast descriptors"):
            lower_event_fields(different, plan.forward.normal)
        uniform = replace(graph, event_tensor=EventTensorIR("unused", (1024,), (0,) * 1024))
        with self.assertRaisesRegex(ValueError, "cannot lower global"):
            lower_event_fields(uniform, plan.forward.normal)
