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
"""受限 MoE 动态域的有效范围、空通知、ABI 拒绝与设备借用元数据检查。"""

import itertools
import json
import struct
import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.backends.events import decode_event_wire, lower_event_fields
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region
from hyper_parallel.core.multicore.ir.dynamic_events import MoeCountSourceIR, MoeTaskDomainIR
from hyper_parallel.core.multicore.ir.events import EventIndexMap
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.backward.storage import prepare_w13_overlap_schedule
from tests.common.mark_utils import arg_mark
from tests.ut.core.multicore.backends.test_events import _moe_spec


def _configs(plan):
    _, forward, _, backward = _build_runtime_artifacts(
        plan.spec, overlap_w13=False, forward_graph_factory=plan.recipe.build_graph)
    if plan.backward_no_replica is None:
        return forward, backward
    overlap = deepcopy(backward)
    prepare_w13_overlap_schedule(overlap, plan.spec.local_experts, plan.spec.num_cube_cores)
    return forward, overlap, backward


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
class TestDynamicEvents(unittest.TestCase):
    """运行时计数只改变数值工作范围，固定任务与通知语义须保持一致。"""

    def test_all_directions_and_replica_variants_roundtrip(self):
        """普通、profiling、副本和回退镜像均绑定专家二维域，并保留精确字节。"""
        for mode, rank, replicas in itertools.product(("push", "pull"), (0, 1), (0, 1)):
            with self.subTest(mode=mode, rank=rank, replicas=replicas):
                plan = moe_region.plan(_moe_spec(mode, rank, replicas))
                for graph in plan.event_graphs:
                    source = graph.count_source
                    self.assertEqual(source.expert_count, plan.spec.local_experts,
                                     f"actual={source.expert_count}, expected={plan.spec.local_experts}")
                    image = getattr(plan, graph.direction)
                    view = decode_event_wire("moe", image.normal)
                    for grid in graph.grids:
                        if grid.dynamic_domain is None:
                            continue
                        self.assertEqual(len(grid.waits.runs), 1, f"actual={grid.waits.runs}, expected_one_run")
                        self.assertEqual(grid.waits.values(), tuple(view.waits[index] for index in grid.native_tasks),
                                         f"actual_stage={grid.name}, expected_native_waits")
                        self.assertEqual(len(grid.notifies.runs), 1, f"actual={grid.notifies.runs}, expected_one_run")
                    for wire in (image.normal, image.profiled):
                        self.assertEqual(lower_event_fields(graph, wire), wire,
                                         f"actual_direction={graph.direction}, expected_exact_wire_roundtrip")
                plan.validate_runtime_configs(_configs(plan))

    def test_zero_and_tail_counts_keep_reserved_notifications(self):
        """空专家、整 tile、部分 tile 和最大单专家负载不缩减事件通知集合。"""
        plan = moe_region.plan(_moe_spec("push", 0))
        graph = plan.event_graphs[0]
        grid = next(grid for grid in graph.grids if grid.name == "swiglu")
        source = graph.count_source
        for prefix, expected_active in (((0, 0), 0), ((1, 1), 1), ((128, 256), 2), ((129, 129), 2),
                                        ((256, 256), 2)):
            with self.subTest(prefix=prefix):
                intervals = graph.resolve_dynamic_events(prefix)["grids"][grid.name]["row_intervals"]
                active = sum(begin < end for begin, end in intervals)
                self.assertEqual(active, expected_active, f"actual={active}, expected={expected_active}")
                self.assertEqual(len(intervals), len(grid.native_tasks),
                                 f"actual={len(intervals)}, expected={len(grid.native_tasks)}")
                external = {event: graph.event_tensor.wait_counts[event]
                            for boundary in graph.boundaries for event in boundary.events}
                graph.simulate(external_counts=external)
                self.assertEqual(len(grid.notifies.values()), len(intervals),
                                 f"actual={len(grid.notifies.values())}, expected={len(intervals)}")
        actual = grid.dynamic_domain.row_intervals(source, (1, 130))
        self.assertEqual(actual, ((0, 1), (1, 1), (1, 129), (129, 130)),
                         f"actual={actual}, expected_explicit_tail_offsets")

    def test_prefix_and_domain_boundaries_are_rejected(self):
        """拒绝 bool、非单调计数、专家越界、总容量越界和未预留足够 Vector tile。"""
        source = MoeCountSourceIR((6, 10), 2, 256, 384)
        for prefix in ((True, 2), (1,), (2, 1), (-1, 0), (257, 257), (256, 512), [0, 0]):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                source.counts(prefix)
        with self.assertRaisesRegex(ValueError, "cover"):
            MoeTaskDomainIR("swiglu", (2, 1), 128).row_intervals(source, (0, 0))
        with self.assertRaisesRegex(ValueError, "expert count source"):
            MoeTaskDomainIR("grouped_matmul", (3, 20)).row_intervals(source, (0, 0))

    def test_borrowed_tensor_checks_do_not_read_or_copy_contents(self):
        """绑定只查 dtype、shape、stride 和 device，禁止隐式 D2H 或标量读取。"""
        source = MoeCountSourceIR((6, 10), 2, 256, 512)
        tensor = torch.tensor([0, 256], dtype=torch.int64)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("Unexpected D2H")), \
                patch.object(torch.Tensor, "item", side_effect=AssertionError("Unexpected scalar read")), \
                patch.object(torch.Tensor, "tolist", side_effect=AssertionError("Unexpected contents read")):
            source.validate_device_binding(tensor, "cpu")
        for bad in (tensor.int(), tensor.reshape(1, 2), torch.zeros(4, dtype=torch.int64)[::2]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                source.validate_device_binding(bad, "cpu")
        with self.assertRaisesRegex(ValueError, "execution device"):
            source.validate_device_binding(tensor, "meta")

    def test_dynamic_header_and_task_indices_are_validated(self):
        """损坏的计数输入槽、类型、专家数、动态标志和任务划分必须在降级时拒绝。"""
        plan = moe_region.plan(_moe_spec("push", 0))
        graph, image = plan.event_graphs[0], plan.forward
        view = decode_event_wire("moe", image.normal)
        dynamic = view.queue_offset + 16 + view.task_capacity * 12
        task = next(grid for grid in graph.grids if grid.dynamic_domain is not None).native_tasks[0]
        base = view.task_offset + task * view.abi.structure("TaskDescC").size
        offsets = (dynamic, dynamic + 4, dynamic + 8, dynamic + 12,
                   base + view.abi.structure("TaskDescC").offset("task_index"))
        for offset in offsets:
            with self.subTest(offset=offset):
                wire = bytearray(image.normal)
                struct.pack_into("<I", wire, offset, 999)
                with self.assertRaisesRegex(ValueError, "dynamic|reserved"):
                    lower_event_fields(graph, bytes(wire))

    def test_mapping_changes_and_materialization_drift_are_rejected(self):
        """前端不能删除空任务通知，或让 materialize 静默执行另一份事件计划。"""
        plan = moe_region.plan(_moe_spec("push", 0))
        graph = plan.event_graphs[0]
        index = next(index for index, grid in enumerate(graph.grids) if grid.dynamic_domain is not None)
        grid = graph.grids[index]
        wrong = replace(grid, notifies=EventIndexMap.from_values(grid.shape, (None,) * len(grid.native_tasks)))
        grids = graph.grids[:index] + (wrong,) + graph.grids[index + 1:]
        with self.assertRaisesRegex(ValueError, "fewer local producers"):
            replace(graph, grids=grids)
        wrong = replace(grid, notifies=EventIndexMap.from_values(grid.shape, tuple(reversed(grid.notifies.values()))))
        changed = replace(graph, grids=graph.grids[:index] + (wrong,) + graph.grids[index + 1:])
        with self.assertRaisesRegex(ValueError, "native reserved task"):
            lower_event_fields(changed, plan.forward.normal)
        changed_plan = replace(plan, event_graphs=(changed, *plan.event_graphs[1:]))
        with self.assertRaisesRegex(ValueError, "bounded event plan"):
            changed_plan.validate_runtime_configs(_configs(plan))
        configs = _configs(plan)
        configs[0].all_tasks[grid.native_tasks[0]].task_split_value += 1
        with self.assertRaisesRegex(ValueError, "different native runtime"):
            plan.validate_runtime_configs(configs)

    def test_emission_seals_dynamic_domains_and_count_sources(self):
        """动态有效域和预留通知语义须进入生成产物与计划身份。"""
        emission = moe_region.compile(_moe_spec("pull", 1, 1))
        files = {item.path: item.content for item in emission.files}
        data = json.loads(files["events/forward.json"])
        self.assertEqual(data["status"], "bounded_dynamic_compatibility_ir",
                         f"actual={data['status']}, expected=bounded_dynamic_compatibility_ir")
        self.assertEqual(data["count_source"]["input_positions"], [6, 10],
                         f"actual={data['count_source']}, expected_forward_aliases")
        self.assertEqual(data["empty_task_notification"], "preserve_reserved_tasks",
                         f"actual={data['empty_task_notification']}, expected=preserve_reserved_tasks")
