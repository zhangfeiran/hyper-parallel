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
"""事件张量映射、汇合和 FIFO 进展的硬件无关语义验证。"""

import unittest
from dataclasses import replace

from hyper_parallel.core.multicore.ir.events import (
    EventBoundaryIR,
    EventGraphIR,
    EventIndexMap,
    EventTensorIR,
    IndexRun,
    TaskGridIR,
    WorkerQueueIR,
)
from tests.common.mark_utils import arg_mark


def _map(values):
    return EventIndexMap.from_values((len(values),), tuple(values))


def _join_graph():
    producers = TaskGridIR("partial", (2, 2), (0, 1, 2, 3),
                           EventIndexMap.from_values((2, 2), (None,) * 4),
                           EventIndexMap((2, 2), (IndexRun(0, 4, 0, step=1, repeat=2),)))
    consumers = TaskGridIR("final", (2,), (4, 5), _map((0, 1)), _map((None, None)))
    queues = (WorkerQueueIR("cube", 0, ((0, 0), (1, 0), (0, 2))),
              WorkerQueueIR("cube", 1, ((0, 1), (0, 3), (1, 1))))
    return EventGraphIR("mhc", "forward", "fixed_stride", EventTensorIR("rows", (2,), (2, 2)),
                        (producers, consumers), queues)


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
class TestEventGraph(unittest.TestCase):
    """汇合只释放对应消费者，外部边界和循环不能伪造进展。"""

    def test_piecewise_affine_maps_cover_the_domain(self):
        """压缩映射保留空依赖、递增、递减与重复事件坐标。"""
        for values in ((None, None, 0, 1, 2), (3, 2, 1, 0), (0, 0, 1, 1), ()):
            with self.subTest(values=values):
                actual = EventIndexMap.from_values((len(values),), values).values()
                self.assertEqual(actual, values, f"actual={actual}, expected={values}")
        actual = _join_graph().grids[0].notifies.values()
        self.assertEqual(actual, (0, 0, 1, 1), f"actual={actual}, expected={(0, 0, 1, 1)}")
        actual = _join_graph().grids[0].notifies.at((1, 0))
        self.assertEqual(actual, 1, f"actual={actual}, expected=1")

    def test_local_join_allows_other_rows_to_progress(self):
        """消费者零行只等待对应两项生产者，同时其他行仍可推进。"""
        result = _join_graph().simulate()
        expected = ((0, 0), (0, 1), (1, 0), (0, 3), (0, 2), (1, 1))
        self.assertEqual(result["visited"], expected, f"actual={result['visited']}, expected={expected}")
        self.assertEqual(result["counters"], (2, 2), f"actual={result['counters']}, expected={(2, 2)}")

    def test_domains_and_partial_maps_are_rejected(self):
        """空域合法，布尔维度、映射空洞和域外事件明确失败。"""
        with self.assertRaisesRegex(ValueError, "static integers"):
            EventTensorIR("bad", (True,), (1,))
        with self.assertRaisesRegex(ValueError, "gaps"):
            EventIndexMap((2,), (IndexRun(1, 2, 0),))
        graph = _join_graph()
        with self.assertRaisesRegex(ValueError, "domain"):
            replace(graph, event_tensor=EventTensorIR("short", (1,), (2,)))

    def test_missing_producers_and_duplicate_work_are_rejected(self):
        """通知不足、遗漏任务或重复执行都会使依赖验证失败。"""
        graph = _join_graph()
        with self.assertRaisesRegex(ValueError, "fewer local producers"):
            replace(graph, event_tensor=EventTensorIR("rows", (2,), (3, 2)))
        with self.assertRaisesRegex(ValueError, "exactly once"):
            replace(graph, queues=(graph.queues[0],))
        with self.assertRaisesRegex(ValueError, "exactly once"):
            replace(graph, queues=graph.queues + (WorkerQueueIR("vector", 0, ((0, 0),)),))

    def test_fifo_deadlock_is_detected(self):
        """通知总数足够仍不能保证固定 FIFO 不会自阻塞。"""
        graph = _join_graph()
        blocked = replace(graph, queues=(WorkerQueueIR("cube", 0, ((1, 0), (0, 0), (0, 1), (0, 2))),
                                         WorkerQueueIR("cube", 1, ((0, 3), (1, 1)))))
        with self.assertRaisesRegex(ValueError, "unsatisfied dependency"):
            blocked.simulate()

    def test_external_counts_require_explicit_assumptions(self):
        """原生通信事件不由本地仿真自动补齐，也不能注入本地汇合计数。"""
        grid = TaskGridIR("remote_consumer", (1,), (0,), _map((0,)), _map((None,)))
        graph = EventGraphIR("moe", "forward", "fixed_stride", EventTensorIR("peer", (1,), (1,)),
                             (grid,), (WorkerQueueIR("cube", 0, ((0, 0),)),),
                             (EventBoundaryIR("peer_publication", (0,)),))
        with self.assertRaisesRegex(ValueError, "external event"):
            graph.simulate()
        result = graph.simulate(external_counts={0: 1})
        self.assertEqual(result["assumed_external_events"], (0,),
                         f"actual={result['assumed_external_events']}, expected={(0,)}")
        with self.assertRaisesRegex(ValueError, "inject a local"):
            _join_graph().simulate(external_counts={0: 2})
