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
"""Analytic network cases and production replica lifecycle contracts."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from examples.torch.multicore.megamoe_simulator.model import (
    Costs,
    Training,
    TrainingGraph,
)
from examples.torch.multicore.megamoe_simulator.network import (
    EventSimulator,
    Fabric,
    Task,
    fair_rates,
)
from examples.torch.multicore.megamoe_simulator.workload import (
    Shape,
    build_plan,
    make_counts,
    validate_counts,
    wave_matrices,
)
from examples.torch.multicore.simulate_mega_moe import parse_args, run


class TestNetworkSimulator(unittest.TestCase):
    """Use closed-form networks to catch shared-capacity and completion errors."""

    def setUp(self) -> None:
        """Create an independent analytic fixture for each case."""
        self.fabric = Fabric((0, 0, 1, 1), inter_gbps=1, inter_latency_us=7)

    def test_two_ranks_share_one_node_exit(self):
        tasks = [Task("a", "data", 0, target=2, size_bytes=1000), Task("b", "data", 1, target=3, size_bytes=1000)]
        result = EventSimulator(self.fabric, tasks).run()
        self.assertEqual(result.finishes_us, [9, 9])
        self.assertEqual(result.resource_bytes["node-tx:0"], 2000)

    def test_receiver_is_also_a_shared_bottleneck(self):
        fabric = replace(self.fabric, node_by_rank=(0, 1, 2, 2))
        tasks = [Task("a", "data", 0, target=2, size_bytes=1000), Task("b", "data", 1, target=3, size_bytes=1000)]
        self.assertEqual(EventSimulator(fabric, tasks).run().finishes_us, [9, 9])

    def test_opposite_directions_are_full_duplex(self):
        tasks = [Task("a", "data", 0, target=2, size_bytes=1000), Task("b", "data", 2, target=0, size_bytes=1000)]
        self.assertEqual(EventSimulator(self.fabric, tasks).run().finishes_us, [8, 8])

    def test_propagation_does_not_hold_bandwidth_but_blocks_consumer(self):
        tasks = [
            Task("a", "data", 0, target=2, size_bytes=1000),
            Task("clock", "cpu", 1, duration_us=1),
            Task("b", "data", 1, (1,), target=3, size_bytes=1000),
            Task("consumer", "compute", 2, (0,), duration_us=2),
        ]
        result = EventSimulator(self.fabric, tasks).run()
        self.assertEqual(result.serialization_ends_us[:3], [1, 1, 2])
        self.assertEqual(result.finishes_us, [8, 1, 9, 10])

    def test_new_flow_joins_an_active_transfer(self):
        fabric = replace(self.fabric, inter_latency_us=0)
        tasks = [
            Task("a", "data", 0, target=2, size_bytes=10000),
            Task("clock", "cpu", 1, duration_us=5),
            Task("b", "data", 1, (1,), target=3, size_bytes=10000),
        ]
        self.assertEqual(EventSimulator(fabric, tasks).run().finishes_us, [15, 5, 20])

    def test_progressive_filling_reassigns_unused_capacity(self):
        self.assertEqual(fair_rates({0: {"shared": 10, "slow": 2}, 1: {"shared": 10}}), {0: 2, 1: 8})

    def test_reject_invalid_graph_and_numeric_inputs(self):
        with self.assertRaisesRegex(ValueError, "earlier"):
            EventSimulator(self.fabric, [Task("cycle", "cpu", 0, (0,))])
        with self.assertRaisesRegex(ValueError, "positive"):
            replace(self.fabric, inter_gbps=float("nan"))
        with self.assertRaisesRegex(ValueError, "outside"):
            EventSimulator(self.fabric, [Task("wrong rank", "cpu", 4)])


class TestReplicaSimulation(unittest.TestCase):
    """Validate byte conservation, weight lifetimes and rank scaling."""

    def setUp(self) -> None:
        """Create an independent analytic fixture for each case."""
        self.shape = Shape(experts=4, tokens_per_rank=128, top_k=1, hidden=8, intermediate=4, capacity_factor=1)
        self.counts = ((128, 0, 0, 0), (128, 0, 0, 0))
        self.plan = build_plan(self.counts, 1, self.shape.capacity)

    def _graph(self, training, plan=None):
        graph = TrainingGraph(self.shape, [self.plan if plan is None else plan], Costs(), training)
        graph.build()
        return graph

    def test_shared_pool_counts_backward_reloads_and_fp32_gradients(self):
        graph = self._graph(Training(layers=2, steps=1))
        result = EventSimulator(Fabric((0, 1)), graph.tasks).run()
        self.assertEqual([event["reused"] for event in graph.cache_events], [False, False, True, True, False, True])
        elements = 3 * self.shape.hidden * self.shape.intermediate
        self.assertEqual(result.traffic["weight_prefetch"]["inter"], 3 * elements * 2)
        self.assertEqual(result.traffic["replica_gradient"]["inter"], 2 * elements * 4)

    def test_unchanged_weights_reuse_but_optimizer_update_invalidates(self):
        for update_every, copies in ((0, 1), (1, 2), (2, 1)):
            with self.subTest(update_every=update_every):
                graph = self._graph(Training(layers=1, steps=2, update_every=update_every))
                self.assertEqual(sum(event["copies"] for event in graph.cache_events), copies)

    def test_per_layer_pool_preserves_distinct_weight_bindings(self):
        graph = self._graph(Training(layers=2, steps=1, pool="per-layer"))
        self.assertEqual(sum(event["copies"] for event in graph.cache_events), 2)

    def test_placement_change_invalidates_unchanged_parameters(self):
        changed = build_plan(((0, 128, 0, 0), (0, 128, 0, 0)), 1, self.shape.capacity)
        graph = TrainingGraph(self.shape, [self.plan, changed], Costs(), Training(layers=1, steps=2, update_every=0))
        graph.build()
        self.assertEqual(sum(event["copies"] for event in graph.cache_events), 2)

    def test_capacity_alignment_matches_runtime_contract(self):
        self.assertEqual(replace(self.shape, capacity_factor=1.1).capacity, 256)
        with self.assertRaisesRegex(ValueError, "local routed"):
            replace(self.shape, tokens_per_rank=256, capacity_factor=0.1)

    def test_wave_slicing_preserves_every_source_and_destination(self):
        for slots in (0, 1):
            plan = build_plan(self.counts, slots, 2)
            waves = wave_matrices(plan, 2)
            for source in range(plan.ep_size):
                self.assertEqual(sum(sum(matrix[source]) for matrix in waves), sum(self.counts[source]))
            for target in range(plan.ep_size):
                loads = [sum(row[target] for row in matrix) for matrix in waves]
                self.assertEqual(sum(loads), plan.destination_loads[target])
                self.assertLessEqual(max(loads), 2)

    def test_multi_wave_backward_adds_hidden_dispatch(self):
        plan = build_plan(self.counts, 0, self.shape.capacity)
        graph = self._graph(Training(layers=1, steps=1, checkpoint=False), plan)
        hidden_bytes = sum(task.size_bytes for task in graph.tasks if task.phase == "backward_hidden_dispatch")
        self.assertEqual(hidden_bytes, 2 * self.shape.tokens_per_rank * self.shape.row_bytes)
        self.assertTrue(any(task.phase == "wave_sync" for task in graph.tasks))

    def test_slow_network_changes_virtual_time_without_changing_plan(self):
        graph = self._graph(Training(layers=1, steps=1))
        fast = EventSimulator(Fabric((0, 1), inter_gbps=10), graph.tasks).run()
        slow = EventSimulator(Fabric((0, 1), inter_gbps=0.1), graph.tasks).run()
        self.assertGreater(slow.makespan_us, fast.makespan_us)
        self.assertEqual(slow.traffic, fast.traffic)

    def test_all_requested_rank_counts_and_topk_conservation(self):
        shape = Shape(experts=64, tokens_per_rank=8, top_k=2, hidden=8, intermediate=4)
        for ranks in (16, 32, 64):
            for routing in ("balanced", "hotspot", "dirichlet"):
                with self.subTest(ranks=ranks, routing=routing):
                    counts = make_counts(shape, ranks, routing=routing, alpha=0.01)
                    plan = build_plan(counts, 1, shape.capacity)
                    self.assertEqual(sum(plan.destination_loads), ranks * 16)
                    self.assertEqual(validate_counts(counts, shape, ranks), counts)

    def test_malformed_counts_fail_before_simulation(self):
        with self.assertRaisesRegex(ValueError, "Top-K"):
            validate_counts(((129, 0, 0, 0), (128, 0, 0, 0)), self.shape, 2)

    def test_cli_saves_replayable_inputs_and_labeled_results(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(
                [
                    "--ranks",
                    "2",
                    "--ranks-per-node",
                    "1",
                    "--slots",
                    "0",
                    "1",
                    "--experts",
                    "4",
                    "--top-k",
                    "1",
                    "--tokens-per-rank",
                    "4",
                    "--hidden",
                    "8",
                    "--intermediate",
                    "4",
                    "--layers",
                    "1",
                    "--steps",
                    "1",
                    "--planner-us",
                    "0",
                    "--trace",
                    "--output",
                    directory,
                ]
            )
            summary = run(args)
            self.assertEqual(summary["evidence_kind"], "simulation")
            self.assertEqual(len(summary["scenarios"]), 2)
            self.assertTrue((Path(directory) / "summary.csv").is_file())
            data = json.loads((Path(directory) / "rank2.counts.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["counts"]), 2)
            self.assertTrue(list(Path(directory).glob("*.trace.json")))


if __name__ == "__main__":
    unittest.main()
