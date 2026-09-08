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
"""Independent arithmetic, lifetime, calibration and conservation contracts."""

import random
import unittest
from dataclasses import replace

from examples.torch.multicore.megamoe_simulator.accelerated import ArrayEventSimulator
from examples.torch.multicore.megamoe_simulator.cost import (
    Hardware,
    Options,
    gemm,
    wave_groups,
)
from examples.torch.multicore.megamoe_simulator.estimate import predict
from examples.torch.multicore.megamoe_simulator.memory import (
    Storage,
    memory_ledger,
    peak,
)
from examples.torch.multicore.megamoe_simulator.model import Training
from examples.torch.multicore.megamoe_simulator.network import (
    EventSimulator,
    Fabric,
    Task,
)
from examples.torch.multicore.megamoe_simulator.profile import (
    calibrate,
    calibrate_transfer_windows,
    evaluate,
    union_us,
)
from examples.torch.multicore.megamoe_simulator.workload import (
    Shape,
    build_plan,
    make_counts,
)


class TestCostModel(unittest.TestCase):
    """Use closed-form cases rather than reimplementing estimator internals."""

    def setUp(self) -> None:
        """Create a hardware-free shape with exact tile boundaries."""
        self.shape = Shape(experts=8, tokens_per_rank=64, top_k=2, hidden=16, intermediate=16)
        self.hardware = Hardware()

    def test_forward_backward_flops(self) -> None:
        """All six GEMMs equal the independent gated-MLP FLOP identity."""
        for rows in (0, 1, 16, 97):
            with self.subTest(rows=rows):
                forward = sum(
                    gemm(phase, [rows], self.shape, self.hardware, 4).flops for phase in ("up_proj", "down_proj")
                )
                backward = sum(
                    gemm(phase, [rows], self.shape, self.hardware, 4).flops
                    for phase in ("act_grad", "gate_dx", "w1_grad", "w2_grad")
                )
                self.assertEqual(forward, 6 * rows * 16 * 16)
                self.assertEqual(backward, 2 * forward)

    def test_wave_expert_intersections_conserve_rows(self) -> None:
        """Overflow slices preserve expert row totals, including empty waves."""
        counts = make_counts(self.shape, 4, hot_fraction=1)
        plan = build_plan(counts, 0, self.shape.capacity)
        groups = wave_groups(plan, self.shape.capacity)
        self.assertGreater(len(groups), 1)
        self.assertEqual(sum(sum(sum(rank) for rank in wave) for wave in groups), 4 * 64 * 2)
        for rank in range(4):
            self.assertEqual(sum(sum(wave[rank]) for wave in groups), plan.destination_loads[rank])

    def test_release_and_alias_prevent_false_peak(self) -> None:
        """Two disjoint buffers never overlap, and aliases cannot be double-counted."""
        result = peak(
            [
                Storage("a", 100, 0, 2),
                Storage("b", 90, 2, 4),
                Storage("shared", 10, 0, 4),
            ]
        )
        self.assertEqual(result["bytes"], 110)
        with self.assertRaises(ValueError):
            peak([Storage("alias", 100, 0, 2), Storage("alias", 100, 1, 2)])

    def test_checkpoint_compute_and_traffic_conservation(self) -> None:
        """One checkpointed layer performs two forwards and one backward."""
        counts = make_counts(self.shape, 4, routing="balanced")
        result = predict(
            self.shape,
            counts,
            0,
            Fabric((0, 0, 1, 1)),
            self.hardware,
            Training(layers=1, steps=1),
            Options(optimizer="none"),
        )
        self.assertAlmostEqual(sum(result["critical_path"]["phase_ms"].values()), result["latency_ms"], places=7)
        work = result["phase_work_all_ranks"]
        flops = sum(row["flops"] for row in work.values())
        self.assertEqual(flops, 4 * 64 * 2 * 24 * 16 * 16)
        forward_bytes = result["traffic_bytes"]["forward_dispatch"]
        self.assertEqual(
            sum(forward_bytes[key] for key in ("inter", "intra", "local")),
            2 * 4 * 64 * 2 * 16 * 2,
        )

    def test_optimizer_and_window_are_explicit(self) -> None:
        """The warmed optimizer's resident state is distinguishable from live receives."""
        counts = make_counts(self.shape, 4, hot_fraction=1)
        plan = build_plan(counts, 1, self.shape.capacity)
        training = Training(layers=2, steps=1)
        bare = memory_ledger(self.shape, plan, training, Options(optimizer="none"))
        adam = memory_ledger(self.shape, plan, training, Options())
        self.assertGreater(adam["named_peak_bytes"], bare["named_peak_bytes"])
        tiny_budget = memory_ledger(self.shape, plan, training, Options(device_gib=0.01, reserve_gib=0))
        self.assertFalse(tiny_budget["fits_budget_with_reserve"])

    def test_roofline_scales_and_holdout_isolation(self) -> None:
        """Fit known service scaling and reject accidental holdout leakage."""
        measured = gemm("up_proj", [64], self.shape, self.hardware, 4).service_us * 2
        sample = {
            "split": "train",
            "shape": vars(self.shape),
            "phase": "up_proj",
            "expert_rows": 64,
            "compute_us": measured,
        }
        fitted = calibrate([sample], self.hardware, identity={"device": "analytic fixture"})
        self.assertEqual(fitted.phase_scales["up_proj"], 2)
        self.assertEqual(gemm("up_proj", [64], self.shape, fitted, 4).service_us, measured)
        with self.assertRaises(ValueError):
            calibrate(
                [{**sample, "split": "holdout"}],
                self.hardware,
                identity={"device": "fixture"},
            )
        self.assertAlmostEqual(evaluate([11, 18], [10, 20])["max_absolute_relative_error"], 0.1)

    def test_overlap_is_not_added(self) -> None:
        """Nested and asynchronous trace spans use an interval union."""
        self.assertEqual(union_us([(0, 10), (2, 4), (8, 12), (20, 25)]), 17)

    def test_row_curves_separate_gradient_dtype_and_step_boundaries(self) -> None:
        """Large-K service can differ by dtype without changing arithmetic work."""
        samples = []
        for rows, width, scale in ((64, 2, 1), (256, 2, 3), (64, 4, 2), (256, 4, 4)):
            samples.append(
                {
                    "split": "train",
                    "shape": vars(self.shape),
                    "phase": "w1_grad",
                    "expert_rows": rows,
                    "gradient_bytes": width,
                    "compute_us": gemm("w1_grad", [rows], self.shape, self.hardware, width).service_us * scale,
                }
            )
        fitted = calibrate(samples, self.hardware, identity={"device": "fixture"})
        for width, expected in ((2, 2), (4, 3)):
            baseline = gemm("w1_grad", [128], self.shape, self.hardware, width)
            predicted = gemm("w1_grad", [128], self.shape, fitted, width)
            self.assertAlmostEqual(predicted.service_us / baseline.service_us, expected)
            self.assertEqual(predicted.flops, baseline.flops)
        counts = make_counts(self.shape, 4, routing="balanced")
        result = predict(
            self.shape,
            counts,
            0,
            Fabric((0,) * 4),
            self.hardware,
            Training(layers=1, steps=2, checkpoint=False),
            Options(optimizer="none"),
        )
        self.assertAlmostEqual(result["step_ms"][0], result["step_ms"][1])
        self.assertAlmostEqual(sum(result["step_ms"]), result["latency_ms"])

    def test_home_only_runtime_preserves_grown_replica_pool(self) -> None:
        """Configured B does not force FP32 dW, but previously allocated pools persist."""
        counts = make_counts(self.shape, 4, routing="balanced")
        args = (self.shape, counts, 1, Fabric((0,) * 4), self.hardware, Training(layers=1, steps=1))
        cold = predict(*args, Options(optimizer="none"))
        warm = predict(*args, Options(optimizer="none", previous_replica_slots=1))
        self.assertEqual(cold["executed_slots"], 0)
        self.assertEqual(cold["copies"], 0)
        self.assertEqual(cold["latency_ms"], warm["latency_ms"])
        self.assertEqual(
            warm["memory"]["named_peak_bytes"] - cold["memory"]["named_peak_bytes"], sum(self.shape.weight_elements) * 6
        )

    def test_isolated_transfer_calibration_keeps_holdout_separate(self) -> None:
        """Known startup is recovered without changing GEMM or network rates."""
        samples = [
            {"split": "train", "base_us": 10, "measured_us": 24, "windows": 2},
            {"split": "train", "base_us": 100, "measured_us": 121, "windows": 3},
        ]
        fitted = calibrate_transfer_windows(samples, self.hardware, identity={"device": "fixture"})
        self.assertEqual(fitted.replica_gradient_window_us, 7)
        self.assertEqual(fitted.phase_scales, self.hardware.phase_scales)
        self.assertEqual(fitted.hbm_gbps, self.hardware.hbm_gbps)
        with self.assertRaises(ValueError):
            calibrate_transfer_windows([{**samples[0], "split": "holdout"}], fitted, identity={"device": "fixture"})

    def test_array_engine_matches_independent_fluid_engine(self) -> None:
        """Vectorized large-rank simulation preserves every event timestamp and byte."""
        rng = random.Random(32)
        fabric = Fabric((0, 0, 1, 1))
        for _ in range(12):
            tasks = []
            for index in range(30):
                dependencies = tuple(sorted(rng.sample(range(index), min(index, rng.randrange(3)))))
                tasks.append(
                    Task(
                        str(index),
                        "flow",
                        rng.randrange(4),
                        dependencies,
                        target=rng.randrange(4),
                        size_bytes=rng.randrange(1, 100000),
                    )
                )
            expected = EventSimulator(fabric, tasks).run()
            actual = ArrayEventSimulator(fabric, tasks).run()
            self.assertEqual(expected.traffic, actual.traffic)
            for first, second in zip(expected.finishes_us, actual.finishes_us):
                self.assertAlmostEqual(first, second, places=7)

    def test_invalid_parameters_fail_at_boundary(self) -> None:
        """Reject unphysical rates and invalid receive windows."""
        for kwargs in (
            {"hbm_gbps": 0},
            {"effective_tflops": float("nan")},
            {"descriptor_us": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Hardware(**kwargs)
        with self.assertRaises(ValueError):
            replace(Options(), gradient_receive_slots=0)
