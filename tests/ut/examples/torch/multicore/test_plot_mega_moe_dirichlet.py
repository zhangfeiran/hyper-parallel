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
"""CPU-only validation of the three-curve MegaMoe plotting example."""

import copy
import json
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

from examples.torch.multicore.plot_mega_moe_dirichlet import (
    GIB,
    load_results,
    write_results,
)


class TestPlotMegaMoeDirichlet(unittest.TestCase):
    """Reject mismatched comparisons and keep memory accounting non-overlapping."""

    def setUp(self) -> None:
        """Own temporary fixtures until unittest finishes the current case."""
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.output = Path(contexts.enter_context(tempfile.TemporaryDirectory()))

    @staticmethod
    def _payload(variant):
        index = ("native", "b0", "b1").index(variant)
        loads = [16, 8, 8, 8, 8, 8, 4, 4]
        ranks = [
            {
                "peak_allocated_bytes": GIB + rank,
                "symmetric_heap_bytes": GIB // 2,
                "conservative_total_peak_bytes": GIB + rank + GIB // 2,
            }
            for rank in range(8)
        ]
        point = {
            "backend": "native" if variant == "native" else "mega_moe",
            "shape": {"local_num_tokens": 4, "hidden_size": 4, "intermediate_size": 8,
                      "num_experts": 16, "top_k": 2, "dtype": "bfloat16"},
            "topology": {"ep": 8, "world_size": 8},
            "route": {
                "profile": "dirichlet", "dirichlet_alpha": 0.1, "dirichlet_seed": 7,
                "expert_loads": [load // 2 for load in loads for _ in range(2)],
                "destination_loads": loads, "realized_destination_skew": 2.0,
                "receive_capacity": 12, "baseline_wave_count": 2,
                "expected_wave_count": 1 if variant == "b1" else 2,
                "physical_destination_skew": 1.0 if variant == "b1" else 2.0,
                "replica_slots_per_rank": int(variant == "b1"),
            },
            "rank_max_fwd_bwd_ms": {"samples": [10 - index * 2, 12 - index * 2, 14 - index * 2]},
            "warmup_steps": 2,
            "measured_steps": 3,
            "validation": {"gradients_present": True, "gradients_finite": True},
            "memory": {
                "per_rank": ranks,
                "rank_max": {"conservative_total_peak_bytes": ranks[-1]["conservative_total_peak_bytes"]},
                "workspace_by_field_rank0": {"diagnostic_view": 100 * GIB},
            },
        }
        return {
            "variant": variant, "kind": "performance", "points": [point],
            "source": {"git_head": "source-sha", "worker_sha256": "worker-sha",
                       "adapter_sha256": "adapter-sha", "native_artifacts": {"kernel.o": "kernel-sha"}},
            "configuration": {
                "replica_slots_per_rank": int(variant == "b1"),
                "compact_physical_weights": variant == "b1", "capacity_factor": 1.5,
                "warmup_steps": 2, "measured_steps": 3, "shuffle_seed": 1, "optimizer_step": False,
                "weight_refresh": "cached", "workspace_lifetime": "retained", "runtime_environment": {},
            },
        }

    def _write(self, variant, payload, suffix="a"):
        path = self.output / f"{variant}_{suffix}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _inputs(self):
        return {variant: [self._write(variant, self._payload(variant))] for variant in ("native", "b0", "b1")}

    def test_original_skew_is_shared_despite_replica_placement(self):
        """Match logical routes while allowing different physical skew and waves."""
        dataset = load_results(self._inputs())
        row = dataset.rows[0]
        self.assertEqual(row["original_skew"], 2.0)
        self.assertEqual(row["b0_planned_waves"], 2)
        self.assertEqual(row["b1_planned_waves"], 1)
        self.assertEqual(row["b1_planned_physical_skew"], 1.0)
        self.assertEqual(row["b1_latency_ms"], 8.0)
        self.assertAlmostEqual(row["b1_hbm_gib"], 1.5 + 7 / GIB)
        self.assertEqual(dataset.metadata["capacity_skew"], 1.5)

    def test_pool_latency_samples_but_keep_maximum_hbm(self):
        """Combine independent repeats without averaging away the memory peak."""
        inputs = self._inputs()
        for variant in inputs:
            payload = self._payload(variant)
            point = payload["points"][0]
            point["rank_max_fwd_bwd_ms"]["samples"] = [
                sample + 10 for sample in point["rank_max_fwd_bwd_ms"]["samples"]
            ]
            point["memory"]["per_rank"][-1]["peak_allocated_bytes"] += GIB
            point["memory"]["per_rank"][-1]["conservative_total_peak_bytes"] += GIB
            point["memory"]["rank_max"]["conservative_total_peak_bytes"] += GIB
            inputs[variant].append(self._write(variant, payload, "b"))
        row = load_results(inputs).rows[0]
        self.assertEqual(row["b1_latency_ms"], 13.0)
        self.assertEqual(row["b1_process_medians_ms"], [8, 18])
        self.assertEqual(row["b1_samples"], 6)
        self.assertAlmostEqual(row["b1_hbm_gib"], 2.5 + 7 / GIB)

    def _stack_inputs(self):
        inputs = {}
        for variant in ("native", "b0", "b1"):
            payload = self._payload(variant)
            payload["configuration"].update(
                layers=16, checkpoint="reentrant", scope="independent MoE layers",
                optimizer={"name": "AdamW", "lr": 1e-4, "weight_decay": 0.01, "foreach": False},
                compact_physical_weights=False, shared_physical_weights=variant == "b1",
                optimizer_step=True,
            )
            payload["points"][0].update(layers=16, checkpoint="reentrant")
            inputs[variant] = [self._write(variant, payload)]
        return inputs

    def test_shared_b_checkpointed_stack_keeps_measured_totals(self):
        """Treat a measured 16-layer stack as one sample, never multiply by 16."""
        dataset = load_results(self._stack_inputs())
        self.assertEqual(dataset.metadata["configuration"]["b1"]["layers"], 16)
        self.assertTrue(dataset.metadata["configuration"]["b1"]["shared_physical_weights"])
        self.assertEqual(dataset.rows[0]["b1_latency_ms"], 8.0)
        self.assertAlmostEqual(dataset.rows[0]["b1_hbm_gib"], 1.5 + 7 / GIB)

    def _train_step_inputs(self):
        inputs = self._stack_inputs()
        for variant, paths in inputs.items():
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            payload["configuration"]["latency_metric"] = "train_step"
            point = payload["points"][0]
            point["rank_max_train_step_ms"] = {
                "samples": [sample + 100 for sample in point["rank_max_fwd_bwd_ms"]["samples"]],
            }
            self._write(variant, payload)
        return inputs

    def test_plot_full_step_and_retain_fwd_bwd_for_export(self):
        """Full-step curves use direct totals while keeping F+B separately in CSV."""
        dataset = load_results(self._train_step_inputs())
        self.assertEqual(dataset.metadata["latency_metric"], "train_step")
        self.assertEqual(dataset.rows[0]["b1_latency_ms"], 108.0)
        self.assertEqual(dataset.rows[0]["b1_process_medians_ms"], [108.0])
        self.assertEqual(dataset.rows[0]["b1_fwd_bwd_latency_ms"], 8.0)
        self.assertAlmostEqual(dataset.rows[0]["b1_hbm_gib"], 1.5 + 7 / GIB)

    def test_reject_missing_or_shorter_full_step_samples(self):
        """Optimizer-inclusive records need valid direct intervals, not estimates."""
        for samples in ([], [7, 8, 9], [10, 11], [math.inf, 20, 30]):
            with self.subTest(samples=samples):
                inputs = self._train_step_inputs()
                payload = json.loads(inputs["native"][0].read_text(encoding="utf-8"))
                payload["points"][0]["rank_max_train_step_ms"]["samples"] = samples
                self._write("native", payload)
                with self.assertRaisesRegex(ValueError, "[Ll]atency"):
                    load_results(inputs)

    def test_reject_mixed_optimizer_timing_in_curves(self):
        """F+B and optimizer-inclusive variants cannot share a comparison curve."""
        inputs = self._train_step_inputs()
        payload = json.loads(inputs["b1"][0].read_text(encoding="utf-8"))
        payload["configuration"]["latency_metric"] = "fwd_bwd"
        self._write("b1", payload)
        with self.assertRaisesRegex(ValueError, "latency_metric"):
            load_results(inputs)

    def test_reject_mixed_layer_checkpoint_and_storage_contracts(self):
        """Different stack sizes or checkpoint lifetimes are not paired work."""
        for key, value in (("layers", 1), ("checkpoint", "none"), ("scope", "connected"),
                           ("optimizer", "SGD"), ("compact_physical_weights", True)):
            with self.subTest(key=key):
                inputs = self._stack_inputs()
                payload = json.loads(inputs["b1"][0].read_text(encoding="utf-8"))
                payload["configuration"][key] = value
                if key in ("layers", "checkpoint"):
                    payload["points"][0][key] = value
                self._write("b1", payload)
                with self.assertRaises(ValueError):
                    load_results(inputs)

    def test_require_post_update_validation_when_declared(self):
        """Do not plot a run that omits or fails its post-update finite checks."""
        inputs = self._stack_inputs()
        for variant, paths in inputs.items():
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            payload["configuration"]["validation_protocol"] = "before and after measurement"
            payload["points"][0]["validation"].update(post_measurement_finite=True, optimizer_states_finite=True)
            payload["points"][0]["optimizer_state_per_rank"] = [{"tensor_bytes": GIB}] * 8
            self._write(variant, payload)
        self.assertEqual(len(load_results(inputs).rows), 1)
        for field in ("post_measurement_finite", "optimizer_states_finite"):
            with self.subTest(field=field):
                payload = json.loads(inputs["b1"][0].read_text(encoding="utf-8"))
                payload["points"][0]["validation"][field] = False
                self._write("b1", payload)
                with self.assertRaisesRegex(ValueError, "after optimizer updates"):
                    load_results(inputs)
                payload["points"][0]["validation"][field] = True
                self._write("b1", payload)

    def test_reject_point_with_different_stack_metadata(self):
        """A point cannot silently inherit an incompatible sweep header."""
        inputs = self._stack_inputs()
        payload = json.loads(inputs["b0"][0].read_text(encoding="utf-8"))
        payload["points"][0]["layers"] = 1
        self._write("b0", payload)
        with self.assertRaisesRegex(ValueError, "Layer count or checkpoint"):
            load_results(inputs)

    def test_reject_duplicate_files_and_duplicate_routes(self):
        """A duplicated file or route must not inflate the sample count."""
        inputs = self._inputs()
        for variant in inputs:
            inputs[variant].append(inputs[variant][0])
        with self.assertRaisesRegex(ValueError, "distinct input files"):
            load_results(inputs)
        inputs = self._inputs()
        payload = self._payload("b0")
        payload["points"].append(copy.deepcopy(payload["points"][0]))
        self._write("b0", payload)
        with self.assertRaisesRegex(ValueError, "Duplicate alpha/seed"):
            load_results(inputs)

    def test_reject_missing_route_and_unequal_repeat_count(self):
        """All variants and repeats must measure the same route set."""
        inputs = self._inputs()
        inputs["b0"].append(self._write("b0", self._payload("b0"), "b"))
        with self.assertRaisesRegex(ValueError, "same number"):
            load_results(inputs)
        inputs = self._inputs()
        payload = self._payload("b0")
        payload["points"][0]["route"]["dirichlet_seed"] = 9
        self._write("b0", payload)
        with self.assertRaisesRegex(ValueError, "Route sets differ"):
            load_results(inputs)

    def test_reject_shape_and_logical_load_mismatch(self):
        """Matching alpha/seed labels alone are insufficient evidence of equal work."""
        for field in ("shape", "loads"):
            with self.subTest(field=field):
                inputs = self._inputs()
                payload = self._payload("b0")
                if field == "shape":
                    payload["points"][0]["shape"]["hidden_size"] = 8
                else:
                    payload["points"][0]["route"]["expert_loads"][0] += 1
                self._write("b0", payload)
                with self.assertRaisesRegex(ValueError, "Logical route or shape mismatch"):
                    load_results(inputs)

    def test_reject_mixed_source_binary_or_configuration(self):
        """Reject stale binaries, different commits and unequal timing protocols."""
        for field in ("git_head", "native_artifacts", "warmup_steps"):
            with self.subTest(field=field):
                inputs = self._inputs()
                payload = self._payload("b1")
                if field == "warmup_steps":
                    payload["configuration"][field] = 5
                    payload["points"][0][field] = 5
                else:
                    payload["source"][field] = "different"
                self._write("b1", payload)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    load_results(inputs)

    def test_reject_invalid_latencies_and_gradients(self):
        """Invalid numeric samples or unchecked gradients must never become curves."""
        for sample in (math.nan, math.inf, -1, 0, True):
            with self.subTest(sample=sample):
                inputs = self._inputs()
                payload = self._payload("native")
                payload["points"][0]["rank_max_fwd_bwd_ms"]["samples"][0] = sample
                self._write("native", payload)
                with self.assertRaisesRegex((TypeError, ValueError), "latency sample"):
                    load_results(inputs)
        inputs = self._inputs()
        payload = self._payload("b1")
        payload["points"][0]["validation"]["gradients_finite"] = False
        self._write("b1", payload)
        with self.assertRaisesRegex(ValueError, "validate gradients"):
            load_results(inputs)

    def test_reject_inconsistent_point_timing(self):
        """The plot caption must agree with each point's actual measurement length."""
        inputs = self._inputs()
        payload = self._payload("b0")
        payload["points"][0]["warmup_steps"] = 1
        self._write("b0", payload)
        with self.assertRaisesRegex(ValueError, "Timing record disagrees"):
            load_results(inputs)

    def test_reject_memory_double_count_and_rank_aggregation_error(self):
        """Check each rank's sum before taking the cross-rank maximum."""
        for field in ("per_rank", "rank_max"):
            with self.subTest(field=field):
                inputs = self._inputs()
                payload = self._payload("native")
                memory = payload["points"][0]["memory"]
                target = memory[field][0] if field == "per_rank" else memory[field]
                target["conservative_total_peak_bytes"] += GIB
                self._write("native", payload)
                with self.assertRaisesRegex(ValueError, "HBM"):
                    load_results(inputs)

    def test_reject_mixed_physical_storage_modes(self):
        """Independent B1 runs must agree on compact versus snapshot semantics."""
        inputs = self._inputs()
        for variant in inputs:
            payload = self._payload(variant)
            if variant == "b1":
                payload["configuration"]["compact_physical_weights"] = False
            inputs[variant].append(self._write(variant, payload, "b"))
        with self.assertRaisesRegex(ValueError, "Mixed source or configuration"):
            load_results(inputs)

    def test_export_without_matplotlib_and_keep_raw_process_medians(self):
        """CSV and provenance exports must remain usable without a renderer."""
        dataset = load_results(self._inputs())
        destination = self.output / "export"
        write_results(dataset, destination, render=False)
        summary = json.loads((destination / "dirichlet_skew_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["by_baseline_wave"]["2"]["points"], 1)
        self.assertTrue((destination / "dirichlet_skew_latency_hbm_points.csv").is_file())
        self.assertFalse((destination / "dirichlet_skew_latency_hbm.png").exists())


if __name__ == "__main__":
    unittest.main()
