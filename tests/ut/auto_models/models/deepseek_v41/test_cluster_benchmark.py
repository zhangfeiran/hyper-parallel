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
"""CPU acceptance for route observations and independent HBM eligibility."""

from datetime import datetime, timedelta, timezone
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
import yaml

from examples.training_demo.deepseek_v41 import benchmark_deepseek_v41_profile as profile
from examples.training_demo.deepseek_v41.benchmark_deepseek_v41_vlm import _recipe

from examples.training_demo.deepseek_v41.benchmark_deepseek_v41_vlm import _RouteDiagnostics
from examples.training_demo.deepseek_v41.monitor_deepseek_v41_hbm import parse_snapshot
from examples.training_demo.deepseek_v41.summarize_deepseek_v41_cluster import (
    _physical_hbm, summarize, summarize_matrix,
)


class TestRouteDiagnostics(unittest.TestCase):
    """Record source rows on native and MegaMoe without a replica dependency."""

    def test_records_actual_rows_and_removes_hooks(self) -> None:
        """Changing local token lengths must be reflected without altering outputs."""
        model = torch.nn.Module()
        model.layer = torch.nn.Module()
        model.layer.mlp = torch.nn.Sequential(torch.nn.Identity())
        model.layer.mlp.gate = torch.nn.Identity()
        diagnostics = _RouteDiagnostics(model)
        for rows in (0, 7, 13):
            inputs = torch.ones(rows, 4)
            self.assertIs(model.layer.mlp(inputs), inputs)
            self.assertEqual(diagnostics.token_rows, {"layer.mlp": rows})
            diagnostics.clear()
            self.assertEqual(diagnostics.token_rows, {})
        diagnostics.close()
        model.layer.mlp(torch.ones(3, 4))
        self.assertEqual(diagnostics.token_rows, {})


class TestClusterRecipe(unittest.TestCase):
    """Parse real recipes for every modality and transport without devices."""

    def test_host_recipes_preserve_dynamic_tokens_and_precision(self) -> None:
        """All backends share dense precision, full decoder replay and Host selection."""
        with TemporaryDirectory() as directory:
            work = Path(directory)
            for modality in ("text", "vlm"):
                for backend, mode in (("owner_ep", "push"), ("megamoe", "push"), ("megamoe", "pull")):
                    for packing in ("dynamic", "packed"):
                        with self.subTest(modality=modality, backend=backend, mode=mode, packing=packing):
                            args = SimpleNamespace(
                                model_dir=directory, engram_assets=str(work / "engram.json"), experts=16,
                                modality=modality, packing_mode=packing, steps=3, empty_cache_steps=1,
                                random_init_seed=42, activation_checkpoint="full", ep_size=8,
                                edp_shard_size=1, data=str(work / "input.jsonl"), validation_crop=True,
                                num_hidden_layers=4, vision_num_hidden_layers=1, full_model=False,
                                engram_storage_backend="host", backend=backend, dispatch_mode=mode,
                                capacity_factor=None,
                            )
                            _recipe(args, work, 8)
                            recipe = yaml.safe_load((work / "recipe.yaml").read_text())
                            self.assertEqual(recipe["model_init_dtype"], "float32")
                            self.assertEqual(recipe["model"]["root_owned_vision"], modality == "vlm")
                            self.assertEqual(recipe["fsdp_config"]["mix_precision"]["reduce_dtype"], "float32")
                            self.assertFalse(recipe["fsdp_config"]["mix_precision"]["cast_forward_inputs"])
                            self.assertTrue(recipe["optimizer"]["fp32_main_params"])
                            self.assertEqual(recipe["activation_checkpoint"]["selection"]["layer_count"], 4)
                            replacements = [entry["replace_module"] for entry in recipe["plan_overrides"]
                                            if "replace_module" in entry]
                            self.assertTrue(any(item["_target_"].endswith("HostEngramModule")
                                                for item in replacements))
                            if backend == "megamoe":
                                moe = next(item for item in replacements
                                           if item["_target_"].endswith("DeepseekV41MegaMoeExperts"))
                                self.assertEqual(moe["local_num_tokens"], 4096)
                                self.assertNotIn("replica_slots_per_rank", moe)
                            if modality == "text":
                                self.assertIsNone(recipe["model"]["vision_num_hidden_layers"])
                            else:
                                expected = 1 if packing == "dynamic" else 4096
                                self.assertEqual(recipe["dataloader"]["token_budget"], expected)

    def test_profiling_preserves_scheduler_horizon(self) -> None:
        """Diagnostic traces use the requested horizon and remain rank-selective."""
        with TemporaryDirectory() as directory:
            work = Path(directory)
            config = SimpleNamespace(training=SimpleNamespace(), profiling=SimpleNamespace())
            args = SimpleNamespace(steps=40, output=str(work / "packed/push"))
            with patch.object(profile, "ORIGINAL_RECIPE", return_value=config), \
                    patch.dict("os.environ", {"PROFILE_START_STEP": "3", "PROFILE_END_STEP": "5",
                                              "SCHEDULE_TRAIN_ITERS": "40"}):
                profile.recipe(args, work, 8)
            self.assertEqual(config.training.train_iters, 40)
            self.assertEqual((config.profiling.start_step, config.profiling.end_step), (3, 5))
            evidence = json.loads((work / "profiling-config.json").read_text())
            self.assertEqual(evidence["purpose"], "performance_diagnosis_not_speedup_acceptance")


class TestHbmParser(unittest.TestCase):
    """Do not promote partial or empty command output to valid HBM evidence."""

    def test_complete_idle_and_busy_snapshots(self) -> None:
        rows = ("| 0 Ascend910 | OK | 100 |\n"
                "| 0 0 | 0000:9D:00.0 | 0 0 / 0 3000 / 65536 |\n"
                "| Process id |\n")
        idle = parse_snapshot(rows + "No running processes found", 1)
        busy = parse_snapshot(rows + "| 0 0 | 123 | python | 100 |\n", 1)
        self.assertTrue(idle["valid"])
        self.assertTrue(busy["valid"])
        self.assertEqual(busy["pids"], [123])
        self.assertEqual(idle["devices"]["0"]["used_mib"], 3000)
        self.assertFalse(parse_snapshot(rows, 1)["valid"])
        self.assertFalse(parse_snapshot(rows + "No running processes found", 2)["valid"])
        self.assertFalse(parse_snapshot("", 1)["valid"])


class TestClusterSummary(unittest.TestCase):
    """Measured statistics exclude warmup and reject incomplete rank sets."""

    def setUp(self) -> None:
        """Create two-rank training and monitor artifacts for each backend."""
        self.root = Path(mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        start = datetime(2026, 10, 14, tzinfo=timezone.utc)
        for label in ("native", "push", "pull"):
            for rank in range(2):
                folder = self.root / label / f"rank{rank}"
                folder.mkdir(parents=True)
                rows = [{"step": index + 1, "phase": "warmup" if index == 0 else "measured",
                         "started_at": (start + timedelta(seconds=2 * index)).isoformat(),
                         "finished_at": (start + timedelta(seconds=2 * index + 2)).isoformat(),
                         "local_seconds": 100.0 if index == 0 else 1.0 + rank * 0.5,
                         "max_rank_seconds": 100.0 if index == 0 else 1.5,
                         "loss": 12.0, "grad_norm": 3.0, "moe_input_rows": {"layer.mlp": 2},
                         "replica_routes": {}} for index in range(3)]
                result = {"rank": rank, "world_size": 2, "steps_detail": rows, "finite": True,
                          "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
                          "measurement": {"peak_allocated_bytes": 90, "peak_reserved_bytes": 190},
                          "shmem_state": None}
                (folder / "result.json").write_text(json.dumps(result))
                (folder / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                (folder / "initialization.json").write_text(json.dumps({"seed": 42, "rank": rank}))
            folder = self.root / "nodes/node_0" / label
            folder.mkdir(parents=True)
            (folder / "case_complete.json").write_text(json.dumps({"training_exit_status": 0}))
            (folder / "monitor_exit_status.txt").write_text("0\n")
            quality = {"monitoring_sufficient": True, "foreign_pids": [], "unresolved_pids": [],
                       "unhealthy_states": [], "max_device_peak_mib": 4000}
            (folder / "hbm_summary.json").write_text(json.dumps(quality))
            samples = [{"time": (start + timedelta(seconds=second)).isoformat(), "valid": True,
                        "devices": {"0": {"used_mib": 3000}}} for second in range(3, 6)]
            (folder / "hbm_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))

    def test_measured_window_and_clean_hbm(self) -> None:
        report = summarize(self.root, 1, 2, 3, 1)
        self.assertEqual(report["cases"]["native"]["mean_seconds"], 1.5)
        self.assertEqual(report["cases"]["native"]["measured_physical_peak_mib"], 3000)
        self.assertTrue(report["comparisons"]["push_vs_native"]["performance_eligible"])

    def test_monitor_failure_does_not_hide_completed_training(self) -> None:
        (self.root / "nodes/node_0/push/monitor_exit_status.txt").write_text("1\n")
        report = summarize(self.root, 1, 2, 3, 1)
        self.assertEqual(report["cases"]["push"]["completed_ranks"], 2)
        self.assertFalse(report["comparisons"]["push_vs_native"]["performance_eligible"])

    def test_missing_rank_cannot_be_reported_complete(self) -> None:
        (self.root / "pull/rank1/result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            summarize(self.root, 1, 2, 3, 1)

    def test_different_token_workloads_are_not_performance_eligible(self) -> None:
        folder = self.root / "push/rank1"
        result = json.loads((folder / "result.json").read_text())
        result["steps_detail"][1]["moe_input_rows"]["layer.mlp"] = 3
        (folder / "result.json").write_text(json.dumps(result))
        (folder / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in result["steps_detail"]))
        report = summarize(self.root, 1, 2, 3, 1)
        self.assertFalse(report["comparisons"]["push_vs_native"]["actual_token_rows_match"])
        self.assertFalse(report["comparisons"]["push_vs_native"]["performance_eligible"])

    def test_long_blind_tail_rejects_measured_hbm(self) -> None:
        start = datetime(2026, 10, 14, tzinfo=timezone.utc)
        record = _physical_hbm(self.root / "nodes/node_0/native", start.isoformat(),
                               (start + timedelta(seconds=65)).isoformat())
        self.assertFalse(record["eligible"])
        self.assertEqual(record["measured_max_gap_seconds"], 60)


    def test_partial_result_keeps_successful_native_push_pair(self) -> None:
        (self.root / 'pull/rank1/result.json').unlink()
        report = summarize(self.root, 1, 2, 3, 1, allow_partial=True)
        self.assertFalse(report['complete'])
        self.assertTrue(report['comparisons']['push_vs_native']['available'])
        self.assertFalse(report['comparisons']['pull_vs_native']['available'])

    def test_matrix_publishes_dynamic_when_packed_is_missing(self) -> None:
        with TemporaryDirectory() as directory:
            matrix = Path(directory)
            shutil.copytree(self.root, matrix / 'dynamic')
            report = summarize_matrix(matrix, 1, 2, 3, 1)
            self.assertTrue(report['groups']['dynamic']['complete'])
            self.assertFalse(report['groups']['packed']['complete'])
            self.assertTrue((matrix / 'dynamic/summary.json').exists())
            self.assertEqual(report['case_order'], [f'{mode}/{label}' for mode in ('dynamic', 'packed')
                                                    for label in ('native', 'push', 'pull')])

    def test_moe_sum_takes_rank_max_after_summing_layers(self) -> None:
        for label in ('native', 'push', 'pull'):
            for rank in range(2):
                folder = self.root / label / f'rank{rank}'
                result = json.loads((folder / 'result.json').read_text())
                scale = 2 if label == 'native' else 1
                for row in result['steps_detail']:
                    row['moe_timing'] = {'incomplete_intervals': [], 'layers': {
                        layer: {'forward': {'current_stream_ms': scale * value, 'calls': 1, 'completed_calls': 1}}
                        for layer, value in (('first.mlp', 10 if rank == 0 else 1),
                                             ('second.mlp', 1 if rank == 0 else 10))}}
                (folder / 'result.json').write_text(json.dumps(result))
                (folder / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in result['steps_detail']))
        report = summarize(self.root, 1, 2, 3, 1)
        timing = report['cases']['native']['moe_timing']
        self.assertEqual(timing['phase_sums']['forward']['mean_ms'], 22)
        self.assertEqual(timing['layers']['first.mlp']['forward']['mean_ms'], 20)
        self.assertEqual(report['comparisons']['push_vs_native']['moe_current_stream_speed_ratios']['forward'], 2)


    def test_corrupt_hbm_does_not_erase_successful_training(self) -> None:
        (self.root / 'nodes/node_0/push/hbm_summary.json').write_text('{broken')
        report = summarize(self.root, 1, 2, 3, 1)
        self.assertEqual(report['cases']['push']['status'], 'complete')
        self.assertFalse(report['cases']['push']['hbm_eligible'])
        self.assertIn('Invalid HBM artifacts', report['cases']['push']['node_hbm'][0]['reason'])
