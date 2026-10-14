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
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from examples.training_demo.deepseek_v41.benchmark_deepseek_v41_vlm import _RouteDiagnostics
from examples.training_demo.deepseek_v41.monitor_deepseek_v41_hbm import parse_snapshot
from examples.training_demo.deepseek_v41.summarize_deepseek_v41_cluster import (
    _physical_hbm, summarize, summarize_matrix,
)
import hyper_parallel.core.expert_parallel.hot_replica.native as native_replica_module


class TestRouteDiagnostics(unittest.TestCase):
    """Observe immutable host plans without changing planner results or weights."""

    def test_records_conserved_rows_and_restores_planner(self) -> None:
        model = torch.nn.Module()
        model.config = SimpleNamespace(num_experts_per_tok=2)
        model.layer = torch.nn.Module()
        model.layer.mlp = torch.nn.Sequential(torch.nn.Identity())
        model.layer.mlp.gate = torch.nn.Identity()
        model.layer.mlp.experts = torch.nn.Identity()
        original = native_replica_module.build_expert_replica_plan
        diagnostics = _RouteDiagnostics(model, True)
        try:
            module = model.layer.mlp
            module(torch.ones(2, 4))
            plan = native_replica_module.build_expert_replica_plan([[2, 2, 0, 0], [2, 2, 0, 0]], 2)
            row = diagnostics.snapshot()["layer.mlp"]
            self.assertEqual(row["source_tokens_by_ep_rank"], [2, 2])
            self.assertEqual(row["home_destination_rows"], [8, 0])
            self.assertEqual(sum(row["balanced_destination_rows"]), 8)
            self.assertEqual(row["balanced_destination_rows"], list(plan.destination_loads))
            self.assertGreater(row["weight_transfers"], 0)
            self.assertLessEqual(max(row["active_guest_slots_by_ep_rank"]), 2)
            diagnostics.clear()
            self.assertEqual(diagnostics.snapshot(), {})
        finally:
            diagnostics.close()
        self.assertIs(native_replica_module.build_expert_replica_plan, original)

    def test_recompute_attribution_does_not_use_last_forward_hook(self) -> None:
        model = torch.nn.Module()
        model.config = SimpleNamespace(num_experts_per_tok=2)
        for name in ("first", "second"):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Sequential()
            layer.mlp.gate = torch.nn.Identity()
            layer.mlp.experts = torch.nn.Identity()
            model.add_module(name, layer)
        diagnostics = _RouteDiagnostics(model, True)

        def recompute(module: torch.nn.Module) -> None:
            """Observe a plan without relying on forward-hook ordering."""
            self.assertIs(module, model.first.mlp)
            native_replica_module.build_expert_replica_plan([[2, 2, 0, 0], [2, 2, 0, 0]], 2)

        try:
            model.first.mlp(torch.ones(2, 4))
            model.second.mlp(torch.ones(3, 4))
            recompute(model.first.mlp)
            self.assertEqual(list(diagnostics.snapshot()), ["first.mlp"])
        finally:
            diagnostics.close()


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
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
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
