# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Synthetic evidence oracles for fresh-process dense performance acceptance."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hyper_parallel.core.multicore.examples.mega_ffn import acceptance, ownership
from tests.common.mark_utils import arg_mark


def _suite(root):
    jobs = acceptance.suite_jobs(3)
    for index, job in enumerate(jobs):
        result = root / f"result-{index}.json"
        owner = root / f"owner-{index}.json"
        record = {"complete": True, "mode": job["mode"], "backend": job["backend"],
                  "identity": {"source": "fixed", "device": "synthetic"},
                  "workload": {"tokens": "fixed", "config": {"num_layers": 2}}}
        payloads = {f"layers.{layer}.mlp": [{"native": {"execution_mode": "resident_dense_tile_candidate"}}]
                    for layer in range(2)} if job["backend"] == "mega_ffn" else {}
        if job["mode"] == "validate":
            checks = {name: {"passed": True} for name in
                      ("logits", "gradients", "parameters", "main", "exp_avg", "exp_avg_sq", "step", "loss")}
            record.update(baseline=job["baseline"], numerical_passed=True, expected_steps=100,
                          native_payloads={"baseline": {}, "candidate": payloads},
                          initial={name: {"passed": True} for name in
                                   ("parameters", "main", "exp_avg", "exp_avg_sq", "step")},
                          steps=[{"step": step + 1, "passed": True, "checks": checks} for step in range(100)])
        else:
            median = 80.0 if job["backend"] == "mega_ffn" else 100.0
            record.update(median_ms=median, samples_ms=[median], measurement_window=[0.5, 1.5],
                          native_payloads=payloads)
        result.write_text(json.dumps(record), encoding="utf-8")
        samples = [{"time": timestamp, "own": [{"pid": index + 100}], "foreign": [], "unresolved": [],
                    "host_cpu_ticks": [0, 0, 0, int(timestamp * 100), 0, 0, 0, 0]} for timestamp in (0.0, 1.0, 2.0)]
        owner.write_text(json.dumps({"clean": True, "exit_code": 0, "worker_pid": index + 100,
                                     "worker_start_ticks": str(index), "samples": samples}), encoding="utf-8")
        job.update(index=index, result=str(result), ownership=str(owner), exit_code=0)
    return {"complete": True, "blocks": 3, "jobs": jobs}


class TestDenseAcceptance(unittest.TestCase):
    """Reject partial, contaminated or noisy evidence before claiming improvements."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_independent_blocks_and_native_controls(self):
        """Feature: Complete-step statistical acceptance.
        Description: Compare flat native controls, a clear gain, no gain and systematic control drift.
        Expectation: Only a gain above a clean native noise envelope passes.
        """
        control = [[100, 100, 100, 100]] * 5
        gain = [[100, 80, 80, 100]] * 5
        self.assertTrue(acceptance.performance_verdict(control, gain)["performance_passed"])
        self.assertFalse(acceptance.performance_verdict(control, control)["performance_passed"])
        drift = [[100, 96, 96, 100]] * 5
        result = acceptance.performance_verdict(drift, gain)
        self.assertFalse(result["native_control_passed"])
        self.assertFalse(result["performance_passed"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_host_adapter_or_missing_layer_cannot_clear_resident_acceptance(self):
        """Feature: Actual execution identity before performance acceptance.
        Description: Substitute a host adapter or omit a candidate decoder's resident payload.
        Expectation: Good synthetic trajectories and timings still cannot pass the resident requirement.
        """
        for failure in ("host", "missing", "empty"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                suite = _suite(Path(directory))
                path = Path(suite["jobs"][1]["result"])
                record = json.loads(path.read_text(encoding="utf-8"))
                payloads = record["native_payloads"]["candidate"]
                if failure == "host":
                    payloads["layers.0.mlp"][0]["native"]["execution_mode"] = "native_host_stream_adapter"
                elif failure == "missing":
                    payloads.pop("layers.0.mlp")
                else:
                    payloads["layers.0.mlp"] = []
                path.write_text(json.dumps(record), encoding="utf-8")
                result = acceptance.analyze_suite(suite)
                self.assertTrue(result["numerical_passed"])
                self.assertFalse(result["resident_forward_present"])
                self.assertFalse(result["performance_passed"])
        for invalid in ([[100] * 4] * 2, [[100, float("inf"), 80, 100]] * 3, [[100, 80, 80]] * 3):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                acceptance.block_effects(invalid)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_complete_synthetic_suite_checks_both_baselines(self):
        """Feature: Fresh-process suite coverage.
        Description: Analyze complete synthetic evidence for both ordinary and packed dense baselines.
        Expectation: The analyzer verifies all jobs and reports the known twenty-percent step-time difference.
        """
        with tempfile.TemporaryDirectory() as directory:
            result = acceptance.analyze_suite(_suite(Path(directory)))
        self.assertTrue(result["numerical_passed"])
        self.assertTrue(result["performance_passed"])
        self.assertEqual(set(result["comparisons"]), {"common", "packed"})
        for comparison in result["comparisons"].values():
            self.assertAlmostEqual(comparison["comparison"]["step_time_saved_percent"], 20.0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_partial_or_contaminated_suite_rejects(self):
        """Feature: Ownership and source identity gates.
        Description: Remove a job, change source identity or retain foreign/unresolved device processes.
        Expectation: Invalid evidence never yields an accepted speedup.
        """
        for failure in ("partial", "foreign", "unresolved", "source", "host", "reused"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                suite = _suite(Path(directory))
                job = suite["jobs"][2]
                result_path, owner_path = Path(job["result"]), Path(job["ownership"])
                record = json.loads(result_path.read_text(encoding="utf-8"))
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
                if failure == "partial":
                    suite["jobs"].pop()
                elif failure in ("foreign", "unresolved"):
                    owner["samples"][0][failure] = [{"pid": 999}]
                elif failure == "source":
                    record["identity"]["source"] = "different"
                elif failure == "host":
                    owner["samples"][1]["host_cpu_ticks"] = [99, 0, 0, 1, 0, 0, 0, 0]
                else:
                    owner.update(worker_pid=100, worker_start_ticks="0")
                result_path.write_text(json.dumps(record), encoding="utf-8")
                owner_path.write_text(json.dumps(owner), encoding="utf-8")
                with self.assertRaises(ValueError):
                    acceptance.analyze_suite(suite)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_numerical_failure_blocks_performance_acceptance(self):
        """Feature: Numerical acceptance precedes performance.
        Description: Change one optimizer check while all synthetic timing results still show gains.
        Expectation: The final acceptance remains false.
        """
        with tempfile.TemporaryDirectory() as directory:
            suite = _suite(Path(directory))
            path = Path(suite["jobs"][1]["result"])
            record = json.loads(path.read_text(encoding="utf-8"))
            record["steps"][1]["checks"]["exp_avg"]["passed"] = False
            path.write_text(json.dumps(record), encoding="utf-8")
            result = acceptance.analyze_suite(suite)
        self.assertFalse(result["numerical_passed"])
        self.assertFalse(result["performance_passed"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_device_process_table_selection_and_missing_header(self):
        """Feature: Physical device ownership parsing.
        Description: Select one board/chip from a table containing another board's processes.
        Expectation: Only selected processes are considered and unknown output formats fail.
        """
        raw = "| 0 910B3 | OK |\n| 4 910B3 | OK |\n| NPU Chip | Process id | Process name |\n"
        raw += "| 0 0 | 1234 | python |\n| 4 0 | 4321 | python |\n"
        self.assertEqual(ownership.process_rows(raw, {(0, 0)}), [(0, 0, 1234)])
        with self.assertRaisesRegex(ValueError, "header"):
            ownership.process_rows("driver initialization failed", {(0, 0)})
        with self.assertRaisesRegex(ValueError, "missing"):
            ownership.process_rows(raw, {(7, 0)})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_linux_process_identity_is_owned_by_ancestry(self):
        """Feature: Fresh worker identity.
        Description: Inspect this process as its own root and as an unrelated root.
        Expectation: Ownership depends on ancestry and includes a stable start timestamp.
        """
        pid = os.getpid()
        own = ownership.process_identity(pid, pid)
        self.assertTrue(own["owned"])
        self.assertTrue(str(own["start_ticks"]).isdecimal())
        self.assertFalse(ownership.process_identity(pid, -1)["owned"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_missing_linux_identity_requires_confirmed_device_release(self):
        """Feature: Unresolved device process rejection.
        Description: Hide the Linux identity of a process still present in the selected NPU table.
        Expectation: It remains unresolved until a second device snapshot confirms release.
        """
        header = "| 0 910B3 | OK |\n| NPU Chip | Process id | Process name |\n"
        active = header + "| 0 0 | 1234 | python |\n"
        for recheck, released in ((active, False), (header, True)):
            with (self.subTest(released=released),
                  patch.object(ownership.subprocess, "check_output", side_effect=[active, recheck]),
                  patch.object(ownership, "process_identity", side_effect=FileNotFoundError())):
                sample = ownership.ownership_sample(1234, {(0, 0)})
            self.assertEqual(bool(sample["released"]), released)
            self.assertEqual(bool(sample["unresolved"]), not released)
