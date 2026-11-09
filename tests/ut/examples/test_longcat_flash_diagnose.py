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
"""CPU regressions for customer diagnosis orchestration and all-rank acceptance."""

import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3] / "examples/longcat_flash_repro"
_TENSORS = ("output", "input_gradient", "probability_gradient", "gate_up_proj", "down_proj")


def _load(name: str, filename: str) -> object:
    """Load a standalone standard-library helper without importing accelerator code."""
    specification = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class TestLongcatDiagnosis(unittest.TestCase):
    """Exercise genuine false-pass and failure-recovery paths without NPU hardware."""

    def setUp(self) -> None:
        """Use a fresh output root and isolated standalone script imports."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.reports = _load("longcat_diagnose_reports_test", "diagnose_reports.py")
        with patch.dict("sys.modules", {"diagnose_reports": self.reports}):
            self.runner = _load("longcat_diagnose_test", "diagnose.py")
        launch = _load("longcat_diagnose_launch_test", "launch.py")
        with patch.dict("sys.modules", {"launch": launch}):
            self.probe = _load("longcat_diagnose_probe_test", "diagnose_probe.py")

    def _fixture(self, stage: str, failed: bool = False) -> list[Path]:
        """Write complete native worker receipts with optional numerical failures."""
        directory = self.root / stage
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for rank in range(1 if stage == "ep1" else 4):
            cases = self.reports.SEMANTIC_CASES if stage == "ep1" else {"push": True, "pull": True}
            rows = []
            for name, primary in cases.items():
                passed = primary and not failed
                comparison = {key: {"strict_pass": passed, "relative_l2": 0 if passed else 0.0045,
                                    "max_abs": 0 if passed else 0.1, "strict_bad": 0 if passed else 7}
                              for key in _TENSORS}
                rows.append({"case" if stage == "ep1" else "mode": name, "primary": primary,
                             "strict_pass": passed, "all_ranks_strict_pass": passed, "comparison": comparison})
            report = {"rank": rank, "native_root_matches": True, "shmem_after_close": {"reference_count": 0},
                      "records" if stage == "ep1" else "results": rows}
            path = directory / f"rank{rank}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            paths.append(path)
        return paths

    def test_device_ids_are_explicit_unique_and_stage_sized(self) -> None:
        """Reject duplicates, negative IDs and an insufficient EP device selection."""
        for value, stage in (("", "ep1"), ("-1", "ep1"), ("0,00,1,2", "all"), ("0,1", "ep4")):
            with self.subTest(value=value, stage=stage), self.assertRaises(ValueError):
                self.runner.parse_devices(value, stage)
        self.assertEqual(self.runner.parse_devices("7, 5,3,1", "all"), ["7", "5", "3", "1"])

    def test_literal_hf_failures_do_not_fail_primary_acceptance(self) -> None:
        """Preserve expected HF diagnostics while requiring all four primary cases."""
        self._fixture("ep1")
        result = self.reports.summarize_stage(self.root, "ep1", 0)
        self.assertEqual(result["status"], "PASS")
        receipt = {"stages": [result], "error": None}
        self.assertEqual(self.reports.write_summary(self.root, receipt), 0)
        self.assertIn("HF参考差异诊断", (self.root / "summary.txt").read_text(encoding="utf-8"))
        self.assertIn("0.450000%", (self.root / "summary.txt").read_text(encoding="utf-8"))

    def test_missing_rank_never_passes_even_with_zero_exit(self) -> None:
        """Three completed reports cannot establish four-rank precision acceptance."""
        self._fixture("ep4-fused")[-1].unlink()
        self.assertEqual(self.reports.summarize_stage(self.root, "ep4-fused", 0)["status"], "ERROR")

    def test_missing_cases_foreign_native_and_leaked_runtime_fail(self) -> None:
        """Require complete comparisons, this checkout and released SHMEM."""
        for mutation in ("case", "tensor", "native", "shmem"):
            with self.subTest(mutation=mutation):
                path = self._fixture("ep4-fused")[0]
                report = json.loads(path.read_text())
                if mutation == "case":
                    report["results"].pop()
                elif mutation == "tensor":
                    report["results"][0]["comparison"].pop("down_proj")
                elif mutation == "native":
                    report["native_root_matches"] = False
                else:
                    report["shmem_after_close"]["reference_count"] = 1
                path.write_text(json.dumps(report))
                self.assertEqual(self.reports.summarize_stage(self.root, "ep4-fused", 0)["status"], "ERROR")

    def test_failed_tensor_cannot_be_hidden_by_passing_row_or_exit(self) -> None:
        """Do not trust a summary boolean that disagrees with tensor verdicts."""
        path = self._fixture("ep4-fused")[1]
        report = json.loads(path.read_text())
        report["results"][0]["comparison"]["output"]["strict_pass"] = False
        path.write_text(json.dumps(report))
        result = self.reports.summarize_stage(self.root, "ep4-fused", 0)
        self.assertEqual(result["status"], "NUMERICAL_FAIL")

    def test_nonzero_worker_exit_with_passing_reports_is_an_error(self) -> None:
        """A runtime/provenance error after numerical reporting must not be ignored."""
        self._fixture("ep4-fused")
        self.assertEqual(self.reports.summarize_stage(self.root, "ep4-fused", 1)["status"], "ERROR")

    def test_partial_run_is_never_reported_as_pass(self) -> None:
        """A process killed between stages must leave a RUNNING receipt, not PASS."""
        self._fixture("ep1")
        receipt = {"stages": [self.reports.summarize_stage(self.root, "ep1", 0)],
                   "complete": False, "error": None}
        self.assertEqual(self.reports.write_summary(self.root, receipt), 2)
        self.assertEqual(receipt["status"], "RUNNING")

    def test_hf_followup_never_masks_original_fused_failure(self) -> None:
        """Run the real controller with mock workers and retain the original failure."""
        output = self.root / "new-run"
        calls = []

        def worker(command: list, env: dict, _log: Path) -> int:
            """Write complete synthetic reports while replacing actual process execution."""
            stage = Path(command[command.index("--output") + 1]).name
            calls.append((stage, env["ASCEND_RT_VISIBLE_DEVICES"]))
            previous, self.root = self.root, output
            try:
                self._fixture(stage, failed=stage == "ep4-fused")
            finally:
                self.root = previous
            return 1 if stage == "ep4-fused" else 0

        argv = ["diagnose.py", "--devices", "7,5,3,1", "--output", str(output), "--cann-env", str(self.root / "env.sh")]
        with (patch("sys.argv", argv), patch.object(self.runner, "_prepare", return_value={}),
              patch.object(self.runner, "_run", side_effect=worker), patch("sys.stdout", new_callable=io.StringIO)):
            self.assertEqual(self.runner.main(), 1)
        self.assertEqual(calls, [("ep1", "7"), ("ep4-fused", "7,5,3,1"), ("ep4-hf", "7,5,3,1")])
        self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "NUMERICAL_FAIL")

    def test_fresh_output_is_required_before_preparation(self) -> None:
        """Reject existing output instead of overwriting customer evidence."""
        argv = ["diagnose.py", "--devices", "0", "--stage", "ep1", "--output", str(self.root),
                "--cann-env", str(self.root / "env.sh")]
        with patch("sys.argv", argv), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            self.runner._arguments()

    def test_environment_script_paths_are_literal_and_support_spaces(self) -> None:
        """Source positional file arguments and preserve multiline environment values."""
        path = self.root / "customer environment.sh"
        path.write_text("export LONGCAT_TEST_VALUE='first\nsecond'\n")
        result = self.runner._source_environment(path, {}, self.root / "prepare.log")
        self.assertEqual(result["LONGCAT_TEST_VALUE"], "first\nsecond")

    def test_editable_verification_accepts_legacy_install_and_rejects_foreign_import(self) -> None:
        """Check actual isolated import behavior instead of assuming PEP660 metadata."""
        expected = str(self.root / "hyper_parallel/__init__.py")
        with patch.object(self.probe.subprocess, "check_output", return_value=expected) as imported:
            self.assertEqual(self.probe.verify_editable(self.root, expected), expected)
        self.assertIn("-I", imported.call_args.args[0])
        self.assertNotIn("PYTHONPATH", imported.call_args.kwargs["env"])
        foreign = str(self.root / "other_checkout/hyper_parallel/__init__.py")
        with (patch.object(self.probe.subprocess, "check_output", return_value=foreign),
              self.assertRaisesRegex(RuntimeError, "Editable install")):
            self.probe.verify_editable(self.root, expected)


if __name__ == "__main__":
    unittest.main()
