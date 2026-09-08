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
"""Reject unobserved, foreign or mismatched device ownership around measured steps."""

import copy
import unittest
from unittest.mock import patch

from examples.torch.multicore import mega_moe_measurement as measurement


def _fixture() -> tuple[dict, list[dict]]:
    """Use independent two-rank intervals with a census before and after the step."""
    host = {"hostname": "host", "boot_id": "boot", "pid_namespace": "pid:[1]"}
    ranks = [{"identity": {"rank": rank, "local_rank": rank, "pid": rank + 100,
                           "start_ticks": 9, "host": host, "run_id": "run", "clock": "host_monotonic_ns"},
              "samples": [{"sample": 0, "start_ns": 12, "fwd_bwd_end_ns": 16, "end_ns": 18}]}
             for rank in range(2)]
    result = {"source": {"measurement_helper_sha256": "digest"}, "points": [
        {"topology": {"world_size": 2}, "measured_steps": 1, "measurement_windows": ranks,
         "rank_max_fwd_bwd_ms": {"samples": [4e-6]}, "rank_max_train_step_ms": {"samples": [6e-6]}}
    ]}
    census = [{"kind": "header", "schema_version": 1, "host": host, "run_id": "run",
               "clock": "host_monotonic_ns", "devices": [[0, 0], [1, 0]], "helper_sha256": "digest"}]
    for start in (0, 10, 20, 30):
        census.append({"kind": "sample", "start_ns": start, "end_ns": start + 1,
                       "processes": [{"board": rank, "chip": 0, "pid": 100 + rank,
                                      "start_ticks": 9, "ownership": "owned"} for rank in range(2)]})
    census.append({"kind": "footer", "return_code": 0})
    return result, census


class TestMeasurementAudit(unittest.TestCase):
    """Clean means sampled evidence with an explicit bounded observation gap."""

    def test_clean_step_has_bracketing_queries_and_exact_rank_durations(self) -> None:
        """Preserve the queried interval and avoid claiming continuous observation."""
        result, census = _fixture()
        report = measurement.audit_windows(result, census, max_gap_ns=15)
        self.assertTrue(report["eligible"])
        self.assertEqual(len(report["windows"]), 2)
        self.assertEqual(report["windows"][0]["bracket_ns"], [10, 21])
        self.assertEqual(report["windows"][0]["max_observation_gap_ns"], 11)
        self.assertIn("sampling", report["limitation"])

    def test_missing_brackets_query_errors_and_long_gaps_are_inconclusive(self) -> None:
        """A successful child process does not fill missing ownership evidence."""
        result, census = _fixture()
        cases = [(census[:3] + census[-1:], 15), (census[:1] + census[3:], 15), (census, 5)]
        failed = copy.deepcopy(census)
        failed[2]["error"] = "npu-smi failed"
        cases.append((failed, 15))
        for records, gap in cases:
            with self.subTest(records=records, gap=gap):
                report = measurement.audit_windows(result, records, max_gap_ns=gap)
                self.assertFalse(report["eligible"])
                self.assertEqual(report["windows"][0]["status"], "inconclusive")

    def test_foreign_unknown_and_reused_pid_never_count_as_owned(self) -> None:
        """PID reuse and a disappearing process cannot silently become a clean sample."""
        for change in ({"ownership": "foreign"}, {"ownership": "unknown"}, {"start_ticks": 10}):
            result, census = _fixture()
            census[2]["processes"][0].update(change)
            with self.subTest(change=change):
                self.assertFalse(measurement.audit_windows(result, census, max_gap_ns=15)["eligible"])

    def test_unseen_rank_is_not_inferred_from_empty_device_rows(self) -> None:
        """Empty endpoints cannot establish the identity of an unobserved worker."""
        result, census = _fixture()
        for record in census[1:-1]:
            record["processes"] = record["processes"][1:]
        report = measurement.audit_windows(result, census, max_gap_ns=15)
        self.assertFalse(report["eligible"])
        self.assertIn("worker was not observed", report["windows"][0]["reasons"])

    def test_identity_clock_rank_coverage_and_reported_timing_must_match(self) -> None:
        """Reject malformed inputs rather than producing plausible aligned timelines."""
        for field, value in (("run_id", "other"), ("clock", "unix_ns"), ("host", {})):
            result, census = _fixture()
            result["points"][0]["measurement_windows"][0]["identity"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                measurement.audit_windows(result, census, max_gap_ns=15)
        for mutate in (
            lambda result: result["source"].update(measurement_helper_sha256="other"),
            lambda result: result["points"][0]["measurement_windows"].pop(),
            lambda result: result["points"][0]["rank_max_train_step_ms"].update(samples=[100]),
            lambda result: result["points"][0]["measurement_windows"][0]["samples"][0].update(end_ns=11),
        ):
            result, census = _fixture()
            mutate(result)
            with self.assertRaises(ValueError):
                measurement.audit_windows(result, census, max_gap_ns=15)

    def test_process_parser_requires_coverage_for_every_requested_device(self) -> None:
        """Missing device rows and unknown output formats fail closed."""
        raw = "Health\nProcess id\n| 0 0 | 123 | python |\n| No running processes found in NPU 1 |\n"
        self.assertEqual(measurement.parse_processes(raw, [(0, 0), (1, 0)]), [(0, 0, 123)])
        with self.assertRaises(ValueError):
            measurement.parse_processes(raw, [(2, 0)])
        with self.assertRaises(ValueError):
            measurement.parse_processes("command unavailable", [(0, 0)])

    def test_malformed_census_and_duplicate_route_windows_are_rejected(self) -> None:
        """Unsupported classifications and reused observations cannot certify a run."""
        for mutate in (
            lambda census: census[2]["processes"][0].update(ownership="probably_owned"),
            lambda census: census[2].update(start_ns=20.0),
            lambda census: census[0].update(devices=[[0, 0], [0, 0]]),
        ):
            result, census = _fixture()
            mutate(census)
            with self.assertRaises(ValueError):
                measurement.audit_windows(result, census, max_gap_ns=15)
        result, census = _fixture()
        result["points"].append(copy.deepcopy(result["points"][0]))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            measurement.audit_windows(result, census, max_gap_ns=15)

    def test_foreign_process_outside_step_still_rejects_whole_run_comparison(self) -> None:
        """Keep per-step attribution while conservatively rejecting a contested run."""
        result, census = _fixture()
        census[1]["processes"][0]["ownership"] = "foreign"
        report = measurement.audit_windows(result, census, max_gap_ns=15)
        self.assertFalse(report["eligible"])
        self.assertTrue(report["foreign_observed_during_run"])
        self.assertTrue(all(window["status"] == "sampled_clean" for window in report["windows"]))

    def test_process_ancestry_checks_root_start_ticks(self) -> None:
        """An unrelated process with a recycled launcher PID is still foreign."""
        rows = {12: (11, 8), 11: (1, 6)}
        with patch.object(measurement, "_process_stat", side_effect=rows.__getitem__):
            self.assertEqual(measurement._ownership(12, (11, 6)), (8, "owned"))
            self.assertEqual(measurement._ownership(12, (11, 7)), (8, "foreign"))
        with patch.object(measurement, "_process_stat", side_effect=FileNotFoundError):
            self.assertEqual(measurement._ownership(12, (11, 6)), (None, "unknown"))
