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
"""Collect and audit sampled Ascend process ownership around exact measured steps.

The launcher requires an already idle-gated command. All clock arithmetic uses
one Linux host, boot and PID namespace. Sampling cannot rule out a short process
that starts and exits between observations; the audit preserves that limitation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import time
import uuid
from itertools import pairwise
from pathlib import Path

RUN_ID_ENV = "MEGAMOE_MEASUREMENT_RUN_ID"
CLOCK = "host_monotonic_ns"


def _process_stat(pid: int) -> tuple[int, int]:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    return int(fields[1]), int(fields[19])


def _host() -> dict:
    return {"hostname": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip(),
            "pid_namespace": os.readlink("/proc/self/ns/pid")}


def measurement_identity(rank: int, run_id: str) -> dict:
    """Read process and wall-clock anchors outside any latency or HBM measurement.

    Monotonic timestamps are exact inputs to durations and ownership joins. The
    realtime anchor is for approximate human-readable dates, not cross-host joins.
    """
    before = time.monotonic_ns()
    realtime = time.time_ns()
    after = time.monotonic_ns()
    return {"rank": rank, "local_rank": int(os.getenv("LOCAL_RANK", str(rank))), "run_id": run_id,
            "pid": os.getpid(), "start_ticks": _process_stat(os.getpid())[1], "host": _host(), "clock": CLOCK,
            "clock_anchor": {"monotonic_ns": (before + after) // 2, "realtime_ns": realtime,
                             "uncertainty_ns": after - before}}


def parse_processes(raw: str, devices: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Decode npu-smi process rows, rejecting missing device coverage or unknown output."""
    if "Process id" not in raw or "Health" not in raw:
        raise ValueError("Unrecognized npu-smi process table")
    rows = [tuple(map(int, row)) for row in re.findall(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|", raw, re.MULTILINE)]
    empty = {int(board) for board in re.findall(r"No running processes found in NPU (\d+)", raw)}
    reported = {(board, chip) for board, chip, _ in rows}
    if any((board, chip) not in reported and board not in empty for board, chip in devices):
        raise ValueError("Missing npu-smi process coverage for requested devices")
    return [row for row in rows if row[:2] in devices]


def _ownership(pid: int, root: tuple[int, int]) -> tuple[int | None, str]:
    """Do not treat exited, unreadable or recycled process identities as owned."""
    seen, start_ticks = set(), None
    for _ in range(32):
        if pid <= 1:
            return start_ticks, "foreign"
        if pid in seen:
            break
        seen.add(pid)
        try:
            parent, started = _process_stat(pid)
        except (OSError, ValueError, IndexError):
            break
        if start_ticks is None:
            start_ticks = started
        if (pid, started) == root:
            return start_ticks, "owned"
        pid = parent
    return start_ticks, "unknown"


def _snapshot(root: tuple[int, int], devices: list[tuple[int, int]]) -> dict:
    started = time.monotonic_ns()
    record = {"kind": "sample", "start_ns": started, "processes": []}
    try:
        record["raw"] = subprocess.check_output(["npu-smi", "info"], text=True, timeout=10)
        for board, chip, pid in parse_processes(record["raw"], devices):
            ticks, ownership = _ownership(pid, root)
            record["processes"].append({"board": board, "chip": chip, "pid": pid,
                                        "start_ticks": ticks, "ownership": ownership})
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record["error"] = str(exc)
    record["end_ns"] = time.monotonic_ns()
    return record


def _window_evidence(window: dict, identity: dict, samples: list[dict], device: list[int], gap: int) -> dict:
    left = [index for index, sample in enumerate(samples) if sample["end_ns"] <= window["start_ns"]]
    right = [index for index, sample in enumerate(samples) if sample["start_ns"] >= window["end_ns"]]
    row = {**window, "rank": identity["rank"], "pid": identity["pid"], "device": device,
           "status": "inconclusive", "reasons": []}
    if not left or not right:
        return {**row, "reasons": ["missing bracketing census"]}
    selected = samples[left[-1]:right[0] + 1]
    worst_gap = max([sample["end_ns"] - sample["start_ns"] for sample in selected] +
                    [later["end_ns"] - earlier["start_ns"] for earlier, later in pairwise(selected)])
    row.update(bracket_ns=[selected[0]["start_ns"], selected[-1]["end_ns"]],
               max_observation_gap_ns=worst_gap, census_indices=[left[-1], right[0]])
    if worst_gap > gap:
        row["reasons"].append("observation gap exceeds policy")
    if any("error" in sample for sample in selected):
        row["reasons"].append("census query failed")
    processes = [process for sample in selected for process in sample["processes"]]
    foreign = any(process["ownership"] == "foreign" for process in processes)
    if foreign:
        row["reasons"].append("foreign process observed")
    if any(process["ownership"] == "unknown" for process in processes):
        row["reasons"].append("unknown process ownership")
    matching = [process for process in processes if process["pid"] == identity["pid"] and
                [process["board"], process["chip"]] == device]
    if not matching:
        row["reasons"].append("worker was not observed")
    elif any(process["start_ticks"] != identity["start_ticks"] for process in matching):
        row["reasons"].append("worker PID identity mismatch")
    row["status"] = "contaminated" if foreign else "inconclusive" if row["reasons"] else "sampled_clean"
    return row


def _point_windows(point: dict, header: dict, samples: list[dict], gap: int) -> list[dict]:
    ranks = point["measurement_windows"]
    if sorted(row["identity"]["rank"] for row in ranks) != list(range(point["topology"]["world_size"])):
        raise ValueError("Missing or duplicate rank measurement windows")
    count = point["measured_steps"]
    if count <= 0:
        raise ValueError("No measured steps")
    if len({row["identity"]["local_rank"] for row in ranks}) != len(ranks):
        raise ValueError("Duplicate local rank device mapping")
    durations = {"fwd_bwd": [[] for _ in range(count)], "train_step": [[] for _ in range(count)]}
    windows = []
    for rank in ranks:
        identity = rank["identity"]
        if any(identity[key] != header[key] for key in ("host", "run_id", "clock")):
            raise ValueError("Measurement/census host, clock or run identity mismatch")
        local = identity["local_rank"]
        if not 0 <= local < len(header["devices"]):
            raise ValueError("Missing device mapping for local rank")
        if [window["sample"] for window in rank["samples"]] != list(range(count)):
            raise ValueError("Missing or duplicate measured steps")
        previous_end = -1
        for window in rank["samples"]:
            start, forward_end, end = (window[key] for key in ("start_ns", "fwd_bwd_end_ns", "end_ns"))
            if (any(not isinstance(value, int) or isinstance(value, bool) for value in (start, forward_end, end))
                    or start < 0
                    or not previous_end <= start < forward_end <= end):
                raise ValueError("Invalid measured step boundaries")
            previous_end = end
            durations["fwd_bwd"][window["sample"]].append((forward_end - start) / 1e6)
            durations["train_step"][window["sample"]].append((end - start) / 1e6)
            windows.append(_window_evidence(window, identity, samples, header["devices"][local], gap))
    for phase, values in durations.items():
        measured = point[f"rank_max_{phase}_ms"]["samples"]
        if len(measured) != count or any(not math.isclose(max(local), value, rel_tol=1e-6, abs_tol=1e-9)
                                         for local, value in zip(values, measured)):
            raise ValueError("Reported rank-max latency differs from measured windows")
    return windows


def audit_windows(result: dict, census: list[dict], *, max_gap_ns: int) -> dict:
    """Join exact local step boundaries to conservative census brackets.

    Args:
        result: One multi-layer benchmark result with all rank measurement windows.
        census: Header, raw timestamped queries and final child return code.
        max_gap_ns: Maximum worst-case interval between observations to accept.

    Returns:
        Per-window evidence and an eligibility decision under the sampling policy.
        Even sampled-clean windows do not prove the absence of short-lived jobs.
    """
    if max_gap_ns <= 0 or len(census) < 3:
        raise ValueError("Positive sampling policy and complete census required")
    header, footer = census[0], census[-1]
    if (header.get("kind") != "header" or header.get("schema_version") != 1 or header.get("clock") != CLOCK
            or footer.get("kind") != "footer" or footer["return_code"] != 0):
        raise ValueError("Unsupported or incomplete census / unsuccessful child")
    if header["helper_sha256"] != result["source"]["measurement_helper_sha256"]:
        raise ValueError("Measurement/census source identity mismatch")
    if not header["devices"] or len({tuple(device) for device in header["devices"]}) != len(header["devices"]):
        raise ValueError("Missing or duplicate census devices")
    samples, previous_end = census[1:-1], -1
    for sample in samples:
        if (sample.get("kind") != "sample" or
                any(not isinstance(sample[key], int) or isinstance(sample[key], bool)
                    for key in ("start_ns", "end_ns")) or
                not previous_end <= sample["start_ns"] <= sample["end_ns"]):
            raise ValueError("Invalid or unordered census interval")
        if any(process["ownership"] not in {"owned", "foreign", "unknown"} for process in sample["processes"]):
            raise ValueError("Unknown ownership classification")
        previous_end = sample["end_ns"]
    windows = [dict(window, point=index) for index, point in enumerate(result["points"])
               for window in _point_windows(point, header, samples, max_gap_ns)]
    rank_ends = {}
    for window in windows:
        if window["start_ns"] < rank_ends.get(window["rank"], -1):
            raise ValueError("Overlapping or unordered route measurements")
        rank_ends[window["rank"]] = window["end_ns"]
    foreign = any(process["ownership"] == "foreign" for sample in samples for process in sample["processes"])
    return {"schema_version": 1, "run_id": header["run_id"], "source": result["source"], "clock": CLOCK,
            "max_gap_policy_ns": max_gap_ns, "foreign_observed_during_run": foreign,
            "eligible": bool(windows) and not foreign and all(row["status"] == "sampled_clean" for row in windows),
            "windows": windows,
            "limitation": "Process sampling can miss short-lived jobs between observations; not continuous exclusion."}


def _run(args: argparse.Namespace) -> None:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    devices = [tuple(map(int, item.split(":"))) for item in args.devices.split(",")]
    if (not command or not math.isfinite(args.interval) or args.interval <= 0 or
            len(set(devices)) != len(devices) or any(len(item) != 2 or min(item) < 0 for item in devices)):
        raise ValueError("Command, positive interval and unique board:chip devices are required")
    run_id = uuid.uuid4().hex
    identity = measurement_identity(0, run_id)
    header = {"kind": "header", "schema_version": 1, "run_id": run_id, "host": identity["host"],
              "clock": CLOCK, "clock_anchor": identity["clock_anchor"], "devices": devices,
              "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interval_seconds": args.interval, "command": command}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:

        def write(record: dict) -> None:
            """Flush each observation so an interrupted job retains its evidence."""
            stream.write(json.dumps(record) + "\n")
            stream.flush()

        write(header)
        with subprocess.Popen(command, env=dict(os.environ, **{RUN_ID_ENV: run_id})) as process:
            while True:
                write(_snapshot((identity["pid"], identity["start_ticks"]), devices))
                try:
                    return_code = process.wait(timeout=args.interval)
                    break
                except subprocess.TimeoutExpired:
                    continue
        write(_snapshot((identity["pid"], identity["start_ticks"]), devices))
        write({"kind": "footer", "return_code": return_code})
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    """Run a census-wrapped command or audit one result without importing a backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", help="Sample processes around an already idle-gated command")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--devices", required=True, help="Physical board:chip pairs, ordered by LOCAL_RANK")
    run.add_argument("--interval", type=float, default=1.0)
    run.add_argument("command", nargs=argparse.REMAINDER)
    audit = commands.add_parser("audit", help="Validate measured windows against a census")
    audit.add_argument("result", type=Path)
    audit.add_argument("census", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--max-gap-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.action == "run":
        _run(args)
        return
    result = json.loads(args.result.read_text(encoding="utf-8"))
    census = [json.loads(line) for line in args.census.read_text(encoding="utf-8").splitlines()]
    report = audit_windows(result, census, max_gap_ns=int(args.max_gap_seconds * 1e9))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"eligible": report["eligible"], "windows": len(report["windows"]), "output": str(args.output)}))
    if not report["eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
