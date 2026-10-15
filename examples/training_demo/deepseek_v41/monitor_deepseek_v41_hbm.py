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
"""Sample node-local physical HBM without changing the training exit status."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from threading import Event

_HBM = re.compile(
    r"^\|\s*\d+\s+(\d+)\s*\|\s*[0-9A-Fa-f:.]+\s*\|"
    r"\s*\d+\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|", re.MULTILINE,
)
_PIDS = re.compile(r"^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|", re.MULTILINE)
_HEALTH = re.compile(r"\|\s*(OK|Warning|Alarm|Critical)\s*\|")
_STOP = Event()


def parse_snapshot(text: str, num_devices: int) -> dict:
    """Parse HBM, health and processes while preserving incomplete samples."""
    devices = {device: {"used_mib": int(used), "total_mib": int(total)}
               for device, used, total in _HBM.findall(text)}
    pids = sorted(set(map(int, _PIDS.findall(text))))
    health = _HEALTH.findall(text)
    valid = (len(devices) == num_devices and len(health) == num_devices
             and "Process id" in text and (bool(pids) or "No running processes found" in text))
    return {"devices": devices, "pids": pids, "health": health, "valid": valid}


def belongs_to_run(pid: int, root_pid: int, root_pgid: int) -> bool | None:
    """Classify a device process by its session or bounded parent chain."""
    try:
        if os.getpgid(pid) == root_pgid:
            return True
    except (PermissionError, ProcessLookupError):
        return None
    for _ in range(64):
        if pid == root_pid:
            return True
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return None
        pid = int(stat[stat.rfind(")") + 2:].split()[1])
        if pid <= 1:
            return False
    return False


def _stop(_signum: int, _frame: object) -> None:
    _STOP.set()


def main() -> None:
    """Keep failed samples and finalize quality independently of training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_devices < 1 or args.interval <= 0:
        parser.error("num-devices and interval must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    root_pgid = os.getpgid(args.root_pid)
    started_at = datetime.now(timezone.utc).isoformat()
    last_valid = time.monotonic()
    valid_count = failures = consecutive = maximum_consecutive = 0
    maximum_gap = 0.0
    owned_all, foreign_all, unresolved_all = set(), set(), set()
    peaks = {}
    unhealthy = set()
    with (args.output / "hbm_samples.jsonl").open("w") as stream:
        while not _STOP.is_set():
            completed = subprocess.run(["timeout", "5", "npu-smi", "info"],
                                       capture_output=True, text=True, check=False)
            raw = completed.stdout + completed.stderr
            sample = parse_snapshot(raw, args.num_devices)
            sample["valid"] = sample["valid"] and completed.returncode == 0
            now = time.monotonic()
            maximum_gap = max(maximum_gap, now - last_valid)
            if sample["valid"]:
                valid_count += 1
                last_valid = now
                consecutive = 0
            else:
                failures += 1
                consecutive += 1
                maximum_consecutive = max(maximum_consecutive, consecutive)
                with (args.output / "hbm_parse_errors.log").open("a") as errors:
                    errors.write(f"\nSample failure {failures}, returncode={completed.returncode}\n{raw}")
            owned, foreign, unresolved = [], [], []
            for pid in sample.pop("pids"):
                classification = belongs_to_run(pid, args.root_pid, root_pgid)
                if classification is None and pid in owned_all and not Path(f"/proc/{pid}").exists():
                    classification = True
                target = owned if classification is True else foreign if classification is False else unresolved
                target.append(pid)
            owned_all.update(owned)
            foreign_all.update(foreign)
            unresolved_all.update(unresolved)
            unresolved_all.difference_update(owned_all | foreign_all)
            unhealthy.update(health for health in sample["health"] if health != "OK")
            for device, memory in sample["devices"].items():
                peaks[device] = max(peaks.get(device, 0), memory["used_mib"])
            stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                     **sample, "owned_pids": owned, "foreign_pids": foreign,
                                     "unresolved_pids": unresolved}) + "\n")
            stream.flush()
            if sample["valid"] or consecutive >= 3:
                time.sleep(args.interval)
    maximum_gap = max(maximum_gap, time.monotonic() - last_valid)
    summary = {"started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
               "root_pid": args.root_pid, "root_pgid": root_pgid, "valid_samples": valid_count,
               "parse_failures": failures, "max_consecutive_failures": maximum_consecutive,
               "max_observation_gap_seconds": maximum_gap,
               "monitoring_sufficient": valid_count > 0 and maximum_consecutive < 3 and maximum_gap <= 30,
               "per_device_peak_mib": peaks, "max_device_peak_mib": max(peaks.values(), default=0),
               "owned_pids": sorted(owned_all), "foreign_pids": sorted(foreign_all),
               "unresolved_pids": sorted(unresolved_all), "unhealthy_states": sorted(unhealthy),
               "peak_interpretation": "Sampled physical peak; short peaks and foreign jobs may be missed"}
    pending = args.output / "hbm_summary.json.tmp"
    pending.write_text(json.dumps(summary, indent=2) + "\n")
    pending.replace(args.output / "hbm_summary.json")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    main()
