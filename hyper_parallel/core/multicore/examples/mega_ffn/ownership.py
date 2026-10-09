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
"""Framework-free, sampled device/process ownership for dense acceptance jobs."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path


def selected_devices(logical_id: int) -> set[tuple[int, int]]:
    """Resolve the same physical board/chip mapping used by the idle helper.

    Args:
        logical_id: Physical logical NPU ID before visibility remapping.
    """
    raw = subprocess.check_output(["npu-smi", "info", "-m"], text=True, timeout=10)
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 3 and all(field.isdecimal() for field in fields[:3]):
            board, chip, logical = map(int, fields[:3])
            if logical == logical_id:
                return {(board, chip)}
    raise RuntimeError(f"Cannot map logical NPU {logical_id} for ownership validation")


def process_rows(raw: str, selected: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Parse validated NPU process tables without accepting unknown output formats.

    Args:
        raw: Complete npu-smi info output.
        selected: Physical board/chip pairs reserved for this job.
    """
    if not re.search(r"\|\s*NPU\s+Chip\s*\|\s*Process id\s*\|", raw, re.IGNORECASE):
        raise ValueError("NPU process-table header is missing")
    rows = [(int(board), int(chip), int(pid)) for board, chip, pid in
            re.findall(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|", raw, re.MULTILINE)]
    boards = {int(board) for board in re.findall(r"^\|\s*(\d+)\s+\S+\s*\|", raw, re.MULTILINE)}
    if not {board for board, _ in selected}.issubset(boards):
        raise ValueError("Selected physical NPUs are missing from the status table")
    return [row for row in rows if row[:2] in selected]


def process_identity(pid: int, worker_pid: int) -> dict[str, object]:
    """Validate Linux process ancestry and retain start-time identity evidence.

    Args:
        pid: Device process to classify.
        worker_pid: Root PID of this fresh benchmark process.
    """
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    current, seen = pid, set()
    while current not in seen and current > 1 and current != worker_pid:
        seen.add(current)
        fields = Path(f"/proc/{current}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        current = int(fields[1])
    result = {"pid": pid, "owned": current == worker_pid, "start_ticks": stat[19],
              "cwd": os.readlink(f"/proc/{pid}/cwd"),
              "command": Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")}
    final_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    if final_stat[19] != stat[19]:
        raise OSError("Device process PID was reused during ownership validation")
    return result


def ownership_sample(worker_pid: int, selected: set[tuple[int, int]]) -> dict[str, object]:
    """Capture one ownership observation, rejecting unresolved device/process state.

    Args:
        worker_pid: Fresh benchmark process whose descendants may own selected NPUs.
        selected: Reserved physical board/chip pairs.
    """
    record = {"time": time.time(), "own": [], "foreign": [], "unresolved": [], "released": []}
    try:
        raw = subprocess.check_output(["npu-smi", "info"], text=True, timeout=10)
        record["raw"] = raw
        for board, chip, pid in process_rows(raw, selected):
            try:
                item = {"board": board, "chip": chip, **process_identity(pid, worker_pid)}
                record["own" if item["owned"] else "foreign"].append(item)
            except FileNotFoundError:
                recheck = subprocess.check_output(["npu-smi", "info"], text=True, timeout=10)
                remaining = process_rows(recheck, selected)
                if any(row[2] == pid for row in remaining):
                    record["unresolved"].append({"pid": pid, "error": "Device process has no readable Linux identity"})
                else:
                    record["released"].append({"pid": pid, "confirmed_absent": True})
            except OSError as error:
                record["unresolved"].append({"pid": pid, "error": str(error)})
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        record["unresolved"].append({"error": str(error)})
    record["host_loadavg"] = Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    record["host_cpu_ticks"] = [int(value) for value in
                                Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:9]]
    return record
