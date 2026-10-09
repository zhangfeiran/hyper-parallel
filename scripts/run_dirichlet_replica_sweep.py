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
"""Run four fresh-process Dirichlet series and record physical HBM/ownership."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

VARIANTS = (("native", 0), ("native", 1), ("megamoe", 0), ("megamoe", 1))


def _process(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return {"pid": pid, "ppid": int(fields[1]), "start_ticks": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "missing": True}


def _owned(pid, launcher):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == launcher:
            return True
        seen.add(pid)
        process = _process(pid)
        if process.get("missing"):
            return False
        pid = process["ppid"]
    return False


def parse_snapshot(raw: str, launcher: int, known_owned: dict | None = None) -> dict:
    """Parse both chips per board, retaining process ownership and physical HBM."""
    processes, chips = [], []
    board = None
    for line in raw.splitlines():
        header = re.search(r"\|\s*(\d+)\s+Ascend\S*\s*\|\s*(\S+)", line)
        if header:
            board = int(header[1])
        memory = re.search(r"\|\s*(\d+)\s+(\d+)\s*\|\s*([\w:.]+)\s*\|\s*(\d+)\s+"
                           r"\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|", line)
        if memory and board is not None:
            chips.append({"board": board, "chip": int(memory[1]), "physical_id": int(memory[2]),
                          "aicore_percent": int(memory[4]), "hbm_mb": int(memory[5]),
                          "capacity_mb": int(memory[6])})
        process = re.search(r"\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*(\d+)\s*\|", line)
        if process:
            pid = int(process[3])
            identity = _process(pid)
            owned = _owned(pid, launcher)
            previous = known_owned.get(pid) if known_owned else None
            if not owned and previous and (identity.get("missing") or
                                            identity.get("start_ticks") == previous.get("start_ticks")):
                identity.update(previous, ownership_from_prior_snapshot=True)
                owned = True
            processes.append({"board": int(process[1]), "chip": int(process[2]), "pid": pid,
                              "memory_mb": int(process[5]), "owned": owned, **identity})
    return {"chips": chips, "processes": processes, "foreign_pids": sorted({item["pid"] for item in processes
                                                                                  if not item["owned"]}),
            "healthy": "Alert" not in raw and "ERROR" not in raw and len(chips) == 16 and raw.count("OK") == 16}


def _monitor(process, directory, stop):
    known_owned = {}
    with (directory / "npu-monitor.jsonl").open("w", encoding="utf-8") as output:
        while not stop.is_set():
            started = time.monotonic_ns()
            snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15, check=False)
            record = {"monotonic_ns": started, "end_ns": time.monotonic_ns(), "wall_time": time.time(),
                      "returncode": snapshot.returncode, "raw": snapshot.stdout + snapshot.stderr,
                      **parse_snapshot(snapshot.stdout, process.pid, known_owned)}
            known_owned.update({item["pid"]: {key: item[key] for key in ("pid", "ppid", "start_ticks")}
                                for item in record["processes"] if item["owned"] and "start_ticks" in item})
            output.write(json.dumps(record) + "\n")
            output.flush()
            stop.wait(1)


def _complete(directory, command=None):
    terminal_path = directory / "terminal.json"
    if not terminal_path.exists():
        return False
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal["returncode"] or not terminal.get("monitor_clean"):
        return False
    if command is not None and terminal.get("command") != command:
        return False
    for rank in range(16):
        if not (directory / f"rank{rank}" / "complete.json").is_file():
            return False
        identity = json.loads((directory / f"rank{rank}" / "identity.json").read_text(encoding="utf-8"))
        source_root = Path(identity["root"])
        files = {source_root / name: digest for name, digest in identity["source_sha256"].items()}
        files.update({Path(item["path"]): item["sha256"] for item in identity["measurement_scripts"].values()})
        if "adapter" in identity:
            files.update({Path(item["path"]): item["sha256"] for item in [identity["adapter"], *identity["kernels"]]})
        if any(not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest
               for path, digest in files.items()):
            return False
    return True


def _run(root, name, backend, budget, extra, resume=False, transport="p2p"):
    directory = root / name
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=16",
               "--module", "tests.torch.expert_parallel._benchmark_dirichlet_replica",
               "--backend", backend, "--budget", str(budget), "--result-dir", str(directory), *extra]
    if backend == "megamoe":
        command.extend(("--replica-transport", transport))
    if directory.exists() and resume:
        if _complete(directory, command):
            print(f"REUSE clean, source-matched {name}", flush=True)
            return
        directory.rename(root / f"{name}.discarded-{time.time_ns()}")
    directory.mkdir()
    metadata = {"command": command, "started": datetime.now(timezone.utc).isoformat(), "name": name}
    (directory / "command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"START {name}", flush=True)
    with (directory / "worker.log").open("w", encoding="utf-8") as output:
        with subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True) as process:
            metadata["launcher_pid"] = process.pid
            stop = threading.Event()
            monitor = threading.Thread(target=_monitor, args=(process, directory, stop), daemon=True)
            monitor.start()
            result = process.wait()
            stop.set()
            monitor.join(timeout=20)
    metadata.update(returncode=result, ended=datetime.now(timezone.utc).isoformat())
    monitor_lines = (directory / "npu-monitor.jsonl").read_text(encoding="utf-8").splitlines()
    snapshots = [json.loads(line) for line in monitor_lines]
    metadata["foreign_pids"] = sorted({pid for snapshot in snapshots for pid in snapshot["foreign_pids"]})
    metadata["monitor_clean"] = bool(snapshots) and not metadata["foreign_pids"] and all(
        snapshot["healthy"] and snapshot["returncode"] == 0 for snapshot in snapshots)
    (directory / "terminal.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if result:
        raise RuntimeError(f"{name} exited {result}; see {directory / 'worker.log'}")
    if not metadata["monitor_clean"]:
        raise RuntimeError(f"{name} physical monitoring was contaminated or unhealthy; discard this attempt")
    missing = [rank for rank in range(16) if not (directory / f"rank{rank}" / "complete.json").is_file()]
    if missing:
        raise RuntimeError(f"{name} incomplete ranks: {missing}")
    print(f"COMPLETE {name}", flush=True)


def main() -> None:
    """Run acceptance then mirrored fresh-process sweeps under an external idle gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "accept", "sweep", "full"), required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse only clean complete runs with matching sources.")
    names = [f"{backend}-b{budget}" for backend, budget in VARIANTS]
    parser.add_argument("--variants", choices=names, nargs="+", default=names)
    parser.add_argument("--replica-transport", choices=("p2p", "shmem_signal_kernel_gradient",
                                                     "shmem_signal_kernel_gradient_adaptive"), default="p2p")
    args = parser.parse_args()
    variants = tuple((backend, budget) for backend, budget in VARIANTS
                     if f"{backend}-b{budget}" in args.variants)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if args.phase == "smoke":
        for backend, budget in variants:
            _run(root, f"smoke-{backend}-b{budget}", backend, budget,
                 ["--layers", "2", "--tokens", "128", "--hidden", "128", "--intermediate", "128",
                  "--warmup", "1", "--measured", "2", "--pairs", "10000:42,0.005:943"],
                 args.resume, args.replica_transport)
    if args.phase in ("accept", "full"):
        acceptance_root = root / "accept" if args.phase == "full" else root
        acceptance_root.mkdir(exist_ok=True)
        for backend, budget in variants:
            _run(acceptance_root, f"accept-{backend}-b{budget}", backend, budget,
                 ["--accept", "--pairs", "10000:42,1:42,0.005:943"], args.resume, args.replica_transport)
    if args.phase in ("sweep", "full"):
        sweep_root = root / "sweep" if args.phase == "full" else root
        sweep_root.mkdir(exist_ok=True)
        for repeat, ordered in enumerate((variants, tuple(reversed(variants))), start=1):
            for backend, budget in ordered:
                _run(sweep_root, f"repeat{repeat}-{backend}-b{budget}", backend, budget, [],
                     args.resume, args.replica_transport)
    if args.phase == "full" and variants == VARIANTS:
        subprocess.run([sys.executable, "scripts/plot_dirichlet_replica_sweep.py", "--input", str(sweep_root),
                        "--output", str(root / "dirichlet_ep16_e96")], check=True)
    completion = {"phase": args.phase, "complete": True, "variants": args.variants,
                  "replica_transport": args.replica_transport}
    (root / f"{args.phase}-complete.json").write_text(json.dumps(completion) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
