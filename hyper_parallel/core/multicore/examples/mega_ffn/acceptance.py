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
"""Gated fresh-process dense trajectories, native controls and ABBA acceptance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Direct-script supervision must not import the framework while waiting for devices.
_SPEC = importlib.util.spec_from_file_location("_megaffn_ownership", Path(__file__).with_name("ownership.py"))
_OWNERSHIP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OWNERSHIP)


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def suite_jobs(blocks: int) -> list[dict[str, object]]:
    """Enumerate separate native controls and candidate jobs for both dense baselines.

    Args:
        blocks: Independent fresh-process four-run blocks per performance phase.
    """
    jobs = []
    for baseline in ("common", "packed"):
        for backend in (baseline, "mega_ffn"):
            jobs.append({"mode": "validate", "baseline": baseline, "backend": backend})
        for phase in ("control", "comparison"):
            for block in range(blocks):
                for slot, role in enumerate(("A", "B", "B", "A")):
                    backend = "mega_ffn" if role == "B" and phase == "comparison" else baseline
                    jobs.append({"mode": "measure", "baseline": baseline, "backend": backend,
                                 "phase": phase, "block": block, "slot": slot})
    return jobs


def block_effects(medians: list[list[float]]) -> list[float]:
    """Compute log baseline/candidate step-time ratios for each independent ABBA block.

    Args:
        medians: Four process-level median times in A, B, B, A order per block.
    """
    if len(medians) < 3:
        raise ValueError("Acceptance requires at least three independent four-run blocks")
    effects = []
    for times in medians:
        if len(times) != 4 or any(not math.isfinite(value) or value <= 0 for value in times):
            raise ValueError("ABBA requires four finite positive process medians")
        effects.append((math.log(times[0]) + math.log(times[3]) - math.log(times[1]) - math.log(times[2])) / 2)
    return effects


def effect_interval(effects: list[float]) -> dict[str, object]:
    """Bootstrap independent blocks, keeping step samples within a process correlated.

    Args:
        effects: Log time ratios per independent process block.
    """
    generator = random.Random(17)
    means = sorted(statistics.mean(generator.choices(effects, k=len(effects))) for _ in range(5000))

    def _percent(effect):
        return 100 * (1 - math.exp(-effect))

    return {"step_time_saved_percent": _percent(statistics.mean(effects)),
            "ci95_percent": [_percent(means[125]), _percent(means[4874])],
            "block_effects": effects}


def performance_verdict(control: list[list[float]], comparison: list[list[float]]) -> dict[str, object]:
    """Require measured gains beyond the complete native-control noise envelope.

    Args:
        control: Native/native four-process medians per independent block.
        comparison: Baseline/candidate ABBA medians per independent block.
    """
    control_effects, comparison_effects = block_effects(control), block_effects(comparison)
    native, candidate = effect_interval(control_effects), effect_interval(comparison_effects)
    noise = max(abs(100 * (1 - math.exp(-effect))) for effect in control_effects)
    native_clean = native["ci95_percent"][0] <= 0 <= native["ci95_percent"][1] and noise <= 2
    return {"native_control": native, "comparison": candidate, "native_control_passed": native_clean,
            "required_gain_percent": max(1.0, noise),
            "performance_passed": native_clean and candidate["ci95_percent"][0] > max(1.0, noise)}


def _worker(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Ownership worker requires an exact benchmark command")
    selected = _OWNERSHIP.selected_devices(args.device)
    samples, launched_at = [], time.time()
    with subprocess.Popen(command) as process:
        start_ticks = _OWNERSHIP.process_identity(process.pid, process.pid)["start_ticks"]
        while process.poll() is None:
            samples.append(_OWNERSHIP.ownership_sample(process.pid, selected))
            try:
                process.wait(timeout=args.poll_seconds)
            except subprocess.TimeoutExpired:
                continue
        samples.append(_OWNERSHIP.ownership_sample(process.pid, selected))
    clean = (len(samples) >= 2 and any(sample["own"] for sample in samples)
             and not any(sample["foreign"] or sample["unresolved"] for sample in samples))
    record = {"clean": clean, "worker_pid": process.pid, "exit_code": process.returncode,
              "worker_start_ticks": start_ticks, "launched_at": launched_at,
              "poll_seconds": args.poll_seconds, "samples": samples, "coverage": "sampled process ownership"}
    _write(args.output, record)
    if process.returncode or not clean:
        raise RuntimeError("Dense benchmark failed or ownership was contaminated/unresolved; evidence retained")


def _launch(args, index, job):
    directory = args.root / f"job-{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    output, ownership = directory / "result.json", directory / "ownership.json"
    benchmark = [sys.executable, str(Path(__file__).with_name("qwen_dense_benchmark.py")),
                 "--mode", job["mode"], "--baseline", job["baseline"], "--backend", job["backend"],
                 "--output", str(output), *args.benchmark_args]
    worker = [sys.executable, str(Path(__file__).resolve()), "worker", "--device", str(args.device),
              "--output", str(ownership), "--", *benchmark]
    environment = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.device),
                       NPU_WAIT_VISIBLE_DEVICES=str(args.device),
                       NPU_WAIT_NUM_CARDS="1", NPU_WAIT_POLL_SECONDS="60", NPU_WAIT_IDLE_SAMPLES="1",
                       NPU_WAIT_FINAL_DELAY_SECONDS="10", NPU_WAIT_LOCK_WAIT_SECONDS="-1")
    with (directory / "run.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(["bash", str(args.helper), *worker], env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    return {**job, "index": index, "result": str(output), "ownership": str(ownership),
            "log": str(directory / "run.log"), "exit_code": result.returncode}


def _run(args):
    if not args.helper.is_file():
        raise ValueError("The existing idle helper must be supplied as a readable file")
    args.root = args.root.resolve()
    journal = args.root / "suite.json"
    if journal.exists():
        raise ValueError("Use a new output directory to preserve previous suite evidence")
    record = {"complete": False, "blocks": args.blocks, "jobs": []}
    _write(journal, record)
    for index, job in enumerate(suite_jobs(args.blocks)):
        result = _launch(args, index, job)
        record["jobs"].append(result)
        _write(journal, record)
        if result["exit_code"]:
            raise RuntimeError(f"Dense suite stopped at job {index}; see {result['log']}")
    record["complete"] = True
    _write(journal, record)
    _write(args.root / "acceptance.json", analyze_suite(record))


def _host_clean(ownership, record):
    window = record.get("measurement_window")
    if window is None:
        return False
    selected = [sample for sample in ownership["samples"] if window[0] <= sample["time"] <= window[1]]
    for previous, current in zip(ownership["samples"], ownership["samples"][1:]):
        if current["time"] < window[0] or previous["time"] > window[1]:
            continue
        before, after = previous["host_cpu_ticks"], current["host_cpu_ticks"]
        total = sum(after) - sum(before)
        idle = after[3] + after[4] - before[3] - before[4]
        if total > 0 and 1 - idle / total > 0.2:
            return False
    return bool(selected)


def _check_ownership(ownership, record):
    if (not record["complete"] or not ownership["clean"] or ownership["exit_code"]
            or not any(sample["own"] for sample in ownership["samples"])
            or any(sample["foreign"] or sample["unresolved"] for sample in ownership["samples"])):
        raise ValueError("Incomplete or contaminated dense acceptance evidence")


def _check_measurement(ownership, record):
    if not _host_clean(ownership, record):
        raise ValueError("Host CPU contamination or missing timed ownership observations")
    if record["median_ms"] != statistics.median(record["samples_ms"]):
        raise ValueError("Recorded process median differs from the retained step samples")


def _checked_result(job, expected_identity):
    if job["exit_code"]:
        raise ValueError("Cannot accept a failed suite process")
    record = json.loads(Path(job["result"]).read_text(encoding="utf-8"))
    ownership = json.loads(Path(job["ownership"]).read_text(encoding="utf-8"))
    _check_ownership(ownership, record)
    identity = {"identity": record["identity"], "workload": record["workload"]}
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Suite source/framework/device/data/optimizer identities differ")
    if (record["mode"] != job["mode"] or record["backend"] != job["backend"]
            or (job["mode"] == "validate" and record["baseline"] != job["baseline"])):
        raise ValueError("Suite result does not belong to its declared workload")
    if job["mode"] == "measure":
        _check_measurement(ownership, record)
    return record, identity, (ownership["worker_pid"], ownership["worker_start_ticks"])


def _performance_blocks(jobs, baseline, phase):
    groups = {}
    for job, record in jobs:
        if job["mode"] != "measure" or job["baseline"] != baseline or job["phase"] != phase:
            continue
        group = groups.setdefault(job["block"], {})
        if job["slot"] in group:
            raise ValueError("Duplicate ABBA process slot")
        group[job["slot"]] = record["median_ms"]
    if any(set(group) != {0, 1, 2, 3} for group in groups.values()):
        raise ValueError("Incomplete ABBA process block")
    return [[group[index] for index in range(4)] for _, group in sorted(groups.items())]


def analyze_suite(suite: dict[str, object]) -> dict[str, object]:
    """Validate complete evidence before comparing two baselines with the candidate.

    Args:
        suite: Journal from the gated fresh-process runner.
    """
    expected_jobs = suite_jobs(suite["blocks"])
    if not suite["complete"] or len(suite["jobs"]) != len(expected_jobs):
        raise ValueError("Dense suite is incomplete")
    identity, jobs, processes = None, [], set()
    for job, expected in zip(suite["jobs"], expected_jobs):
        if any(job[key] != value for key, value in expected.items()):
            raise ValueError("Suite job order/assignment differs from the requested experiment")
        record, identity, process = _checked_result(job, identity)
        if process in processes:
            raise ValueError("Every suite workload must use a fresh process")
        processes.add(process)
        jobs.append((job, record))
    numerical = all(_numerical_passed(record) for job, record in jobs if job["mode"] == "validate")
    resident = all(_resident_candidate(record) for job, record in jobs if job["backend"] == "mega_ffn")
    comparisons = {baseline: performance_verdict(_performance_blocks(jobs, baseline, "control"),
                                                _performance_blocks(jobs, baseline, "comparison"))
                   for baseline in ("common", "packed")}
    return {"identity": identity, "numerical_passed": numerical, "comparisons": comparisons,
            "resident_forward_present": resident,
            "performance_passed": numerical and resident
            and all(entry["performance_passed"] for entry in comparisons.values()),
            "scope": "Fixed-data random-initialized dense-Qwen training; FSDP and long convergence are separate"}


def _resident_candidate(record):
    payloads = record.get("native_payloads", {})
    if record["mode"] == "validate":
        payloads = payloads.get("candidate", {})
    layers = record["workload"].get("config", {}).get("num_layers", 0)
    if layers <= 0 or set(payloads) != {f"layers.{index}.mlp" for index in range(layers)}:
        return False
    return all(entries and all(entry.get("native", {}).get("execution_mode") == "resident_dense_tile_candidate"
                               for entry in entries) for entries in payloads.values())


def _numerical_passed(record):
    kinds = {"logits", "gradients", "parameters", "main", "exp_avg", "exp_avg_sq", "step", "loss"}
    expected = record["expected_steps"]
    if expected < 100 or len(record["steps"]) != expected or not record["numerical_passed"]:
        return False
    if set(record["initial"]) != {"parameters", "main", "exp_avg", "exp_avg_sq", "step"}:
        return False
    if not all(check["passed"] for check in record["initial"].values()):
        return False
    return all(row["step"] == index + 1 and row["passed"] and set(row["checks"]) == kinds
               and all(check["passed"] for check in row["checks"].values())
               for index, row in enumerate(record["steps"]))


def main() -> None:
    """Run or analyze dense acceptance without importing device frameworks in the supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--helper", type=Path, required=True)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--blocks", type=int, default=5)
    run.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--device", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--poll-seconds", type=float, default=1)
    worker.add_argument("command", nargs=argparse.REMAINDER)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        if args.blocks < 3 or args.device < 0:
            parser.error("blocks must be at least three and device must be nonnegative")
        args.benchmark_args = args.benchmark_args[1:] if args.benchmark_args[:1] == ["--"] else args.benchmark_args
        reserved = ("--mode", "--backend", "--baseline", "--output")
        if any(argument.split("=", 1)[0] in reserved for argument in args.benchmark_args):
            parser.error("The suite owns mode/backend/baseline/output arguments")
        _run(args)
    elif args.action == "worker":
        if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
            parser.error("ownership poll interval must be finite and positive")
        _worker(args)
    else:
        suite = json.loads((args.root / "suite.json").read_text(encoding="utf-8"))
        _write(args.root / "acceptance.json", analyze_suite(suite))


if __name__ == "__main__":
    main()
