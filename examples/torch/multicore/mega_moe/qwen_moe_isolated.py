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
"""Framework-free parent for fresh-process, single-resident-backend Qwen comparisons."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.torch.multicore.mega_moe.qwen_moe_config import make_config, optimizer_contract, parse_args

_PHASES = {"inputs", "initial_parameters", "forward", "gradients", "updated_parameters"}


def parse_isolated_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Extend the shared workload CLI without importing a training framework."""
    parser = argparse.ArgumentParser(description="Compare single-resident Qwen backends in fresh EP8 processes.")
    parser.add_argument("--order", choices=("abba", "ab"), default="abba")
    parser.add_argument("--artifacts-dir", help="parent directory for a new run directory and retained CPU references")
    parser.add_argument("--timeout-seconds", type=int, default=3600,
                        help="maximum duration of each child process group")
    args = parse_args(argv, parser=parser)
    if args.backend != "both" or args.snapshot_dir or args.reference_dir:
        raise ValueError("--backend/--snapshot-dir/--reference-dir are worker-only options; "
                         "run qwen_moe_benchmark.py.")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive.")
    return args


def run_order(order: str) -> list[tuple[str, str]]:
    """Return the requested sequence of independent backend runs."""
    if order not in ("ab", "abba"):
        raise ValueError(f"unsupported Qwen comparison order: {order}")
    result = [("A1", "common"), ("B1", "mega_moe")]
    return result + [("B2", "mega_moe"), ("A2", "common")] if order == "abba" else result


def worker_arguments(args: argparse.Namespace) -> list[str]:
    """Preserve all parsed workload options while excluding parent/output controls."""
    excluded = {"order", "artifacts_dir", "timeout_seconds", "output", "backend", "snapshot_dir", "reference_dir"}
    arguments = []
    for name, value in sorted(vars(args).items()):
        if name in excluded or value is False:
            continue
        arguments.append("--" + name.replace("_", "-"))
        if value is not True:
            arguments.append("none" if value is None else str(value))
    return arguments


def launch_topology() -> dict[str, Any]:
    """Validate the parent context and the existing single-/multi-node EP8 environment."""
    if any(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise ValueError("launch the isolated parent directly, not under torchrun or an inherited rank environment.")
    topology = {
        "nnodes": int(os.getenv("NNODES", "1")), "nproc_per_node": int(os.getenv("NPROC_PER_NODE", "8")),
        "node_rank": int(os.getenv("NODE_RANK", "0")), "master_addr": os.getenv("MASTER_ADDR", "127.0.0.1"),
        "master_port": int(os.getenv("MASTER_PORT", "29500")),
    }
    if topology["nnodes"] <= 0 or topology["nproc_per_node"] <= 0:
        raise ValueError(f"positive process counts required: {topology}")
    if topology["nnodes"] * topology["nproc_per_node"] != 8 or not 0 <= topology["node_rank"] < topology["nnodes"]:
        raise ValueError(f"Qwen comparison requires EP8 and a valid node rank: {topology}")
    if not 1 <= topology["master_port"] <= 65535:
        raise ValueError(f"invalid master port: {topology['master_port']}")
    return topology


def _terminate_group(child: subprocess.Popen) -> None:
    """Reap only the newly created child process group, including its torchrun workers."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def run_process(command: list[str], environment: dict[str, str], log: Path, timeout: int) -> None:
    """Run one child group to completion and clean up on timeout or parent cancellation."""
    with log.open("x", encoding="utf-8") as stream, \
            subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                             start_new_session=True) as child:
        try:
            result = child.wait(timeout=timeout)
        except BaseException:
            _terminate_group(child)
            raise
        if result:
            _terminate_group(child)
            raise subprocess.CalledProcessError(result, command)


def _validate_result(result: dict[str, Any], args: argparse.Namespace, backend: str, export: bool) -> None:
    """Reject missing accuracy, stale workloads, invalid timings, and paired residency."""
    expected = {"mode": "isolated_backend", "backend": backend, "configuration": asdict(make_config(args)),
                "initialization_seed": args.seed, "world_size": 8,
                "warmup_steps": args.warmup_steps, "measured_steps": args.measured_steps}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Qwen child result does not match the requested workload: {expected}")
    if result["optimizer"] != optimizer_contract(args):
        raise ValueError("Qwen child optimizer does not match the comparison.")
    accuracy = result["accuracy"]
    if accuracy["mode"] != ("export" if export else "compare") or set(accuracy["phases"]) != _PHASES:
        raise ValueError("Qwen child did not complete the required accuracy phases.")
    for name, phase in accuracy["phases"].items():
        if (export and phase.get("status") != "reference_saved") or (not export and phase.get("passed") is not True):
            raise ValueError(f"Qwen child accuracy failed: {phase}")
        if phase.get("tensor_count", 0) <= 0:
            raise ValueError(f"Qwen child accuracy phase has no tensors: {phase}")
        if not export:
            expected_tolerance = (0.0, 0.0) if name in ("inputs", "initial_parameters") else (0.02, 0.002)
            if (phase.get("rtol"), phase.get("atol")) != expected_tolerance:
                raise ValueError(f"Qwen child changed the original accuracy tolerance: {phase}")
    validation = result["measurement"]["validation"]
    if not all(validation.get(key) is True for key in
               ("loss_finite", "gradients_finite", "dense_parameter_updated", "expert_parameter_updated")):
        raise ValueError(f"Qwen optimizer validation failed: {validation}")
    samples = result["measurement"]["steady_state_optimizer_step_ms"]["samples"]
    if len(samples) != args.measured_steps or not all(math.isfinite(value) and value > 0 for value in samples):
        raise ValueError(f"invalid Qwen timing samples: {samples}")
    residents = result["resident_training_per_rank"]
    memory = result["measurement"]["steady_state_memory_per_rank"]
    if len(residents) != 8 or len(memory) != 8 or any(value["resident_backend_count"] != 1 for value in residents):
        raise ValueError("Qwen child is missing its eight-rank single-backend memory ledger.")


def _summary(args: argparse.Namespace, records: list[dict[str, Any]], artifacts: Path) -> dict[str, Any]:
    """Average per-run medians without pooling samples from different fresh processes."""
    reference_runtime = records[0]["result"]["runtime"]
    if any(record["result"]["runtime"] != reference_runtime for record in records):
        raise ValueError("Qwen runtime/source/native identity differs between comparison runs.")
    medians = {
        backend: [statistics.median(record["result"]["measurement"]["steady_state_optimizer_step_ms"]["samples"])
                  for record in records if record["result"]["backend"] == backend]
        for backend in ("common", "mega_moe")
    }
    common_mean, mega_mean = (statistics.mean(medians[backend]) for backend in ("common", "mega_moe"))
    return {
        "mode": "isolated_comparison", "configuration": asdict(make_config(args)), "initialization_seed": args.seed,
        "artifacts_dir": str(artifacts), "runtime": reference_runtime,
        "comparison": {
            "accuracy": {"passed": True, "runs": {record["label"]: record["result"]["accuracy"]["phases"]
                                                   for record in records[1:]}},
            "performance": {"order": [record["label"] for record in records],
                            "metric": "mean of fresh-process rank-max optimizer-step medians (milliseconds)",
                            "run_medians_ms": medians, "common_mean_ms": common_mean, "mega_moe_mean_ms": mega_mean,
                            "mega_over_common": mega_mean / common_mean},
        },
        "runs": records,
    }


def _stop_on_signal(signum: int, _frame: Any) -> None:
    """Turn outer timeout termination into normal child-group cleanup."""
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    """Run separate EP8 processes, retaining CPU references and all raw evidence."""
    args = parse_isolated_args(argv)
    topology = launch_topology()
    output = Path(args.output).resolve()
    parent = Path(args.artifacts_dir).resolve() if args.artifacts_dir else output.parent
    parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="qwen_isolated_", dir=parent))
    reference = artifacts / "reference"
    launcher = os.getenv("TORCHRUN_BIN")
    prefix = [launcher] if launcher else [sys.executable, "-m", "torch.distributed.run"]
    prefix.extend(f"--{name}={value}" for name, value in topology.items())
    prefix.append(str(Path(__file__).with_name("qwen_moe_benchmark.py")))
    records = []
    previous_handler = signal.signal(signal.SIGTERM, _stop_on_signal)
    try:
        for label, backend in run_order(args.order):
            result_path, log = artifacts / f"{label}.json", artifacts / f"{label}.log"
            command = prefix + worker_arguments(args) + ["--backend", backend, "--output", str(result_path),
                                                        "--snapshot-dir" if label == "A1" else "--reference-dir",
                                                        str(reference)]
            print(f"QWEN_RUN={label} backend={backend} log={log}", flush=True)
            run_process(command, dict(os.environ), log, args.timeout_seconds)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            _validate_result(result, args, backend, label == "A1")
            records.append({"label": label, "result_path": str(result_path), "log_path": str(log),
                            "result": result})
        summary = _summary(args, records, artifacts)
        if topology["node_rank"] == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"RESULT_JSON={output}", flush=True)
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    sys.exit(main())
