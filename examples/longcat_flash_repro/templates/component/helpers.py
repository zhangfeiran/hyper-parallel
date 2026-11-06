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
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
REPO = Path("__REPO__")
ROUTING_BACKEND = "triton"
DIRECT_GRADIENTS = True
DEVICES = list(map(int, os.environ["HP_REPRO_COMPONENT_DEVICES"].split(",")))
PYTHON = sys.executable


def save(path: Path, data: Any) -> None:
    """Write a rank report outside the measurement interval.

    Args:
        path: Artifact path to read or write.
        data: Structured report contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def seal() -> dict[str, str]:
    "Hash the selected model sources and native payload."
    paths = []
    for name in [
        "hyper_parallel/models/longcat_flash",
        "hyper_parallel/core/multicore",
        "hyper_parallel/components/functional",
        "tests/torch/models/longcat_flash",
    ]:
        paths.extend(
            p
            for p in (REPO / name).rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".cpp", ".h", ".hpp", ".patch", ".json", ".yaml", ".cmake")
            and "__pycache__" not in p.parts
        )
    native_root = Path(os.environ["HP_REPRO_NATIVE_ROOT"])
    paths.extend(p for p in native_root.rglob("*") if p.is_file())
    return {
        (
            "native/" + str(p.relative_to(native_root)) if p.is_relative_to(native_root) else str(p.relative_to(REPO))
        ): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def percentile(rows: Any, q: float) -> float:
    """Interpolate the requested percentile from all samples.

    Args:
        rows: Raw aligned per-step records.
        q: Requested percentile in the closed unit interval.
    """
    values = sorted(rows)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = math.ceil(pos)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo) if hi != lo else values[lo]


def parse_processes(raw: str, mapping: Any) -> list[dict[str, Any]]:
    """Map NPU process samples to logical device IDs.

    Args:
        raw: Complete NPU process inventory text.
        mapping: Physical board/chip to logical device mapping.
    """
    pattern = r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+)\|\s*(\d+)\s*\|"
    result = []
    for card, chip, pid, name, memory in re.findall(pattern, raw, re.M):
        result.append(
            {"device": mapping[(int(card), int(chip))], "pid": int(pid), "name": name.strip(), "mib": int(memory)}
        )
    if "Process id" not in raw or not result and "No running processes found" not in raw:
        raise RuntimeError("incomplete npu-smi process sample")
    return result


def belongs(pid: int, root: int) -> Optional[bool]:
    """Resolve whether a process descends from this case launcher.

    Args:
        pid: Process ID whose ownership is being checked.
        root: Source root or owning launcher identity to inspect.
    """
    for _ in range(64):
        if pid == root:
            return True
        try:
            s = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            pid = int(s[s.rfind(")") + 2 :].split()[1])
        except FileNotFoundError:
            return None
        if pid <= 1:
            return False
    return False


def monitor(stop: threading.Event, case: Path, launcher: int, mapping: Any) -> None:
    """Capture device process ownership until the case finishes.

    Args:
        stop: Monitor shutdown event.
        case: Fresh case output directory.
        launcher: Owning case launcher process ID.
        mapping: Physical board/chip to logical device mapping.
    """
    rows = []
    while not stop.is_set():
        try:
            out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5, check=True).stdout
            processes = parse_processes(out, mapping)
            for p in processes:
                p["belongs_to_case"] = belongs(p["pid"], launcher)
            rows.append({"utc": datetime.now(timezone.utc).isoformat(), "processes": processes, "valid": True})
            (case / f"npu-{len(rows):04}.txt").write_text(out, encoding="utf-8")
        except Exception as error:
            rows.append({"valid": False, "error": repr(error)})
        stop.wait(0.5)
    save(case / "monitor.json", rows)


def wait_owned(process: subprocess.Popen, stop: threading.Event, thread: threading.Thread) -> int:
    """Stop this case's process group and monitor on interruption or failure.

    Args:
        process: Owned launcher process handle.
        stop: Monitor shutdown event.
        thread: Case monitor thread to join.
    """
    try:
        return process.wait()
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        stop.set()
        thread.join()


def validate_ownership(case: Path, known: set[int]) -> None:
    """Check complete samples and reject foreign device processes.

    Args:
        case: Fresh case output directory.
        known: Worker process IDs owned by this case.
    """
    samples = json.loads((case / "monitor.json").read_text(encoding="utf-8"))
    observed = {p["pid"] for sample in samples if sample["valid"] for p in sample["processes"]}
    owned = known | {
        p["pid"] for sample in samples if sample["valid"] for p in sample["processes"] if p["belongs_to_case"] is True
    }
    foreign = observed - owned
    if foreign or not known.issubset(observed) or not samples or any(not s["valid"] for s in samples):
        raise RuntimeError("incomplete ownership or foreign NPU processes: " + str(case) + " " + str(foreign))


def validate_component_rank(row: Any, rank: int, mode: str) -> None:
    """Check component shapes, operator selection and terminal state.

    Args:
        row: Complete rank report.
        rank: Logical rank within the selected process group.
        mode: Selected expert implementation.
    """
    terminal_checks = (
        row["rank"] == rank,
        row["backend"] == mode,
        row["finite_gradients"],
        row["unused_gate_rows_gradient_zero"],
        not row["shmem_after_close"]["reference_count"],
        not Path(f"/proc/{row['pid']}").exists(),
    )
    if not all(terminal_checks):
        raise RuntimeError("incomplete rank terminal record")
    if row["hyper_parallel_source"] != str(REPO / "hyper_parallel/__init__.py"):
        raise RuntimeError("different imported checkout")
    if row["local_experts"] != {"real": 8, "identity": 4} or row["parameter_local_shapes"] != {
        "gate_up_proj": [12, 4096, 6144],
        "down_proj": [8, 6144, 2048],
    }:
        raise RuntimeError("local expert crop changed")
    if mode == "native" and any(
        row["native_operator_census"][name] <= 0
        for name in ["npu_grouped_matmul", "npu_swiglu", "npu_moe_token_permute", "npu_moe_token_unpermute"]
    ):
        raise RuntimeError("native big-operator path did not execute")
    if row["swiglu_precision"] != "fused":
        raise RuntimeError("SwiGLU not fused")
    activation = row["activation_execution"]
    if mode == "native" and (row["native_operator_census"]["silu"] or activation["operator_calls"] <= 0):
        raise RuntimeError("native npu_swiglu absent")
    if mode != "native" and (activation["forward_hf_bf16"] or activation["backward_hf_bf16"]):
        raise RuntimeError("MegaMoE HF mode unexpectedly enabled")


def component_phases(ranks: Any) -> dict[str, Any]:
    """Recompute phase statistics from every raw rank sample.

    Args:
        ranks: Complete ordered rank reports.
    """
    phases = {}
    for label in ["forward", "forward_backward"]:
        n = len(ranks[0]["phases"][label]["local_ms"])
        values = [max(r["phases"][label]["local_ms"][i] for r in ranks) for i in range(n)]
        if n != 30:
            raise RuntimeError("incorrect sample count")
        for row in ranks:
            if any(abs(a - b) > max(1e-5, abs(b) * 1e-6) for a, b in zip(row["phases"][label]["rank_max_ms"], values)):
                raise RuntimeError("rank-max aggregation differs from raw samples")
        phases[label] = {
            "p50_ms": statistics.median(values),
            "p90_ms": percentile(values, 0.9),
            "rank_max_ms": values,
            "peak_allocated_bytes": max(r["phases"][label]["peak_allocated_bytes"] for r in ranks),
            "peak_reserved_bytes": max(r["phases"][label]["peak_reserved_bytes"] for r in ranks),
            "shmem_heap_bytes": max(r["phases"][label]["memory"]["shmem_heap_bytes"] for r in ranks),
            "source_rows": [r["routing"]["real_source_rows"] for r in ranks],
            "owner_rows": [r["routing"]["owner_received_rows"] for r in ranks],
        }
    return phases


def run_case(token_count: int, mode: str, index: int, phase: str, source_seal: Any, mapping: Any) -> dict[str, Any]:
    """Measure one isolated expert implementation and validate every rank.

    Args:
        token_count: Unmodified token count per rank.
        mode: Selected expert implementation.
        index: Process index in the frozen experiment order.
        phase: Pilot or formal measurement phase.
        source_seal: Frozen source and native payload fingerprints.
        mapping: Physical board/chip to logical device mapping.
    """
    case = ROOT / "component-runs" / f"t{token_count}" / f"{phase}-{index:02}-{mode}"
    case.mkdir(parents=True, exist_ok=False)
    if seal() != source_seal:
        raise RuntimeError("source changed before benchmark case")
    command = [
        PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "--module",
        "tests.torch.models.longcat_flash._benchmark_longcat_flash",
        "--routing-backend",
        ROUTING_BACKEND,
        "--backend",
        mode,
        "--ep",
        "4",
        "--real",
        "32",
        "--zero",
        "16",
        "--tokens",
        str(token_count),
        "--initial-token-capacity",
        "24576",
        "--output",
        str(case / "result"),
    ]
    if not DIRECT_GRADIENTS:
        command.append("--no-direct-weight-gradients")
    save(
        case / "command.json",
        {"command": command, "cwd": str(REPO), "devices": DEVICES, "utc": datetime.now(timezone.utc).isoformat()},
    )
    with (case / "launcher.log").open("w", encoding="utf-8") as stream:
        with subprocess.Popen(
            command, cwd=REPO, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        ) as process:
            stop = threading.Event()
            thread = threading.Thread(target=monitor, args=(stop, case, process.pid, mapping))
            thread.start()
            status = wait_owned(process, stop, thread)
    save(case / "exit.json", {"launcher_pid": process.pid, "exit_code": status})
    if status:
        raise RuntimeError("benchmark process failed: " + str(case))
    ranks = [json.loads((case / f"result/rank{r}.json").read_text(encoding="utf-8")) for r in range(4)]
    if seal() != source_seal:
        raise RuntimeError("source changed during benchmark case")
    identities = [row["identities"] for row in ranks]
    validate_ownership(case, {row["pid"] for row in ranks})
    for rank, row in enumerate(ranks):
        validate_component_rank(row, rank, mode)
    phases = component_phases(ranks)
    result = {
        "case": str(case.relative_to(ROOT)),
        "backend": mode,
        "tokens": token_count,
        "phase": phase,
        "routing_backend": ROUTING_BACKEND,
        "direct_weight_gradients": DIRECT_GRADIENTS,
        "identities": identities,
        "phases": phases,
        "ownership_verified": True,
        "all_rank_pids_exited": True,
    }
    save(case / "summary.json", result)
    return result
