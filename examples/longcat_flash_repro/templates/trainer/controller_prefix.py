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
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = Path("__REPO__")
spec = importlib.util.spec_from_file_location("component_controller", ROOT.parent / "component/helpers.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def validate_terminal_reports(reports: Any, name: str, expected_steps: int, expected_calls: int) -> None:
    """Require all ranks to complete the declared recomputation scope.

    Args:
        reports: All-rank terminal records.
        name: Artifact or case label.
        expected_steps: Required final optimizer step.
        expected_calls: Required recomputation entry count.
    """
    for r in reports:
        if (
            r["global_step"] != expected_steps
            or not r["all_main_fp32_finite"]
            or r["shmem_after_close"]["reference_count"]
            or Path(f"/proc/{r['pid']}").exists()
        ):
            raise RuntimeError("incomplete terminal record: " + name)
        if (
            len(r["layer_forward_entry_counts"]) != 1
            or set(r["layer_forward_entry_counts"].values()) != {expected_calls}
            or len(r["gradient_checkpointing_layers"]) != 1
        ):
            raise RuntimeError("incomplete full-depth recomputation coverage: " + name)
    if any(
        r["source"] != str(REPO / "hyper_parallel/__init__.py") or r["HCCL_DETERMINISTIC"] != "true" for r in reports
    ):
        raise RuntimeError("different source or HCCL determinism")


def validate_update_coverage(initial: Any, steps: Any, name: str) -> None:
    """Require complete consecutive updates and main gradient evidence.

    Args:
        initial: Restored model and training-state records.
        steps: Aligned rank-indexed update records.
        name: Artifact or case label.
    """
    if name != "initialize" and any(row["global_step"] != 1 or not row["parameters"] for row in initial):
        raise RuntimeError("common step1 state was not restored")
    for rank, rows in enumerate(steps):
        expected_fqns = set(initial[rank]["parameters"])
        for row in rows:
            if row.get("objective_dtypes") != ["torch.float32"]:
                raise RuntimeError("missing FP32 objective evidence")
            if name.startswith("numerical-"):
                for key in ["updated_model_sha256", "updated_main_sha256"]:
                    if set(row.get(key, {})) != expected_fqns:
                        raise RuntimeError("missing complete updated-state hash coverage")
                if row.get("main_gradient_metadata", {}).get("step") != row["step"]:
                    raise RuntimeError("missing main-gradient evidence")


def validate_step_sequence(name: str, steps: list, expected_steps: int) -> None:
    """Require the declared uninterrupted update window.

    Args:
        name: Case label.
        steps: Rank-indexed update records.
        expected_steps: Required final update index.
    """
    if name != "initialize" and any([r["step"] for r in rows] != list(range(2, expected_steps + 1)) for rows in steps):
        raise RuntimeError("nonconsecutive steps")


def run(name: str, config_path: Path, mapping: dict) -> dict[str, Any]:
    """Execute and validate one fresh Trainer process group.

    Args:
        name: Artifact or case label.
        config_path: Resolved Trainer recipe.
        mapping: Physical board/chip to logical device mapping.
    """
    target = ROOT / name
    target.mkdir(exist_ok=False)
    env = dict(
        os.environ,
        HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR=str(target),
        HCCL_HOST_SOCKET_PORT_RANGE="auto",
        HP_LONGCAT_EXACT_STATE_PROBES="1" if name.startswith("numerical-") else "0",
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=16",
        str(ROOT / "audit_train.py"),
        str(config_path),
    ]
    c.save(
        target / "command.json",
        {
            "command": command,
            "cwd": str(REPO),
            "devices": list(map(int, os.environ["HP_REPRO_TRAINER_DEVICES"].split(","))),
            "HCCL_HOST_SOCKET_PORT_RANGE": env["HCCL_HOST_SOCKET_PORT_RANGE"],
        },
    )
    with (target / "launcher.log").open("w", encoding="utf-8") as stream:
        with subprocess.Popen(
            command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        ) as proc:
            stop = threading.Event()
            thread = threading.Thread(target=c.monitor, args=(stop, target, proc.pid, mapping))
            thread.start()
            status = c.wait_owned(proc, stop, thread)
    c.save(target / "exit.json", {"exit_code": status, "launcher_pid": proc.pid})
    if status:
        raise RuntimeError("trainer failed: " + name)
    reports = [json.loads((target / f"terminal-rank{r}.json").read_text(encoding="utf-8")) for r in range(16)]
    c.validate_ownership(target, {r["pid"] for r in reports})
    expected_steps = 1 if name == "initialize" else (11 if name.startswith("numerical-") else 16)
    expected_calls = 2 if name == "initialize" else 2 * (expected_steps - 1)
    validate_terminal_reports(reports, name, expected_steps, expected_calls)
    initial = [json.loads((target / f"initial-rank{r}.json").read_text(encoding="utf-8")) for r in range(16)]
    steps = [
        [json.loads(line) for line in (target / f"steps-rank{r}.jsonl").read_text(encoding="utf-8").splitlines()]
        for r in range(16)
    ]
    validate_step_sequence(name, steps, expected_steps)
    summary = {
        "name": name,
        "initial": initial,
        "steps": steps,
        "all_rank_pids_exited": True,
        "ownership_verified": True,
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for rows in steps for r in rows),
        "peak_reserved_bytes": max(r["peak_reserved_bytes"] for rows in steps for r in rows),
    }
    validate_update_coverage(initial, steps, name)
    c.save(target / "summary.json", summary)
    return summary


def compare_initial_states(a: Any, b: Any) -> None:
    """Require common checkpoint, optimizer, scheduler and RNG states.

    Args:
        a: Reference checkpoint and update records.
        b: Candidate checkpoint and update records.
    """
    keys = [
        "norm_classes",
        "global_step",
        "model_architecture",
        "parameters",
        "optimizer_sha256",
        "scheduler_sha256",
        "python_rng_sha256",
        "cpu_rng_sha256",
        "npu_rng_sha256",
        "optimizer_classes",
    ]
    for x, y in zip(a["initial"], b["initial"]):
        for key in keys:
            if x[key] != y[key]:
                raise RuntimeError("warm-start state mismatch: " + key)


def compare_step_inputs(xs: list, ys: list) -> None:
    """Require input, learned route and learning-rate identity.

    Args:
        xs: Reference per-rank update rows.
        ys: Candidate per-rank update rows.
    """
    if any(
        x["input_sha256"] != y["input_sha256"] or x["routing_sha256"] != y["routing_sha256"] or x["lr"] != y["lr"]
        for x, y in zip(xs, ys)
    ):
        raise RuntimeError("step input/LR mismatch")


def compare(a: Any, b: Any) -> dict[str, Any]:
    """Compare aligned loss, gradient norm and exact update records.

    Args:
        a: Reference checkpoint and update records.
        b: Candidate checkpoint and update records.
    """
    compare_initial_states(a, b)
    rows = []
    count = min(len(a["steps"][0]), len(b["steps"][0]))
    for i in range(count):
        xs = [r[i] for r in a["steps"]]
        ys = [r[i] for r in b["steps"]]
        compare_step_inputs(xs, ys)
        for rank, (x, y) in enumerate(zip(xs, ys)):
            loss = abs(x["loss"] - y["loss"])
            norm = abs(x["grad_norm"] - y["grad_norm"])
            for key in ["updated_model_sha256", "updated_main_sha256", "main_gradient_metadata"]:
                if key in x and key in y and x[key] != y[key]:
                    raise RuntimeError("updated exact state mismatch: " + key)
            rows.append(
                {
                    "step": x["step"],
                    "rank": rank,
                    "reference_loss": x["loss"],
                    "candidate_loss": y["loss"],
                    "loss_absolute_delta": loss,
                    "reference_preclip_norm": x["grad_norm"],
                    "candidate_preclip_norm": y["grad_norm"],
                    "preclip_norm_absolute_delta": norm,
                    "pass": loss <= 0.001 + 0.001 * abs(x["loss"]) and norm <= 0.01 + 0.01 * abs(x["grad_norm"]),
                    "input_identical": True,
                    "routing_identical": True,
                    "lr_identical": True,
                }
            )
    return {
        "reference": a["name"],
        "candidate": b["name"],
        "status": "PASS" if all(r["pass"] for r in rows) else "FAIL",
        "consecutive_steps_per_rank": count,
        "measured_steps_per_rank": sum(x["phase"] == "measured" for x in b["steps"][0][:count]),
        "state_identical": True,
        "rows": rows,
    }
