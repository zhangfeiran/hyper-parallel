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
"""Unified, identity-checked imports for cost-model diagnostics and task services."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from examples.torch.multicore.mega_moe_gradient_windows import analyze_windows
from examples.torch.multicore.mega_moe_p2p_profile import _validate_payload_balance
from examples.torch.multicore.mega_moe_profile_context import source_id
from examples.torch.multicore.mega_moe_trace import Trace

PHASES = {17: "up_proj", 19: "down_proj", 20: "act_grad", 21: "gate_dx", 22: "w1_grad", 23: "w2_grad"}


def _validate_task_queues(profile: dict, expected_cores: int) -> None:
    """Validate the raw clock and coverage even when sampling prevents a GEMM fit."""
    if profile.get("schema_version") != 1 or profile.get("unit") != "system_cycle":
        raise ValueError("Unsupported task schema or clock unit")
    for kind in ("cube", "vector"):
        queues = [worker for worker in profile["workers"] if worker["kind"] == kind]
        if sorted(worker["core"] for worker in queues) != list(range(expected_cores)) or expected_cores <= 0:
            raise ValueError(f"Missing or duplicate {kind} queues")
        for worker in queues:
            if (worker["stride"] < 1 or worker["recorded"] != len(worker["records"])
                    or not 0 <= worker["recorded"] <= worker["seen"]):
                raise ValueError("Missing task records or invalid queue coverage")
            for record in worker["records"]:
                times = [record[name] for name in ("start_cycle", "ready_cycle", "compute_end_cycle", "end_cycle")]
                if times != sorted(times) or times[0] < 0:
                    raise ValueError("Nonmonotonic task cycles")


def cube_services(profile: dict, *, expected_cores: int, cycles_per_us: float = 50.0) -> list[dict]:
    """Require all unsampled Cube queues and one record per worker for each GEMM."""
    if profile.get("schema_version") != 1 or profile.get("unit") != "system_cycle":
        raise ValueError("Unsupported task schema or clock unit")
    if not math.isfinite(cycles_per_us) or cycles_per_us <= 0:
        raise ValueError("Invalid system cycle frequency")
    cubes = [worker for worker in profile["workers"] if worker["kind"] == "cube"]
    if sorted(worker["core"] for worker in cubes) != list(range(expected_cores)) or expected_cores <= 0:
        raise ValueError("GEMM calibration requires every Cube queue exactly once")
    if any(worker["stride"] != 1 for worker in cubes):
        raise ValueError("GEMM calibration requires unsampled Cube queues")
    groups = defaultdict(dict)
    for worker in cubes:
        if worker["recorded"] != len(worker["records"]) or worker["recorded"] > worker["seen"]:
            raise ValueError("Missing task records or invalid queue coverage")
        for record in worker["records"]:
            times = [record[name] for name in ("start_cycle", "ready_cycle", "compute_end_cycle", "end_cycle")]
            if times != sorted(times):
                raise ValueError("Nonmonotonic task cycles")
            if record["task_type"] != 104 or record["expert_rows"] <= 0:
                continue
            if record["tiling_position"] not in PHASES:
                raise ValueError("Unknown GEMM phase")
            key = (record["tiling_position"], record["wave"], record["task_index"] // expected_cores,
                   record["expert_rows"])
            if worker["core"] in groups[key]:
                raise ValueError("Duplicate worker record for the same expert GEMM")
            groups[key][worker["core"]] = record
    results = []
    for (position, wave, expert, rows), workers in groups.items():
        if len(workers) != expected_cores:
            raise ValueError("GEMM calibration needs all workers for each expert")
        records = list(workers.values())
        results.append({"phase": PHASES[position], "tiling_position": position, "wave": wave,
                        "expert": expert, "expert_rows": rows,
                        "compute_us": max(record["compute_end_cycle"] - record["ready_cycle"]
                                          for record in records) / cycles_per_us,
                        "ready_cycle": max(record["ready_cycle"] for record in records),
                        "complete_cycle": max(record["end_cycle"] for record in records)})
    return results


def task_attribution(profile: dict, context: dict) -> dict:
    """Report last-launch coverage and local expert readiness without cross-rank clock arithmetic."""
    expected = context["num_cube_cores"]
    _validate_task_queues(profile, expected)
    strides = {kind: [worker["stride"] for worker in profile["workers"] if worker["kind"] == kind]
               for kind in ("cube", "vector")}
    if any(len(values) != expected or any(value < 1 for value in values) for values in strides.values()):
        raise ValueError("Missing task queues or invalid sampling strides")
    # Sampled queues remain useful diagnostic evidence but cannot enter a GEMM fit.
    services = cube_services(profile, expected_cores=expected) if all(value == 1 for value in strides["cube"]) else []
    for item in services:
        expert = item["expert"]
        if not 0 <= expert < len(context["logical_experts"]):
            raise ValueError("Task expert is outside its actual placement")
        item["logical_expert"] = context["logical_experts"][expert]
        item["stage"] = ("dW" if item["tiling_position"] in (22, 23) else
                         "dX" if item["tiling_position"] in (20, 21) else
                         "recompute" if context["direction"] == "recompute" else "forward")
    return {"context": context, "coverage": "last launch per resource group and direction; not the whole step",
            "clock_domain": "rank_local_device_system_cycle", "unit": "system_cycle", "cycles_per_us": 50.0,
            "strides": strides, "gemm_calibration_eligible": all(value == 1 for value in strides["cube"]),
            "expert_services": services, "profile": profile}


def _validate_identity(identity: dict, expected: dict) -> None:
    if identity != expected:
        raise ValueError("Profile source/native identity mismatch")
    hashes = identity.get("native_artifacts", identity.get("sources_and_payload", {}))
    native = [value for name, value in hashes.items() if name.endswith((".so", ".o"))]
    if not native or any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in native):
        raise ValueError("Profile import requires hashed native artifacts")


def import_profiles(documents: list[dict], *, expected_identity: dict, read_traces: bool = True) -> dict[str, Any]:
    """Import both benchmark families while keeping each execution and measurement domain separate.

    Args:
        documents: Layer benchmark JSON or per-rank cost-profile JSON documents.
        expected_identity: Exact reviewed source/native identity; no implicit best-effort match.
        read_traces: Read raw calibrated trace artifacts when present.

    Returns:
        Profile-off measurements and separate contextual diagnostic executions.
        This schema has no combined timestamp or estimated profile-off peak from diagnostics.
    """
    executions, measurements, timelines = {}, [], []

    def register(context: dict) -> dict:
        """Keep executions separate and reject duplicate per-rank captures."""
        if context.get("source_id") != source_id(expected_identity):
            raise ValueError("Diagnostic capture source/native identity mismatch")
        key = context["capture_id"]
        execution = executions.setdefault(
            key, {"capture_id": key, "ranks": {}, "clock_domains": context["clock_domains"]},
        )
        if context["rank"] in execution["ranks"]:
            raise ValueError("Duplicate rank in the same diagnostic execution")
        row = {"context": context, "tasks": [], "scoped_events": [], "p2p": None}
        execution["ranks"][context["rank"]] = row
        return row

    def tasks(profile: dict, metadata: dict, context: dict) -> dict:
        """Require a task arena to match a captured last launch."""
        if metadata not in context["last_launches"] or any(metadata.get(name) != context.get(name) for name in (
            "capture_id", "source_id", "rank", "world_size", "step",
        )):
            raise ValueError("Task profile does not identify an actual last launch in this capture")
        return task_attribution(profile, metadata)

    for document in documents:
        _validate_identity(document.get("source", document.get("identity", {})), expected_identity)
        if "results" in document:
            if document.get("profile_schema_version") != 2:
                raise ValueError("Legacy cost profiles lack execution identities; retain the legacy calibration reader")
            for result in document["results"]:
                measurements.append({"rank": document["rank"], "pair": result["pair"],
                                     "latency_ms": result["profile_off_samples_ms"], "memory": result["memory"],
                                     "source": "profile_off", "shape": result["shape"]})
                for channel, context in result["captures"].items():
                    row = register(context)
                    if channel == "scopes":
                        for event in result["scoped_events"]:
                            if any(event["context"].get(name) != context.get(name) for name in (
                                "capture_id", "source_id", "rank", "world_size", "step",
                            )):
                                raise ValueError("Scope came from a different diagnostic step")
                        row["scoped_events"] = result["scoped_events"]
                    elif channel == "tasks":
                        row["tasks"] = [tasks(profile, result["task_contexts"][direction], context)
                                        for direction, profile in result["task_profiles"].items()]
                        row["cycles"] = result["cycle_profiles"]
            continue
        measurements.extend({"source": "profile_off", "point": point} for point in document["points"])
        for payload in document.get("diagnostic_kernel_cycles", {}).get("captures", []):
            row = register(payload["context"])
            row["resource_group_cycles"] = [group["cycles"] for group in payload["resource_groups"]]
            row["tasks"] = [tasks(item["profile"], item["context"], payload["context"])
                            for group in payload["resource_groups"] for item in group["tasks"].values()]
        p2p = document.get("diagnostic_p2p", {})
        trace = document.get("diagnostic_trace", {})
        for channel in (p2p, trace.get("capture", {})):
            if not channel:
                continue
            _validate_payload_balance(channel["per_rank"])
            for payload in channel["per_rank"]:
                row = register(payload["context"])
                row["p2p"] = {name: value for name, value in payload.items() if name != "context"}
        if trace and read_traces:
            if "capture" not in trace:
                raise ValueError("Legacy trace has no same-execution payload capture")
            if "per_rank" in trace:
                timelines.append(analyze_windows(document))
            else:
                payload = trace["capture"]["per_rank"][trace["rank"]]
                timelines.append({"rank": trace["rank"],
                                  "memory": Trace(Path(trace["path"]), payload["context"]).memory(
                                      payload.get("memory_events"),
                                  ),
                                  "coverage": "rank-zero trace only; no cross-rank wait attribution"})
    for execution in executions.values():
        execution["ranks"] = [row for _, row in sorted(execution["ranks"].items())]
        worlds = {row["context"]["world_size"] for row in execution["ranks"]}
        if len(worlds) != 1:
            raise ValueError("Ranks disagree on the capture world size")
        execution["missing_ranks"] = sorted(set(range(next(iter(worlds)))) -
                                             {row["context"]["rank"] for row in execution["ranks"]})
    return {"schema_version": 2, "identity": expected_identity, "measurements": measurements,
            "executions": list(executions.values()), "timelines": timelines,
            "units": {"measurements": "ms and allocator bytes", "trace": "host monotonic raw ns",
                      "tasks": "rank-local system cycles", "p2p_events": "stream elapsed ms"},
            "restriction": ("diagnostic steps are distinct; local cycles and stream elapsed times "
                            "have no cross-rank origin")}


def main() -> None:
    """Write a unified offline artifact using an explicit reference identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path, nargs="+")
    parser.add_argument("--identity-from", type=Path, required=True,
                        help="Reviewed benchmark/profile JSON whose source/native identity must match.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-traces", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads(args.identity_from.read_text(encoding="utf-8"))
    expected = reference.get("source", reference.get("identity", reference))
    result = import_profiles([json.loads(path.read_text(encoding="utf-8")) for path in args.documents],
                             expected_identity=expected, read_traces=not args.skip_traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
