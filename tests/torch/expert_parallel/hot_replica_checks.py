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
"""Shared correctness fixtures; importing these helpers never initializes devices."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute


def deferred_ids(tokens: int, top_k: int, experts: int, invocation: int,
                 device: torch.device) -> torch.Tensor:
    """Rotate a legal hot set while retaining K distinct experts per token."""
    if not 1 <= top_k <= experts or invocation not in (0, 1):
        raise ValueError("Deferred routes require 1 <= K <= E and invocation 0 or 1")
    positions = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k)
    return (positions.remainder(top_k) + invocation * (experts // 2)).remainder(experts).long()


def saved_route(output: torch.Tensor) -> ReplicaRoute:
    """Read the route actually saved by autograd, without replacing its planner."""
    pending, visited = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        for name in ("replica_route", "route"):
            route = getattr(node, name, None)
            if isinstance(route, ReplicaRoute):
                return route
        pending.extend(parent for parent, _ in node.next_functions)
    raise AssertionError("Candidate output has no saved replica route")


def route_evidence(output: torch.Tensor, planner: str) -> dict:
    """Fail if the requested planner was bypassed, and describe its actual plan."""
    route = saved_route(output)
    actual = "device" if route.device_plan is not None else "cpu"
    if actual != planner:
        raise AssertionError(f"Requested planner {planner}, executed {actual}")
    return {"planner": actual, "slots": route.plan.slot_to_logical,
            "counts": route.plan.destination_counts, "transfers": len(route.plan.transfers)}


def file_identity(path: Path) -> dict:
    """Identify the file actually loaded rather than an intended build directory."""
    path = path.resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def execution_identity() -> dict:
    """Record source and environment identity outside any measured execution."""
    root = Path(__file__).resolve().parents[3]
    def _git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--binary"])
    sources = {}
    for directory in ("hyper_parallel/core/expert_parallel", "hyper_parallel/core/multicore",
                      "tests/torch/expert_parallel", "tests/ut/core/expert_parallel"):
        for path in sorted((root / directory).rglob("*")):
            if path.suffix in (".py", ".cpp", ".h", ".md"):
                sources[str(path.relative_to(root))] = file_identity(path)["sha256"]
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
    versions = {str(path): path.read_text() for path in
                (cann / "version.info", cann / "ascend_toolkit_install.info") if path.is_file()}
    return {"root": str(root), "sha": _git("rev-parse", "HEAD"), "status": _git("status", "--porcelain"),
            "diff_sha256": hashlib.sha256(diff).hexdigest(), "source_sha256": sources,
            "cann": str(cann), "cann_versions": versions,
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__, "source_manifest_sha256":
                hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()}
