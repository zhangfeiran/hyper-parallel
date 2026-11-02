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

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute


def deferred_ids(tokens: int, top_k: int, experts: int, invocation: int,
                 device: torch.device) -> torch.Tensor:
    """Rotate a legal hot set while retaining K distinct experts per token.

    Args:
        tokens: Local token count.
        top_k: Distinct experts selected for each token.
        experts: Logical expert count.
        invocation: First or second retained invocation.
        device: Device hosting the routing IDs.
    """
    if not 1 <= top_k <= experts or invocation not in (0, 1):
        raise ValueError("Deferred routes require 1 <= K <= E and invocation 0 or 1")
    positions = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k)
    return (positions.remainder(top_k) + invocation * (experts // 2)).remainder(experts).long()


def saved_route(output: torch.Tensor) -> ReplicaRoute:
    """Read the route actually saved by autograd, without replacing its planner.

    Args:
        output: Expert output with a live backward graph.
    """
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


def replica_forward(executor: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Observe the real planner when the home fast path drops its autograd route.

    Args:
        executor: Replica-enabled expert module.
        *args: Positional forward arguments passed unchanged.
        **kwargs: Keyword forward arguments passed unchanged.
    """
    observed = []
    prepare = executor._prepare_replica_route

    @wraps(prepare)
    def _record(*planner_args: Any, **planner_kwargs: Any) -> ReplicaRoute:
        route = prepare(*planner_args, **planner_kwargs)
        observed[:] = [route]
        return route

    with patch.object(executor, "_prepare_replica_route", side_effect=_record):
        output = executor(*args, **kwargs)
    if observed and observed[0].home_only:
        output._replica_home_route = observed[0]
    return output


def route_evidence(output: torch.Tensor, planner: str, *, home_route: ReplicaRoute | None = None) -> dict:
    """Fail if the requested planner was bypassed, and describe its actual plan.

    Args:
        output: Actual expert output.
        planner: Requested CPU or device planner.
        home_route: Observed home-only route when autograd omits that metadata.
    """
    if home_route is None:
        home_route = getattr(output, "_replica_home_route", None)
    if home_route is not None and home_route.plan.transfers:
        raise AssertionError("A home-only route cannot bypass saved replica-plan checks")
    route = saved_route(output) if home_route is None else home_route
    actual = "device" if route.device_plan is not None else "cpu"
    if actual != planner:
        raise AssertionError(f"Requested planner {planner}, executed {actual}")
    return {"planner": actual, "slots": route.plan.slot_to_logical,
            "counts": route.plan.destination_counts, "transfers": len(route.plan.transfers)}


def file_identity(path: Path) -> dict:
    """Identify the file actually loaded rather than an intended build directory.

    Args:
        path: Loaded source or native library path.
    """
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
