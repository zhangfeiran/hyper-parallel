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
"""Common source emission for Gate WorkerPipeline and MoE/MHC TaskDAG plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from hyper_parallel.core.multicore.backends.schema import (
    cpp_calls,
    cpp_schema,
    python_calls,
    python_schema,
)
from hyper_parallel.core.multicore.ir.program import ProgramIR
from hyper_parallel.core.multicore.runtime.abi import family_abi
from hyper_parallel.core.multicore.runtime.mhc import MhcKernelPlan
from hyper_parallel.core.multicore.runtime.moe import MoeKernelPlan
from hyper_parallel.core.multicore.runtime.plan import KernelPlan

CORE = Path(__file__).resolve().parents[1]


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"


def _hash(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class EmittedFile:
    """One immutable generated source, descriptor or mapping artifact."""

    path: str
    content: bytes


@dataclass(frozen=True)
class BackendEmission:
    """Host plan and reproducible source artifacts; no device pointers or binary claim."""

    plan: KernelPlan | MoeKernelPlan | MhcKernelPlan
    definition_key: str
    plan_key: str
    files: tuple[EmittedFile, ...]

    @property
    def artifact_key(self) -> str:
        """Include source mappings in disk identity without polluting native definition identity."""
        return _hash(_json({"definition_key": self.definition_key, "plan_key": self.plan_key,
                           "family": self.plan.export_manifest()["family"],
                           "artifacts": {file.path: _hash(file.content) for file in self.files}}))

    def export_manifest(self) -> dict[str, object]:
        """Expose separate definition, static plan and source-mapped artifact identities."""
        return {"format_version": 1, "status": "source_only", "definition_key": self.definition_key,
                "plan_key": self.plan_key, "artifact_key": self.artifact_key,
                "family": self.plan.export_manifest()["family"],
                "artifacts": {file.path: _hash(file.content) for file in self.files}}


def emit_plan(plan: KernelPlan | MoeKernelPlan | MhcKernelPlan, ir: ProgramIR) -> BackendEmission:
    """Emit each supported family through the same typed schema and artifact pipeline.

    Args:
        plan: Complete family-specific host schedule with original backward/finalize contracts.
        ir: Typed semantic computation defining this plan.
    """
    family = plan.export_manifest()["family"]
    if family not in {"gate", "mhc", "moe"}:
        raise ValueError("Ascend emission requires a supported native family")
    definition_key = _hash(_json(_definition(ir, family)))
    images = _images(plan)
    plan_key = _hash(_json({"definition": definition_key, "static": plan.export_manifest(),
                           "images": {name: _hash(content) for name, content in images.items()}}))
    calls = json.loads((CORE / "runtime/native_calls.json").read_text())
    bindings = {name: entry for name, entry in calls["entries"].items() if entry["family"] == family}
    files = {f"schema/{family}_abi.py": python_schema(family).encode(),
             f"schema/{family}_schema.hpp": cpp_schema(family).encode(),
             "bindings/native_calls.py": python_calls().encode(),
             "bindings/native_calls.hpp": cpp_calls().encode(), "bindings/schema.json": _json(bindings),
             "plan.json": _json(plan.export_manifest()), "source_map.json": _json(json.loads(plan.explain())),
             **images}
    return BackendEmission(plan, definition_key, plan_key,
                           tuple(EmittedFile(name, content) for name, content in sorted(files.items())))


def _definition(ir, family):
    semantic = json.loads(ir.dump())
    for operation in semantic["operations"]:
        operation.pop("source")
        operation.pop("call_chain")
    inputs = ("backends/schema.py", "backends/ascend.py", "runtime/bindings.py", "runtime/native_calls.json")
    return {"generator_version": 1, "target": "ascend", "semantic": semantic,
            "abi": family_abi(family).export_manifest(),
            "generator_inputs": {name: _hash((CORE / name).read_bytes()) for name in inputs}}


def _images(plan):
    result = {}
    for direction in ("forward", "backward", "backward_no_replica"):
        image = getattr(plan, direction, None)
        if image is not None:
            result[f"schedule/{direction}.bin"] = image.normal
            result[f"schedule/{direction}.profiled.bin"] = image.profiled
    return result
