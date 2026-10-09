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
"""Reproducible dense provider and task-plan emission without native binary claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hyper_parallel.core.multicore.backends.dense_codegen import generate_dense_cpp
from hyper_parallel.core.multicore.runtime.dense import DenseKernelPlan

CORE = Path(__file__).resolve().parents[1]


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def dense_artifacts(plan: DenseKernelPlan) -> tuple[str, str, dict[str, bytes]]:
    """Seal semantic definitions separately from bound shapes and source locations.

    Args:
        plan: Canonically lowered dense task DAG.
    """
    semantic = json.loads(plan.ir.dump())
    for operation in semantic["operations"]:
        operation.pop("source")
        operation.pop("call_chain")
    inputs = ("primitives/dense.py", "compiler/dense.py", "runtime/dense.py", "runtime/dense_native.py",
              "runtime/dense_compiled.py", "runtime/dense_execution.py", "_build/build_dense.py",
              "backends/providers.py",
              "backends/dense.py", "backends/dense_codegen.py")
    providers = {task.provider.name: task.provider.export_manifest() for task in plan.tasks}
    definition = {"semantic": semantic, "providers": providers,
                  "sources": {name: hashlib.sha256((CORE / name).read_bytes()).hexdigest() for name in inputs}}
    definition_key = hashlib.sha256(_json(definition)).hexdigest()
    manifest = plan.export_manifest()
    plan_key = hashlib.sha256(_json({"definition_key": definition_key, "plan": manifest})).hexdigest()
    files = {"plan.json": _json(manifest), "bindings/providers.json": _json(providers),
             "source_map.json": plan.explain().encode(), "definition.json": _json(definition)}
    files["native/dense.cpp"] = generate_dense_cpp(plan, "hp_dense_" + definition_key[:24]).encode()
    return definition_key, plan_key, files
