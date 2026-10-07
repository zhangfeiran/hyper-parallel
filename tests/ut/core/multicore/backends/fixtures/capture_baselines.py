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
"""Reproduce ABI and Gate wire snapshots from the design's exact Git objects.

This developer tool executes trusted, fixed-revision legacy builders in an
isolated namespace. It never executes user DSL source and requires no NPU imports.
"""

from __future__ import annotations

import ast
import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[6]
CORE = "hyper_parallel/core/multicore/"
REVISIONS = {
    "moe": "befaa01069e997d511987feec82464c5f4549b95",
    "mhc": "979e2a9ac913413e361f4fc2dd9987766af8ddb4",
    "gate": "890a00ac71226594a7c69c4980ad7ac50bd8896c",
    "little_kernel": "d9c29909d7fcd1202d3f676e84c9c3d456308ca2",
}
STRUCTURES = ("TensorDescC", "TaskDescC", "EventDescC", "DynamicDataC", "RuntimeConfigC")
SCALAR_TYPES = {ctypes.c_uint32: "uint32", ctypes.c_int32: "int32", ctypes.c_int64: "int64", ctypes.c_uint64: "uint64"}
COMMON_NAMES = {
    "TASK_TERMINATE": "runtime.terminate",
    "TASK_BEGIN_TASK_GRAPH": "runtime.begin",
    "TASK_ADD_CUSTOM": "runtime.add_custom",
}


def _git(*arguments):
    return subprocess.run(["git", *arguments], cwd=ROOT, check=True, capture_output=True).stdout


def _read(family, path):
    return _git("show", f"{REVISIONS[family]}:{CORE}{path}")


def _trusted_module(content, namespace, names=None):
    tree = ast.parse(content)
    tree.body = [node for node in tree.body if _selected_node(node, names)]
    # pylint: disable=exec-used
    exec(compile(tree, "<pinned-legacy-builder>", "exec"), namespace)  # noqa: S102 - trusted pinned Git objects
    # pylint: enable=exec-used


def _selected_node(node, names):
    if names is not None:
        if isinstance(node, ast.FunctionDef):
            return node.name in names
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        )
    if isinstance(node, ast.ImportFrom):
        return not (node.module or "").startswith("hyper_parallel")
    if isinstance(node, ast.Import):
        return all(alias.name != "torch_npu" for alias in node.names)
    return True


def _logical_name(native):
    if native in COMMON_NAMES:
        return COMMON_NAMES[native] + ".v1"
    if native.startswith("TASK_GATE_"):
        return "gate." + native.removeprefix("TASK_GATE_").lower().replace("grad_", "grad.", 1) + ".v1"
    if native == "TASK_MEGA_GATE_ROUTE":
        return "gate.route.v1"
    if native.startswith(("TASK_MHC_", "TASK_RMS_")):
        return "mhc." + native.removeprefix("TASK_MHC_").removeprefix("TASK_").lower() + ".v1"
    return "moe." + native.removeprefix("TASK_").lower() + ".v1"


def _struct_schema(structure):
    fields = []
    for name, field_type in structure._fields_:
        count = field_type._length_ if issubclass(field_type, ctypes.Array) else 1
        element = field_type._type_ if issubclass(field_type, ctypes.Array) else field_type
        fields.append(
            {
                "name": name,
                "type": SCALAR_TYPES.get(element, element.__name__),
                "count": count,
                "offset": getattr(structure, name).offset,
            }
        )
    return {"name": structure.__name__, "size": ctypes.sizeof(structure), "fields": fields}


def _source_closure(family):
    paths = _git("ls-tree", "-r", "--name-only", REVISIONS[family], CORE).decode().splitlines()
    prefixes = (
        CORE + "ops/runtime/",
        CORE + f"ops/hyper_mega_{family}",
        CORE + f"modules/mega_{family}",
        CORE + "scheduler/",
        CORE + "profiler/",
        CORE + "_build/",
    )
    selected = [path for path in paths if path.startswith(prefixes) or path.endswith(".patch")]
    return {path: hashlib.sha256(_git("show", f"{REVISIONS[family]}:{path}")).hexdigest() for path in selected}


def _family_schema(family):
    module = ModuleType(f"pinned_{family}_config")
    sys.modules[f"pinned_{family}_config"] = module
    namespace = vars(module)
    _trusted_module(_read(family, "scheduler/config.py"), namespace)
    lock = json.loads(_read(family, "_build/dependencies.lock.json"))
    dependencies = lock["components"]["multicore"]
    for dependency in dependencies.values():
        for patch in dependency.get("patches", []):
            actual = hashlib.sha256(_git("show", f"{REVISIONS[family]}:{patch['path']}")).hexdigest()
            if actual != patch["sha256"]:
                raise ValueError(f"Pinned dependency patch mismatch: {family}, {patch['path']}")
    native = _read(family, "ops/runtime/runtime_config.hpp").decode()
    prefix = native[: native.index("__aicore__")]
    comment = prefix[: prefix.index("#ifndef")]
    declarations = prefix[prefix.index("namespace MulticoreRuntime {") :]
    fixture = ROOT / f"tests/ut/core/multicore/backends/fixtures/{family}_runtime_layout.txt"
    fixture.write_text(comment + "#include <cstdint>\n#include <cstddef>\n" + declarations + "\n}\n", encoding="utf-8")
    schema = {
        "family": family,
        "runtime_abi_version": 1,
        "source_revision": REVISIONS[family],
        "task_types": [
            {"native_name": item.name, "numeric_id": item.value, "logical_name": _logical_name(item.name)}
            for item in namespace["TaskType"]
        ],
        "structures": [_struct_schema(namespace[name]) for name in STRUCTURES],
        "constants": {
            name: namespace[name]
            for name in ("MIN_EVENT_CAPACITY", "NUM_WORKERS_VECTOR", "NUM_WORKERS_CUBE", "ATOMIC_ADD_VALUE_LEN")
        },
        "scratch_policy": "dynamic_local_experts" if family == "moe" else "fixed_512_int64",
        "dependencies": dependencies,
        "source_hashes": _source_closure(family),
    }
    return schema, namespace


def _gate_snapshots(namespace):
    _trusted_module(_read("gate", "scheduler/runtime.py"), namespace)
    _trusted_module(_read("gate", "profiler/profiling.py"), namespace)
    profile_names = {
        "MEGA_GATE_PROFILE_OWNER_LABEL",
        "MEGA_GATE_STAGE_NAMES",
        "MEGA_GATE_ROUTE_GRAD_STAGE_NAMES",
        "MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES",
        "_configure_mega_gate_profile_metadata",
        "_configure_mega_gate_grad_profile_metadata",
    }
    _trusted_module(_read("gate", "modules/mega_gate/profiling.py"), namespace, profile_names)
    plan_names = {
        "MEGA_GATE_AIV_WORKER_CAPACITY",
        "MEGA_GATE_TASK_TYPES",
        "MEGA_GATE_ROUTE_GRAD_TASK_TYPES",
        "MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES",
        "_build_pipeline_runtime_config",
        "_build_runtime_config",
        "_build_grad_runtime_config",
        "_partition_token_rows",
    }
    _trusted_module(_read("gate", "modules/mega_gate/plan.py"), namespace, plan_names)
    sequences = {
        "forward": ("MEGA_GATE_TASK_TYPES", "MEGA_GATE_STAGE_NAMES", "HyperMegaGateRoute"),
        "backward": ("MEGA_GATE_ROUTE_GRAD_TASK_TYPES", "MEGA_GATE_ROUTE_GRAD_STAGE_NAMES", "HyperMegaGateRouteGrad"),
        "backward_k1": (
            "MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES",
            "MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES",
            "HyperMegaGateRouteGrad",
        ),
    }
    result = {"source_revision": REVISIONS["gate"], "sequences": {}}
    fixture_path = ROOT / "tests/ut/core/multicore/backends/fixtures"
    for name, (task_key, stage_key, kernel) in sequences.items():
        tasks, stages = namespace[task_key], namespace[stage_key]
        config = namespace["_build_pipeline_runtime_config"](tasks, stages, kernel_name=kernel)
        # Execute the original host profile-layout calculation, including row-worker broadcasting.
        namespace["_configure_profile_layout"](config)
        images = {}
        for profiled in (False, True):
            config.cycle_profiling_enabled = int(profiled)
            image = namespace["serialize_runtime_config"](config)
            suffix = "profiled" if profiled else "normal"
            (fixture_path / f"gate_{name}_{suffix}.bin").write_bytes(image)
            images[suffix] = {"bytes": len(image), "sha256": hashlib.sha256(image).hexdigest()}
        result["sequences"][name] = {
            "kernel_name": kernel,
            "stage_names": stages,
            "task_types": [task.name for task in tasks],
            "images": images,
        }
    return result


def main() -> None:
    """Write deterministic schemas and original Gate descriptor images from pinned Git sources."""
    families = {}
    for family in ("moe", "mhc", "gate"):
        schema, namespace = _family_schema(family)
        families[family] = schema
        if family == "gate":
            snapshots = _gate_snapshots(namespace)
            schema["pipelines"] = snapshots["sequences"]
            fixture = ROOT / "tests/ut/core/multicore/backends/fixtures/gate_snapshots.json"
            fixture.write_text(json.dumps(snapshots, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"manifest_version": 1, "design_revisions": REVISIONS, "families": families}
    destination = ROOT / (CORE + "runtime/baselines/families.json")
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
