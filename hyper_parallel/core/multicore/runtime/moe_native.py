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
"""Verify the native MoE family and private SHMEM artifacts before AST materialization."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields
from importlib.metadata import version
from pathlib import Path

import torch

from hyper_parallel.core.multicore._loader import get_multicore_paths
from hyper_parallel.core.multicore.runtime.abi import NativeManifest, family_abi


def verify_moe_native() -> dict[str, object]:
    """Check activated component paths, native ABI and all build-sealed artifacts."""
    vendor, _ = get_multicore_paths()
    configured = [Path(value).resolve() for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
                  if value]
    if not configured or configured[0] != vendor:
        raise ValueError("AST MoE requires its sealed vendor first in ASCEND_CUSTOM_OPP_PATH")
    multicore = vendor.parent.parent
    data = json.loads((multicore / "frontend_manifest.json").read_text(encoding="utf-8"))
    family_abi("moe").verify_native(NativeManifest(**{
        field.name: data[field.name] for field in fields(NativeManifest)}))
    expected = hashlib.sha256(json.dumps({"build": data["build"], "artifacts": data["artifacts"]},
                                         sort_keys=True).encode()).hexdigest()
    if expected != data["build_fingerprint"]:
        raise ValueError("MoE native build fingerprint mismatch")
    roots = {"multicore": multicore, "shmem": (multicore.parent / "shmem/lib").resolve()}
    if set(data["artifacts"]) != set(roots):
        raise ValueError("MoE native manifest requires both component artifact sets")
    _verify_artifacts(data["artifacts"], roots)
    _verify_framework(data["build"])
    return data


def _verify_artifacts(components, roots):
    for name, root in roots.items():
        artifacts = components[name]
        if not artifacts:
            raise ValueError(f"Empty MoE artifact set: {name}")
        for relative, digest in artifacts.items():
            path = (root / relative).resolve()
            if (not path.is_relative_to(root) or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
                raise ValueError(f"MoE artifact missing, escaping or corrupted: {name}/{relative}")
        actual = {str(path.relative_to(root)) for path in root.rglob("*")
                  if path.is_file() and path.name != "frontend_manifest.json"}
        if actual != set(artifacts):
            raise ValueError(f"MoE artifact set contains unrecorded files: {name}")


def _verify_framework(build):
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "")).resolve()
    if str(cann) != build["cann_root"] or (cann / "opp/version.info").read_text() != build["cann_version"]:
        raise ValueError("MoE native CANN identity mismatch")
    if (torch.__version__ != build["torch"] or version("torch-npu") != build["torch_npu"]
            or torch.compiled_with_cxx11_abi() != build["torch_cxx11_abi"]):
        raise ValueError("MoE native framework/C++ ABI mismatch")
