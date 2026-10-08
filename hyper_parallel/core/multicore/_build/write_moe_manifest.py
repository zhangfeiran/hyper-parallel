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
"""Seal the current MoE/SHMEM native closure for AST compatibility binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import fields
from importlib.metadata import version
from pathlib import Path

import torch

from hyper_parallel.core.multicore.backends.schema import adapt_cpp_schema
from hyper_parallel.core.multicore.runtime.abi import NativeManifest, family_abi

_REPO = Path(__file__).resolve().parents[4]
_COMPONENT = _REPO / "hyper_parallel/core/multicore"


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources():
    baseline = json.loads((_COMPONENT / "runtime/baselines/families.json").read_text())["families"]["moe"]
    for relative, expected in baseline["source_hashes"].items():
        if "/ops/" in relative and _hash(_REPO / relative) != expected:
            _verify_generated_header(relative, expected, baseline["source_revision"])
    sources = {}
    for directory in ("ops", "torch/csrc", "cmake", "shmem/ccsrc", "shmem/cmake"):
        for path in sorted((_COMPONENT / directory).rglob("*")):
            if path.is_file():
                sources[str(path.relative_to(_COMPONENT))] = _hash(path)
    for relative in ("build.sh", "_build/dependencies.lock.json", "_build/write_moe_manifest.py",
                     "backends/schema.py", "runtime/native_calls.json"):
        sources[relative] = _hash(_COMPONENT / relative)
    return sources


def _verify_generated_header(relative, expected, revision):
    if relative != "hyper_parallel/core/multicore/ops/runtime/runtime_config.hpp":
        raise ValueError(f"MoE pinned native source mismatch: {relative}")
    original = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=_REPO)
    if hashlib.sha256(original).hexdigest() != expected:
        raise ValueError("MoE original header source identity mismatch")
    if adapt_cpp_schema(original.decode(), "moe") != (_REPO / relative).read_text():
        raise ValueError("MoE generated header adapter mismatch")


def write_manifest(multicore: Path, shmem: Path, socs: str) -> None:
    """Seal native sources, dependencies, toolchain and both component artifact sets.

    Args:
        multicore: Built multicore library root.
        shmem: Built private SHMEM library root.
        socs: Comma-separated compiled CANN SoCs.
    """
    cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    build = {
        "socs": socs.split(","), "sources": _sources(), "host_arch": platform.machine(),
        "torch": str(torch.__version__), "torch_npu": version("torch-npu"),
        "torch_cxx11_abi": torch.compiled_with_cxx11_abi(), "python": platform.python_version(),
        "cann_root": str(cann), "cann_version": (cann / "opp/version.info").read_text(),
        "bisheng": subprocess.check_output(["bisheng", "--version"], text=True),
    }
    artifacts = {name: {str(path.relative_to(root)): _hash(path) for path in sorted(root.rglob("*"))
                        if path.is_file() and path.name != "frontend_manifest.json"}
                 for name, root in (("multicore", multicore), ("shmem", shmem))}
    if not artifacts["multicore"] or not artifacts["shmem"]:
        raise ValueError("Both MoE and private SHMEM artifact sets must be present")
    abi = family_abi("moe")
    identity = {field.name: getattr(abi, field.name) for field in fields(NativeManifest)
                if field.name != "build_fingerprint"}
    data = {**identity, "source_revision": abi.source_revision, "build": build, "artifacts": artifacts,
            "build_fingerprint": hashlib.sha256(json.dumps(
                {"build": build, "artifacts": artifacts}, sort_keys=True).encode()).hexdigest()}
    (multicore / "frontend_manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True)+"\n")


def main() -> None:
    """Emit provenance after the existing native build and ELF checks succeed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multicore-root", required=True, type=Path)
    parser.add_argument("--shmem-root", required=True, type=Path)
    parser.add_argument("--soc-list", required=True)
    args = parser.parse_args()
    write_manifest(args.multicore_root, args.shmem_root, args.soc_list)


if __name__ == "__main__":
    main()
