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
"""Build sealed dense host-stream native adapters from frontend source artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from tempfile import NamedTemporaryFile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from hyper_parallel.core.multicore.backends.dense import dense_artifacts
from hyper_parallel.core.multicore.runtime.dense import DenseKernelPlan


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_build_identity() -> dict[str, object]:
    """Capture native adapter toolchain and framework ABI without initializing devices."""
    compiler = os.environ.get("CXX", "c++")
    return {"torch": torch.__version__, "torch_cxx11_abi": torch.compiled_with_cxx11_abi(),
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "flags": ["-O2", "-std=c++17"]}


def build_dense_payload(plan: DenseKernelPlan, root: Path) -> Path:
    """Compile generated provider calls and VJPs in a caller-owned cache directory.

    Args:
        plan: Canonical dense plan whose definition determines the generated source.
        root: Writable native cache root, independent of framework site-packages.

    Returns:
        Manifest path sealing the generated source and native shared library.
    """
    definition_key, _, files = dense_artifacts(plan)
    source = files["native/dense.cpp"]
    build = dense_build_identity()
    identity = {"definition_key": definition_key, "source_sha256": hashlib.sha256(source).hexdigest(),
                "build": build}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(root).resolve() / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        library = (directory / recorded["library"]).resolve()
        if (not library.is_relative_to(directory) or recorded["identity"] != identity
                or _hash(directory / "dense.cpp") != identity["source_sha256"]
                or _hash(library) != recorded["library_sha256"]):
            raise ValueError("Dense native cache identity/source/library mismatch")
        return manifest
    source_path = directory / "dense.cpp"
    source_path.write_bytes(source)
    name = "hp_dense_" + definition_key[:24]
    if getattr(getattr(torch.ops, name), "forward", None) is not None:
        raise ValueError("Dense operator namespace is already registered; reuse its owning native cache")
    library = Path(load(name=name, sources=[str(source_path)], build_directory=str(directory),
                        extra_cflags=build["flags"], is_python_module=False, verbose=False))
    record = {"format_version": 1, "namespace": name, "identity": identity,
              "execution_mode": "native_host_stream_adapter", "library": library.name,
              "library_sha256": _hash(library)}
    with NamedTemporaryFile(mode="w", dir=directory, suffix=".manifest.tmp", delete=False, encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
        temporary = Path(stream.name)
    temporary.replace(manifest)
    return manifest
