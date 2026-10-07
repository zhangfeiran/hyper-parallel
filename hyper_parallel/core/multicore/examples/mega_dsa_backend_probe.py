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
"""Print DSA package identity and registered ABI without launching device work."""

import hashlib
import json
import os
import platform
import subprocess
from importlib import import_module, metadata
from pathlib import Path

import torch

_REQUIRED_OPS = (
    "npu_lightning_indexer_enhance",
    "npu_sparse_flash_attention_enhance",
    "npu_sparse_flash_attention_grad_enhance",
    "npu_sparse_lightning_indexer_grad_kl_loss_enhance",
)


def probe_environment() -> dict:
    """Record exact schemas and library hashes; registration is not numerical validation."""
    root = Path(__file__).resolve().parents[4]
    lock_path = root / "hyper_parallel/core/multicore/_build/dependencies.lock.json"
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    versions = {}
    for package in ("torch", "torch-npu", "omni_training_custom_ops", "hyper_parallel"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "checkout": str(root),
        "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True).strip()),
        "source_lock_commits": {name: spec["commit"] for name, spec in
                                source_lock["components"]["multicore"].items()},
        "python": platform.python_version(),
        "torch_runtime": torch.__version__,
        "versions": versions,
        "cann_environment": {name: os.environ.get(name) for name in
                             ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_TOOLKIT_HOME",
                              "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH")},
        "device_execution": False,
    }
    # The optional registration package is inspected explicitly, never loaded by the CPU oracle.
    try:
        package = import_module("omni_training_custom_ops")
    except (ImportError, OSError, RuntimeError) as error:
        result["registration_error"] = f"{type(error).__name__}: {error}"
        return result
    package_root = Path(package.__file__).resolve().parent
    result["omni_package"] = str(package_root)
    libraries = {}
    for library in sorted(package_root.rglob("*.so")):
        digest = hashlib.sha256()
        with library.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        libraries[str(library.relative_to(package_root))] = digest.hexdigest()
    result["library_sha256"] = libraries
    schemas = {}
    for name in _REQUIRED_OPS:
        op = getattr(torch.ops.custom, name, None)
        schemas[name] = None if op is None else {overload: str(getattr(op, overload)._schema)
                                               for overload in op.overloads()}
    result["schemas"] = schemas
    result["all_required_ops_registered"] = all(schemas.values())
    return result


if __name__ == "__main__":
    print(json.dumps(probe_environment(), indent=2, sort_keys=True))
