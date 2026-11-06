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
"""Validate customer imports and native selection without running NPU kernels."""

# Standalone accelerator workers use the native Torch APIs required by the project.
# pylint: disable=forbidden-backend-import
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import torch
import torch_npu
import transformers
import triton
from launch import host_architecture, verify_elf_architecture
from transformers import LongcatFlashConfig
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashExperts

import hyper_parallel
from hyper_parallel.core.multicore._loader import get_multicore_paths
from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts
from hyper_parallel.trainer.text_trainer import TextTrainer


def dependency_versions() -> dict[str, str]:
    """Require the dependency profile used by the reference experiment."""
    versions = {}
    for line in (Path(__file__).parent / "requirements-no-torch.lock").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        actual = metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Private dependency mismatch: {name} expected={expected} actual={actual}")
        versions[name] = actual
    return versions


def verify_runtime_contract(repo: Path, native_root: Path) -> tuple[Path, Path]:
    """Verify native selection and required public NPU operator entrypoints.

    Args:
        repo: Editable model checkout.
        native_root: Selected compiled payload root.
    """
    architecture = host_architecture()
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("This supplement requires Linux CPython3.11 on x86_64 or aarch64")
    if Path(hyper_parallel.__file__).resolve() != repo / "hyper_parallel/__init__.py":
        raise RuntimeError("Python imports another checkout; install this checkout editable with --no-deps first")
    if inspect.signature(LongcatFlashMegaMoeExperts.__init__).parameters["swiglu_precision"].default != "fused":
        raise RuntimeError("LongCat fused default is missing")
    if not str(torch.__version__).startswith("2.9."):
        raise RuntimeError("This comparison requires an already working Torch/PTA2.9 native build")
    names = ["npu_grouped_matmul", "npu_swiglu", "npu_moe_token_permute", "npu_moe_token_unpermute"]
    if any(not callable(getattr(torch_npu, name, None)) for name in names):
        raise RuntimeError("One or more required native NPU big operators are unavailable")
    if not callable(TextTrainer) or not callable(LongcatFlashExperts) or LongcatFlashConfig().moe_topk != 12:
        raise RuntimeError("LongCat or Trainer import contract changed")
    vendor, adapter = get_multicore_paths()
    for path in (vendor, adapter):
        if not Path(path).resolve().is_relative_to(native_root):
            raise RuntimeError(f"Loaded native path is outside --native-root: {path}")
    host_libraries = [
        adapter,
        vendor / "op_api/lib/libcust_opapi.so",
        vendor / "op_impl/ai_core/tbe/op_tiling/liboptiling.so",
    ]
    shmem_root = adapter.parents[3] / "shmem/lib"
    host_libraries.extend(
        (
            shmem_root / "libhyper_parallel_shmem_ops.so",
            shmem_root / "shmem/libhyper_parallel_shmem.so",
            shmem_root / "framework/torch/hyper_parallel_shmem_torch.so",
        )
    )
    for library in host_libraries:
        verify_elf_architecture(library, architecture)
    return vendor, adapter


def main() -> None:
    """Print one JSON receipt; leave all device execution to the precision phase."""
    repo, native_root = map(lambda value: Path(value).resolve(), sys.argv[1:3])
    vendor, adapter = verify_runtime_contract(repo, native_root)
    versions = dependency_versions()
    # Check editable registration without relying on a repository PYTHONPATH entry.
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    imported = subprocess.check_output(
        [sys.executable, "-c", "import hyper_parallel; print(hyper_parallel.__file__)"],
        cwd=tempfile.gettempdir(),
        env=clean_env,
        text=True,
    ).strip()
    if Path(imported).resolve() != repo / "hyper_parallel/__init__.py":
        raise RuntimeError(
            "Editable installation points elsewhere. Run: PYTHON -m pip install --no-deps --no-build-isolation -e REPO"
        )
    cann_root = Path(os.environ["ASCEND_HOME_PATH"])
    cann_versions = {
        str(path.relative_to(cann_root)): path.read_text(encoding="utf-8")
        for path in [cann_root / "version.info", cann_root / "opp/version.info"]
        if path.is_file()
    }
    result = {
        "repo": str(repo),
        "python": platform.python_version(),
        "host_architecture": host_architecture(),
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "transformers": transformers.__version__,
        "triton": triton.__version__,
        "dependency_mode": os.environ.get("HP_REPRO_DEPENDENCY_MODE", "existing"),
        "dependency_versions": versions,
        "cann_root": str(cann_root),
        "cann_versions": cann_versions,
        "vendor": str(vendor),
        "adapter": str(adapter),
        "adapter_sha256": hashlib.sha256(Path(adapter).read_bytes()).hexdigest(),
        "customer_torch_and_pta_preserved": True,
        "npu_kernels_executed_by_preflight": False,
        "reference_runtime": {"torch": "2.9.1+cpu", "torch_npu": "2.9.1", "cann": "8.5.0"},
        "numerical_precision_phase_required": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
