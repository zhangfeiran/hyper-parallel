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
"""Collect actual customer runtime identity without pinning or replacing packages."""

import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from launch import host_architecture, verify_elf_architecture


def verify_editable(repo: Path, imported_file: str) -> str:
    """Verify actual editable imports outside the source directory and PYTHONPATH.

    Args:
        repo: Required source checkout.
        imported_file: Package file imported by the probe itself.
    """
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    clean_import = subprocess.check_output(
        [sys.executable, "-I", "-c", "import hyper_parallel; print(hyper_parallel.__file__)"],
        cwd=tempfile.gettempdir(), env=environment, text=True,
    ).strip()
    expected = repo / "hyper_parallel/__init__.py"
    if Path(clean_import).resolve() != expected or Path(imported_file).resolve() != expected:
        raise RuntimeError("Editable install must point to this checkout; "
                           "rerun with --build or install it --no-deps -e")
    return clean_import


def main() -> None:
    """Verify editable provenance, NPU entries and selected native host architecture."""
    repo, output = (Path(value).resolve() for value in sys.argv[1:3])
    receipt = {"python": sys.version, "host_arch": platform.machine(), "cann_root": os.getenv("ASCEND_HOME_PATH")}
    try:
        modules = {name: importlib.import_module(name) for name in
                   ("torch", "torch_npu", "transformers", "triton", "hyper_parallel")}
        receipt["packages"] = {name: {"version": str(getattr(module, "__version__", "unknown")),
                                       "file": str(getattr(module, "__file__", "unknown"))}
                               for name, module in modules.items()}
        receipt["editable_import_file"] = verify_editable(repo, modules["hyper_parallel"].__file__)
        native = importlib.import_module("hyper_parallel.core.multicore._loader")
        vendor, adapter = native.get_multicore_paths()
        shmem = adapter.parents[3] / "shmem/lib"
        libraries = [adapter, vendor / "op_api/lib/libcust_opapi.so",
                     vendor / "op_impl/ai_core/tbe/op_tiling/liboptiling.so",
                     shmem / "libhyper_parallel_shmem_ops.so", shmem / "shmem/libhyper_parallel_shmem.so",
                     shmem / "framework/torch/hyper_parallel_shmem_torch.so"]
        for library in libraries:
            if not library.resolve().is_relative_to(repo):
                raise RuntimeError(f"Selected a native library outside this checkout: {library}")
            verify_elf_architecture(library, host_architecture())
        receipt.update(vendor=str(vendor), adapter=str(adapter))
        npu = modules["torch"].npu
        if npu.device_count() < int(sys.argv[3]):
            raise RuntimeError("Fewer visible logical NPUs than requested")
        operators = ("npu_grouped_matmul", "npu_swiglu", "npu_moe_token_permute", "npu_moe_token_unpermute")
        if any(not callable(getattr(modules["torch_npu"], name, None)) for name in operators):
            raise RuntimeError("Required native MoE operator entrypoint is unavailable")
        receipt.update(chip=npu.get_device_name(0), device_limits=npu.get_device_limit(0), status="PASS")
        cann = Path(os.environ["ASCEND_HOME_PATH"])
        receipt["cann_versions"] = {str(path.relative_to(cann)): path.read_text(encoding="utf-8")
                                    for path in (cann / "version.info", cann / "opp/version.info") if path.is_file()}
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError, KeyError,
            TypeError, subprocess.SubprocessError) as error:
        receipt.update(status="ERROR", error=str(error))
        raise
    finally:
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
