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
"""Build an isolated pinned Gate vendor and emit artifact-bound provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import torch

from hyper_parallel.core.multicore._build.prepare_dependencies import (
    verify_git_dependency,
)
from hyper_parallel.core.multicore.backends.launchers import install_launcher_glue
from hyper_parallel.core.multicore.backends.schema import install_cpp_schema
from hyper_parallel.core.multicore.backends.workers import install_worker_glue
from hyper_parallel.core.multicore.runtime.abi import family_abi

_ROOT = Path(__file__).resolve().parents[4]
_MULTICORE = Path("hyper_parallel/core/multicore")
_CMAKE = Path(__file__).with_name("gate")
_VENDOR = "hyper_parallel_multicore_gate_v1"


def _hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export(repository, revision, paths, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "archive", "--format=tar", revision, *map(str, paths)],
            cwd=repository, stdout=archive, check=True,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as source:
            source.extractall(destination, filter="data")


def _assemble(repository, ops_nn, work):
    baseline = json.loads((_ROOT / _MULTICORE / "runtime/baselines/families.json").read_text())["families"]["gate"]
    pinned = work / "pinned"
    paths = set(baseline["source_hashes"])
    paths.update(str(_MULTICORE / path) for path in (
        "cmake/hardening.cmake", "torch/csrc/mega_gate.cpp", "torch/csrc/mega_gate_registration.cpp",
    ))
    _export(repository, baseline["source_revision"], sorted(paths), pinned)
    for relative, expected in baseline["source_hashes"].items():
        if _hash_file(pinned / relative) != expected:
            raise ValueError(f"Pinned Gate source hash mismatch: {relative}")
    dependency = baseline["dependencies"]["ops_nn"]
    verify_git_dependency(dependency, ops_nn, dependency_name="ops_nn")
    source = work / "source"
    source.mkdir()
    for name in ("hyper_mega_gate", "hyper_mega_gate_grad"):
        shutil.copytree(pinned / _MULTICORE / "ops" / name, source / name)
        shutil.copytree(pinned / _MULTICORE / "ops/runtime", source / name / "op_kernel/runtime")
        install_cpp_schema(source / name / "op_kernel/runtime", "gate")
        install_worker_glue(source / name, "gate", "backward" if name.endswith("_grad") else "forward")
    upstream = work / "upstream"
    _export(ops_nn, dependency["commit"], ["index/linear_index/op_host/op_api"], upstream)
    shutil.copytree(
        upstream / "index/linear_index/op_host/op_api",
        source / "hyper_mega_gate_grad/op_host/op_api/linear_index",
    )
    shutil.copytree(pinned / _MULTICORE / "torch/csrc", work / "torch/csrc")
    install_launcher_glue(work / "torch/csrc", "gate")
    return pinned, source


def _run_build(cmake_source, build, options, jobs):
    subprocess.run(["cmake", "-S", str(cmake_source), "-B", str(build), *options], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], check=True)


def _validate_libraries(payload):
    libraries = sorted(payload.rglob("*.so"))
    if len(libraries) != 5:
        raise ValueError(f"Gate payload requires exactly five host ELF libraries, found {len(libraries)}")
    for library in libraries:
        dynamic = subprocess.check_output(["readelf", "-d", str(library)], text=True)
        headers = subprocess.check_output(["readelf", "-W", "-l", str(library)], text=True)
        if "RPATH" in dynamic or "RUNPATH" in dynamic or "TEXTREL" in dynamic:
            raise ValueError(f"Unsafe dynamic ELF section: {library}")
        if "GNU_RELRO" not in headers or "BIND_NOW" not in dynamic or "RWE" in headers:
            raise ValueError(f"Missing ELF hardening: {library}")
    vendor = payload / "vendors" / _VENDOR
    symbols = subprocess.check_output(["nm", "-D", str(vendor / "op_api/lib/libcust_opapi.so")], text=True)
    for name in ("aclnnHyperMegaGateRoute", "aclnnHyperMegaGateRouteGrad"):
        for suffix in ("", "GetWorkspaceSize"):
            if name + suffix not in symbols:
                raise ValueError(f"Missing Gate ACLNN symbol: {name}{suffix}")
    if "aclnnHyperMegaMoe" in symbols:
        raise ValueError("Gate payload must not export legacy MoE entry points")


def _write_manifest(payload, work, soc, cann):
    abi = family_abi("gate")
    inputs = {}
    for directory in (work / "pinned", work / "source", work / "torch", _CMAKE):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                inputs[f"{directory.name}/{path.relative_to(directory)}"] = _hash_file(path)
    inputs["build_gate.py"] = _hash_file(Path(__file__))
    inputs["families.json"] = _hash_file(_ROOT / _MULTICORE / "runtime/baselines/families.json")
    for relative in ("backends/schema.py", "backends/workers.py", "backends/contexts.py",
                     "backends/launchers.py",
                     "runtime/native_calls.json", "runtime/worker_calls.json"):
        inputs[relative] = _hash_file(_ROOT / _MULTICORE / relative)
    identity = {
        "soc": soc, "cann_root": str(cann), "cann_version": (cann / "opp/version.info").read_text(),
        "torch": torch.__version__, "torch_npu": subprocess.check_output(
            [sys.executable, "-c", "from importlib.metadata import version; print(version('torch-npu'))"], text=True,
        ).strip(),
        "torch_cxx11_abi": torch.compiled_with_cxx11_abi(), "host_arch": platform.machine(),
        "python": platform.python_version(), "inputs": inputs,
        "bisheng": subprocess.check_output(["bisheng", "--version"], text=True),
        "gcc": subprocess.check_output(["gcc", "--version"], text=True),
    }
    artifacts = {
        str(path.relative_to(payload)): _hash_file(path)
        for path in sorted(payload.rglob("*")) if path.is_file()
    }
    data = {
        **abi.export_manifest(), "vendor": _VENDOR, "build": identity, "artifacts": artifacts,
        "build_fingerprint": hashlib.sha256(
            json.dumps({"build": identity, "artifacts": artifacts}, sort_keys=True).encode(),
        ).hexdigest(),
    }
    data.pop("native_build_fingerprint")
    (payload / "manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Export verified baseline inputs and build only Route/RouteGrad for one SoC."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repository", type=Path, default=_ROOT)
    parser.add_argument("--ops-nn-source", type=Path, required=True)
    parser.add_argument("--soc", choices=("ascend910b", "ascend910_93"), default="ascend910b")
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "")).resolve()
    if not (cann / "opp/version.info").is_file():
        parser.error("Source the selected CANN set_env.sh before building")
    version = (cann / "opp/version.info").read_text().splitlines()[0].removeprefix("Version=")
    if tuple(int(number) for number in version.split(".")[:2]) < (9, 2):
        parser.error("The pinned Gate tiling uses TensorShape/TensorDataType and requires CANN >=9.2")
    output = _ROOT / "build/native/gate"
    work = output / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    pinned, source = _assemble(args.source_repository, args.ops_nn_source, work)
    options = [
        "-DCMAKE_BUILD_TYPE=Release", f"-DASCEND_CANN_PACKAGE_PATH={cann}",
        f"-DASCEND_COMPUTE_UNIT={args.soc}", f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
        f"-DHP_MULTICORE_SOURCE_ROOT={source}", f"-DHP_GATE_PINNED_ROOT={pinned}",
        f"-DHP_VENDOR_STAGE_ROOT={work / 'stage'}",
    ]
    _run_build(_CMAKE, work / "vendor-build", options, args.jobs)
    subprocess.run(["cmake", "--build", str(work / "vendor-build"), "--parallel", str(args.jobs),
                    "--target", "binary"], check=True)
    subprocess.run(["cmake", "--install", str(work / "vendor-build")], check=True)
    payload = work / "payload"
    vendor = payload / "vendors" / _VENDOR
    shutil.copytree(work / "stage/packages/vendors" / _VENDOR, vendor)
    _run_build(_CMAKE / "torch", work / "torch-build", [
        "-DCMAKE_BUILD_TYPE=Release", f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DHP_GATE_PINNED_ROOT={pinned}", f"-DHP_MULTICORE_VENDOR_ROOT={vendor}",
        f"-DHP_GATE_TORCH_SOURCE_ROOT={work / 'torch'}",
        f"-DCMAKE_INSTALL_PREFIX={payload / 'framework/torch'}",
    ], args.jobs)
    subprocess.run(["cmake", "--install", str(work / "torch-build")], check=True)
    shutil.copy2(Path(__file__).parent / "gate/set_env.bash", payload / "set_env.bash")
    _validate_libraries(payload)
    for name in ("hyper_mega_gate_route", "hyper_mega_gate_route_grad"):
        if not list(vendor.glob(f"op_impl/ai_core/tbe/kernel/{args.soc}/{name}/*.o")):
            raise ValueError(f"Missing compiled Gate device binary: {name}")
    _write_manifest(payload, work, args.soc, cann)
    installed = output / "payload"
    if installed.exists():
        shutil.rmtree(installed)
    shutil.move(payload, installed)
    print(f"Gate payload ready: {installed}; activate set_env.bash before starting Python")


if __name__ == "__main__":
    main()
