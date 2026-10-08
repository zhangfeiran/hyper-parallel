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
"""Build an isolated pinned MHC vendor and emit artifact-bound provenance."""

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
    _git_archive_sha256,
    verify_git_dependency,
)
from hyper_parallel.core.multicore.backends.schema import install_cpp_schema
from hyper_parallel.core.multicore.runtime.abi import family_abi

_ROOT = Path(__file__).resolve().parents[4]
_MULTICORE = Path("hyper_parallel/core/multicore")
_CMAKE = Path(__file__).with_name("mhc")
_VENDOR = "hyper_parallel_multicore_mhc_v1"


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


def _verify_dependency(dependency, repository, name):
    archive = _git_archive_sha256(repository, dependency["commit"],
                                 f"{Path(dependency['repository']).stem}-{dependency['version']}/")
    allowed = {dependency["git_archive_tar_sha256"], *dependency.get("git_archive_tar_sha256_compat", [])}
    if archive not in allowed:
        raise ValueError(f"MHC dependency archive mismatch: {name}: {archive}")
    verify_git_dependency({**dependency, "git_archive_tar_sha256": archive}, repository, dependency_name=name)


def _assemble(repository, ops_nn, ops_mhc, work):
    baseline = json.loads((_ROOT / _MULTICORE / "runtime/baselines/families.json").read_text())["families"]["mhc"]
    pinned = work / "pinned"
    paths = set(baseline["source_hashes"])
    paths.update(str(_MULTICORE / path) for path in (
        "cmake/hardening.cmake", "torch/csrc/mega_mhc.cpp", "torch/csrc/registration_mhc.cpp",
    ))
    _export(repository, baseline["source_revision"], sorted(paths), pinned)
    for relative, expected in baseline["source_hashes"].items():
        if _hash_file(pinned / relative) != expected:
            raise ValueError(f"Pinned MHC source hash mismatch: {relative}")
    exports = {}
    selected = {"ops_nn": (ops_nn, ["norm/rms_norm/op_kernel", "norm/rms_norm_grad/op_kernel"]),
                "ops_transformer_mhc": (ops_mhc, ["common/include", "mhc"])}
    for name, (dependency_repository, directories) in selected.items():
        dependency = baseline["dependencies"][name]
        _verify_dependency(dependency, dependency_repository, name)
        target = work / name
        _export(dependency_repository, dependency["commit"], directories, target)
        for patch in dependency["patches"]:
            if name == "ops_nn" and "rms_norm" not in patch["path"]:
                continue
            patch_file = pinned / patch["path"]
            if _hash_file(patch_file) != patch["sha256"]:
                raise ValueError(f"MHC adapter hash mismatch: {patch['path']}")
            environment = {**os.environ, "GIT_CEILING_DIRECTORIES": str(target.parent)}
            command = ["git", "apply", "--ignore-space-change"]
            for options in (("--check",), (), ("--reverse", "--check")):
                subprocess.run([*command, *options, str(patch_file)], cwd=target,
                               env=environment, check=True)
        exports[name] = target
    source = work / "source"
    source.mkdir()
    nn, mhc = exports["ops_nn"], exports["ops_transformer_mhc"]
    shutil.copytree(mhc / "common/include", source / "ops_transformer_common")
    shutil.copytree(mhc / "mhc", source / "mhc")
    # The locked common headers include an obsolete SDK header without using its declarations.
    for header in (source / "ops_transformer_common").rglob("*.h"):
        text = header.read_text()
        if '#include "op_host/util/op_const_def.h"' in text:
            header.write_text(text.replace('#include "op_host/util/op_const_def.h"', ""))
    for name, norm, post, pre in (
        ("hyper_mega_mhc", "rms_norm", "mhc_post", "mhc_pre_sinkhorn"),
        ("hyper_mega_mhc_grad", "rms_norm_grad", "mhc_post_backward", "mhc_pre_sinkhorn_backward"),
    ):
        operator = source / name
        shutil.copytree(pinned / _MULTICORE / "ops" / name, operator)
        shutil.copytree(pinned / _MULTICORE / "ops/runtime", operator / "op_kernel/runtime")
        install_cpp_schema(operator / "op_kernel/runtime", "mhc")
        shutil.copytree(nn / "norm" / norm / "op_kernel", operator / "op_kernel" / norm)
        shutil.copytree(mhc / "mhc" / post / "op_kernel/arch22", operator / "op_kernel" / post)
        shutil.copytree(mhc / "mhc" / pre / "op_kernel", operator / "op_kernel" / pre)
        shutil.copytree(mhc / "mhc" / pre / "op_host/op_tiling", operator / "op_host" / (pre + "_tiling"))
    sdk_record = json.loads((_ROOT / "build/native/work/multicore/shmem/sdk.json").read_text())
    sdk = Path(sdk_record["install_root"]) / "shmem"
    for directory in ("include", "src/device", "src/host_device"):
        shutil.copytree(sdk / directory, source / "shmem_sdk" / directory)
    return pinned, source


def _run_build(cmake_source, build, options, jobs):
    subprocess.run(["cmake", "-S", str(cmake_source), "-B", str(build), *options], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], check=True)


def _validate_libraries(payload):
    libraries = sorted(payload.rglob("*.so"))
    if len(libraries) != 5:
        raise ValueError(f"MHC payload requires exactly five host ELF libraries, found {len(libraries)}")
    for library in libraries:
        dynamic = subprocess.check_output(["readelf", "-d", str(library)], text=True)
        headers = subprocess.check_output(["readelf", "-W", "-l", str(library)], text=True)
        if "RPATH" in dynamic or "RUNPATH" in dynamic or "TEXTREL" in dynamic:
            raise ValueError(f"Unsafe dynamic ELF section: {library}")
        if "GNU_RELRO" not in headers or "BIND_NOW" not in dynamic or "RWE" in headers:
            raise ValueError(f"Missing ELF hardening: {library}")
    vendor = payload / "vendors" / _VENDOR
    symbols = subprocess.check_output(["nm", "-D", str(vendor / "op_api/lib/libcust_opapi.so")], text=True)
    for name in ("aclnnHyperMegaMhc", "aclnnHyperMegaMhcGrad"):
        for suffix in ("", "GetWorkspaceSize"):
            if name + suffix not in symbols:
                raise ValueError(f"Missing MHC ACLNN symbol: {name}{suffix}")
    if "aclnnHyperMegaMoe" in symbols:
        raise ValueError("MHC payload must not export legacy MoE entry points")


def _write_manifest(payload, work, soc, cann):
    abi = family_abi("mhc")
    inputs = {}
    for directory in (work / "pinned", work / "source", _CMAKE):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                inputs[f"{directory.name}/{path.relative_to(directory)}"] = _hash_file(path)
    inputs["build_mhc.py"] = _hash_file(Path(__file__))
    inputs["families.json"] = _hash_file(_ROOT / _MULTICORE / "runtime/baselines/families.json")
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
    """Export verified baseline inputs and build only MHC forward/backward for one SoC."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repository", type=Path, default=_ROOT)
    parser.add_argument("--ops-nn-source", type=Path, required=True)
    parser.add_argument("--ops-mhc-source", type=Path, required=True)
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
        parser.error("The pinned MHC tiling uses TensorShape/TensorDataType and requires CANN >=9.2")
    output = _ROOT / "build/native/mhc"
    work = output / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    pinned, source = _assemble(args.source_repository, args.ops_nn_source, args.ops_mhc_source, work)
    options = [
        "-DCMAKE_BUILD_TYPE=Release", f"-DASCEND_CANN_PACKAGE_PATH={cann}",
        f"-DASCEND_COMPUTE_UNIT={args.soc}", f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
        f"-DHP_MULTICORE_SOURCE_ROOT={source}", f"-DHP_MHC_PINNED_ROOT={pinned}",
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
        f"-DHP_MHC_PINNED_ROOT={pinned}", f"-DHP_MULTICORE_VENDOR_ROOT={vendor}",
        f"-DCMAKE_INSTALL_PREFIX={payload / 'framework/torch'}",
    ], args.jobs)
    subprocess.run(["cmake", "--install", str(work / "torch-build")], check=True)
    shutil.copy2(Path(__file__).parent / "mhc/set_env.bash", payload / "set_env.bash")
    _validate_libraries(payload)
    for name in ("hyper_mega_mhc", "hyper_mega_mhc_grad"):
        if not list(vendor.glob(f"op_impl/ai_core/tbe/kernel/{args.soc}/{name}/*.o")):
            raise ValueError(f"Missing compiled MHC device binary: {name}")
    _write_manifest(payload, work, args.soc, cann)
    installed = output / "payload"
    if installed.exists():
        shutil.rmtree(installed)
    shutil.move(payload, installed)
    print(f"MHC payload ready: {installed}; activate set_env.bash before starting Python")


if __name__ == "__main__":
    main()
