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
"""Build resident dense candidates with sealed SDK, kernel and launcher identities."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import torch
from torch.utils.cpp_extension import load

from hyper_parallel.core.multicore._build.build_dense import dense_build_identity

CORE = Path(__file__).resolve().parents[1]
SOURCES = ("ops/dense/dense_tile_abi.h", "ops/dense/dense_tile_kernel.cpp", "ops/dense/dense_tile_tiling.cpp",
           "ops/dense/dense_tile_launch.cpp", "_build/dense_tile/CMakeLists.txt", "_build/build_dense_tile.py",
           "compiler/dense_tile.py", "backends/dense_tile.py", "runtime/dense_tile.py")
SDK_LIBRARIES = ("libtiling_api.a", "libplatform.so", "libregister.so", "libascendc_runtime.a",
                 "libascendcl.so", "libruntime.so")
SDK_VERSIONS = ("runtime", "bisheng-compiler", "asc-devkit", "metadef")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _npu_package():
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise ValueError("Resident dense native builds require the installed TorchNPU package")
    return Path(spec.origin).resolve().parent


def _cmake():
    executable = shutil.which("cmake")
    if executable is None:
        raise ValueError("Resident dense builds require CMake")
    return str(Path(executable).resolve())


def dense_tile_build_identity(cann_root: Path, soc: str) -> dict[str, object]:
    """Hash device toolchain, SDK inputs and TorchNPU ABI without initializing NPU.

    Args:
        cann_root: Exact selected SDK; independent of site-packages and vendor installs.
        soc: Exact standalone Ascend910B target name.
    """
    if soc not in ("Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4"):
        raise ValueError("Resident dense candidates currently compile for explicit Ascend910B1..4 targets")
    cann_root = Path(cann_root).resolve()
    arch = platform.machine()
    sdk_sources = {}
    for directory in (cann_root / f"{arch}-linux/asc", cann_root / "include",
                      cann_root / f"{arch}-linux/tikcpp/tikcfw",
                      cann_root / "tools/tikcpp/ascendc_kernel_cmake"):
        if not directory.is_dir():
            raise ValueError(f"Dense standalone SDK input directory is missing: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".h", ".hpp", ".cpp", ".c", ".py", ".cmake", ".sh", ".txt"):
                sdk_sources[str(path.relative_to(cann_root))] = _hash(path)
    versions = {}
    for name in SDK_VERSIONS:
        path = cann_root / "share/info" / name / "version.info"
        if path.exists():
            versions[name] = path.read_text(encoding="utf-8")
    compiler = cann_root / f"{arch}-linux/ccec_compiler/bin/bisheng"
    npu = _npu_package()
    return {"format_version": 1, "soc": soc, "architecture": arch, "cann_root": str(cann_root),
            "sources": {name: _hash(CORE / name) for name in SOURCES}, "sdk_sources": sdk_sources,
            "sdk_versions": versions, "sdk_libraries": {name: _hash(cann_root / "lib64" / name)
                                                          for name in SDK_LIBRARIES},
            "device_compiler": {"path": str(compiler), "sha256": _hash(compiler)},
            "torch_npu": importlib.metadata.version("torch-npu"),
            "torch_npu_library_sha256": _hash(npu / "lib/libtorch_npu.so"), "host": dense_build_identity(),
            "cmake": {"path": _cmake(), "version": subprocess.check_output([_cmake(), "--version"],
                                                                           text=True).splitlines()[0]}}


def verify_dense_tile_payload(manifest: Path, identity: dict[str, object]) -> dict[str, object]:
    """Reject cache escape, source/toolchain drift or modified libraries before load.

    Args:
        manifest: Native candidate manifest within its caller-owned build cache.
        identity: Freshly computed exact source and toolchain identity.
    """
    manifest = Path(manifest).resolve()
    record = json.loads(manifest.read_text(encoding="utf-8"))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if (record["format_version"] != 1 or record["identity"] != identity
            or record["execution_mode"] != "resident_dense_tile_candidate"
            or record["namespace"] != "hp_dense_tile_" + key[:24]):
        raise ValueError("Resident dense source/SDK/framework identity mismatch")
    for name, digest in record["files"].items():
        path = (manifest.parent / name).resolve()
        if not path.is_relative_to(manifest.parent) or _hash(path) != digest:
            raise ValueError("Resident dense artifact path or integrity mismatch")
    required = {record["library"], record["tiling_library"], record["binary"]}
    if not required.issubset(record["files"]):
        raise ValueError("Resident dense manifest omits a required native artifact")
    return record


def _extract_binary(archive, directory, soc):
    stub = directory / "host_stub.o"
    blob = directory / "kernel-section.bin"
    with stub.open("wb") as stream:
        subprocess.run(["/usr/bin/ar", "p", str(archive), "host_stub.cpp.o"], stdout=stream, check=True)
    section = f".ascend.kernel.{soc.lower()}.hyper_parallel_dense_tile"
    subprocess.run(["/usr/bin/objcopy", "--dump-section", f"{section}={blob}", str(stub)], check=True)
    data = blob.read_bytes()
    if len(data) < 20:
        raise ValueError("Dense standalone SDK mixed-binary section is truncated")
    version, count, kind, padded, length = struct.unpack_from("<5I", data)
    if version != 1 or count != 1 or kind != 0 or length == 0 or padded != length or len(data) != 20 + length:
        raise ValueError("Dense standalone SDK mixed-binary section ABI is unsupported")
    binary = directory / "dense_tile.bin"
    binary.write_bytes(data[20:])
    literals = [", ".join(f"0x{value:02x}" for value in data[first:first + 16])
                for first in range(20, len(data), 16)]
    include = directory / "dense_tile_binary.inc"
    include.write_text(f'constexpr char HP_DENSE_SOC[] = "{soc.lower()}";\n'
                       + "alignas(64) constexpr unsigned char HP_DENSE_BINARY[] = {\n"
                       + ",\n".join(literals) + "\n};\n", encoding="utf-8")
    return binary


def _build_sdk(cann_root, directory, soc):
    build = directory / ("sdk-" + uuid4().hex)
    commands = [[_cmake(), "-S", str(CORE / "_build/dense_tile"), "-B", str(build),
                 f"-DASCEND_CANN_PACKAGE_PATH={cann_root}", f"-DSOC_VERSION={soc}", "-DCMAKE_BUILD_TYPE=Release",
                 f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}"],
                [_cmake(), "--build", str(build), "--parallel", "2"]]
    for index, command in enumerate(commands):
        with (directory / f"sdk-{index}.log").open("w", encoding="utf-8") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    return build


def _build_payload(cann_root, directory, identity):
    soc = identity["soc"]
    build = _build_sdk(cann_root, directory, soc)
    archive = build / "lib/libhyper_parallel_dense_tile.a"
    binary = _extract_binary(archive, directory, soc)
    npu = _npu_package()
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    namespace = "hp_dense_tile_" + key[:24]
    flags = ["-O2", "-std=c++17", f"-DHP_DENSE_NAMESPACE={namespace}"]
    links = [str(cann_root / "lib64/libascendc_runtime.a"), f"-L{cann_root}/lib64", "-lascendcl", "-lruntime",
             f"-L{npu}/lib", "-ltorch_npu", "-ldl", f"-Wl,-rpath,{cann_root}/lib64", f"-Wl,-rpath,{npu}/lib"]
    library = Path(load(name=namespace, sources=[str(CORE / "ops/dense/dense_tile_launch.cpp")],
                        extra_include_paths=[str(directory), str(npu / "include"), str(cann_root / "include")],
                        extra_cflags=flags, extra_ldflags=links, build_directory=str(directory),
                        is_python_module=False, verbose=False))
    schema = getattr(torch.ops, namespace).launch.default._schema
    if schema.arguments[0].alias_info is not None or not schema.arguments[1].alias_info.is_write:
        raise ValueError("Resident dense native schema must preserve read-only inputs and declare output writes")
    tiler = build / "libhyper_parallel_dense_tiling.so"
    artifacts = (library, tiler, binary, archive, directory / "dense_tile_binary.inc")
    return {"format_version": 1, "execution_mode": "resident_dense_tile_candidate", "device_status": "unvalidated",
            "identity": identity, "namespace": namespace, "library": str(library.relative_to(directory)),
            "tiling_library": str(tiler.relative_to(directory)), "binary": str(binary.relative_to(directory)),
            "files": {str(path.relative_to(directory)): _hash(path) for path in artifacts}}


def build_dense_tile_payload(cann_root: Path, root: Path, soc: str) -> Path:
    """Compile one kernel/provider implementation independently of token counts.

    Args:
        cann_root: Selected CANN standalone SDK root.
        root: Caller-owned cache; no vendor or framework installation is modified.
        soc: Exact device target, checked again by lazy native registration.

    Returns:
        Sealed candidate manifest. This is build evidence, not device acceptance.
    """
    cann_root = Path(cann_root).resolve()
    identity = dense_tile_build_identity(cann_root, soc)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(root).resolve() / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    with (directory / ".build.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            verify_dense_tile_payload(manifest, identity)
            return manifest
        if getattr(getattr(torch.ops, "hp_dense_tile_" + key[:24]), "launch", None) is not None:
            raise ValueError("Resident dense namespace is already registered without its sealed cache")
        record = _build_payload(cann_root, directory, identity)
        if identity != dense_tile_build_identity(cann_root, soc):
            raise ValueError("Resident dense sources/toolchain changed during native build")
        with NamedTemporaryFile(mode="w", dir=directory, suffix=".manifest.tmp", delete=False,
                                encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
            temporary = Path(stream.name)
        temporary.replace(manifest)
    return manifest
