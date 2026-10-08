# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Assemble the isolated HyperMegaMoe source closure from pinned kernel sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from prepare_dependencies import verify_git_dependency

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LOCK_PATH = Path(__file__).with_name("dependencies.lock.json")
_MULTICORE_OPS = _REPO_ROOT / "hyper_parallel" / "core" / "multicore" / "ops"
_SHMEM_CCSRC = _REPO_ROOT / "hyper_parallel" / "core" / "multicore" / "shmem" / "ccsrc"
_OPS_NN_PATHS = (
    "activation/swi_glu/op_kernel",
    "activation/swi_glu_grad/op_kernel",
)
_OPS_TRANSFORMER_PATHS = (
    "gmm/grouped_matmul/op_kernel",
    "attention/sparse_flash_attention/op_kernel",
    "attention/sparse_flash_attention/op_host",
    "attention/lightning_indexer/op_kernel",
    "attention/lightning_indexer/op_host",
    "attention/sparse_flash_attention_grad/op_kernel",
    "attention/sparse_flash_attention_grad/op_host",
    "attention/sparse_flash_attention_grad/basic_modules",
    "common/include/err",
    "common/include/op_host/tiling_util.h",
    "common/include/op_host/tiling_base.h",
    "common/include/op_host/tiling_type.h",
)
_HYPER_OPERATORS = ("hyper_mega_moe", "hyper_mega_moe_grad")


def _parse_args() -> argparse.Namespace:
    """Parse the isolated source assembly contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-nn-source", required=True)
    parser.add_argument("--ops-transformer-source", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args()


def main() -> int:
    """Verify, export, adapt, and compose the selected kernel sources."""
    args = _parse_args()
    lock = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))["components"]["multicore"]
    ops_nn_source = Path(args.ops_nn_source).resolve()
    ops_transformer_source = Path(args.ops_transformer_source).resolve()
    work_dir = Path(args.work_dir).resolve()
    _validate_new_work_dir(work_dir, (ops_nn_source, ops_transformer_source))

    verify_git_dependency(lock["ops_nn"], ops_nn_source, dependency_name="ops_nn")
    verify_git_dependency(
        lock["ops_transformer"],
        ops_transformer_source,
        dependency_name="ops_transformer",
    )

    ops_nn_copy = work_dir / "adapter-inputs" / "ops-nn"
    transformer_copy = work_dir / "adapter-inputs" / "ops-transformer"
    _export_git_tree(
        ops_nn_source,
        ops_nn_copy,
        lock["ops_nn"]["commit"],
        _OPS_NN_PATHS,
    )
    _export_git_tree(
        ops_transformer_source,
        transformer_copy,
        lock["ops_transformer"]["commit"],
        _OPS_TRANSFORMER_PATHS,
    )
    _apply_locked_adapters(ops_nn_copy, lock["ops_nn"])
    _apply_locked_adapters(transformer_copy, lock["ops_transformer"])

    source_root = work_dir / "source"
    _compose_hyper_parallel_ops(source_root, ops_nn_copy, transformer_copy)
    _require_assembled_files(source_root)
    print(json.dumps({"source_root": str(source_root)}, sort_keys=True))
    return 0


def _validate_new_work_dir(work_dir: Path, inputs: tuple[Path, ...]) -> None:
    """Reject broad, existing, or source-overlapping output paths."""
    protected = {Path("/").resolve(), _REPO_ROOT.resolve(), _REPO_ROOT.parent.resolve(), *inputs}
    if work_dir in protected or any(work_dir == path.parent for path in inputs):
        raise ValueError(f"Refusing unsafe multicore work directory: {work_dir}")
    if work_dir.exists():
        raise ValueError(f"Multicore work directory must not already exist: {work_dir}")
    work_dir.mkdir(parents=True)


def _export_git_tree(
    source_root: Path,
    destination_root: Path,
    commit: str,
    relative_paths: tuple[str, ...],
) -> None:
    """Export selected committed files from one verified dependency revision."""
    if destination_root.exists():
        raise ValueError(f"Git export destination already exists: {destination_root}")
    destination_root.mkdir(parents=True)
    command = ["git", "archive", "--format=tar", commit, *relative_paths]
    with tempfile.TemporaryFile() as archive_file:
        result = subprocess.run(
            command,
            cwd=source_root,
            check=False,
            stdout=archive_file,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(
                f"Failed to export locked dependency {source_root} at {commit}: {result.stderr.strip()}"
            )
        archive_file.seek(0)
        with tarfile.open(fileobj=archive_file, mode="r:") as source:
            _validate_tar_members(source.getmembers())
            source.extractall(destination_root)


def _apply_locked_adapters(source_root: Path, dependency_lock: dict[str, Any]) -> None:
    """Verify and apply fusion adapters only to the isolated source copy."""
    for adapter in dependency_lock.get("patches", []):
        adapter_path = (_REPO_ROOT / adapter["path"]).resolve()
        actual_hash = _sha256(adapter_path)
        if actual_hash != adapter["sha256"]:
            raise ValueError(
                f"Adapter hash mismatch for {adapter_path}: expected={adapter['sha256']}, actual={actual_hash}"
            )
        _run_git_apply(source_root, adapter_path, check_only=True)
        _run_git_apply(source_root, adapter_path, check_only=False)
        _run_git_apply(source_root, adapter_path, check_only=True, reverse=True)


def _run_git_apply(
    source_root: Path,
    adapter_path: Path,
    check_only: bool,
    reverse: bool = False,
) -> None:
    """Run Git's patch parser against one isolated source copy."""
    command = ["git", "apply", "--ignore-space-change"]
    if reverse:
        command.append("--reverse")
    if check_only:
        command.append("--check")
    command.append(str(adapter_path))
    environment = os.environ.copy()
    environment["GIT_CEILING_DIRECTORIES"] = str(source_root.parent)
    result = subprocess.run(
        command,
        cwd=source_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        phase = "reverse check" if reverse else "check" if check_only else "apply"
        raise ValueError(f"Adapter {phase} failed for {adapter_path}: {result.stdout.strip()}")


def _compose_hyper_parallel_ops(
    source_root: Path,
    ops_nn_copy: Path,
    transformer_copy: Path,
) -> None:
    """Compose HP operator code with selected adapted upstream kernel sources."""
    shmem_root = source_root / "shmem"
    shmem_root.mkdir(parents=True)
    (shmem_root / "cann").mkdir()
    shutil.copy2(_SHMEM_CCSRC / "cann" / "device.h", shmem_root / "cann" / "device.h")
    shutil.copy2(_SHMEM_CCSRC / "cann" / "signal.h", shmem_root / "cann" / "signal.h")
    (shmem_root / "data_plane").mkdir()
    shutil.copy2(_SHMEM_CCSRC / "data_plane" / "rma.h", shmem_root / "data_plane" / "rma.h")
    shutil.copy2(_SHMEM_CCSRC / "data_plane" / "sync.h", shmem_root / "data_plane" / "sync.h")
    for operator_name in _HYPER_OPERATORS:
        operator_root = source_root / operator_name
        shutil.copytree(_MULTICORE_OPS / operator_name, operator_root)
        shutil.copytree(_MULTICORE_OPS / "runtime", operator_root / "op_kernel" / "runtime")
        shutil.copytree(
            ops_nn_copy / "activation" / "swi_glu" / "op_kernel",
            operator_root / "op_kernel" / "swi_glu",
        )
        shutil.copytree(
            transformer_copy / "gmm" / "grouped_matmul" / "op_kernel",
            operator_root / "op_kernel" / "grouped_matmul",
        )
    shutil.copytree(
        ops_nn_copy / "activation" / "swi_glu_grad" / "op_kernel",
        source_root / "hyper_mega_moe_grad" / "op_kernel" / "swi_glu_grad",
    )

    mixed_root = source_root / "hyper_dsa_mixed_tile"
    shutil.copytree(_MULTICORE_OPS / "hyper_dsa_mixed_tile", mixed_root)
    shutil.copytree(transformer_copy / "common" / "include" / "err", mixed_root / "op_host" / "err")
    upstream_host = transformer_copy / "attention" / "sparse_flash_attention" / "op_host"
    for filename in ("sparse_flash_attention_def.cpp", "sparse_flash_attention_tiling.cpp",
                     "sparse_flash_attention_tiling.h", "sparse_flash_attention_infershape.cpp"):
        shutil.copy2(upstream_host / filename, mixed_root / "op_host" / filename)
    upstream_kernel = transformer_copy / "attention" / "sparse_flash_attention" / "op_kernel"
    shutil.copytree(upstream_kernel / "arch22", mixed_root / "op_kernel" / "arch22")
    for filename in ("sparse_flash_attention_common.h", "sparse_flash_attention_template_tiling_key.h"):
        shutil.copy2(upstream_kernel / filename, mixed_root / "op_kernel" / filename)
    _compose_mixed_indexer(source_root, transformer_copy)
    _compose_mixed_grad(source_root, transformer_copy)
    _compose_fused_forward(source_root, transformer_copy)
    _compose_fused_grad(source_root, transformer_copy)
    for name in ("hyper_dsa_mixed_tile", "hyper_dsa_mixed_indexer", "hyper_dsa_mixed_grad",
                 "hyper_dsa_fused_forward", "hyper_dsa_fused_grad"):
        runtime = source_root / name / "op_kernel" / "runtime"
        runtime.mkdir()
        shutil.copy2(_MULTICORE_OPS / "runtime" / "dsa_mixed_group.h", runtime / "dsa_mixed_group.h")
        if name == "hyper_dsa_fused_forward":
            shutil.copy2(_MULTICORE_OPS / "runtime" / "dsa_cp_transport.h", runtime / "dsa_cp_transport.h")
        if name == "hyper_dsa_fused_grad":
            shutil.copy2(_MULTICORE_OPS / "runtime" / "dsa_cp_grad_transport.h", runtime / "dsa_cp_grad_transport.h")


def _compose_mixed_indexer(source_root: Path, transformer_copy: Path) -> None:
    """Keep retained logical LI partials separate from reusable physical scratch."""
    mixed_root = source_root / "hyper_dsa_mixed_indexer"
    shutil.copytree(_MULTICORE_OPS / "hyper_dsa_mixed_indexer", mixed_root)
    shutil.copytree(transformer_copy / "common" / "include" / "err", mixed_root / "op_host" / "err")
    upstream = transformer_copy / "attention" / "lightning_indexer"
    (mixed_root / "op_host" / "op_host").mkdir()
    shutil.copy2(transformer_copy / "common" / "include" / "op_host" / "tiling_util.h",
                 mixed_root / "op_host" / "op_host" / "tiling_util.h")
    for filename in ("lightning_indexer_def.cpp", "lightning_indexer_tiling.cpp",
                     "lightning_indexer_tiling.h", "lightning_indexer_infershape.cpp"):
        shutil.copy2(upstream / "op_host" / filename, mixed_root / "op_host" / filename)
    shutil.copytree(upstream / "op_kernel" / "arch22", mixed_root / "op_kernel" / "arch22")
    for filename in ("lightning_indexer_common.h", "lightning_indexer_template_tiling_key.h"):
        shutil.copy2(upstream / "op_kernel" / filename, mixed_root / "op_kernel" / filename)


def _compose_mixed_grad(source_root: Path, transformer_copy: Path) -> None:
    """Retain the locked gradient math with separately ordered initialization and post."""
    mixed_root = source_root / "hyper_dsa_mixed_grad"
    shutil.copytree(_MULTICORE_OPS / "hyper_dsa_mixed_grad", mixed_root)
    shutil.copytree(transformer_copy / "common" / "include" / "err", mixed_root / "op_host" / "err")
    (mixed_root / "op_host" / "op_host").mkdir()
    for name in ("tiling_base.h", "tiling_type.h"):
        shutil.copy2(transformer_copy / "common" / "include" / "op_host" / name,
                     mixed_root / "op_host" / "op_host" / name)
    upstream = transformer_copy / "attention" / "sparse_flash_attention_grad"
    for name in ("sparse_flash_attention_grad_def.cpp", "sparse_flash_attention_grad_tiling_common.cpp",
                 "sparse_flash_attention_grad_tiling_common.h", "sparse_flash_attention_grad_tiling.h",
                 "sparse_flash_attention_grad_infershape.cpp"):
        shutil.copy2(upstream / "op_host" / name, mixed_root / "op_host" / name)
    shutil.copytree(upstream / "op_host" / "arch22", mixed_root / "op_host" / "arch22")
    shutil.copytree(upstream / "op_kernel" / "arch22", mixed_root / "op_kernel" / "arch22")
    shutil.copytree(upstream / "basic_modules", mixed_root / "basic_modules")


def _export_tiling_declarations(source: Path, destination: Path) -> None:
    """Copy locked tiling schemas without unrelated host classes or constants."""
    text = source.read_text(encoding="utf-8")
    blocks = re.findall(r"BEGIN_TILING_DATA_DEF\([^\n]+\).*?REGISTER_TILING_DATA_CLASS\([^\n]+\)", text, re.DOTALL)
    if not blocks:
        raise ValueError(f"No locked tiling schemas in {source}")
    license_end = text.index("*/") + 2
    destination.write_text(text[:license_end] + '\n#include "register/tilingdata_base.h"\n'
                           + "namespace optiling {\n" + "\n\n".join(blocks) + "\n}\n", encoding="utf-8")


def _export_sfa_template_modes(source: Path, destination: Path) -> None:
    """Export locked mode constants without importing another operator's template registration."""
    text = source.read_text(encoding="utf-8")
    declarations = re.findall(r"^#define (?:C_TEMPLATE|V_TEMPLATE) [01]$", text, re.MULTILINE)
    if len(declarations) != 2:
        raise ValueError(f"Missing locked SFA template modes in {source}")
    destination.write_text(text[:text.index("*/") + 2] + "\n" + "\n".join(declarations) + "\n", encoding="utf-8")


def _compose_fused_forward(source_root: Path, transformer_copy: Path) -> None:
    """Compose both locked tile closures and their generated native tiling schemas."""
    root = source_root / "hyper_dsa_fused_forward"
    shutil.copytree(_MULTICORE_OPS / "hyper_dsa_fused_forward", root)
    for short, operator in (("li", "lightning_indexer"), ("sfa", "sparse_flash_attention")):
        upstream = transformer_copy / "attention" / operator
        shutil.copytree(upstream / "op_kernel", root / "op_kernel" / short)
        _export_tiling_declarations(upstream / "op_host" / f"{operator}_tiling.h",
                                    root / "op_host" / f"{short}_tiling_data.h")
    _export_sfa_template_modes(
        root / "op_kernel" / "sfa" / "sparse_flash_attention_template_tiling_key.h",
        root / "op_kernel" / "sfa_template_modes.h")


def _compose_fused_grad(source_root: Path, transformer_copy: Path) -> None:
    """Use the same locked gradient tiles and one exported original tiling schema."""
    root = source_root / "hyper_dsa_fused_grad"
    shutil.copytree(_MULTICORE_OPS / "hyper_dsa_fused_grad", root)
    upstream = transformer_copy / "attention" / "sparse_flash_attention_grad"
    shutil.copytree(upstream / "op_kernel" / "arch22", root / "op_kernel" / "arch22")
    shutil.copytree(upstream / "basic_modules", root / "basic_modules")
    _export_tiling_declarations(upstream / "op_host" / "sparse_flash_attention_grad_tiling.h",
                                root / "op_host" / "grad_tiling_data.h")


def _require_assembled_files(source_root: Path) -> None:
    """Reject incomplete source closures before invoking the CANN toolchain."""
    required_paths = (
        source_root / "hyper_mega_moe" / "op_host" / "hyper_mega_moe_def.cpp",
        source_root / "hyper_mega_moe" / "op_kernel" / "hyper_mega_moe.cpp",
        source_root / "hyper_mega_moe" / "op_kernel" / "swi_glu" / "swi_glu.cpp",
        source_root / "hyper_mega_moe" / "op_kernel" / "grouped_matmul" / "grouped_matmul.cpp",
        source_root / "hyper_mega_moe_grad" / "op_host" / "hyper_mega_moe_grad_def.cpp",
        source_root / "hyper_mega_moe_grad" / "op_kernel" / "hyper_mega_moe_grad.cpp",
        source_root / "hyper_mega_moe_grad" / "op_kernel" / "swi_glu_grad" / "swi_glu_grad.cpp",
        source_root / "hyper_dsa_fused_forward" / "op_kernel" / "hyper_dsa_fused_forward.cpp",
        source_root / "hyper_dsa_fused_grad" / "op_kernel" / "hyper_dsa_fused_grad.cpp",
        source_root / "shmem" / "data_plane" / "rma.h",
        source_root / "shmem" / "data_plane" / "sync.h",
        source_root / "hyper_dsa_mixed_tile" / "op_kernel" / "hyper_dsa_mixed_tile.cpp",
        source_root / "hyper_dsa_mixed_tile" / "op_kernel" / "arch22" / "sparse_flash_attention_kernel_mla.h",
        source_root / "hyper_dsa_mixed_indexer" / "op_kernel" / "arch22" / "lightning_indexer_kernel.h",
        source_root / "hyper_dsa_mixed_indexer" / "op_kernel" / "runtime" / "dsa_mixed_group.h",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete HyperMegaMoe source closure: {missing}")


def _validate_archive_names(names: Any) -> None:
    """Reject absolute or parent-traversing archive members."""
    for name in names:
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe path in Git archive: {name}")


def _validate_tar_members(members: list[tarfile.TarInfo]) -> None:
    """Allow only safe regular files and directories before archive extraction."""
    _validate_archive_names(member.name for member in members)
    for member in members:
        if member.issym() or member.islnk():
            raise ValueError(
                f"Archive links are not allowed in native build inputs: {member.name} -> {member.linkname}"
            )
        if not member.isfile() and not member.isdir():
            raise ValueError(
                f"Archive special files are not allowed in native build inputs: {member.name}"
            )


def _sha256(path: Path) -> str:
    """Return the SHA256 digest for one fusion adapter."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"[HP-MULTICORE-SOURCE-ERROR] {error}\n")
        raise SystemExit(1) from error
