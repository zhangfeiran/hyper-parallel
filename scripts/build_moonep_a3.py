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
"""Build MoonEP's MTE kernels for A3, rejecting unsupported A5 entry points.

Run after sourcing CANN and building MoonEP's pinned SHMEM submodule with
``bash scripts/build.sh -soc_type Ascend910B``. The source checkout is patched
locally; upstream MTE algorithms are retained with explicit A3 FFTS setup.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

UPSTREAM_REVISION = "b4632790350631d1e526e85d617ee13086dcac7f"


def prepare_a3(source: Path) -> None:
    """Select upstream MTE sources and make A5-only calls fail explicitly."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected MoonEP {UPSTREAM_REVISION}, found {revision}")
    native = source / "npu_patch/native"
    header = (native / "include/ascend_moonep/kernels/launch.h").read_text(encoding="utf-8")
    declarations = re.findall(r"ASCEND_MOONEP_EXPORT (?:void|int) ([\s\S]*?);", header)
    definitions = []
    for declaration in declarations:
        if "udma" in declaration or declaration.startswith("planning_kernel_do"):
            return_type = "int" if declaration.startswith("moonep_shmem_grad_reduce_udma") else "void"
            definitions.append(f'extern "C" {return_type} {declaration} {{\n'
                               '    throw std::runtime_error("This MoonEP entry point requires Ascend950/A5");\n'
                               '}\n')
    (native / "kernels/a3_unsupported.cpp").write_text(
        '#include <stdexcept>\n#include "ascend_moonep/kernels/launch.h"\n\n' + "\n".join(definitions),
        encoding="utf-8")
    original = subprocess.check_output(["git", "show", "HEAD:npu_patch/native/CMakeLists.txt"],
                                       cwd=source, text=True)
    for name in ("moonep_combine_udma", "moonep_dispatch_udma", "moonep_grad_reduce_udma",
                 "planning", "prefetch_udma"):
        original = original.replace(f"        kernels/{name}.cpp\n", "")
    original = original.replace("        kernels/barrier_bench.cpp\n",
                                "        kernels/barrier_bench.cpp\n        kernels/a3_unsupported.cpp\n")
    original = original.replace("        USE_SIMT\n", "")
    (native / "CMakeLists.txt").write_text(original, encoding="utf-8")
    api_path = source / "npu_patch/api.py"
    api = subprocess.check_output(["git", "show", "HEAD:npu_patch/api.py"], cwd=source, text=True)
    for name in ("projection", "full_grad"):
        api = api.replace(f"assert int({name}.shape[0]) == int(ctx['E']) + int(ctx['B'])",
                          f"assert int({name}.shape[0]) in (int(ctx['E']) + int(ctx['B']), "
                          "int(ctx['E']) // int(ctx['R']))")
    api_path.write_text(api, encoding="utf-8")
    _prepare_a3_heap(source)
    _prepare_a3_ffts(source)


def _prepare_a3_ffts(source: Path) -> None:
    """Initialize A3 SyncAll explicitly, as upstream grad-reduce already does."""
    for filename, entry in (("moonep_dispatch", "moonep_dispatch_kernel"),
                            ("moonep_combine", "moonep_combine_kernel"),
                            ("barrier_bench", "device_barrier_bench_kernel")):
        relative = f"npu_patch/native/kernels/{filename}.cpp"
        original = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=source, text=True)
        signature = re.search(rf"void\s+{entry}\([\s\S]*?\)\s*\{{", original)
        if signature is None:
            raise ValueError(f"MoonEP kernel entry changed: {entry}")
        replacement = signature.group().replace(f"{entry}(", f"{entry}(uint64_t ffts_addr, ", 1)
        replacement += "\n    util_set_ffts_config(ffts_addr);"
        original = original.replace(signature.group(), replacement, 1)
        original, launches = re.subn(rf"({entry}<<<[^>]*>>>\(\s*)",
                                    r"\g<1>shmemx_get_ffts_config(), ", original)
        if launches != 1:
            raise ValueError(f"MoonEP kernel launcher changed: {entry}")
        (source / relative).write_text(original, encoding="utf-8")


def _prepare_a3_heap(source):
    """Use A3 SHMEM allocation instead of unsupported caller-mapped VMM export."""
    path = source / "npu_patch/buffer_c.py"
    original = subprocess.check_output(["git", "show", "HEAD:npu_patch/buffer_c.py"], cwd=source, text=True)
    original = original.replace("import math\n", "import math\nimport os\n")
    allocation = '''class A3HeapAllocation:
    """A SHMEM-owned allocation whose Tensor releases it before finalization."""

    def __init__(self, tensor, size, device):
        self.source_va = tensor.data_ptr()
        self.size = size
        self.mem_handle = 0
        self.fabric_handle_bytes = b""
        self.device_id = device
        self._destroyed = False
        self._registered = False
        self.owner = tensor

    def destroy(self):
        if not self._destroyed:
            self.owner = None
            self._destroyed = True


'''
    original = original.replace("def create_vmm_physical(shape, dtype: torch.dtype, device: int):",
                                allocation + "def create_vmm_physical(shape, dtype: torch.dtype, device: int):")
    marker = "    nbytes = _chunk_nbytes(shape, dtype)\n    device_id = int(device)\n"
    replacement = '''    nbytes = _chunk_nbytes(shape, dtype)
    device_id = int(device)
    if os.environ.get("ASCEND_MOONEP_A3_HEAP") == "1":
        _require_current_device(device_id)
        tensor, _, _ = acl_dist_alloc(list(shape), dtype)
        allocation = A3HeapAllocation(tensor, nbytes, device_id)
        tensor._vmm_allocation = allocation
        register_symmetric_descriptor(allocation)
        return tensor, 0, tensor.data_ptr()
'''
    if original.count(marker) != 1:
        raise ValueError("MoonEP allocation implementation changed")
    original = original.replace(marker, replacement)
    original = original.replace("    sym = int(acl_shmem_get_buffer_ptr(int(va)))\n",
                                '    if os.environ.get("ASCEND_MOONEP_A3_HEAP") == "1":\n'
                                '        return int(va)\n    sym = int(acl_shmem_get_buffer_ptr(int(va)))\n')
    path.write_text(original, encoding="utf-8")
    path = source / "npu_patch/shmem_runtime.py"
    original = subprocess.check_output(["git", "show", "HEAD:npu_patch/shmem_runtime.py"], cwd=source, text=True)
    original = original.replace("from ._C import acl_shmem_barrier_all, acl_shmem_get_unique_id",
                                "from ._C import acl_shmem_barrier_all, acl_shmem_get_unique_id, acl_shmem_init_uid")
    method = '''    @classmethod
    def init_legacy_a3(cls, group, heap_bytes):
        """Initialize the A3 MTE heap before constructing any Buffer/pool."""
        if cls._initialized or cls._poisoned:
            raise RuntimeError("MoonEP SHMEM runtime is already initialized or poisoned")
        rank, size = dist.get_rank(group), dist.get_world_size(group)
        root = dist.get_global_rank(group, 0) if group is not None else 0
        uid = acl_shmem_get_unique_id() if rank == 0 else None
        device = torch.device("npu", torch.npu.current_device())
        length = torch.tensor([len(uid) if uid is not None else 0], dtype=torch.int64, device=device)
        dist.broadcast(length, src=root, group=group)
        length = int(length.cpu())
        payload = (torch.tensor(list(uid), dtype=torch.uint8, device=device) if rank == 0
                   else torch.empty(length, dtype=torch.uint8, device=device))
        dist.broadcast(payload, src=root, group=group)
        acl_shmem_init_uid(rank, size, int(heap_bytes), bytes(payload.cpu().tolist()), torch.npu.current_device())
        cls._initialized = True

'''
    marker = "    @classmethod\n    def init_with_buffer("
    if original.count(marker) != 1:
        raise ValueError("MoonEP SHMEM initialization implementation changed")
    path.write_text(original.replace(marker, method + marker), encoding="utf-8")
    path = source / "npu_patch/native/csrc/acl_shared_buffer.cpp"
    original = subprocess.check_output(["git", "show", "HEAD:npu_patch/native/csrc/acl_shared_buffer.cpp"],
                                       cwd=source, text=True)
    original = original.replace("    const int init_rc = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);",
                                "    attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;\n"
                                "    const int init_rc = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);")
    path.write_text(original, encoding="utf-8")


def main() -> None:
    """Compile the external MoonEP checkout and install its optional extension."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    prepare_a3(source)
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    shmem = source / "npu_patch/3rdparty/shmem/install"
    build = source / "build_ascend_a3"
    subprocess.run(["cmake", "-S", str(source / "npu_patch/native"), "-B", str(build),
                    f"-DCMAKE_CXX_COMPILER={cann / 'tools/bisheng_compiler/bin/bisheng'}",
                    "-DASCEND_MOONEP_BISHENG_ARCH=a2", f"-DSHMEM_HOME_PATH={shmem}"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "ascend_moonep_kernels"], check=True)
    target = source / "npu_patch/lib"
    target.mkdir(exist_ok=True)
    shutil.copy2(build / "lib/libascend_moonep_kernels.so", target)
    for library in (shmem / "shmem/lib").glob("*.so*"):
        shutil.copy2(library, target / library.name)
    environment = dict(os.environ, MOONEP_BUILD_BACKEND="npu", SHMEM_HOME_PATH=str(shmem),
                       ASCEND_MOONEP_KERNEL_LIB_DIR=str(target))
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(source / "npu_patch"),
                    "--no-build-isolation", "--no-deps"], env=environment, check=True)
    (source / "a3-adaptation.json").write_text(json.dumps({
        "revision": UPSTREAM_REVISION, "architecture": "dav-2201", "transport": "MTE",
        "planning": "HyperParallel CPU port of MoonEP reference", "dedup": "rank", "allocation": "A3 MTE SHMEM heap",
        "ffts_sync_base": "explicit dispatch, combine and barrier kernel initialization",
        "unsupported": ["A5 SIMT planning", "UDMA", "server dedup"]}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
