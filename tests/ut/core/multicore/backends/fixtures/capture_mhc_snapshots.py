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
"""Reproduce MHC images using only the design's trusted fixed-revision Git objects."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

from tests.ut.core.multicore.backends.fixtures import capture_baselines as capture

OUTPUT = Path(__file__).resolve().parent


def _modules():
    base = ModuleType("mhc_pinned_reference")
    sys.modules["mhc_pinned_reference"] = base
    namespace = vars(base)
    for source in ("scheduler/config.py", "scheduler/runtime.py", "scheduler/graph.py", "tasks/task_base.py",
                   "scheduler/builder.py", "profiler/profiling.py"):
        capture._trusted_module(capture._read("mhc", source), namespace)
    result = {}
    for direction, source in (("forward", "modules/mega_mhc/graph.py"),
                              ("backward", "modules/mega_mhc_grad/graph.py")):
        name = "mhc_pinned_" + direction
        module = ModuleType(name)
        sys.modules[name] = module
        values = vars(module)
        values.update({key: value for key, value in namespace.items() if key != "__name__"})
        capture._trusted_module(capture._read("mhc", source), values)
        result[direction] = values
    return namespace, result


def _case(tokens, tile, cores, namespace, modules):
    record = {"T": tokens, "H": 128, "C": cores, "tile": tile, "images": {}}
    for direction, module in modules.items():
        if direction == "forward":
            graph, topology = module["build_mega_mhc_graph"](tokens, 128, cores, tile)
        else:
            graph, topology = module["build_mega_mhc_grad_graph"](tokens, 128, tile, num_vector_cores=2 * cores)
        config = namespace["build_runtime_config"](graph, topology, num_cube_cores=cores)
        if direction == "backward":
            module["order_vector_tasks_by_stage"](config)
        namespace["_apply_mega_kernel_profile_graph"](config, graph, namespace["_ProfileSpec"](
            kernel_name="HyperMegaMhc" if direction == "forward" else "HyperMegaMhcGrad",
            owner_label="TokenPartition" if direction == "forward" else "NativeKernel"))
        namespace["_configure_profile_layout"](config)
        for profiled in (False, True):
            config.cycle_profiling_enabled = int(profiled)
            wire = namespace["serialize_runtime_config"](config)
            name = f"mhc_t{tokens}_tile{tile}_{direction}_{int(profiled)}.bin.gz"
            (OUTPUT / name).write_bytes(gzip.compress(wire, mtime=0))
            record["images"][f"{direction}_{int(profiled)}"] = {
                "file": name, "bytes": len(wire), "sha256": hashlib.sha256(wire).hexdigest()}
    return record


def main() -> None:
    """Capture normal/profiled forward/backward for tails, ring reuse and macro joins."""
    namespace, modules = _modules()
    cases = [_case(tokens, tile, cores, namespace, modules)
             for tokens, tile, cores in ((48, 32, 24), (129, 64, 20), (2593, 32, 20), (12000, 96, 20))]
    (OUTPUT / "mhc_snapshots.json").write_text(
        json.dumps({"revision": capture.REVISIONS["mhc"], "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
