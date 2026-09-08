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
"""Construct real MegaMoe schedules on Torch meta tensors; never launch a kernel.

Unlike the framework-free predictor, this optional audit loads the installed
Torch adapter. It checks descriptor/event capacity using the real graph builders
and reports static tensor bytes, without creating an NPU process group or heap.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.torch.multicore.megamoe_simulator.cost import Options
from examples.torch.multicore.megamoe_simulator.workload import Shape


def audit(shape: Shape, ranks: int, slots: int, options: Options) -> dict[str, Any]:
    """Build the current implementation's static schedules in one host process."""
    # The framework is an optional dependency of this diagnostic command. The
    # ordinary predictor never initializes HyperParallel or either framework.
    os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
    plan_module = importlib.import_module("hyper_parallel.core.multicore.modules.mega_moe.plan")
    spec_module = importlib.import_module("hyper_parallel.core.multicore.modules.mega_moe.spec")
    switches = {
        "HP_MEGA_MOE_INPLACE_WAVE_BACKWARD": int(options.inplace_backward),
        "HP_MEGA_MOE_ATOMIC_DW_ACCUMULATION": int(options.atomic_dw),
        "HP_MEGA_MOE_TILE_PHASE_ACK": int(options.inplace_backward),
        "HP_MEGA_MOE_DESTINATION_WAVE_FENCE": int(options.inplace_backward),
        "HP_MEGA_MOE_REUSE_RETAINED_DW_SCRATCH": 0,
    }
    previous = {name: os.environ.get(name) for name in switches}
    for name, value in switches.items():
        os.environ[name] = str(value)
    result = {"shape": asdict(shape), "ranks": ranks, "slots": slots, "options": asdict(options)}
    try:
        spec = spec_module.MegaMoeSpec(
            expert_capacity_factor=shape.capacity_factor,
            local_num_tokens=shape.tokens_per_rank,
            hidden_size=shape.hidden,
            intermediate_size=shape.intermediate,
            num_experts=shape.experts,
            top_k=shape.top_k,
            receive_capacity=shape.capacity,
            ep_size=ranks,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
            replica_slots_per_rank=slots,
            forward_fusion_mode="permute",
            backward_fusion_mode="both",
        )
        plan = plan_module.build_mega_moe_plan(spec, "meta")
        tensors = {
            name: value.numel() * value.element_size() for name, value in vars(plan).items() if hasattr(value, "numel")
        }
        result.update(
            buildable=True,
            plan_tensor_bytes=tensors,
            total_plan_tensor_bytes=sum(tensors.values()),
            scope="rank-zero graph construction only; numerical kernels and other ranks are not executed",
        )
    except (IndexError, ValueError) as error:
        frames = traceback.extract_tb(error.__traceback__)
        result.update(
            buildable=False,
            error_type=type(error).__name__,
            error=str(error),
            failure={"file": frames[-1].filename, "line": frames[-1].lineno, "code": frames[-1].line},
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return result


def main() -> None:
    """Audit a source-matched rank/budget matrix with no device allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", default="16,32,64")
    parser.add_argument("--slots", default="0,1,2,3")
    parser.add_argument("--shape", type=Path, help="Shape JSON; defaults to E384/T4096/H5120/I1792/K8")
    parser.add_argument("--options", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a fresh artifact path")
    shape = Shape(**json.loads(args.shape.read_text())) if args.shape else Shape()
    options = Options(**json.loads(args.options.read_text())) if args.options else Options()
    results = [
        audit(shape, ranks, slots, options)
        for ranks in map(int, args.ranks.split(","))
        for slots in map(int, args.slots.split(","))
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
