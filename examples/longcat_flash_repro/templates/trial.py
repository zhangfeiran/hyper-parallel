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
"""Original-width precision and preliminary timing of exact bounded schedules."""

import json
import os
import statistics
import time

# Standalone accelerator workers use the native Torch APIs required by the project.
# pylint: disable=forbidden-backend-import
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from tests.torch.models.longcat_flash._benchmark_longcat_flash import _arguments, _build, _step
from tests.torch.models.longcat_flash._compare_longcat_flash import _compare, _snapshot, _values
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime


def measure(tensors: tuple) -> dict[str, Any]:
    """Collect diagnostic forward and backward timings.

    Args:
        tensors: Expert module inputs and upstream gradient.
    """
    for _ in range(3):
        _step(*tensors, True)
    torch.npu.synchronize()
    dist.barrier()
    local, maximum = ([], [])
    for _ in range(10):
        start = time.perf_counter()
        _step(*tensors, True)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        value = torch.tensor([elapsed], device="npu", dtype=torch.float32)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        local.append(elapsed)
        maximum.append(value.item())
    return {"local_ms": local, "rank_max_ms": maximum, "p50_ms": statistics.median(maximum)}


args = _arguments()
args.backend = "native"
rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(rank)
dist.init_process_group("hccl")
start_shmem_lifetime()
case_tensors = _build(args, rank)
experts = case_tensors[0]
experts.direct_weight_gradients = False
expected = _snapshot(case_tensors, _step(*case_tensors, True))
baseline = measure(case_tensors)
results = []
experts.routing_backend = "triton"
for label in ("native-old-grad", "native", "push-old-grad", "push", "pull"):
    mode = "native" if label.startswith("native") else "push" if label == "push-old-grad" else label
    experts.direct_weight_gradients = not label.endswith("old-grad")
    experts.close_runtime()
    start_shmem_lifetime()
    experts.backend = "native" if mode == "native" else "megamoe"
    experts.dispatch_mode = "push" if mode == "native" else mode
    experts.configure(dist.group.WORLD, args.ep, args.top_k)
    output = _step(*case_tensors, True)
    comparison = {name: _compare(value, expected[name]) for name, value in _values(case_tensors, output).items()}
    passed = all((row["strict_pass"] for row in comparison.values()))
    timing = measure(case_tensors) if passed else None
    results.append({"mode": label, "comparison": comparison, "strict_pass": passed, "timing": timing})
experts.close_runtime()
args.output.mkdir(exist_ok=True, parents=True)
(args.output / f"rank{rank}.json").write_text(
    json.dumps(
        {
            "rank": rank,
            "pid": os.getpid(),
            "baseline": baseline,
            "results": results,
            "shmem_after_close": shmem.debug_state(),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
dist.barrier()
dist.destroy_process_group()
if not all((row["strict_pass"] for row in results)):
    raise RuntimeError("bounded schedule precision failed")
