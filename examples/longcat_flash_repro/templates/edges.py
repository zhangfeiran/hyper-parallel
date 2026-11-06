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
# Standalone accelerator workers use the native Torch APIs required by the project.
# pylint: disable=forbidden-backend-import
import json
import os

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from tests.torch.models.longcat_flash._benchmark_longcat_flash import _arguments, _build, _step
from tests.torch.models.longcat_flash._compare_longcat_flash import _compare, _values
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime

args = _arguments()
args.backend = "native"
rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(rank)
dist.init_process_group("hccl")
start_shmem_lifetime()
original = _build(args, rank)
experts, hidden, learned_ids, probabilities, upstream = original
versions = {name: value._version for name, value in experts.named_parameters()}
results = []
for case in ["empty_source", "empty_owners", "identity_only"]:
    ids = learned_ids.clone()
    if case == "empty_source" and rank == 0 or case == "identity_only":
        ids = torch.arange(args.real, args.real + args.top_k, device="npu").expand(args.tokens, -1).contiguous()
    if case == "empty_owners":
        ids = (
            torch.tensor([0, 1, 2, 3, 4, 5] + list(range(args.real, args.real + 6)), device="npu")
            .expand(args.tokens, -1)
            .contiguous()
        )
    tensors = (experts, hidden, ids, probabilities, upstream)
    experts.close_runtime()
    start_shmem_lifetime()
    experts.backend, experts.routing_backend = "native", "torch"
    experts.direct_weight_gradients = False
    experts.configure(dist.group.WORLD, args.ep, args.top_k)
    output = _step(*tensors, True)
    expected = {
        name: None if value is None else value.detach().cpu() for name, value in _values(tensors, output).items()
    }
    for mode in ["native", "push", "pull"]:
        experts.close_runtime()
        start_shmem_lifetime()
        experts.backend = "native" if mode == "native" else "megamoe"
        experts.dispatch_mode = "push" if mode == "native" else mode
        experts.routing_backend = "triton"
        experts.direct_weight_gradients = True
        experts.configure(dist.group.WORLD, args.ep, args.top_k)
        output = _step(*tensors, True)
        comparison = {}
        for name, value in _values(tensors, output).items():
            target = expected[name]
            if target is None or value is None:
                comparison[name] = {
                    "unused_gradient_identical": target is None and value is None,
                    "pass": target is None and value is None,
                }
            else:
                current = _compare(value, target)
                current["pass"] = current["strict_pass"] and current["max_abs"] == 0
                comparison[name] = current
        if versions != {name: value._version for name, value in experts.named_parameters()}:
            raise RuntimeError("a boundary test modified expert parameters")
        results.append(
            {
                "case": case,
                "backend": mode,
                "comparison": comparison,
                "pass": all(value["pass"] for value in comparison.values()),
            }
        )
    del expected
experts.close_runtime()
torch.npu.synchronize()
args.output.mkdir(exist_ok=True, parents=True)
(args.output / f"rank{rank}.json").write_text(
    json.dumps(
        {
            "rank": rank,
            "pid": os.getpid(),
            "scope": "Original-width forced-routing diagnostics; not performance acceptance",
            "shape": {
                "tokens": args.tokens,
                "hidden": 6144,
                "intermediate": 2048,
                "real": args.real,
                "zero": args.zero,
                "top_k": 12,
                "ep": args.ep,
            },
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
if not all(row["pass"] for row in results):
    raise RuntimeError("original-width empty-route precision failed")
