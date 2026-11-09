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
"""Full-width native/MegaMoE component parity with CPU-held reference gradients."""

import json
import os
import platform

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from tests.torch.models.longcat_flash._benchmark_longcat_flash import _arguments, _build, _hash, _step
from tests.torch.models.longcat_flash._precision import STRICT_GATE, compare_tensors as _compare, require_strict_pass
from tests.torch.multicore._mega_moe_utils import environment_identity, start_shmem_lifetime


def _snapshot(tensors: tuple, output: torch.Tensor) -> dict:
    """Retain reference values on CPU so only one device gradient set is live."""
    experts, hidden, _, probabilities, _ = tensors
    result = {"output": output.detach().cpu(), "input_gradient": hidden.grad.detach().cpu(),
              "probability_gradient": probabilities.grad.detach().cpu()}
    result.update({name: value.grad.detach().cpu() for name, value in experts.named_parameters()})
    return result


def _values(tensors: tuple, output: torch.Tensor) -> dict:
    """Resolve the same named candidate tensors without copying them."""
    experts, hidden, _, probabilities, _ = tensors
    result = {"output": output, "input_gradient": hidden.grad, "probability_gradient": probabilities.grad}
    result.update({name: value.grad for name, value in experts.named_parameters()})
    return result


def main() -> None:
    """Compare two transports against native operators at one identical EP topology."""
    args = _arguments()
    args.backend = "native"
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != args.ep:
        raise ValueError("component comparison requires WORLD_SIZE to match EP")
    start_shmem_lifetime()
    tensors = _build(args, rank)
    experts = tensors[0]
    identities = {name: _hash(value) for name, value in experts.named_parameters()}
    versions = {name: value._version for name, value in experts.named_parameters()}
    expected = _snapshot(tensors, _step(*tensors, True))
    experts.zero_grad(set_to_none=True)
    results = []
    for mode in ("push", "pull"):
        experts.close_runtime()
        start_shmem_lifetime()
        experts.backend, experts.dispatch_mode = "megamoe", mode
        experts.configure(dist.group.WORLD, args.ep, args.top_k)
        output = _step(*tensors, True)
        comparison = {name: _compare(value, expected[name]) for name, value in _values(tensors, output).items()}
        if versions != {name: value._version for name, value in experts.named_parameters()}:
            raise RuntimeError("the component comparison modified a parameter")
        results.append({"mode": mode, "comparison": comparison,
                        "strict_pass": all(row["strict_pass"] for row in comparison.values()),
                        "component_pass": all(row["component_pass"] for row in comparison.values())})
        status = torch.tensor([results[-1]["strict_pass"]], device="npu", dtype=torch.int32)
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        results[-1]["all_ranks_strict_pass"] = bool(status.item())
        results[-1]["failed_tensors"] = [name for name, row in comparison.items() if not row["strict_pass"]]
        experts.zero_grad(set_to_none=True)
    experts.close_runtime()
    report = {"rank": rank, "pid": os.getpid(), "ep": args.ep,
              "shape": {name: getattr(args, name) for name in
                        ("tokens", "hidden", "intermediate", "real", "zero", "top_k")},
              "oracle": "native NPU GMM/activation/permute/AllToAll/unpermute with identical parameters",
              "swiglu_precision": args.swiglu_precision,
              "gate": {"name": "strict", **STRICT_GATE, "component_pass": "diagnostic only"},
              "strict_pass": all(row["all_ranks_strict_pass"] for row in results),
              "host_arch": platform.machine(), "environment": environment_identity(),
              "identities": identities, "results": results, "shmem_after_close": shmem.debug_state()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    require_strict_pass(results)


if __name__ == "__main__":
    main()
