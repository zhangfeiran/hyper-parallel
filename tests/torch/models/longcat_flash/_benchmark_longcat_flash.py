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
"""Same-topology LongCat expert-component forward/backward benchmark worker."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from torch.nn import functional as F
from transformers import LongcatFlashConfig
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashExperts, LongcatFlashTopkRouter

import hyper_parallel
from hyper_parallel.core.multicore import shmem
from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts
from tests.torch.multicore._mega_moe_utils import environment_identity, memory_sample, start_shmem_lifetime


def _hash(tensor: torch.Tensor) -> str:
    """Hash every byte in bounded host chunks, outside the timed interval."""
    digest = hashlib.sha256()
    flattened = tensor.detach().reshape(-1)
    for offset in range(0, flattened.numel(), 32 * 1024 * 1024):
        part = flattened[offset:offset + 32 * 1024 * 1024].cpu().contiguous().view(torch.uint8)
        digest.update(part.numpy().tobytes())
    return digest.hexdigest()


def _build(args: argparse.Namespace, rank: int) -> tuple:
    """Materialize only the owner shard, including retained unused Gate/Up rows."""
    if Path(hyper_parallel.__file__).resolve().parents[1] != Path.cwd():
        raise RuntimeError("the editable install must resolve to the benchmark checkout")
    config = LongcatFlashConfig.from_dict({
        "hidden_size": args.hidden, "expert_ffn_hidden_size": args.intermediate,
        "n_routed_experts": args.real, "zero_expert_num": args.zero, "moe_topk": args.top_k,
    })
    with torch.device("meta"):
        source = LongcatFlashExperts(config)
        experts = LongcatFlashMegaMoeExperts(module=source,
            backend="native" if args.backend == "native" else "megamoe",
            swiglu_precision=args.swiglu_precision,
            dispatch_mode="push" if args.backend == "native" else args.backend,
            context={"ep_size": args.ep, "ep": args.ep > 1})
    torch.npu.manual_seed(22000 + rank)
    dimensions = {"gate_up_proj": ((args.real + args.zero) // args.ep, 2 * args.intermediate, args.hidden),
                  "down_proj": (args.real // args.ep, args.hidden, args.intermediate)}
    for name, shape in dimensions.items():
        local = torch.randn(shape, device="npu", dtype=torch.bfloat16).mul_(0.02)
        setattr(experts, name, torch.nn.Parameter(local))
    if args.ep > 1 and args.zero:
        experts._checkpoint_ep_size = torch.tensor([args.ep], device="npu", dtype=torch.int64)
    experts.configure(dist.group.WORLD, args.ep, args.top_k)
    torch.manual_seed(12000)
    router = LongcatFlashTopkRouter(config).to(device="npu", dtype=torch.bfloat16)
    torch.npu.manual_seed(32000 + rank)
    hidden = torch.randn(args.tokens, args.hidden, device="npu", dtype=torch.bfloat16)
    with torch.no_grad():
        probabilities, indices = router(hidden)[-2:]
    upstream = torch.randn_like(hidden)
    return experts, hidden.requires_grad_(), indices, probabilities.detach().requires_grad_(), upstream


@contextmanager
def _operator_census(enabled: bool) -> Any:
    """Observe one untimed native call, then restore all public operator functions."""
    names = ("npu_moe_token_permute", "npu_moe_token_unpermute", "npu_grouped_matmul", "npu_swiglu")
    owners = {name: torch_npu for name in names}
    owners["silu"] = F
    original, counts = {}, dict.fromkeys(owners, 0)
    if enabled:
        for name, owner in owners.items():
            original[name] = getattr(owner, name)

            def observed(*args: Any, _name: str = name, **kwargs: Any) -> Any:
                """Count and execute the selected public NPU primitive.

                Args:
                    *args: Original operator arguments.
                    _name: Bound public operator name.
                    **kwargs: Original operator keyword arguments.
                """
                counts[_name] += 1
                return original[_name](*args, **kwargs)

            setattr(owner, name, observed)
    try:
        yield counts
    finally:
        for name, operator in original.items():
            setattr(owners[name], name, operator)


def _step(experts: torch.nn.Module, hidden: torch.Tensor, ids: torch.Tensor,
          probabilities: torch.Tensor, upstream: torch.Tensor, backward: bool) -> torch.Tensor:
    """Execute one complete expert forward or forward/backward with fresh gradients."""
    experts.zero_grad(set_to_none=True)
    hidden.grad = None
    probabilities.grad = None
    with torch.set_grad_enabled(backward):
        output = experts(hidden, ids, probabilities)
        if backward:
            output.backward(upstream)
    return output


def _phase(args: argparse.Namespace, tensors: tuple, backward: bool) -> dict:
    """Warm the same implementation, then retain every local and rank-max sample."""
    for _ in range(args.warmup):
        _step(*tensors, backward)
    torch.npu.synchronize()
    dist.barrier()
    torch.npu.reset_peak_memory_stats()
    local_samples, rank_samples = [], []
    for _ in range(args.samples):
        started = time.perf_counter()
        output = _step(*tensors, backward)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        maximum = torch.tensor([elapsed], device="npu", dtype=torch.float32)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        local_samples.append(elapsed)
        rank_samples.append(float(maximum.item()))
    return {"local_ms": local_samples, "rank_max_ms": rank_samples,
            "output_finite": bool(torch.isfinite(output).all()),
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(), "memory": memory_sample()}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "push", "pull"), required=True)
    parser.add_argument("--swiglu-precision", choices=("hf", "fused"), default="hf")
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--real", type=int, default=512)
    parser.add_argument("--zero", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one isolated backend process and persist rank-indexed terminal evidence."""
    args = _arguments()
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != args.ep:
        raise ValueError("benchmark world size must equal --ep")
    start_shmem_lifetime()
    tensors = _build(args, rank)
    experts, hidden, ids, probabilities, upstream = tensors
    identities = {name: _hash(value) for name, value in experts.named_parameters()}
    identities.update(hidden=_hash(hidden), ids=_hash(ids),
                      probabilities=_hash(probabilities), upstream=_hash(upstream))
    source_counts = torch.bincount(ids.reshape(-1).long(), minlength=args.real + args.zero)
    global_counts = source_counts.clone()
    dist.all_reduce(global_counts)
    local_experts = args.real // args.ep
    routing = {"real_source_rows": int(source_counts[:args.real].sum()),
               "identity_source_rows": int(source_counts[args.real:].sum()),
               "owner_received_rows": int(global_counts[rank * local_experts:(rank + 1) * local_experts].sum()),
               "global_real_expert_counts": global_counts[:args.real].tolist()}
    with _operator_census(args.backend == "native") as operators:
        _step(*tensors, True)
        torch.npu.synchronize()
    phases = {name: _phase(args, tensors, backward) for name, backward
              in (("forward", False), ("forward_backward", True))}
    finite_gradients = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                           for parameter in experts.parameters())
    unused_gradient_zero = not bool(experts.gate_up_proj.grad[args.real // args.ep:].count_nonzero())
    parameter_shapes = {name: list(value.shape) for name, value in experts.named_parameters()}
    experts.close_runtime()
    torch.npu.synchronize()
    report = {"rank": rank, "pid": os.getpid(), "backend": args.backend,
              "swiglu_precision": args.swiglu_precision,
              "shape": {name: getattr(args, name) for name in
                        ("ep", "tokens", "hidden", "intermediate", "real", "zero", "top_k")},
              "scope": "expert component; learned routing precomputed; no optimizer or Router projection",
              "hyper_parallel_source": hyper_parallel.__file__,
              "warmup": args.warmup, "samples": args.samples, "identities": identities,
              "parameter_local_shapes": parameter_shapes, "native_operator_census": operators, "routing": routing,
              "phases": phases, "finite_gradients": finite_gradients,
              "unused_gate_rows_gradient_zero": unused_gradient_zero,
              "shmem_after_close": shmem.debug_state(), "environment": environment_identity()}
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if not finite_gradients or not unused_gradient_zero or any(not row["output_finite"] for row in phases.values()):
        raise RuntimeError("non-finite result or a gradient reached an unused Gate/Up row")


if __name__ == "__main__":
    main()
