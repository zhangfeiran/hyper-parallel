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
"""Measure one fixed-shape MegaMoE plus shared expert and export internal traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe import module as module_impl
from hyper_parallel.core.multicore.profiler import mega_kernel_profile, schedule


def signature(tensor: torch.Tensor) -> dict:
    """Fingerprint values independently of the device storage layout."""
    host = tensor.detach().float().cpu().contiguous()
    return {"sha256": hashlib.sha256(host.numpy().tobytes()).hexdigest(),
            "sum": host.double().sum().item(), "norm": host.double().norm().item()}


def main() -> None:
    """Measure one complete layer and optionally check or profile an extra step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=1792)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--transport", choices=("push", "pull"), default="push")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--route", choices=("balanced", "dirichlet"), default="balanced")
    parser.add_argument("--placement", choices=("tail", "dispatch", "overlap", "idle"), default="idle")
    args = parser.parse_args()
    if args.warmup < 0 or args.steps <= 0:
        parser.error("warmup must be nonnegative and steps must be positive")
    if args.fused:
        original_bind = module_impl.bind_mega_moe_spec
        module_impl.bind_mega_moe_spec = lambda spec, x: replace(
            original_bind(spec, x), shared_placement=args.placement,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("npu", local_rank)
    own_pids = [None] * world
    dist.all_gather_object(own_pids, os.getpid())
    if rank == 0:
        root = Path(__file__).resolve().parents[3]
        benchmark_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
        (args.output / "source.patch").write_bytes(patch)
        binaries = {}
        for path in (root / "build/native/payload").rglob("*"):
            if path.is_file() and path.suffix in (".so", ".o"):
                binaries[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance = dict(source=source, benchmark_sha256=benchmark_sha, patch_sha256=hashlib.sha256(patch).hexdigest(),
                          binaries=binaries, python=sys.version, torch=torch.__version__,
                          torch_npu=torch_npu.__version__, own_pids=own_pids,
                          command=sys.argv, environment={key: os.environ.get(key) for key in
                          ("HYPER_ROOT", "ASCEND_HOME_PATH", "LD_LIBRARY_PATH", "ASCEND_CUSTOM_OPP_PATH")})
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))

    def record_health(phase: str) -> None:
        """Reject foreign NPU workers before and after the measured window."""
        foreign = []
        if rank == 0:
            health = subprocess.check_output(["npu-smi", "info"], text=True)
            (args.output / f"npu-smi-{phase}.txt").write_text(health)
            active = set(map(int, re.findall(r"\|\s*\d+\s+\d+\s+(\d+)\s+", health)))
            foreign = sorted(active - set(own_pids))
            if foreign:
                (args.output / "CONTAMINATED.json").write_text(json.dumps(dict(phase=phase, foreign_pids=foreign)))
        contaminated = torch.tensor(len(foreign), dtype=torch.int64, device=device)
        dist.all_reduce(contaminated)
        if contaminated.item():
            raise RuntimeError(f"Foreign NPU workers contaminated this run: rank={rank} local={foreign}")

    record_health("start")
    dist.barrier()
    torch.manual_seed(42 + rank)
    layer = MegaMoeExperts(local_num_tokens=args.tokens, hidden_size=args.hidden,
                           intermediate_size=args.intermediate, num_experts=args.experts,
                           top_k=args.topk, expert_capacity_factor=1.5,
                           ep_size=world, ep_group=dist.group.WORLD,
                           dispatch_mode=args.transport,
                           **({"shared_intermediate_size": args.intermediate} if args.fused else {}),
                           ).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16).requires_grad_()
    dy = torch.randn_like(x) * 0.01
    w1 = torch.nn.Parameter(torch.randn(2 * args.intermediate, args.hidden, device=device,
                                       dtype=torch.bfloat16) * 0.01)
    w2 = torch.nn.Parameter(torch.randn(args.hidden, args.intermediate, device=device,
                                       dtype=torch.bfloat16) * 0.01)
    ids = ((torch.arange(args.tokens * args.topk, device=device).reshape(args.tokens, args.topk)
            + rank * args.topk) % args.experts).to(torch.int32)
    if args.route == "dirichlet":
        # One deterministic global load vector, then distinct local token samples.
        load = torch.from_numpy(np.random.default_rng(193).dirichlet(np.full(args.experts, 10.0))).float()
        generator = torch.Generator().manual_seed(913 + rank)
        sample_weights = load.expand(args.tokens, -1)
        ids = torch.multinomial(sample_weights, args.topk, replacement=False, generator=generator)
        ids = ids.to(device=device, dtype=torch.int32)
    probs = torch.full((args.tokens, args.topk), 1 / args.topk, device=device,
                       dtype=torch.float32, requires_grad=True)

    reference = None
    if args.check:
        reference = MegaMoeExperts(local_num_tokens=args.tokens, hidden_size=args.hidden,
                                  intermediate_size=args.intermediate, num_experts=args.experts,
                                  top_k=args.topk, expert_capacity_factor=1.5, ep_size=world,
                                  ep_group=dist.group.WORLD, dispatch_mode=args.transport,
                                  ).to(device=device, dtype=torch.bfloat16)
        reference.load_state_dict(layer.state_dict())

    def step() -> torch.Tensor:
        """Run forward and backward, including the shared branch in both modes."""
        layer.zero_grad(set_to_none=True)
        x.grad = probs.grad = w1.grad = w2.grad = None
        if args.fused:
            output = layer(x, ids, probs, shared_gate_up_weight=w1, shared_down_weight=w2)
        else:
            routed = layer(x, ids, probs)
            shared = F.linear(torch_npu.npu_swiglu(F.linear(x, w1)), w2)
            output = routed + shared
        output.backward(dy)
        return output

    try:
        if args.check:
            actual = step()
            torch.npu.synchronize()
            actual_tensors = [tensor.detach().clone() for tensor in
                              (actual, x.grad, probs.grad, layer.gate_up_weight.grad, layer.down_weight.grad,
                               w1.grad, w2.grad)]
            x.grad = probs.grad = w1.grad = w2.grad = None
            expected = reference(x, ids, probs) + F.linear(torch_npu.npu_swiglu(F.linear(x, w1)), w2)
            expected.backward(dy)
            expected_tensors = (expected, x.grad, probs.grad, reference.gate_up_weight.grad,
                                reference.down_weight.grad, w1.grad, w2.grad)
            errors = []
            failures = []
            for name, observed, wanted in zip(("output", "dx", "dprobs", "dw1", "dw2", "shared_dw1", "shared_dw2"),
                                               actual_tensors, expected_tensors):
                max_error = (observed.float() - wanted.float()).abs().max().item()
                errors.append({"name": name, "max_abs_error": max_error})
                try:
                    torch.testing.assert_close(observed, wanted, atol=2e-3, rtol=2e-2, msg=f"rank={rank} {name}")
                except AssertionError as error:
                    failures.append(str(error))
            (args.output / f"rank{rank}_correctness.json").write_text(json.dumps(errors, indent=2))
            failure_count = torch.tensor(len(failures), dtype=torch.int64, device=device)
            dist.all_reduce(failure_count)
            if failure_count.item():
                raise AssertionError(f"Distributed correctness comparison failed: rank={rank}, local={failures}")
            reference.close()
            reference = None
            del actual_tensors, expected_tensors, actual, expected
            dist.barrier()
        for _ in range(args.warmup):
            step()
        torch.npu.synchronize()
        dist.barrier()
        torch.npu.reset_peak_memory_stats()
        times = []
        for _ in range(args.steps):
            start = time.perf_counter()
            output = step()
            torch.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        stats = {"rank": rank, "shape": vars(args) | {"output": str(args.output)},
                 "world_size": world, "times_ms": times,
                 "max_allocated_gib": torch.npu.max_memory_allocated() / 2**30,
                 "max_reserved_gib": torch.npu.max_memory_reserved() / 2**30,
                 "signature": {name: signature(tensor) for name, tensor in
                               (("output", output), ("dx", x.grad), ("dprobs", probs.grad),
                                ("dw1", layer.gate_up_weight.grad), ("dw2", layer.down_weight.grad),
                                ("shared_dw1", w1.grad), ("shared_dw2", w2.grad))}}
        (args.output / f"rank{rank}.json").write_text(json.dumps(stats, indent=2))
        dist.barrier()
        if args.profile:
            dropped = torch.zeros((), dtype=torch.int64, device=device)
            if rank in (0, world - 1):
                with mega_kernel_profile(schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
                                         detailed_task_names=True) as profiler:
                    step()
                    profiler.step()
                trace = profiler.export_chrome_trace(args.output / f"rank{rank}_trace.json")
                dropped.fill_(trace["megaKernelCycleTrace"]["droppedRecordCount"])
            else:
                step()
            torch.npu.synchronize()
            dist.all_reduce(dropped, op=dist.ReduceOp.MAX)
            if dropped.item():
                raise RuntimeError("Incomplete cycle trace")
            dist.barrier()
        record_health("end")
        dist.barrier()
        if rank == 0:
            rank_stats = [json.loads((args.output / f"rank{peer}.json").read_text()) for peer in range(world)]
            maxima = [max(item["times_ms"][i] for item in rank_stats) for i in range(args.steps)]
            summary = {"rank_max_median_ms": statistics.median(maxima),
                       "rank_max_p95_ms": float(torch.tensor(maxima).quantile(0.95)),
                       "max_allocated_gib": max(item["max_allocated_gib"] for item in rank_stats),
                       "max_reserved_gib": max(item["max_reserved_gib"] for item in rank_stats),
                       "times_ms": maxima}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
    finally:
        if reference is not None:
            reference.close()
        layer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
