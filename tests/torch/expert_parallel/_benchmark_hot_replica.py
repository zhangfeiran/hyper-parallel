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
"""Fresh-process replica F+B+AdamW samples with untimed plan and payload evidence."""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica.device_kernel import (
    _planner_library,
)
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore._loader import get_multicore_paths
from tests.common.port_utils import allocate_port
from tests.torch.expert_parallel.hot_replica_checks import (
    execution_identity,
    file_identity,
    route_evidence,
)


def _step(module, optimizer, values, ids, probabilities, gradient, *, recompute=False, inspect=False):
    optimizer.zero_grad(set_to_none=True)
    values.grad = None
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    plans = []

    def _capture(_module, _inputs, output):
        if torch.is_grad_enabled():
            plans.append(route_evidence(output, module.replica_planner))

    handle = module.register_forward_hook(_capture) if inspect and module.replica_slots_per_rank else None
    try:
        # Reentrant recompute owns its new plan if push capacity grew after the first forward.
        output = (checkpoint(module, values, ids, probabilities, use_reentrant=True) if recompute else
                  module(values, ids, probabilities))
        output.backward(gradient)
    finally:
        if handle is not None:
            handle.remove()
    plan = plans[-1] if plans else None
    optimizer.step()
    torch.npu.synchronize()
    local_ms = (time.perf_counter() - start) * 1000
    maximum = torch.tensor([local_ms], device=values.device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if inspect:
        if module.replica_slots_per_rank and plan is None:
            raise AssertionError("Audit did not observe the gradient-enabled forward plan")
        if not torch.isfinite(output).all() or not torch.isfinite(values.grad).all():
            raise AssertionError("Benchmark produced nonfinite outputs or input gradients")
        provider = module._get_execution_resources(values).workspace.replica_provider
        if plan is not None:
            plan["kernel_gradient_matrices"] = getattr(provider, "last_kernel_gradient_matrices", ())
    return {"local_step_ms": local_ms, "rank_max_step_ms": float(maximum.cpu()[0]), "plan": plan}


def run(args: argparse.Namespace) -> None:
    """Separate cold/audit calls from warmed samples and record every rank."""
    rank, size = dist.get_rank(), dist.get_world_size()
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    endpoint = [f"tcp://127.0.0.1:{allocate_port()}" if rank == 0 else None]
    dist.broadcast_object_list(endpoint, src=0)
    os.environ["HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT"] = endpoint[0]
    calibration = None if args.cost_model is None else ExpertReplicaCostModel(
        **json.loads(Path(args.cost_model).read_text(encoding="utf-8"))["model"])
    torch.manual_seed(715)
    module = MegaMoeExperts(
        local_num_tokens=args.tokens, hidden_size=args.hidden, intermediate_size=args.intermediate,
        num_experts=size * 6, top_k=args.top_k, ep_size=size, ep_group=dist.group.WORLD,
        dispatch_mode=args.backend, replica_slots_per_rank=args.budget, replica_transport=args.replica_transport,
        replica_min_rows=args.replica_min_rows, replica_target_load=args.replica_target_load,
        replica_planner=args.replica_planner, replica_cost_model=calibration,
        replica_minimum_load_reduction=args.replica_minimum_load_reduction,
    ).to(device=device, dtype=torch.bfloat16)
    try:
        torch.manual_seed(712 + rank)
        values = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16).requires_grad_()
        gradient = torch.randn_like(values) / args.tokens
        probabilities = torch.full((args.tokens, args.top_k), 1 / args.top_k, device=device)
        positions = torch.arange(args.tokens * args.top_k, device=device).reshape(args.tokens, args.top_k)
        ids = positions.remainder(size * 6 if args.route == "balanced" else args.top_k).long()
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.001)

        def _execute(iteration, *, inspect=False):
            route = ids if args.route != "rotating_hot" else (ids + 6 * (iteration % size)).remainder(size * 6)
            return _step(module, optimizer, values, route, probabilities, gradient,
                         recompute=args.checkpoint, inspect=inspect)

        cold = _execute(0, inspect=True)
        for iteration in range(args.warmup):
            _execute(iteration)
        torch.npu.reset_peak_memory_stats()
        samples = [_execute(iteration) for iteration in range(args.iterations)]
        memory = {"peak_allocated_bytes": torch.npu.max_memory_allocated(),
                  "peak_reserved_bytes": torch.npu.max_memory_reserved()}
        audit = _execute(0, inspect=True)
        identity = execution_identity()
        vendor, adapter = get_multicore_paths()
        payloads = [adapter, vendor / "op_api/lib/libcust_opapi.so", *sorted(vendor.rglob("*.o"))]
        identity.update(torch_npu=torch_npu.__version__, rank=rank, ep_size=size,
                        ep_members=dist.get_process_group_ranks(dist.group.WORLD),
                        payload=[file_identity(path) for path in payloads],
                        transport_abi=torch.ops.hyper_parallel.mega_moe_transport_version())
        if args.replica_planner == "device" and args.budget:
            library = _planner_library()
            identity["planner_payload"] = {**file_identity(Path(library._name)), "abi": library.planner_abi_version()}
        directory = Path(args.result_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"rank{rank}.json").write_text(json.dumps(
            {"identity": identity, "configuration": vars(args), "cold": cold, "samples": samples,
             "audit": audit, "memory": memory, "scope": "Warmed F+B+AdamW; cold/audit calls are excluded. "
                 "Run fresh-process A/B/B/A with identical route, seed and ownership."}, indent=2) + "\n",
            encoding="utf-8")
    finally:
        module.close()


def main() -> None:
    """Run one frozen configuration per fresh distributed process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("push", "pull"), default="push")
    parser.add_argument("--replica-transport", choices=("p2p", "shmem_signal_kernel_gradient"), default="p2p")
    parser.add_argument("--replica-planner", choices=("cpu", "device"), default="cpu")
    parser.add_argument("--route", choices=("balanced", "hot", "rotating_hot"), default="hot")
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--replica-min-rows", type=int, default=1024)
    parser.add_argument("--replica-target-load", type=int)
    parser.add_argument("--replica-minimum-load-reduction", type=float)
    parser.add_argument("--cost-model")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=1792)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("Benchmark requires at least one warmup and measured step")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
