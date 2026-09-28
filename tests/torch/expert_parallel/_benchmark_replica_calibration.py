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
"""Offline conservative replica calibration using actual FP32 partial execution."""

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaCostModel, build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute
from hyper_parallel.core.expert_parallel.hot_replica.transport import prefetch_weights, return_gradients
from hyper_parallel.core.multicore import MegaMoeExperts
from tests.common.port_utils import allocate_port
from tests.torch.expert_parallel.hot_replica_checks import execution_identity


def _maximum(value, device):
    maximum = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.cpu()[0])


def _spread(values):
    deciles = statistics.quantiles(values, n=10)
    return deciles[8] - deciles[0]


def _compute(args, rows, device):
    rank, size = dist.get_rank(), dist.get_world_size()
    module = MegaMoeExperts(local_num_tokens=rows, hidden_size=args.hidden,
                            intermediate_size=args.intermediate, num_experts=size, top_k=1,
                            ep_size=size, ep_group=dist.group.WORLD, dispatch_mode=args.backend,
                            replica_slots_per_rank=1, replica_transport=args.replica_transport).to(
                                device=device, dtype=torch.bfloat16)
    values = torch.randn(rows, args.hidden, device=device, dtype=torch.bfloat16).requires_grad_()
    gradient = torch.randn_like(values) / rows
    probabilities = torch.ones(rows, 1, device=device)
    routes = {"local": torch.full((rows, 1), rank, device=device, dtype=torch.long)}
    if rows == args.maximum_rows:
        routes["remote"] = torch.full((rows, 1), (rank + 1) % size, device=device, dtype=torch.long)
    samples = {}
    for name, ids in routes.items():
        forward, backward = [], []
        for iteration in range(args.iterations + args.warmup):
            module.zero_grad(set_to_none=True)
            values.grad = None
            dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            output = module(values, ids, probabilities)
            torch.npu.synchronize()
            middle = time.perf_counter()
            output.backward(gradient)
            torch.npu.synchronize()
            end = time.perf_counter()
            forward_ms = _maximum((middle - start) * 1000, device)
            backward_ms = _maximum((end - middle) * 1000, device)
            if iteration >= args.warmup:
                forward.append(forward_ms)
                backward.append(backward_ms)
        samples[name] = {"forward_ms": forward, "backward_ms": backward}
    return module, values, {"rows": rows, **samples}


def _transfers(args, module, values):
    size, rank = dist.get_world_size(), dist.get_rank()
    weights = (module.gate_up_weight, module.down_weight)
    provider = module._get_execution_resources(values).workspace.replica_provider
    plan = build_expert_replica_plan([[128] + [0] * (size - 1)] * size, 1)
    route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), rank, dist.group.WORLD, provider)
    fanout = max(sum(transfer.owner_rank == peer for transfer in plan.transfers) for peer in range(size))
    if not fanout:
        raise AssertionError("Calibration did not produce a replica transfer")
    weight_samples, gradient_samples = [], []
    for iteration in range(args.iterations + args.warmup):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        with prefetch_weights(weights, route, provider=provider):
            pass
        torch.npu.synchronize()
        weight_ms = _maximum((time.perf_counter() - start) * 1000, values.device) / fanout
        with prefetch_weights(weights, route, provider=provider, backward=True) as pool:
            for guest in pool.gradients:
                guest.zero_()
            home = tuple(torch.zeros_like(weight, dtype=torch.float32) for weight in weights)
            dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            # Measure both projections without assuming a kernel overlap window.
            home = return_gradients(home, route, pool.gradients, provider, consume=True)
            torch.npu.synchronize()
            gradient_ms = _maximum((time.perf_counter() - start) * 1000, values.device) / fanout
        if iteration >= args.warmup:
            weight_samples.append(weight_ms)
            gradient_samples.append(gradient_ms)
    return weight_samples, gradient_samples


def _model(args, samples, weights, gradients):
    forward, backward = [(0, 0.0)], [(0, 0.0)]
    for row in samples:
        forward.append((row["rows"], max(forward[-1][1], statistics.median(row["local"]["forward_ms"]))))
        backward.append((row["rows"], max(backward[-1][1], statistics.median(row["local"]["backward_ms"]))))
    local = sum(statistics.median(samples[-1]["local"][key]) for key in ("forward_ms", "backward_ms"))
    remote = sum(statistics.median(samples[-1]["remote"][key]) for key in ("forward_ms", "backward_ms"))
    model = ExpertReplicaCostModel(calibration_version=2, backend=args.backend, hidden_size=args.hidden,
                                   intermediate_size=args.intermediate, ep_size=dist.get_world_size(),
                                   transport=args.replica_transport, forward_ms=tuple(forward),
                                   backward_ms=tuple(backward), weight_ms=statistics.median(weights),
                                   gradient_ms=statistics.median(gradients),
                                   token_ms_per_row=max(0, remote - local) / (2 * args.maximum_rows))
    counts = [[512] * 8 + [0] * 16] * dist.get_world_size()

    def _host_cost(calibration):
        timings = []
        for _ in range(60):
            start = time.perf_counter()
            build_expert_replica_plan(counts, 1, capacity_limit=10926, cost_model=calibration)
            timings.append((time.perf_counter() - start) * 1000)
        return statistics.quantiles(timings, n=10)[8]

    extra = max(0, _host_cost(model) - _host_cost(None))
    uncertainty = max(_spread(row["local"][key]) for row in samples for key in ("forward_ms", "backward_ms"))
    return replace(model, minimum_gain_ms=extra + uncertainty), extra, uncertainty


def run(args: argparse.Namespace) -> None:
    """Collect raw distributions and an explicitly versioned conservative model."""
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    endpoint = [f"tcp://127.0.0.1:{allocate_port()}" if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(endpoint, src=0)
    os.environ["HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT"] = endpoint[0]
    torch.manual_seed(715)
    samples = []
    weights, gradients = [], []
    rows = 128
    while rows <= args.maximum_rows:
        module, values, sample = _compute(args, rows, device)
        try:
            samples.append(sample)
            if rows == args.maximum_rows:
                weights, gradients = _transfers(args, module, values)
        finally:
            module.close()
        del module, values
        rows *= 2
    if dist.get_rank() == 0:
        model, extra, uncertainty = _model(args, samples, weights, gradients)
        directory = Path(args.result_dir)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"model": asdict(model), "samples": samples, "weight_samples_ms": weights,
                   "gradient_samples_ms": gradients, "cpu_extra_p90_ms": extra,
                   "compute_p90_p10_ms": uncertainty, "identity": execution_identity(),
                   "torch_npu": torch_npu.__version__, "configuration": vars(args),
                   "scope": "Single active expert F/B includes launch/control. Both gradient projections are exposed. "
                            "Fan-out normalized transfer costs approximate contention. No schedule-prefix discount. "
                            "Calibration has one home expert; target has six. Validate candidate plans end to end."}
        (directory / "cost-model.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    dist.barrier()


def main() -> None:
    """Run calibration in a fresh four-rank process, separate from timed ablations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("push", "pull"), default="push")
    parser.add_argument("--replica-transport", default="shmem_signal_sdma_projection")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=1792)
    parser.add_argument("--maximum-rows", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    if args.maximum_rows < 128 or args.maximum_rows & (args.maximum_rows - 1):
        raise ValueError("maximum-rows must be a power of two >= 128")
    if args.iterations < 2 or args.warmup < 1:
        raise ValueError("Calibration requires at least two measured and one warmup iteration")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        if dist.get_world_size() != 4:
            raise ValueError("The reference refinement workload requires EP4")
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
