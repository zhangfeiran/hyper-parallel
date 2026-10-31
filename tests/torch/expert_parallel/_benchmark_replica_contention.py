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
"""Measure complete-matrix fanout and owner contention without a CANN profiler."""

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute
from hyper_parallel.core.expert_parallel.hot_replica.transport import prefetch_weights, return_gradients
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore._loader import get_multicore_paths
from tests.torch.expert_parallel.replica_contention_cases import ContentionCase, case_plan, make_case, summarize_ranks


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Publish complete JSON files atomically on shared storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def _check_gradients(returned: tuple[torch.Tensor, ...], case: ContentionCase, rank: int) -> int:
    """Check all home slots, including zero gradients on guest and idle ranks."""
    expected_sum = sum(target + 1 for owner, target in case.edges if owner == rank)
    for gradient in returned:
        expected = torch.zeros_like(gradient)
        expected[0].fill_(expected_sum)
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
    return len(returned)


def _measure_case(module: MegaMoeExperts, values: torch.Tensor, case: ContentionCase,
                  args: argparse.Namespace, group: object) -> dict[str, Any]:
    """Retain local timings; gather and validate outside all measured intervals."""
    rank = dist.get_rank(group)
    resources = module._get_execution_resources(values)
    resources.workspace.ensure(resources.spec, values.dtype, values.device)
    provider = resources.workspace.replica_provider
    weights = module.gate_up_weight, module.down_weight
    plan = case_plan(case, args.ep_size, args.home_experts)
    route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), rank, group, provider)
    home_gradients = tuple(torch.zeros_like(weight, dtype=torch.float32) for weight in weights)
    incoming = next((owner for owner, target in case.edges if target == rank), None)
    samples = []
    weight_checks, gradient_checks = 0, 0
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for iteration in range(args.warmup + args.iterations):
        dist.barrier(group=group)
        torch.npu.synchronize()
        started = time.perf_counter()
        begin.record()
        with prefetch_weights(weights, route, provider=provider) as pool:
            pass
        end.record()
        torch.npu.synchronize()
        weight_wall_ms = (time.perf_counter() - started) * 1000
        weight_ms = begin.elapsed_time(end)
        if incoming is not None and iteration == args.warmup:
            for guest in pool.weights:
                torch.testing.assert_close(guest[0], torch.full_like(guest[0], incoming + 1), rtol=0, atol=0)
                weight_checks += 1
        for gradient in home_gradients:
            gradient.zero_()
        with prefetch_weights(weights, route, provider=provider, backward=True) as pool:
            if incoming is not None:
                for gradient in pool.gradients:
                    gradient[0].fill_(rank + 1)
            dist.barrier(group=group)
            torch.npu.synchronize()
            started = time.perf_counter()
            begin.record()
            returned = return_gradients(home_gradients, route, pool.gradients, provider, consume=True)
            end.record()
            torch.npu.synchronize()
            gradient_wall_ms = (time.perf_counter() - started) * 1000
            gradient_ms = begin.elapsed_time(end)
        if iteration == args.warmup:
            gradient_checks = _check_gradients(returned, case, rank)
        if iteration >= args.warmup:
            samples.append(dict(weight_ms=weight_ms, physical_gradient_ms=gradient_ms,
                                weight_wall_ms=weight_wall_ms, gradient_wall_ms=gradient_wall_ms))
    return dict(case=case.name, ep_rank=rank, global_rank=dist.get_rank(), role=case.role(rank),
                weight_matrix_checks=weight_checks, home_gradient_matrix_checks=gradient_checks,
                checks_passed=gradient_checks == 2 and weight_checks == (2 if incoming is not None else 0),
                samples=samples)


def _cases(args: argparse.Namespace, nodes: tuple[int, ...]) -> list[ContentionCase]:
    """Rotate placement order across shapes to avoid always measuring local first."""
    cases = []
    for owner_index, owners in enumerate(args.owners):
        for fanout_index, fanout in enumerate(args.fanouts):
            offset = (owner_index + fanout_index) % len(args.placements)
            placements = args.placements[offset:] + args.placements[:offset]
            for placement in placements:
                if placement == 'mixed' and fanout == 1:
                    continue
                cases.append(make_case(nodes, fanout, owners, placement))
    return cases


def _collect(module: MegaMoeExperts, values: torch.Tensor, args: argparse.Namespace,
             group: object) -> dict[str, Any]:
    """Write per-case EP evidence and one terminal record from every global rank."""
    rank, ep_rank = dist.get_rank(), dist.get_rank(group)
    output = Path(args.result_dir)
    resources = module._get_execution_resources(values)
    nodes = tuple(resources.rank_nodes)
    cases = _cases(args, nodes)
    for weight in (module.gate_up_weight, module.down_weight):
        weight.fill_(ep_rank + 1)
    vendor, adapter = get_multicore_paths()
    identity = dict(global_rank=rank, ep_rank=ep_rank, global_leader=rank - ep_rank,
                    global_ep_members=dist.get_process_group_ranks(group), rank_nodes=nodes,
                    hostname=socket.gethostname(), pid=os.getpid(),
                    soc_name=torch.npu.get_device_name(values.device), torch_npu=torch_npu.__version__,
                    native_adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(), native_vendor=str(vendor))
    completed, summaries = [], []
    for index, case in enumerate(cases):
        print(f'Rank {rank}: starting case {index + 1}/{len(cases)} {case.name}', flush=True)
        record = _measure_case(module, values, case, args, group)
        records = [None] * args.ep_size
        dist.all_gather_object(records, record, group=group)
        summary = summarize_ranks(records)
        completed.append(record)
        if ep_rank == 0:
            weight_bytes = 3 * args.hidden * args.intermediate * 2
            payload = dict(case=asdict(case), name=case.name, identity=identity, ranks=records, summary=summary,
                           bf16_weight_bytes_per_edge=weight_bytes, fp32_gradient_bytes_per_edge=weight_bytes * 2,
                           intra_edges=sum(nodes[owner] == nodes[target] for owner, target in case.edges),
                           inter_edges=sum(nodes[owner] != nodes[target] for owner, target in case.edges))
            relative = f'ep{rank:05d}/{index:02d}-{case.name}.json'
            _write(output / relative, payload)
            summaries.append(dict(name=case.name, file=relative,
                                  medians=summary['median_of_iteration_rank_max_ms']))
    _write(output / f'ranks/rank_{rank:05d}.json',
           dict(identity=identity, cases=completed, completed=True, expected_cases=len(cases)))
    return dict(identity=identity, cases=summaries, expected_cases=len(cases))


def run(args: argparse.Namespace) -> None:
    """Run identical shapes in disjoint EP groups using production transport providers."""
    size, rank = dist.get_world_size(), dist.get_rank()
    if size % args.ep_size:
        raise ValueError('World size must be divisible by EP size')
    group = None
    for start in range(0, size, args.ep_size):
        candidate = dist.new_group(list(range(start, start + args.ep_size)))
        if start <= rank < start + args.ep_size:
            group = candidate
    device = torch.device('npu', int(os.environ['LOCAL_RANK']))
    module = MegaMoeExperts(hidden_size=args.hidden, intermediate_size=args.intermediate,
        num_experts=args.ep_size * args.home_experts, top_k=1, ep_size=args.ep_size, ep_group=group,
        dispatch_mode=args.backend, replica_slots_per_rank=1,
        replica_transport='shmem_signal_kernel_gradient_bands').to(device=device, dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            values = torch.zeros(128, args.hidden, device=device, dtype=torch.bfloat16)
            report = _collect(module, values, args, group)
    finally:
        module.close()
    reports = [None] * size
    dist.all_gather_object(reports, report if dist.get_rank(group) == 0 else None)
    if rank == 0:
        _write(Path(args.result_dir) / 'contention-measurements.json',
               dict(configuration=vars(args), ep_groups=[row for row in reports if row is not None],
                    completed_global_ranks=list(range(size)), performance_eligible=False,
                    profiler_enabled=False, cost_tables_changed=False,
                    measurement='Caller-stream NPU events after peer-stream joins; CPU wall time saved separately',
                    scope='Isolated full-matrix protocol, not full-network training. Eager FP32 gradient return '
                          'is not the exposed residual of fused gradient bands. No timing samples are trimmed.'))
    dist.barrier()


def main() -> None:
    """Require an activated native payload and a routable SHMEM bootstrap endpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('push', 'pull'), default='push')
    parser.add_argument('--ep-size', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--intermediate', type=int, default=2304)
    parser.add_argument('--home-experts', type=int, default=6)
    parser.add_argument('--fanouts', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--owners', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--placements', choices=('intra', 'inter', 'mixed'), nargs='+',
                        default=['intra', 'inter', 'mixed'])
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--result-dir', required=True)
    args = parser.parse_args()
    if min(args.ep_size, args.hidden, args.intermediate, args.home_experts, args.iterations,
           args.warmup, *args.fanouts, *args.owners) <= 0 or args.ep_size < 2:
        raise ValueError('EP >= 2 and positive shapes, sample counts, fanouts and owners are required')
    if any(len(values) != len(set(values)) for values in (args.fanouts, args.owners, args.placements)):
        raise ValueError('Duplicate fanouts, owner counts or placements are not allowed')
    if not os.environ.get('HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT'):
        host = socket.gethostbyname(os.environ['MASTER_ADDR'])
        port = int(os.environ['MASTER_PORT']) + 1
        if port > 65535:
            raise ValueError('MASTER_PORT must leave room for SHMEM')
        os.environ['HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT'] = f'tcp://{host}:{port}'
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=7200))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
