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
"""Measure complete replica matrices on actual intra-node and inter-node peers."""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica.planner import _materialize
from hyper_parallel.core.expert_parallel.hot_replica.routing import ReplicaRoute
from hyper_parallel.core.expert_parallel.hot_replica.transport import prefetch_weights, return_gradients
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore._loader import get_multicore_paths


def _link_pairs(nodes: tuple[int, ...], limit: int) -> list[tuple[str, int, int]]:
    """Sample one representative owner per node without assuming a rank stride."""
    members = {node: [rank for rank, label in enumerate(nodes) if label == node] for node in dict.fromkeys(nodes)}
    labels = list(members)
    pairs = []
    for kind in ('intra_node', 'inter_node'):
        selected = []
        for index, node in enumerate(labels):
            owner = members[node][0]
            if kind == 'intra_node' and len(members[node]) > 1:
                selected.append((kind, owner, members[node][len(members[node]) // 2]))
            if kind == 'inter_node' and len(labels) > 1:
                selected.append((kind, owner, members[labels[(index + 1) % len(labels)]][0]))
        pairs.extend(selected[:limit])
    return pairs


def _pair_plan(size: int, home: int, owner: int, target: int) -> Any:
    """Force exactly one complete expert copy with independently validated source counts."""
    if not 0 <= owner < size or not 0 <= target < size or owner == target:
        raise ValueError('Replica calibration needs distinct owner and target ranks in the EP group')
    config = ExpertReplicaConfig(size * home, size, 1)
    expert = owner * home
    counts = [[0] * config.num_experts for _ in range(size)]
    counts[owner][expert] = 1
    copies = [{logical: int(logical == expert) for logical in range(rank * home, (rank + 1) * home)}
              for rank in range(size)]
    copies[owner][expert] = 0
    copies[target][expert] = 1
    plan = _materialize(tuple(map(tuple, counts)), copies, config)
    plan.validate()
    if len(plan.transfers) != 1:
        raise ValueError('Replica calibration must contain exactly one transfer')
    return plan


def _maximum(milliseconds: float, device: torch.device, group: object) -> float:
    """Reduce host wall time after synchronization, outside the measured interval."""
    value = torch.tensor([milliseconds], device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
    return float(value.cpu()[0])


def _measure_pair(module: MegaMoeExperts, values: torch.Tensor, kind: str,
                  owner: int, target: int, args: argparse.Namespace, group: object) -> dict[str, Any]:
    """Time the production provider, excluding checks and gradient-buffer clearing."""
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    resources = module._get_execution_resources(values)
    resources.workspace.ensure(resources.spec, values.dtype, values.device)
    provider = resources.workspace.replica_provider
    weights = module.gate_up_weight, module.down_weight
    for weight in weights:
        weight.fill_(rank + 1)
    plan = _pair_plan(size, args.home_experts, owner, target)
    route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), rank, group, provider)
    home_gradients = tuple(torch.zeros_like(weight, dtype=torch.float32) for weight in weights)
    samples = []
    for iteration in range(args.warmup + args.iterations):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        dist.barrier(group=group)
        torch.npu.synchronize()
        started = time.perf_counter()
        begin.record()
        with prefetch_weights(weights, route, provider=provider) as pool:
            pass
        end.record()
        torch.npu.synchronize()
        weight_wall_ms = _maximum((time.perf_counter() - started) * 1000, values.device, group)
        weight_ms = _maximum(begin.elapsed_time(end), values.device, group)
        if rank == target and iteration == args.warmup:
            for guest in pool.weights:
                torch.testing.assert_close(guest[0], torch.full_like(guest[0], owner + 1), rtol=0, atol=0)
        for gradient in home_gradients:
            gradient.zero_()
        with prefetch_weights(weights, route, provider=provider, backward=True) as pool:
            if rank == target:
                for guest in pool.gradients:
                    guest[0].fill_(1)
            dist.barrier(group=group)
            torch.npu.synchronize()
            started = time.perf_counter()
            begin.record()
            returned = return_gradients(home_gradients, route, pool.gradients, provider, consume=True)
            end.record()
            torch.npu.synchronize()
            gradient_wall_ms = _maximum((time.perf_counter() - started) * 1000, values.device, group)
            gradient_ms = _maximum(begin.elapsed_time(end), values.device, group)
        if rank == owner and iteration == args.warmup:
            for gradient in returned:
                torch.testing.assert_close(gradient[0], torch.ones_like(gradient[0]), rtol=0, atol=0)
        if iteration >= args.warmup:
            samples.append(dict(weight_ms=weight_ms, physical_gradient_ms=gradient_ms,
                                weight_wall_ms=weight_wall_ms, gradient_wall_ms=gradient_wall_ms))
    weight_bytes = sum(weight[0].numel() * weight.element_size() for weight in weights)
    return dict(kind=kind, owner=owner, target=target, samples=samples,
                bf16_weight_bytes=weight_bytes, fp32_gradient_bytes=weight_bytes * 2)


def _publish(args: argparse.Namespace, reports: list[dict[str, Any]]) -> None:
    """Emit measured link patches and matching candidate files without changing installed tables."""
    output = Path(args.result_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = [pair for report in reports for pair in report['pairs']]
    if any(report['soc_name'] != reports[0]['soc_name'] for report in reports):
        raise ValueError('Replica link calibration requires the same device model across EP groups')
    patch = {}
    summaries = {}
    for kind in ('intra_node', 'inter_node'):
        pairs = [pair for pair in records if pair['kind'] == kind]
        weights = [sample['weight_ms'] for pair in pairs for sample in pair['samples']]
        gradients = [sample['physical_gradient_ms'] for pair in pairs for sample in pair['samples']]
        if weights:
            patch[f'{kind}_weight_ms'] = statistics.median(weights)
            summaries[kind] = dict(pairs=len(pairs), samples=len(weights), weight_ms=statistics.median(weights),
                                   physical_gradient_ms=statistics.median(gradients))
    payload = dict(configuration=vars(args), reports=reports, communication_patch=patch, summary=summaries,
        performance_eligible=False, gradient_patch_applied=False,
        measurement='NPU events on the caller stream after production peer-stream joins; CPU wall time is separate',
        scope='Isolated complete-matrix transfer and protocol latency. Eager FP32 gradient return is physical '
              'communication, not the exposed residual of fused gradient bands; it is not installed as gradient_ms.')
    (output / 'link-measurements.json').write_text(json.dumps(payload, indent=2)+'\n')
    source = Path(__file__).resolve().parents[3] / 'hyper_parallel/core/expert_parallel/hot_replica/calibrations'
    for path in source.glob('*.json'):
        candidate = json.loads(path.read_text())
        model, selector = candidate['model'], candidate['selector']
        if (model['backend'], model['hidden_size'], model['intermediate_size'], model['ep_size'], model['transport'],
            selector['num_experts'], selector['top_k'], selector['soc_name']) != (
                args.backend, args.hidden, args.intermediate, args.ep_size, args.replica_transport,
                args.ep_size * args.home_experts, args.reference_top_k, reports[0]['soc_name']):
            continue
        model.update(patch)
        ExpertReplicaCostModel(**model)
        candidate['provenance']['node_link_measurements'] = summaries
        candidate['provenance']['node_link_gradient_patch_applied'] = False
        candidate['provenance']['node_link_scope'] = payload['scope']
        (output / path.name).write_text(json.dumps(candidate, indent=2)+'\n')


def run(args: argparse.Namespace) -> None:
    """Use disjoint target-size EP groups and aggregate their representative links."""
    size, rank = dist.get_world_size(), dist.get_rank()
    if size % args.ep_size:
        raise ValueError('World size must be divisible by the calibrated EP size')
    group = None
    for start in range(0, size, args.ep_size):
        candidate = dist.new_group(list(range(start, start + args.ep_size)))
        if start <= rank < start + args.ep_size:
            group = candidate
    device = torch.device('npu', int(os.environ['LOCAL_RANK']))
    module = MegaMoeExperts(hidden_size=args.hidden, intermediate_size=args.intermediate,
        num_experts=args.ep_size * args.home_experts, top_k=1, ep_size=args.ep_size, ep_group=group,
        dispatch_mode=args.backend, replica_slots_per_rank=1, replica_transport=args.replica_transport).to(
            device=device, dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            values = torch.zeros(128, args.hidden, device=device, dtype=torch.bfloat16)
            resources = module._get_execution_resources(values)
            pairs = [_measure_pair(module, values, kind, owner, target, args, group)
                     for kind, owner, target in _link_pairs(resources.rank_nodes, args.pairs_per_kind)]
            vendor, adapter = get_multicore_paths()
            report = dict(global_leader=rank - dist.get_rank(group), rank_nodes=resources.rank_nodes,
                          global_ep_members=dist.get_process_group_ranks(group), pairs=pairs,
                          soc_name=torch.npu.get_device_name(device), torch_npu=torch_npu.__version__,
                          native_adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                          native_vendor=str(vendor)) if dist.get_rank(group) == 0 else None
    finally:
        module.close()
    reports = [None] * size
    dist.all_gather_object(reports, report)
    if rank == 0:
        _publish(args, [item for item in reports if item is not None])
    dist.barrier()


def main() -> None:
    """Run with an activated native payload and a routable shared bootstrap endpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('push', 'pull'), default='push')
    parser.add_argument('--replica-transport', choices=('p2p', 'shmem_signal_kernel_gradient_bands'),
                        default='shmem_signal_kernel_gradient_bands')
    parser.add_argument('--ep-size', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--intermediate', type=int, default=2304)
    parser.add_argument('--home-experts', type=int, default=6)
    parser.add_argument('--reference-top-k', type=int, default=6)
    parser.add_argument('--pairs-per-kind', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--result-dir', required=True)
    args = parser.parse_args()
    if min(args.ep_size, args.hidden, args.intermediate, args.home_experts, args.pairs_per_kind,
           args.iterations, args.warmup, args.reference_top_k) <= 0 or args.ep_size < 2:
        raise ValueError('Replica link calibration requires EP>=2 and positive shape/sample counts')
    if not os.environ.get('HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT'):
        host = socket.gethostbyname(os.environ['MASTER_ADDR'])
        port = int(os.environ['MASTER_PORT']) + 1
        if port > 65535:
            raise ValueError('MASTER_PORT must leave room for the SHMEM bootstrap port')
        os.environ['HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT'] = f'tcp://{host}:{port}'
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=300))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
