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
"""Fresh-process MoE F+B+SGD controls using identical historical DSV routing counts."""

import argparse
from datetime import timedelta
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.activation_memory import checkpoint
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.expert_parallel import AllToAllTokenDispatcher
from hyper_parallel.core.expert_parallel.hot_replica import policy
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore._loader import get_multicore_paths
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import deepseek_v41_swiglu
from tests.torch.expert_parallel.hot_replica_checks import route_evidence


def histogram_ids(counts: list[int], tokens: int, top_k: int) -> torch.Tensor:
    """Realize exact per-expert counts with K distinct experts in every token row."""
    if len(counts) < top_k or sum(counts) != tokens * top_k or any(type(n) is not int or not 0 <= n <= tokens
                                                                              for n in counts):
        raise ValueError('Routing histogram must conserve N*K entries and have expert counts in [0,N]')
    flat = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    return flat.reshape(top_k, tokens).T.contiguous()


class NativeGroupedExperts(torch.nn.Module):
    """Owner-only HCCL dispatch and BF16 grouped-MM autograd with packed local weights."""

    def __init__(self, hidden: int, intermediate: int, size: int, group: object,
                 device: torch.device, limit: float) -> None:
        super().__init__()
        self.group = group
        self.mesh = SimpleNamespace(get_group=lambda: group, size=lambda: size)
        self.limit = limit
        self.config = ExpertReplicaConfig(size * 6, size, 0)
        self.gate_up_weight = torch.nn.Parameter(torch.empty(6, hidden, 2 * intermediate,
                                                           device=device, dtype=torch.bfloat16))
        self.down_weight = torch.nn.Parameter(torch.empty(6, intermediate, hidden,
                                                        device=device, dtype=torch.bfloat16))

    def forward(self, values: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        """Use standard grouped-MM backward; no expert copies or replica gradients are created."""
        order = torch.argsort(ids.flatten(), stable=True)
        expanded = values[:, None, :].expand(-1, ids.shape[1], -1).reshape(-1, values.shape[-1])
        counts = torch.bincount(ids.flatten(), minlength=self.config.num_experts)
        routed, local_counts, state = AllToAllTokenDispatcher.dispatch(self, (expanded[order], counts), self.mesh)
        groups = local_counts.to(torch.int64).cumsum(0)
        up = torch.ops.npu.npu_grouped_matmul([routed], [self.gate_up_weight], bias=None,
            group_list=groups, split_item=3, group_type=0, group_list_type=0)[0]
        gate, up = up.float().chunk(2, dim=-1)
        activation = deepseek_v41_swiglu(gate, up, self.limit).to(values.dtype)
        output = torch.ops.npu.npu_grouped_matmul([activation], [self.down_weight], bias=None,
            group_list=groups, split_item=3, group_type=0, group_list_type=0)[0]
        output = AllToAllTokenDispatcher.combine(self, output, self.mesh, state)
        restored = torch.empty_like(output)
        restored[order] = output
        return (restored.reshape(values.shape[0], ids.shape[1], -1) * probabilities.unsqueeze(-1)).sum(1)


def _step(module: torch.nn.Module, optimizer: torch.optim.Optimizer, values: torch.Tensor, ids: torch.Tensor,
          probabilities: torch.Tensor, gradient: torch.Tensor, budget: int, inspect: bool) -> dict:
    optimizer.zero_grad(set_to_none=True)
    values.grad = None
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    plans = []

    def capture(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        if torch.is_grad_enabled():
            plans.append(route_evidence(output, 'cpu'))

    handle = module.register_forward_hook(capture) if inspect and budget else None
    try:
        output = checkpoint(module, values, ids, probabilities)
        output.backward(gradient)
    finally:
        if handle is not None:
            handle.remove()
    evidence = plans[-1] if plans else None
    if inspect and budget and evidence is None:
        raise AssertionError('Audit did not capture a gradient-enabled replica plan')
    optimizer.step()
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    maximum = torch.tensor([elapsed], device=values.device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if inspect:
        finite = (torch.isfinite(output).all() & torch.isfinite(values.grad).all()).int()
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite.cpu()):
            raise AssertionError('MoE benchmark produced nonfinite output or input gradients')
    return dict(local_step_ms=elapsed, rank_max_step_ms=float(maximum.cpu()[0]), plan=evidence)


def run(args: argparse.Namespace) -> None:
    """Keep data, weights, routes and update order identical across fresh controls."""
    size, rank = dist.get_world_size(), dist.get_rank()
    if size % args.ep_size:
        raise ValueError('World size must be divisible by EP size')
    group = None
    for start in range(0, size, args.ep_size):
        candidate = dist.new_group(list(range(start, start + args.ep_size)))
        if start <= rank < start + args.ep_size:
            group = candidate
    ep_rank = dist.get_rank(group)
    device = torch.device('npu', int(os.environ['LOCAL_RANK']))
    if args.backend == 'native':
        if args.budget:
            raise ValueError('The native control requires B0')
        module = NativeGroupedExperts(args.hidden, args.intermediate, args.ep_size, group, device, args.swiglu_limit)
    else:
        module = MegaMoeExperts(hidden_size=args.hidden,
            intermediate_size=args.intermediate, num_experts=args.ep_size * 6, top_k=args.top_k,
            ep_group=group, ep_size=args.ep_size, dispatch_mode=args.backend,
            replica_slots_per_rank=args.budget, replica_min_rows=args.replica_min_rows,
            replica_minimum_load_reduction=args.replica_minimum_load_reduction,
            swiglu_limit=args.swiglu_limit,
            replica_target_load=(int(args.tokens * args.top_k * 1.25) if args.backend == 'push' else None),
            replica_transport='shmem_signal_kernel_gradient_bands',
            **({'initial_capacity_factor': 2.5} if args.backend == 'push' else {})).to(
                device=device, dtype=torch.bfloat16)
    try:
        torch.manual_seed(42 + ep_rank)
        with torch.no_grad():
            module.gate_up_weight.normal_(0, 1 / math.sqrt(args.hidden))
            module.down_weight.normal_(0, 1 / math.sqrt(args.intermediate))
        torch.manual_seed(712 + rank)
        values = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16).requires_grad_()
        gradient = torch.randn_like(values) / args.tokens
        probabilities = torch.full((args.tokens, args.top_k), 1 / args.top_k, device=device)
        if args.route_file:
            raw = Path(args.route_file).read_bytes()
            all_routes = json.loads(raw)
            key = str(rank - ep_rank)
            frames = all_routes['groups'][key]
            ids = [histogram_ids(frame['logical_counts'][ep_rank], args.tokens, args.top_k).to(device)
                   for frame in frames]
            route_hash = hashlib.sha256(raw).hexdigest()
        else:
            counts = [args.tokens * args.top_k // (args.ep_size * 6)] * (args.ep_size * 6)
            counts[0] += args.tokens * args.top_k - sum(counts)
            ids = [histogram_ids(counts, args.tokens, args.top_k).to(device)]
            route_hash = hashlib.sha256(json.dumps(counts).encode()).hexdigest()
        optimizer = torch.optim.SGD(module.parameters(), lr=.0001)
        cost_enabled = bool(getattr(module, '_cost_model_enabled', False))
        cost_model = policy._runtime_cost_model(cost_enabled, args.backend, args.hidden, args.intermediate, args.ep_size,
            'shmem_signal_kernel_gradient_bands', args.ep_size * 6, args.top_k, device) if args.budget else None
        if args.budget and args.ep_size == 64 and cost_enabled and cost_model is None:
            raise RuntimeError(f'EP64 cost control has no matching calibration for {torch.npu.get_device_name(device)}')
        cold = _step(module, optimizer, values, ids[0], probabilities, gradient, args.budget, True)
        for iteration in range(args.warmup):
            _step(module, optimizer, values, ids[iteration % len(ids)], probabilities, gradient, args.budget, False)
        torch.npu.reset_peak_memory_stats()
        samples = [_step(module, optimizer, values, ids[iteration % len(ids)], probabilities, gradient,
                         args.budget, False) for iteration in range(args.iterations)]
        audits = [dict(route_index=index, **_step(module, optimizer, values, route, probabilities, gradient,
                                                args.budget, True))
                  for index, route in enumerate(ids[:args.iterations])]
        _, adapter = get_multicore_paths()
        payload = dict(configuration=vars(args), global_rank=rank, ep_rank=ep_rank,
            ep_members=dist.get_process_group_ranks(group), hostname=socket.gethostname(), pid=os.getpid(),
            soc_name=torch.npu.get_device_name(device), torch_npu=torch_npu.__version__,
            source_root=str(Path(__file__).resolve().parents[3]),
            native_adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
            cost_model_matched=cost_model is not None,
            cost_model_enabled=cost_enabled,
            cost_fanout_active=bool(cost_model and cost_model.weight_fanout_ms),
            route_sha256=route_hash, cold=cold, audit=audits[0], route_audits=audits, samples=samples,
            peak_allocated_bytes=torch.npu.max_memory_allocated(),
            peak_reserved_bytes=torch.npu.max_memory_reserved(), completed=True,
            checkpoint_api='hyper_non_reentrant',
            scope='One MoE layer F+checkpoint recompute+B+SGD with replayed count histograms. '
                  'No attention, Engram, expert FSDP or full-network optimizer. No CANN profiler.')
        directory = Path(args.result_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'rank{rank:05d}.json').write_text(json.dumps(payload, indent=2) + '\n')
    finally:
        if isinstance(module, MegaMoeExperts):
            module.close()
    dist.barrier()


def main() -> None:
    """Run one isolated benchmark configuration per torchrun invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('native', 'push', 'pull'), required=True)
    parser.add_argument('--ep-size', type=int, default=64)
    parser.add_argument('--budget', type=int, default=0)
    parser.add_argument('--replica-min-rows', type=int, default=1024)
    parser.add_argument('--replica-minimum-load-reduction', type=float)
    parser.add_argument('--swiglu-limit', type=float, default=10.0)
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--intermediate', type=int, default=2304)
    parser.add_argument('--tokens', type=int, default=4096)
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--route-file')
    parser.add_argument('--result-dir', required=True)
    args = parser.parse_args()
    if min(args.ep_size, args.hidden, args.intermediate, args.tokens, args.top_k, args.warmup, args.iterations) < 1:
        raise ValueError('Positive topology, shape and iteration counts are required')
    if not 0 <= args.budget <= 6 or args.top_k > args.ep_size * 6:
        raise ValueError('Budget must be in [0,6] and top_k must fit the experts')
    handler = logging.StreamHandler()
    policy._LOGGER.addHandler(handler)
    policy._LOGGER.setLevel(logging.INFO)
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=7200))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
