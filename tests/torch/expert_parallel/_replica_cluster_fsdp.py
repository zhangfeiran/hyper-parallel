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
"""Cross-node eight-layer EP64/FSDP lifecycle and metadata feature gates."""

from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import traceback
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

import hyper_parallel
from hyper_parallel import fully_shard, init_device_mesh, SkipDTensorDispatch
from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts, is_recomputing
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.expert_parallel.hot_replica import _metadata
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.multicore import MegaMoeExperts
from tests.torch.expert_parallel.replica_cluster_observer import Ledger
from tests.torch.expert_parallel.replica_cluster_metadata import MetadataLedger, device_metadata_gate


def local(value: torch.Tensor) -> torch.Tensor:
    """Read the actual local shard without gathering state."""
    return value.to_local() if isinstance(value, DTensor) else value


def digest(value: torch.Tensor) -> str:
    """Keep only a digest; no parameter or optimizer matrices are written to disk."""
    value = local(value).detach().contiguous().cpu()
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError('Nonfinite FSDP tensor')
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class ExpertLayer(nn.Module):
    """Own real FSDP parameters and call a parameterless MegaMoe executor."""

    def __init__(self, index: int, args: argparse.Namespace, ep_group: Any, ep_rank: int, ledger: Ledger) -> None:
        super().__init__()
        self.index, self.ledger = index, ledger
        torch.manual_seed(715 + index + ep_rank * 17)
        self.gate_up = nn.Parameter(torch.randn(6, args.hidden, 2 * args.intermediate, device='npu',
                                                dtype=torch.bfloat16) * .02)
        self.down = nn.Parameter(torch.randn(6, args.intermediate, args.hidden, device='npu',
                                             dtype=torch.bfloat16) * .02)
        options = {'initial_capacity_factor': 2.0} if args.mode == 'push' else {}
        self.executor = MegaMoeExperts(hidden_size=args.hidden, intermediate_size=args.intermediate,
                                       num_experts=args.ep_size * 6, top_k=6, ep_size=args.ep_size, ep_group=ep_group,
                                       create_parameters=False, dispatch_mode=args.mode, swiglu_limit=10.,
                                       replica_slots_per_rank=1, replica_min_rows=0,
                                       replica_target_load=args.tokens * 6,
                                       replica_minimum_load_reduction=0.,
                                       replica_transport='shmem_signal_kernel_gradient_bands', **options)

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor,
                probabilities: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        weights = local(self.gate_up), local(self.down)
        self.ledger.register(weights, self.index)
        previous = self.ledger.active_layer
        self.ledger.active_layer = self.index
        phase = 'recompute' if is_recomputing() else 'forward'
        self.ledger.rows.append({**self.ledger.context, 'kind': 'layer_call', 'phase': phase,
                                 'layer': self.index, 'parameter_versions': [value._version for value in weights],
                                 'materialization_ids': [id(getattr(value, '_hp_materialization_token', None))
                                                         for value in weights]})
        try:
            return hidden + self.executor(hidden, ids, probabilities, tokens_per_expert=counts,
                                           expert_weights=weights)
        finally:
            self.ledger.active_layer = previous


class Stack(nn.Module):
    """Use genuine checkpointed FSDP layer calls with residuals to avoid vanishing test signals."""

    def __init__(self, layers: list[ExpertLayer], early_stop: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.early_stop = early_stop

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor,
                probabilities: list[torch.Tensor], counts: torch.Tensor) -> torch.Tensor:
        for layer, probs in zip(self.layers, probabilities):
            with set_checkpoint_early_stop(self.early_stop):
                hidden = checkpoint(layer, hidden, ids, probs, counts, use_reentrant=False,
                                    context_fn=create_recompute_contexts)
        return hidden


def build(args: argparse.Namespace, mesh: Any, ep_rank: int, ledger: Ledger) -> Stack:
    layers = [ExpertLayer(index, args, mesh['ep'].get_group(), ep_rank, ledger) for index in range(8)]
    MegaMoeExperts.share_execution_resources([layer.executor for layer in layers])
    model = Stack(layers, early_stop=args.feature == 'metadata')
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    options = {'mesh': mesh['dp'], 'reshard_after_forward': args.policy == 'reshard',
               'mp_policy': policy, 'shard_placement_fn': lambda parameter: Shard(1)}
    for layer in layers:
        fully_shard(layer, **options)
    fully_shard(model, **options)
    return model


def execute(args: argparse.Namespace, mesh: Any, ep_rank: int, dp_rank: int, enabled: bool) -> dict:
    torch.npu.reset_peak_memory_stats()
    ledger = Ledger(enabled if args.feature == 'residency' else True)
    os.environ['HYPER_PARALLEL_REPLICA_METADATA_REUSE'] = str(int(enabled and args.feature == 'metadata'))
    _metadata._metadata_reuse_enabled.cache_clear()
    metadata = MetadataLedger()
    model = build(args, mesh, ep_rank, ledger)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.001)
    updates = []
    try:
        with ledger.observe(), metadata.observe():
            for update in range(2):
                optimizer.zero_grad(set_to_none=True)
                hashes = []
                for micro in range(args.gas):
                    ledger.context = {'update': update, 'micro': micro}
                    last = micro == args.gas - 1
                    model.set_requires_gradient_sync(last)
                    model.set_reshard_after_backward(args.policy == 'reshard' or last)
                    pattern = ('hot', 'rotate', 'balanced', 'ragged')[micro % 4]
                    tokens = (0 if pattern == 'ragged' and ep_rank % 4 == 1 else args.tokens)
                    if update == 1 and micro == args.gas - 1:
                        tokens = 0
                    owner = (update * args.gas + micro) % args.ep_size if pattern == 'rotate' else 0
                    torch.manual_seed(230 + dist.get_rank() * 31 + micro * 19 + update * 7)
                    hidden = (torch.randn(tokens, args.hidden, device='npu', dtype=torch.bfloat16) * .2)
                    hidden.requires_grad_()
                    upstream = torch.randn_like(hidden) / max(args.tokens * args.gas, 1)
                    if pattern == 'balanced':
                        ids = torch.arange(tokens * 6, device='npu', dtype=torch.int32).reshape(tokens, 6)
                        ids = ids % (args.ep_size * 6)
                    else:
                        ids = torch.arange(6, device='npu', dtype=torch.int32) + 6 * owner
                        ids = ids.expand(tokens, 6).contiguous()
                    ids = ids.roll(update + micro, dims=0)
                    counts = torch.bincount(ids.long().flatten(), minlength=args.ep_size * 6).int()
                    probabilities = [torch.full((tokens, 6), 1/6, device='npu', requires_grad=True)
                                     for _ in range(8)]
                    if args.feature == 'metadata':
                        consumer = torch.npu.Stream()
                        consumer.wait_stream(torch.npu.current_stream())
                        with torch.npu.stream(consumer):
                            output = model(hidden, ids, probabilities, counts)
                            output.backward(upstream)
                        torch.npu.current_stream().wait_stream(consumer)
                        torch.npu.synchronize()
                    else:
                        output = model(hidden, ids, probabilities, counts)
                        output.backward(upstream)
                    hashes.append({'output': digest(output), 'dx': digest(hidden.grad),
                                   'router': [digest(value.grad) for value in probabilities]})
                gradients = [digest(parameter.grad) for parameter in parameters]
                versions_before = [local(parameter)._version for parameter in parameters]
                with SkipDTensorDispatch(no_skip={torch.zeros_like}):
                    optimizer.step()
                ledger.rows.append({**ledger.context, 'kind': 'optimizer_update', 'layer': 'all',
                                    'sharded_versions_before': versions_before,
                                    'sharded_versions_after': [local(parameter)._version for parameter in parameters]})
                updates.append({'microbatches': hashes, 'gradients': gradients,
                                'parameters': [digest(parameter) for parameter in parameters],
                                'optimizer': [{key: digest(value) for key, value in optimizer.state[p].items()
                                               if isinstance(value, torch.Tensor)} for p in parameters]})
        return {'updates': updates, 'ledger': ledger.rows, 'metadata': metadata.rows,
                'weight_put_bytes': ledger.put_bytes,
                'sharded_parameter_shapes': [list(local(parameter).shape) for parameter in parameters],
                'peak_allocated_bytes': torch.npu.max_memory_allocated(),
                'peak_reserved_bytes': torch.npu.max_memory_reserved()}
    except BaseException:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f'error-rank{dist.get_rank()}-enabled{enabled}.json').write_text(
            json.dumps({'traceback': traceback.format_exc(), 'ledger': ledger.rows}, indent=2) + '\n')
        raise
    finally:
        for layer in model.layers:
            layer.executor.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--bootstrap-base', type=int, default=47100)
    parser.add_argument('--ep-size', type=int, default=64)
    parser.add_argument('--feature', choices=('residency', 'metadata'), default='residency')
    parser.add_argument('--mode', choices=('push', 'pull'), default='push')
    parser.add_argument('--gas', type=int, default=4)
    parser.add_argument('--policy', choices=('reshard', 'retain'), default='reshard')
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--intermediate', type=int, default=256)
    parser.add_argument('--tokens', type=int, default=512)
    args = parser.parse_args()
    if Path(hyper_parallel.__file__).resolve().parents[1] != args.source.resolve():
        raise RuntimeError('Unexpected editable source')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=180))
    try:
        if dist.get_world_size() != 2 * args.ep_size or args.ep_size != 64 or args.gas < 4:
            raise ValueError('This cross-node gate requires EP64 x DP2 and GAS at least four')
        addresses = [None] * dist.get_world_size()
        hosts = os.environ.get('VC_WORKER_HOSTS', '').split(',')
        own_host = (hosts[int(os.environ['NODE_RANK'])] if len(hosts) == 8 else socket.gethostname())
        address = socket.gethostbyname(own_host)
        if address.startswith('127.'):
            raise ValueError('Cross-node FSDP requires a reachable non-loopback worker address')
        dist.all_gather_object(addresses, address)
        mesh = init_device_mesh('npu', (2, args.ep_size), mesh_dim_names=('dp', 'ep'))
        dp_rank, ep_rank = mesh.get_coordinate()
        probe = (device_metadata_gate(torch.device('npu', int(os.environ['LOCAL_RANK'])))
                 if args.feature == 'metadata' else None)
        results = []
        for variant, enabled in enumerate((False, True)):
            os.environ['HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT'] = (
                f'tcp://{addresses[dp_rank * args.ep_size]}:{args.bootstrap_base + variant * 4 + dp_rank}')
            results.append(execute(args, mesh, ep_rank, dp_rank, enabled))
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f'rank{dist.get_rank()}-variant{variant}.json').write_text(
                json.dumps(results[-1], indent=2) + '\n')
            print(json.dumps({'event': 'variant_complete', 'rank': dist.get_rank(),
                              'enabled': enabled, 'weight_put_bytes': results[-1]['weight_put_bytes']}), flush=True)
            gc.collect()
            torch.npu.empty_cache()
            dist.barrier()
        identical = results[0]['updates'] == results[1]['updates']
        accepted = torch.tensor([int(identical)], device='npu')
        dist.all_reduce(accepted, op=dist.ReduceOp.MIN)
        if not bool(accepted.cpu()[0]):
            raise RuntimeError('Real FSDP off/on training hashes differ')
        cross = [0, 0]
        for row in results[1]['ledger']:
            if row['kind'] != 'prepare':
                continue
            for fact in row['view']['facts']:
                if fact['status'] == 'unknown':
                    continue
                owner, target = (dp_rank * args.ep_size + fact[key] for key in ('owner', 'target'))
                if addresses[owner] != addresses[target]:
                    cross[0] += 1
                    cross[1] += int(fact['status'] == 'valid')
        coverage = torch.tensor(cross, device='npu', dtype=torch.int64)
        dist.all_reduce(coverage)
        if not int(coverage[0].cpu()):
            raise RuntimeError('The lifecycle gate did not exercise cross-node replicas')
        if args.feature == 'residency' and args.policy == 'retain' and not int(coverage[1].cpu()):
            raise RuntimeError('Retain diagnostics did not exercise a cross-node resident hit')
        args.output.mkdir(parents=True, exist_ok=True)
        record = {'configuration': vars(args), 'rank': dist.get_rank(), 'ep_rank': ep_rank, 'dp_rank': dp_rank,
                  'ep_members': dist.get_process_group_ranks(mesh['ep'].get_group()),
                  'dp_members': dist.get_process_group_ranks(mesh['dp'].get_group()),
                  'metadata_device_probe': probe, 'cross_node_prepares_and_hits': coverage.cpu().tolist(),
                  'all_tensors_identical': identical, 'variants': results}
        (args.output / f'rank{dist.get_rank()}.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
