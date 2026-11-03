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
"""Collect shape-owned F/R/B samples and verify execution-signed empirical models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hyper_parallel.core.activation_memory import checkpoint
from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts, get_recompute_state
from hyper_parallel.core.expert_parallel.hot_replica import policy, routing
from hyper_parallel.core.expert_parallel.hot_replica.autocalibration import calibration_ids, load_manifest
from hyper_parallel.core.expert_parallel.hot_replica.calibration_catalog import runtime_contract
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica.planner import _home_layout, _materialize
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe import module as mega
from hyper_parallel.core.multicore.modules.mega_moe.inactive_tasks import UNMAPPED_GUEST_SUFFIX_SCHEDULE


def _checkpoint_operation(operation: Callable, *arguments: torch.Tensor, contract: str) -> torch.Tensor:
    """Match the model's checkpoint F/R grad modes while sharing Hyper invocation state."""
    if contract == 'nonreentrant':
        return checkpoint(operation, *arguments, early_stop=False)
    if contract != 'reentrant':
        raise ValueError(f'Unsupported checkpoint contract: {contract}')
    forward, replay = create_recompute_contexts()

    def wrapped(*values: torch.Tensor) -> torch.Tensor:
        with replay if torch.is_grad_enabled() else forward:
            return operation(*values)

    return torch_checkpoint(wrapped, *arguments, use_reentrant=True, preserve_rng_state=True)


def _profiles(directory: Path | None) -> dict:
    result = {}
    if directory is not None:
        for path in sorted(directory.glob('*-b*.json')):
            payload = json.loads(path.read_text())
            model = ExpertReplicaCostModel(**payload['model'])
            selector = payload['selector']
            key = (model.backend, model.hidden_size, model.intermediate_size, model.ep_size, model.transport,
                   selector['num_experts'], selector['top_k'], selector['soc_name'], selector['execution_signature'])
            result[key] = model
    return result


def _capture(manifest: dict[str, Any], mode: str, budget: int, group: object,
             device: torch.device, output: Path) -> None:
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    global_rank = dist.get_rank()
    module = MegaMoeExperts(hidden_size=manifest['hidden_size'], intermediate_size=manifest['intermediate_size'],
        num_experts=manifest['num_experts'], top_k=manifest['top_k'], ep_group=group, dispatch_mode=mode,
        replica_slots_per_rank=budget, swiglu_limit=manifest['swiglu_limit'],
        replica_minimum_load_reduction=None if os.environ['HYPER_PARALLEL_REPLICA_COST_MODEL'] == '1' else 0.,
        replica_transport='shmem_signal_kernel_gradient_bands').to(device).bfloat16()
    torch.manual_seed(42 + rank)
    with torch.no_grad():
        module.gate_up_weight.normal_(0, 1 / math.sqrt(manifest['hidden_size']))
        module.down_weight.normal_(0, 1 / math.sqrt(manifest['intermediate_size']))
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-4)
    initial = torch.empty(manifest['local_num_tokens'], manifest['hidden_size'], device=device, dtype=torch.bfloat16)
    resources = module._get_execution_resources(initial)
    nodes = resources.rank_nodes
    del initial
    active: dict[str, Any] = {}
    events = []
    records = []
    original_plan, original_cost = routing._plan_replica_route, mega._runtime_cost_model

    def plan_hook(*arguments: Any, **options: Any) -> Any:
        begin = time.perf_counter()
        result = original_plan(*arguments, **options)
        active['planner_cpu_ms'] = active.get('planner_cpu_ms', 0.) + (time.perf_counter() - begin) * 1000
        active['plan'] = result[0]
        return result

    def cost_hook(*arguments: Any, **options: Any) -> Any:
        model = original_cost(*arguments, **options)
        expected = options.get('execution_signature')
        key = (*arguments[1:8], policy._soc_name(device), expected)
        candidate = policy._calibrations().get(key)
        active['cost_model_matched'] = model is not None and model == candidate
        return model

    try:
        with (patch.object(routing, '_plan_replica_route', plan_hook),
              patch.object(mega, '_runtime_cost_model', cost_hook)):
            for frame in range(manifest['frames']):
                tokens = manifest['local_num_tokens'] // (2 if 4 <= frame % 12 < 8 and rank % 2 == 0 else 1)
                if frame % 4 == 3 and rank % 4 == 0:
                    tokens = 0
                frame_tokens = [None] * size
                dist.all_gather_object(frame_tokens, tokens, group=group)
                maximum_tokens = max(frame_tokens)
                ids = calibration_ids(tokens, manifest, frame, nodes).to(device)
                local_counts = torch.bincount(ids.flatten().long(), minlength=manifest['num_experts']).cpu().tolist()
                for sample in range(-manifest['warmup'], manifest['samples']):
                    torch.manual_seed(113 + frame * 11 + sample + rank)
                    values = torch.randn(tokens, manifest['hidden_size'], device=device,
                                         dtype=torch.bfloat16, requires_grad=True)
                    probabilities = torch.full(ids.shape, 1 / manifest['top_k'], device=device, requires_grad=True)
                    gradient = torch.randn_like(values) / max(1, tokens)
                    optimizer.zero_grad(set_to_none=True)
                    active.clear()
                    events.clear()

                    def operation(hidden: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
                        state = get_recompute_state()
                        phase = 'recompute' if state is not None and state.is_recomputing else 'forward'
                        first, last = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        first.record()
                        result = module(hidden, ids, weights)
                        last.record()
                        events.append((phase, first, last))
                        active['signature'] = mega._critical_execution_signature(
                            module, module._get_execution_resources(hidden),
                            execution_tokens=maximum_tokens)
                        return result

                    dist.barrier(group=group)
                    torch.npu.synchronize()
                    started = time.perf_counter()
                    result = _checkpoint_operation(operation, values, probabilities,
                                                   contract=manifest['checkpoint_contract'])
                    backward_first = torch.npu.Event(enable_timing=True)
                    backward_last = torch.npu.Event(enable_timing=True)
                    backward_first.record()
                    result.backward(gradient)
                    backward_last.record()
                    optimizer.step()
                    torch.npu.synchronize()
                    wall_ms = (time.perf_counter() - started) * 1000
                    if sample < 0:
                        continue
                    phases = {name: sum(first.elapsed_time(last) for phase, first, last in events if phase == name)
                              for name in ('forward', 'recompute', 'backward')}
                    phases['backward'] = backward_first.elapsed_time(backward_last) - phases['recompute']
                    if phases['backward'] <= 0:
                        raise RuntimeError('Invalid nonpositive backward duration after exact replay subtraction')
                    finite = all(bool(torch.isfinite(value).all()) for value in
                                 (result, values.grad, probabilities.grad, *[p.grad for p in module.parameters()]))
                    local = {'phase_ms': phases, 'wall_ms': wall_ms, 'counts': local_counts, 'finite': finite,
                             'signature': active['signature'], 'matched': active.get('cost_model_matched', False),
                             'planner_cpu_ms': active.get('planner_cpu_ms', 0.)}
                    gathered = [None] * size
                    dist.all_gather_object(gathered, local, group=group)
                    if not all(row['finite'] for row in gathered):
                        raise RuntimeError(f'Nonfinite calibration tensors: {mode}/B{budget}/frame{frame}')
                    if len({row['signature'] for row in gathered}) != 1:
                        raise RuntimeError('EP ranks observed incompatible execution signatures')
                    if rank == 0:
                        counts = tuple(tuple(row['counts']) for row in gathered)
                        config = ExpertReplicaConfig(manifest['num_experts'], size, budget)
                        plan = active.get('plan')
                        if plan is None:
                            _, _, copies = _home_layout(counts, config)
                            plan = _materialize(counts, copies, config, rank_nodes=nodes)
                        plan.validate()
                        records.append({'group_leader': global_rank, 'frame': frame, 'sample': sample,
                            'execution_signature': active['signature'], 'maximum_source_tokens': maximum_tokens,
                            'config': asdict(plan.config), 'source_counts': plan.logical_counts,
                            'slots': plan.slot_to_logical, 'dispatch': plan.dispatch_counts, 'rank_nodes': nodes,
                            'rank_phase_ms': [row['phase_ms'] for row in gathered],
                            'wall_rank_max_ms': max(row['wall_ms'] for row in gathered),
                            'planner_cpu_ms': max(row['planner_cpu_ms'] for row in gathered),
                            'finite': True, 'profile_on': False,
                            'cost_model_matched': all(row['matched'] for row in gathered),
                            'balanced_owner_bypass': len(set(sum(sum(source[start:start + config.home_experts])
                                for source in counts) for start in range(0, config.num_experts,
                                                                         config.home_experts))) == 1})
    finally:
        module.close()
    if rank == 0:
        payload = {'manifest': manifest, 'backend': mode, 'budget': budget, 'rank_nodes': nodes,
            'world_size': dist.get_world_size(),
            'native_payload_sha256': mega._critical_native_fingerprint(),
            'ep_members': dist.get_process_group_ranks(group), 'soc_name': torch.npu.get_device_name(device),
            'schedule': UNMAPPED_GUEST_SUFFIX_SCHEDULE, 'execution_signature': active['signature'],
            'runtime_contract': runtime_contract(), 'records': records, 'completed': True}
        (output / f'{mode}-b{budget}-group{global_rank}.json').write_text(json.dumps(payload) + '\n')


def main() -> None:
    """Run the same bounded manifest matrix for collection and private model-in-loop verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-directory', type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if os.environ.get('HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY') != '0':
        raise ValueError('Automatic calibration requires force_normal_miss residency')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    if not os.environ.get('HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT'):
        os.environ['HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT'] = (
            f"tcp://{socket.gethostbyname(os.environ['MASTER_ADDR'])}:{int(os.environ['MASTER_PORT']) + 1}")
    dist.init_process_group('hccl', timeout=timedelta(seconds=7200))
    try:
        size, rank, ep = dist.get_world_size(), dist.get_rank(), manifest['ep_size']
        if size % ep:
            raise ValueError('World size must be divisible by the manifest EP size')
        group = None
        for first in range(0, size, ep):
            candidate = dist.new_group(list(range(first, first + ep)))
            if first <= rank < first + ep:
                group = candidate
        args.output.mkdir(parents=True, exist_ok=True)
        models = _profiles(args.candidate_directory)
        os.environ['HYPER_PARALLEL_REPLICA_COST_MODEL'] = '1' if args.candidate_directory else '0'
        with patch.object(policy, '_calibrations', new=lambda: models):
            for function in (policy._calibration_available, policy._critical_path_available,
                             policy._critical_path_identity, policy._resolve_cost_model):
                function.cache_clear()
            for mode in manifest['dispatch_modes']:
                for budget in [0] + manifest['replica_caps']:
                    device = torch.device('npu', int(os.environ['LOCAL_RANK']))
                    _capture(manifest, mode, budget, group, device, args.output)
        (args.output / f'status-rank{rank}.json').write_text(json.dumps({'rank': rank, 'completed': True}) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
