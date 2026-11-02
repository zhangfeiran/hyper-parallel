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
"""Collect device phase events and verify live critical-path admission and training.

The opt-in correctness profile is explicitly synthetic. It exercises automatic
profile selection, not production calibration or accepted performance. A supplied
profile directory instead captures real resolved decisions and device events with
the deployed tables, without changing any installed calibration files.
"""

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # Register the selected Ascend backend in this worker.

from hyper_parallel.core.activation_memory import checkpoint
from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state
from hyper_parallel.core.expert_parallel.hot_replica import policy, routing
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica.critical_path import DeviceCriticalPathCalibration
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe import function as executor
from hyper_parallel.core.multicore.modules.mega_moe import module as mega


def _clear_profiles() -> None:
    for function in (policy._calibration_available, policy._critical_path_available,
                     policy._critical_path_identity, policy._resolve_cost_model):
        function.cache_clear()


def _synthetic_profile(module: MegaMoeExperts, values: torch.Tensor) -> ExpertReplicaCostModel:
    resources = module._get_execution_resources(values)
    signature = mega._critical_execution_signature(module, resources)
    table = ((0, 0.), (1000000, 1000.))
    calibration = DeviceCriticalPathCalibration(
        signature, ((0, 0.), (1000000, 2000.)), ((0, 0.), (1000000, 2000.)),
        .01, .02, tuple(range(module.ep_size)), planner_host_ms=.01)
    return ExpertReplicaCostModel(
        calibration_version=2, backend=module.dispatch_mode, hidden_size=module.hidden_size,
        intermediate_size=module.intermediate_size, ep_size=module.ep_size, transport=module.replica_transport,
        forward_ms=table, backward_ms=table, weight_ms=.01, gradient_ms=0., backward_weight_ms=0.,
        forward_repeats=2, _rank_nodes=resources.rank_nodes, critical_path=calibration)


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    value = actual.detach().float().cpu()
    return float((value - expected).norm() / expected.norm().clamp_min(1e-8))


def _routes(tokens: int, experts: int, top_k: int, pattern: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build distinct TopK fixtures without depending on an experiment branch."""
    ids = torch.arange(tokens*top_k).reshape(tokens, top_k) % experts
    hot = torch.arange(top_k).expand(tokens, top_k)
    if pattern=='middle':
        ids[:tokens//2] = hot[:tokens//2]
    elif pattern!='balanced':
        ids = hot.contiguous()
    return ids, torch.bincount(ids.flatten(), minlength=experts)


def main() -> None:
    """Capture F/R/B separately, with numerical checks outside measured intervals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--intermediate', type=int, default=128)
    parser.add_argument('--tokens', type=int, default=4096)
    parser.add_argument('--experts', type=int, default=24)
    parser.add_argument('--ep-size', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--observations-only', action='store_true')
    parser.add_argument('--synthetic-correctness', action='store_true')
    parser.add_argument('--profile-directory', type=Path)
    args = parser.parse_args()
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local)
    dist.init_process_group('hccl', timeout=timedelta(seconds=180))
    world, device = dist.get_world_size(), torch.device('npu', local)
    size, group = args.ep_size, None
    if size <= 0 or world % size or args.experts % size or not 1 <= args.top_k <= args.experts:
        raise ValueError('Expert count, EP size and TopK must form a valid distributed topology')
    for start in range(0, world, size):
        candidate = dist.new_group(list(range(start, start+size)))
        if start <= rank < start+size:
            group = candidate
    if args.profile_directory is not None and args.synthetic_correctness:
        raise ValueError('Measured and synthetic profiles cannot be combined')
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    original_profiles = policy._calibrations
    models = dict(original_profiles())
    if args.profile_directory is not None:
        for file in sorted(args.profile_directory.glob('*.json')):
            payload = json.loads(file.read_text())
            model = ExpertReplicaCostModel(**payload['model'])
            key = (model.backend, model.hidden_size, model.intermediate_size, model.ep_size, model.transport,
                   payload['selector']['num_experts'], payload['selector']['top_k'], payload['selector']['soc_name'])
            if model.critical_path is not None:
                key += (model.critical_path.execution_signature,)
            models[key] = model
    baselines: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    reports = []
    original_plan = routing._plan_replica_route
    original_forward, original_backward = executor._launch_forward_kernel, executor._launch_backward_kernel
    original_topk = mega.prepare_topk_route
    for mode in ('push', 'pull'):
        for budget in (0, 1, 6):
            torch.manual_seed(42 + rank)
            module = MegaMoeExperts(hidden_size=args.hidden, intermediate_size=args.intermediate,
                num_experts=args.experts, top_k=args.top_k, ep_group=group, dispatch_mode=mode,
                replica_slots_per_rank=budget, replica_transport='shmem_signal_kernel_gradient_bands',
                swiglu_limit=10.).to(device).bfloat16()
            optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
            events: list[tuple[str, Any, Any]] = []
            decisions: dict[str, dict[str, Any]] = {}
            active: dict[str, Any] = {}
            steps = []

            def plan_observer(*inputs: Any, **options: Any) -> Any:
                started = time.perf_counter()
                plan, reused = original_plan(*inputs, **options)
                planner_cpu_ms = (time.perf_counter()-started)*1000
                model = options.get('cost_model')
                if model is not None and model.critical_path is not None:
                    model = replace(model, _rank_nodes=tuple(options['_rank_nodes']))
                identity = hashlib.sha256(repr((plan.logical_counts, plan.slot_to_logical,
                    options['receive_capacity'], model)).encode()).hexdigest()
                if identity not in decisions:
                    predicted = None
                    if model is not None and model.critical_path is not None:
                        copies = [{expert: rows for expert, rows in zip(slots, counts) if expert >= 0}
                                  for slots, counts in zip(plan.slot_to_logical, plan.destination_counts)]
                        estimate = model.critical_path.estimate(model, copies, plan.logical_counts,
                                                               plan.config, trace=True)
                        predicted = asdict(estimate) | {'total_ms':estimate.total_ms}
                    decisions[identity] = dict(
                        model=None if model is None else asdict(model),
                        options={key:value for key,value in options.items() if key!='cost_model'},
                        config=asdict(plan.config), source_counts=plan.logical_counts,
                        slots=plan.slot_to_logical, dispatch=plan.dispatch_counts, estimate=predicted,
                        plan_reuses=0, calls=0, planner_cpu_wall_ms=[])
                decisions[identity]['plan_reuses'] += int(reused)
                decisions[identity]['calls'] += 1
                decisions[identity]['planner_cpu_wall_ms'].append(planner_cpu_ms)
                active['decision'] = identity
                return plan, reused

            def topk_observer(*inputs: Any, **options: Any) -> Any:
                selector = options.get('_select_replica_route')
                if selector is None:
                    return original_topk(*inputs, **options)

                def selected(*arguments: Any) -> Any:
                    proposal = selector(*arguments)
                    if proposal is None:
                        active['native_bypasses'] = active.get('native_bypasses', 0)+1
                    return proposal

                return original_topk(*inputs, **(options | {'_select_replica_route':selected}))

            def launch_forward(*inputs: Any, **options: Any) -> None:
                state = get_recompute_state()
                phase = 'recompute' if state is not None and state.is_recomputing else 'forward'
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                original_forward(*inputs, **options)
                end.record()
                events.append((phase, begin, end))

            def launch_backward(*inputs: Any, **options: Any) -> None:
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                original_backward(*inputs, **options)
                end.record()
                events.append(('backward', begin, end))

            try:
                with (patch.object(policy, '_calibrations', new=lambda: models),
                      patch.object(routing, '_plan_replica_route', new=plan_observer),
                      patch.object(mega, 'prepare_topk_route', new=topk_observer),
                      patch.object(executor, '_launch_forward_kernel', new=launch_forward),
                      patch.object(executor, '_launch_backward_kernel', new=launch_backward)):
                    _clear_profiles()
                    patterns = [pattern for pattern in ('balanced','middle','hot','ragged','empty','hot','balanced')
                                for _ in range(args.repeats)]
                    for step, pattern in enumerate(patterns):
                        tokens = args.tokens
                        if pattern=='ragged':
                            tokens = (0, 128, args.tokens, 512)[rank % 4]
                        if pattern=='empty':
                            tokens = 0
                        ids, counts = _routes(tokens, args.experts, args.top_k, pattern)
                        generator = torch.Generator().manual_seed(1000 + rank * 100 + step)
                        values = torch.randn(tokens, args.hidden, generator=generator).to(device).bfloat16()
                        values.requires_grad_()
                        gradient = torch.randn(tokens,args.hidden,generator=generator).to(device).bfloat16()
                        probabilities = torch.full((tokens,args.top_k),1/args.top_k,device=device,requires_grad=True)
                        ids, counts = ids.to(device).int(), counts.to(device).int()
                        events.clear()
                        active.clear()

                        def operation(hidden: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
                            if budget and args.synthetic_correctness:
                                profile = _synthetic_profile(module, hidden)
                                key = (mode,args.hidden,args.intermediate,size,module.replica_transport,
                                       args.experts,args.top_k,
                                       policy._soc_name(device),profile.critical_path.execution_signature)
                                if key not in models:
                                    models[key] = profile
                                    _clear_profiles()
                            output = module(hidden, ids, weights, tokens_per_expert=counts)
                            if budget:
                                resources = module._get_execution_resources(hidden)
                                active['execution_signature'] = mega._critical_execution_signature(module, resources)
                            return output

                        optimizer.zero_grad(set_to_none=True)
                        dist.barrier()
                        torch.npu.synchronize()
                        started = time.perf_counter()
                        result = checkpoint(operation, values, probabilities, early_stop=False)
                        result.backward(gradient)
                        optimizer.step()
                        torch.npu.synchronize()
                        elapsed = time.perf_counter()-started
                        phase_events = [{'phase':phase,'device_ms':begin.elapsed_time(end)}
                                        for phase,begin,end in events]
                        tensors = dict(output=result,input_grad=values.grad,router_grad=probabilities.grad)
                        tensors.update({name+'.grad':p.grad for name,p in module.named_parameters()})
                        tensors.update({name+'.updated':p for name,p in module.named_parameters()})
                        finite = all(t is not None and bool(torch.isfinite(t).all()) for t in tensors.values())
                        errors = {}
                        if budget==0 and not args.observations_only:
                            baselines[mode,step] = {name:t.detach().float().cpu() for name,t in tensors.items()}
                        elif not args.observations_only:
                            errors = {name:_relative(t,baselines[mode,step][name]) for name,t in tensors.items()}
                        worst = max(errors.values(),default=0.)
                        check = torch.tensor([float(not finite),worst],device=device)
                        dist.all_reduce(check,op=dist.ReduceOp.MAX)
                        check = check.cpu().tolist()
                        if check[0] or check[1]>.03:
                            raise RuntimeError(
                                f'Critical-path training precision failed: {mode}/{budget}/{pattern}/{check}')
                        steps.append(dict(step=step,pattern=pattern,tokens=tokens,wall_seconds=elapsed,
                            device_events=phase_events,errors=errors,decision=active.get('decision'),finite=finite,
                            execution_signature=active.get('execution_signature'),
                            native_bypasses=active.get('native_bypasses',0)))
                        del result, values, probabilities, gradient, tensors
                    if budget and args.synthetic_correctness:
                        if not any(d['model'] is not None and d['model']['critical_path'] is not None
                                   for d in decisions.values()):
                            raise RuntimeError('Critical-path calibration never reached the actual planner')
                        home = args.experts // size
                        if not any(sum(expert>=0 for slots in d['slots'] for expert in slots[home:])
                                   for d in decisions.values()):
                            raise RuntimeError('Critical-path planner never executed a replica')
                report = dict(mode=mode,budget=budget,rank=rank,steps=steps,decisions=decisions,
                    synthetic_calibration=args.synthetic_correctness,
                    observations_only=args.observations_only,
                    scope='Component correctness and device observations; not performance acceptance',
                    peak_allocated_bytes=torch.npu.max_memory_allocated())
                (args.output/f'{mode}-b{budget}-rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
                reports.append(dict(mode=mode,budget=budget,passed=True))
            finally:
                module.close()
    (args.output/f'status-rank{rank}.json').write_text(json.dumps(reports,indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':
    main()
