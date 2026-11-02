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
"""Capture bounded replica dependencies with fixed inputs and checked actual plans.

This is an executor component experiment. It is not full DSV training or an
admission-policy acceptance test. Host and each Device clock remain independent.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts, is_recomputing
from hyper_parallel.core.expert_parallel.hot_replica.residency import _ReplicaWeightCache
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.function import _MegaMoeFunction
from hyper_parallel.core.multicore.diagnostics import profiler, schedule
from hyper_parallel.core.multicore._loader import get_multicore_paths
from tests.torch.expert_parallel.replica_cluster_fixture import load_fixture, local_routes, plan_digest
from tests.torch.expert_parallel.replica_cluster_fixed_probe import FixedPlanProbe
from tests.torch.expert_parallel.replica_schedule_capture import weight_submission_summary
from hyper_parallel.core.multicore.modules.mega_moe.module import (
    _critical_execution_signature, _critical_native_fingerprint,
)


class CheckedPlans(MegaMoeExperts):
    """Retain only immutable CPU plans for comparison after the captured step."""

    def _prepare_replica_route(self, topk_ids: Any, resources: Any, hidden: Any,
                               autograd_mask: int, counts: Any, gathered: Any = None) -> Any:
        probe = self.fixed_probe
        context = probe.module_call(resources, hidden, autograd_mask) if probe is not None else nullcontext()
        with context:
            route = super()._prepare_replica_route(topk_ids, resources, hidden, autograd_mask, counts, gathered)
            if probe is not None:
                probe.accept(route)
            if probe is None or probe.record_observations:
                self.checked_plans.append(route.plan)
                self.actual_execution_signatures.add(_critical_execution_signature(self, resources))
            return route


def bind_cpu() -> list[int]:
    """Retain the affinity assigned by the pre-import cluster launcher."""
    return sorted(os.sched_getaffinity(0))


def tensor_hash(tensor: torch.Tensor) -> str:
    """Hash exact bytes outside the timing/capture window."""
    value = tensor.detach().cpu().contiguous()
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise RuntimeError("Nonfinite tensor in fixed-plan numerical validation")
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Reduce only same-device intervals; keep pending windows distinct from pure waits."""
    invocations = []
    for invocation in trace["megaKernelCycleTrace"]["invocations"]:
        rows = [event for event in trace["traceEvents"] if event.get("args", {}).get("invocation_id")
                == invocation["invocationId"] and event["ph"] == "X"]
        by_name: dict[str, list[dict[str, Any]]] = {}
        for event in rows:
            by_name.setdefault(event["args"].get("task_stage", event["name"]), []).append(event)
        cube_done = [event["args"]["end_cycle"] for event in rows if event["args"]["core_type"] == "AIC"
                     and event["args"]["desc_id"] == 0x1001E]
        returned = [event["args"]["end_cycle"] for event in by_name.get("GradientReturn", [])]
        frequency = invocation["cycleFrequencyMHz"]
        invocations.append({**invocation, "event_counts": {name: len(events) for name, events in by_name.items()},
                            "max_ack_observed_us": max((event["dur"] for event in by_name.get("AckObserved", [])),
                                                       default=0),
                            "local_return_tail_after_cube_tasks_us":
                            max(0, max(returned) - max(cube_done)) / frequency if cube_done and returned else None,
                            "pending_scope": "HeadPendingWindow includes service of other bands; not pure wait"})
    host_events = trace.get('replicaHostObservations', {}).get('events', [])
    return {"invocations": invocations, "dropped_records": trace["megaKernelCycleTrace"]["droppedRecordCount"],
            "weight_submissions": weight_submission_summary(host_events),
            "host_dropped": trace.get("replicaHostObservations", {}).get("dropped_events", 0),
            "cross_rank_latency": None, "clock_alignment": "not calibrated across dies or Host/Device"}


def training_hashes(output: torch.Tensor, hidden: torch.Tensor, probs: torch.Tensor,
                    module: CheckedPlans) -> dict[str, str]:
    """Audit output, all gradients, and the optimizer update outside capture."""
    return {"output": tensor_hash(output), "input_grad": tensor_hash(hidden.grad),
            "router_grad": tensor_hash(probs.grad),
            **{name + ".grad": tensor_hash(parameter.grad) for name, parameter in module.named_parameters()},
            **{name + ".updated": tensor_hash(parameter) for name, parameter in module.named_parameters()}}


def write_observations(args: argparse.Namespace, module: CheckedPlans, cpus: list[int],
                       cpu_ids: torch.Tensor, markers: dict[str, Any], origin: Any,
                       trace: dict[str, Any], actual_hashes: dict[str, str]) -> None:
    """Retain reproducible normalized facts and remove the redundant raw trace."""
    rank, ep_rank = dist.get_rank(), dist.get_rank() % args.ep_size
    group_ranks = list(range(rank - ep_rank, rank - ep_rank + args.ep_size))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    output_dir = Path(args.output)
    raw = output_dir / f"rank{rank}.raw.json"
    summary = summarize_trace(trace)
    plans = module.checked_plans
    if plans and any(plan != plans[0] for plan in plans[1:]):
        raise RuntimeError("Actual plan changed across fixed-route steps")
    plan_payload = asdict(plans[0]) if plans else None
    plan_bytes = json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode()
    with raw.open('rb') as source, gzip.open(output_dir / f"rank{rank}.events.json.gz", "wb") as stream:
        shutil.copyfileobj(source, stream)
    with gzip.open(output_dir / f"rank{rank}.plan.json.gz", "wt") as stream:
        stream.write(plan_bytes.decode())
    marker_facts = {name: {"device_ms_after_step_origin": origin.elapsed_time(event), "host_enqueue_ns": host_ns,
                          "scope": "parameter post-accumulate stream marker, no FSDP/optimizer readiness claim"}
                    for name, (event, host_ns) in markers.items()}
    if module.fixed_probe is not None:
        if hashlib.sha256(plan_bytes).hexdigest() != module.fixture_plan_sha256:
            raise RuntimeError("Serialized final plan SHA differs from the actual fixture")
        facts = module.fixed_probe.facts()
        measured_calls = 2 * (args.timing_warmup + args.measure_iterations) if args.measure_iterations else 0
        expected_calls = 2 * (args.warmup + 2)
        if (facts['guard_calls'] != expected_calls + measured_calls
                or facts['accepted_routes'] != expected_calls + measured_calls
                or facts['metadata_uploads'] != expected_calls or facts['host_stage_drops']):
            raise RuntimeError(f"Fixed-plan replay reachability is incomplete: {facts}")
    else:
        facts = {'mode': 'dynamic_policy' if args.dynamic_plan else 'owner_only_control',
                 'source_route_fixture': module.fixture_file.name,
                 'scope': 'Actual runtime decisions; no literal plan injection'}
    source_manifest = json.loads((Path(os.environ['BUNDLE_ROOT']) / 'native-fingerprint.json').read_text())
    native_identity = _critical_native_fingerprint()
    if native_identity != source_manifest['native_fingerprint']:
        raise RuntimeError("Loaded native fingerprint differs from the frozen observation payload")
    certificates = {mode: hashlib.sha256(json.dumps({
        'base_signatures': sorted(module.actual_execution_signatures), 'native': native_identity,
        'mode': mode, 'fixture_plan': module.fixture_plan_sha256,
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'probe_sha256': hashlib.sha256(
            (Path(__file__).parent / 'replica_cluster_fixed_probe.py').read_bytes()).hexdigest(),
        'fixture_loader_sha256': hashlib.sha256(
            (Path(__file__).parent / 'replica_cluster_fixture.py').read_bytes()).hexdigest(),
        'runtime_config': {'hidden': args.hidden, 'intermediate': args.intermediate, 'ep': args.ep_size,
                           'experts': args.ep_size * 6, 'top_k': 6, 'tokens': args.tokens,
                           'budget': args.budget, 'weight_layout': 'EIO', 'swiglu_limit': 10.},
        'source_counts_sha256': hashlib.sha256(json.dumps(
            module.fixture_provenance['plan']['logical_counts'], separators=(',', ':')).encode()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest() for mode in ('OFF', 'causal_v1')}
    manifest = {"rank": rank, "ep_rank": ep_rank, "ep_group_global_ranks": group_ranks, "cpus": cpus,
                "args": vars(args), "fixed_route_sha256": tensor_hash(cpu_ids),
                "pure_timing": dict(samples=module.pure_samples, warmup=args.timing_warmup,
                    iterations=args.measure_iterations, observations=False, profiler=False,
                    scope='Fixed-plan F+full recompute+B+AdamW; strict plan binding remains, no dynamic planner'),
                "actual_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "actual_plan_checks": len(plans),
                "runtime_calls": module.runtime_calls, "backend": args.backend,
                "tensor_hashes_exact": True, "tensor_sha256": actual_hashes, "observations": summary,
                "gradient_markers": marker_facts, "fixed_plan_replay": facts,
                "fixture_file_sha256": hashlib.sha256(module.fixture_file.read_bytes()).hexdigest(),
                "fixture_origin_execution_signature": module.fixture_provenance['observed_execution_signature'],
                "runtime_base_signatures": sorted(module.actual_execution_signatures),
                "diagnostic_execution_certificates": certificates, "native_fingerprint": native_identity,
                "original_fixture_ep_members": module.fixture_provenance['global_ep_members'],
                "group_relocation": group_ranks != module.fixture_provenance['global_ep_members'],
                "native_paths": str(get_multicore_paths()),
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "raw_bytes": raw.stat().st_size, "raw_deleted_after_lossless_compression": True,
                "terminal_status": 0, "max_memory_allocated": torch.npu.max_memory_allocated(device)}
    (output_dir / f"rank{rank}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    raw.unlink()
    print(json.dumps({"rank": rank, "event": "complete", "dropped": 0, "tensor_hashes_exact": True}), flush=True)


def outer_capture(args: argparse.Namespace, ep_rank: int, rank: int) -> Any:
    """Capture the runtime prefix/kernel envelope on a bounded set of key ranks."""
    if not args.outer_capture or ep_rank not in args.outer_ranks:
        return nullcontext()
    native_profiler = torch_npu.profiler
    return native_profiler.profile(
        activities=[native_profiler.ProfilerActivity.CPU, native_profiler.ProfilerActivity.NPU],
        on_trace_ready=native_profiler.tensorboard_trace_handler(str(Path(args.output) / "outer" / f"rank{rank}")),
        record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
        experimental_config=native_profiler._ExperimentalConfig(  # pylint: disable=protected-access
            profiler_level=native_profiler.ProfilerLevel.Level0,
            aic_metrics=native_profiler.AiCMetrics.AiCoreNone, data_simplification=False),
    )


def run(args: argparse.Namespace, cpus: list[int]) -> None:
    """Run fixed F/R/B/AdamW, collect all ranks, and fail closed on plan drift or drops."""
    rank, world = dist.get_rank(), dist.get_world_size()
    if world % args.ep_size:
        raise ValueError("WORLD_SIZE must be a multiple of EP size")
    group = None
    group_ranks = []
    for start in range(0, world, args.ep_size):
        ranks = list(range(start, start + args.ep_size))
        candidate = dist.new_group(ranks, backend="hccl")
        if rank in ranks:
            group, group_ranks = candidate, ranks
    ep_rank = group_ranks.index(rank)
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    if args.budget and not args.dynamic_plan:
        fixed_probe = FixedPlanProbe(Path(args.fixture))
        fixture, fixture_plan = fixed_probe.fixture, fixed_probe.plan
        if args.budget != fixture_plan.config.replica_slots_per_rank:
            raise ValueError("Requested B cap differs from the actual fixture")
    else:
        fixture, fixture_plan = load_fixture(Path(args.fixture), verify_original_planner=False)
        fixed_probe = None
    expected_pattern = 'middle' if args.skew == 'moderate' else 'hot'
    if fixture['route_spec']['pattern'] != expected_pattern:
        raise ValueError('Fixture route pattern differs from the requested label')
    model = fixture['model_configuration']
    if (args.ep_size, args.hidden, args.intermediate, args.tokens) != (
            model['ep_size'], model['hidden_size'], model['intermediate_size'], fixture['route_spec']['tokens']):
        raise ValueError("Runtime shape differs from the fixed fixture")
    torch.manual_seed(42 + rank)
    module = CheckedPlans(hidden_size=args.hidden, intermediate_size=args.intermediate,
                          num_experts=args.ep_size * 6, top_k=6, ep_group=group, ep_size=args.ep_size,
                          dispatch_mode=args.backend, swiglu_limit=10., replica_slots_per_rank=args.budget,
                          replica_transport="shmem_signal_kernel_gradient_bands").to(
                              device=device, dtype=torch.bfloat16)
    module.checked_plans = []
    module.fixed_probe = fixed_probe
    module.actual_execution_signatures = set()
    module.fixture_provenance = fixture
    module.fixture_plan_sha256 = fixture['plan_sha256'] if args.budget and not args.dynamic_plan else None
    module.runtime_calls = []
    module.pure_samples = []
    module.fixture_file = Path(args.fixture)
    generator = torch.Generator().manual_seed(1000 + rank * 100 + fixture['original_step'])
    hidden = torch.randn(args.tokens, args.hidden, generator=generator).to(device).bfloat16().requires_grad_()
    upstream = torch.randn(args.tokens, args.hidden, generator=generator).to(device).bfloat16()
    probs = torch.full((args.tokens, 6), 1 / 6, device=device, requires_grad=True)
    cpu_ids = torch.tensor(local_routes(fixture, ep_rank), dtype=torch.int32)
    cpu_counts = torch.bincount(cpu_ids.long().flatten(), minlength=args.ep_size * 6).int()
    ids, counts = cpu_ids.to(device), cpu_counts.to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=.0001)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    markers: dict[str, Any] = {}
    hooks = []
    marked = False

    def gradient_ready(parameter: torch.Tensor, name: str) -> None:
        if marked:
            event = torch.npu.Event(enable_timing=True)
            event.record()
            markers[name] = (event, time.perf_counter_ns())

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        hidden.grad = probs.grad = None
        with set_checkpoint_early_stop(False):
            output = checkpoint(module, hidden, ids, probs, tokens_per_expert=counts,
                                use_reentrant=False, context_fn=create_recompute_contexts)
        output.backward(upstream)
        optimizer.step()
        torch.npu.synchronize()
        return output

    try:
        if args.measure_iterations:
            with fixed_probe.installed(observations=False):
                for index in range(args.timing_warmup + args.measure_iterations):
                    dist.barrier()
                    torch.npu.synchronize()
                    started = time.perf_counter()
                    value = step()
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    del value
                    if index >= args.timing_warmup:
                        module.pure_samples.append(dict(sample_index=index - args.timing_warmup,
                                                       local_step_ms=elapsed_ms))
        for name, parameter in module.named_parameters():
            hooks.append(parameter.register_post_accumulate_grad_hook(
                lambda value, name=name: gradient_ready(value, name)))
        context = fixed_probe.installed() if fixed_probe is not None else nullcontext()
        original_forward = _MegaMoeFunction.forward

        def record_runtime(ctx: Any, *values: Any) -> Any:
            route, spec = values[3].replica_route, values[4].spec
            module.runtime_calls.append(dict(replica_route_present=route is not None,
                transfer_edges=0 if route is None else len(route.plan.transfers),
                physical_replica_slots=spec.replica_slots_per_rank,
                phase='recompute' if is_recomputing() else 'forward'))
            return original_forward(ctx, *values)

        with context, patch.object(_MegaMoeFunction, 'forward', staticmethod(record_runtime)):
            for _ in range(args.warmup):
                step()
            reference_weights = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
            reference_state = deepcopy(optimizer.state_dict())
            # Exact instrumentation-on/off comparison uses the same state and inputs.
            expected = step().detach().clone()
            expected_hashes = training_hashes(expected, hidden, probs, module)
            with torch.no_grad():
                for name, parameter in module.named_parameters():
                    parameter.copy_(reference_weights[name])
            # Restore AdamW state separately so diagnostics do not change the update sequence.
            optimizer.load_state_dict(reference_state)
            dist.barrier()
            torch.npu.synchronize()
            marked = True
            origin = torch.npu.Event(enable_timing=True)
            origin.record()
            with outer_capture(args, ep_rank, rank):
                with profiler.mega_kernel_profile(schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
                                                  detailed_task_names=True, max_pending_calls=3) as capture:
                    actual = step()
                    capture.step()
            marked = False
            raw = output_dir / f"rank{rank}.raw.json"
            trace = capture.export_chrome_trace(raw)
            summary = summarize_trace(trace)
            if summary["dropped_records"] or summary["host_dropped"]:
                raise RuntimeError(f"Diagnostic event budget exceeded: {summary}")
            actual_hashes = training_hashes(actual, hidden, probs, module)
            if actual_hashes != expected_hashes:
                raise RuntimeError("Profiling changed output or gradient bytes for the same input and weights")
            write_observations(args, module, cpus, cpu_ids, markers, origin, trace, actual_hashes)
            dist.barrier()
    finally:
        for hook in hooks:
            hook.remove()
        module.close()


def main() -> None:
    """Launch one explicit workload; cluster launcher provides fresh processes per case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--budget", type=int, choices=(0, 1, 6), required=True)
    parser.add_argument("--backend", choices=("push", "pull"), default="push")
    parser.add_argument("--dynamic-plan", action="store_true")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--skew", choices=("moderate", "strong"), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--outer-capture", action="store_true")
    parser.add_argument("--force-residency-miss", action="store_true")
    parser.add_argument("--measure-iterations", type=int, default=0)
    parser.add_argument("--timing-warmup", type=int, default=6)
    parser.add_argument("--outer-ranks", nargs="+", type=int, default=[0, 16, 63])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.ep_size, args.hidden, args.intermediate, args.tokens) <= 0 or args.warmup < 1:
        raise ValueError("Positive dimensions and at least one warmup step are required")
    if args.measure_iterations < 0 or args.timing_warmup < 1:
        raise ValueError('Timing iterations must be nonnegative and timing warmup positive')
    if args.measure_iterations and (args.dynamic_plan or not args.budget):
        raise ValueError('Uninstrumented fixed-plan timing requires a positive literal-plan budget')
    if not args.dynamic_plan and os.environ.get("HYPER_PARALLEL_REPLICA_COST_MODEL") != "0":
        raise ValueError("This literal-plan diagnostic disables production scoring; the saved plan is injected exactly")
    if args.dynamic_plan and os.environ.get("HYPER_PARALLEL_REPLICA_COST_MODEL") != "1":
        raise ValueError('Dynamic-plan diagnostics require the real cost policy to remain enabled')
    cpus = bind_cpu()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=600))
    original_prepare = _ReplicaWeightCache.prepare

    def forced_miss(cache: Any, weights: Any, route: Any) -> Any:
        cache.clear()
        return original_prepare(cache, weights, route)

    try:
        with patch.object(_ReplicaWeightCache, 'prepare', forced_miss if args.force_residency_miss
                          else original_prepare):
            run(args, cpus)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
