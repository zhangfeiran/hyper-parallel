# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Diagnostic-only exact plan injection; production policy and remapping stay unchanged."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterator
from unittest.mock import patch

from tests.torch.expert_parallel.replica_cluster_fixture import load_fixture, plan_digest
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.expert_parallel.hot_replica import routing
from hyper_parallel.core.expert_parallel.hot_replica._observer import _get_replica_observer
from hyper_parallel.core.multicore.modules.mega_moe import function as kernels
from hyper_parallel.core.multicore.modules.mega_moe import route as native_routes


class FixedPlanProbe:
    """Execute one saved immutable plan after all original input validation and strict guards."""

    def __init__(self, path: Path, verify_original_planner: bool = False) -> None:
        """Load the full actual layout, never a new cost-off solution."""
        self.path = path
        self.fixture, self.plan = load_fixture(path, verify_original_planner=verify_original_planner)
        self.digest = plan_digest(self.plan)
        self.original_options = self.fixture['original_decision']['options']
        self.topology = tuple(self.fixture['original_topology'])
        self.tokens = self.fixture['route_spec']['tokens']
        self.top_k = self.fixture['route_spec']['top_k']
        self.guard_calls = 0
        self.allowance_calls = 0
        self.accepted_routes = 0
        self.upload_calls = 0
        self.kernel_submissions = 0
        self._active_ordinal: int | None = None
        self._call_prepare_ns = 0
        self._last_accepted_ordinal: int | None = None
        self._runtime_options: dict[str, Any] | None = None
        self._stage_drops = 0
        self._runtime_resources: dict[str, Any] | None = None
        self._stages: list[dict[str, Any]] = []
        self._stage_limit = 128

    def capacity_allowance(self, maximum_tokens: int, top_k: int, target_load: int | None,
                           receive_capacity: int | None, factor: float | None,
                           budget: Any = None) -> tuple[int, int]:
        """Restore only the recorded allowance; keep downstream hard limits and heap growth."""
        del budget
        expected_target = self.original_options['target_load']
        expected_receive = self.original_options['receive_capacity']
        if (maximum_tokens != self.tokens or top_k != self.top_k or factor != 1.25
                or target_load not in (None, expected_target)
                or receive_capacity not in (None, expected_receive)):
            raise ValueError('Fixed-plan capacity input differs from the recorded diagnostic domain')
        self.allowance_calls += 1
        return expected_target, expected_receive

    def select(self, counts: Any, config: Any, group: Any, rank: int, **options: Any) -> tuple[Any, bool]:
        """Return the saved layout only after exact source/config/topology/hard-bound agreement."""
        del group
        started = perf_counter_ns()
        if self._call_prepare_ns:
            self._stamp('CountPreparationToReplay', self._call_prepare_ns,
                        scope='Remaining compact/count preparation, may include full D2H; not pure copy duration')
        if config != self.plan.config or not 0 <= rank < config.ep_size:
            raise ValueError('Fixed-plan configuration/rank mismatch')
        current = tuple(tuple(row) for row in counts)
        if current != self.plan.logical_counts:
            raise ValueError('Fixed-plan full per-source expert counts mismatch')
        if tuple(options.get('_rank_nodes', ())) != self.topology:
            raise ValueError('Fixed-plan normalized topology mismatch')
        for name in ('target_load', 'receive_capacity', 'capacity_limit', 'minimum_replica_rows', 'adaptive'):
            if options.get(name) != self.original_options[name]:
                raise ValueError(f'Fixed-plan runtime bound/control mismatch: {name}')
        if options.get('cost_model') is not None:
            raise ValueError('Fixed-plan probe cannot install or evaluate a production cost model')
        if max(self.plan.destination_loads) > options['capacity_limit']:
            raise ValueError('Fixed-plan layout exceeds the active hard capacity limit')
        self._runtime_options = dict(options)
        self.guard_calls += 1
        self._stamp('FixtureReplayGuard', started, plan_sha256=self.digest,
                    original_capacity_limit=self.original_options['capacity_limit'],
                    receive_capacity=self.original_options['receive_capacity'])
        return self.plan, bool(is_recomputing())

    @contextmanager
    def module_call(self, resources: Any, hidden: Any, autograd_mask: int) -> Iterator[None]:
        """Check the actual source/shape before a planner shortcut can be accepted."""
        model = self.fixture['model_configuration']
        spec = resources.spec
        if (hidden.shape[0] != self.tokens or hidden.shape[-1] != model['hidden_size']
                or spec.hidden_size != model['hidden_size'] or spec.intermediate_size != model['intermediate_size']
                or spec.ep_size != model['ep_size'] or spec.top_k != model['top_k']
                or spec.logical_num_experts != model['num_experts']
                or spec.replica_slots_per_rank != model['replica_slots_per_rank']
                or spec.expert_weight_layout != 'EIO' or str(hidden.dtype) != 'torch.bfloat16'
                or spec.swiglu_limit != model['swiglu_limit'] or autograd_mask != 31
                or tuple(resources.rank_nodes) != self.topology):
            raise ValueError('Fixed-plan runtime shape/topology/autograd domain mismatch')
        previous = self._active_ordinal
        self._active_ordinal = self.accepted_routes
        before = self.guard_calls
        self._call_prepare_ns = perf_counter_ns()
        self._runtime_resources = {'physical_experts': spec.num_experts, 'local_experts': spec.local_experts,
                                   'replica_slots_per_rank': spec.replica_slots_per_rank,
                                   'rank_nodes': list(resources.rank_nodes), 'autograd_mask': autograd_mask}
        try:
            yield
            if self.guard_calls != before + 1:
                raise RuntimeError('Fixed-plan injection was bypassed before execution')
        finally:
            self._active_ordinal = previous
            self._call_prepare_ns = 0

    def accept(self, route: Any) -> None:
        """Reject home/compact shortcuts and verify the final plan passed to metadata upload."""
        if route.home_only or plan_digest(route.plan) != self.digest:
            raise RuntimeError('Final execution layout differs from the actual saved fixture')
        self._last_accepted_ordinal = self.accepted_routes
        self.accepted_routes += 1

    def _stamp(self, kind: str, start_ns: int, **metadata: Any) -> None:
        """Retain bounded Host intervals; no SDMA completion or CPU exposure claim."""
        observer = _get_replica_observer()
        if observer is None:
            return
        phase = metadata.pop('phase', 'recompute' if is_recomputing() else 'forward')
        ordinal = self._active_ordinal
        if ordinal is None:
            ordinal = self.accepted_routes if kind == 'CountReadbackAndValidation' else self._last_accepted_ordinal
        record = observer.begin(kind, phase=phase, fixture_call_ordinal=ordinal, **metadata)
        if record is not None:
            record['start_ns'] = start_ns
            observer.end(record)
        if len(self._stages) < self._stage_limit:
            self._stages.append({'kind': kind, 'start_ns': start_ns, 'end_ns': perf_counter_ns(),
                                 'fixture_call_ordinal': ordinal})
        else:
            self._stage_drops += 1

    @contextmanager
    def installed(self) -> Iterator[FixedPlanProbe]:
        """Replace policy only inside this diagnostic; retain all normal remap/allocation code."""
        read = native_routes._read_dynamic_route
        upload = routing._upload_route_metadata
        forward = kernels._launch_forward_kernel
        backward = kernels._launch_backward_kernel

        def readback(*args: Any, **kwargs: Any) -> Any:
            start = perf_counter_ns()
            result = read(*args, **kwargs)
            self._stamp('CountReadbackAndValidation', start,
                        scope='Compact/full readback plus validation, excludes earlier gather submission/wait')
            return result

        def metadata(*args: Any, **kwargs: Any) -> Any:
            start = perf_counter_ns()
            result = upload(*args, **kwargs)
            self.upload_calls += 1
            self._stamp('MetadataPackUploadSubmit', start, scope='CPU pack and H2D enqueue, not Device completion')
            self._stamp('MetadataViewsConstructed', perf_counter_ns(),
                        scope='Host views ready, not Device transfer completion')
            return result

        def submit(original: Any, direction: str, *args: Any, **kwargs: Any) -> Any:
            self.kernel_submissions += 1
            self._stamp('KernelLaunchFunctionEntry', perf_counter_ns(), direction=direction,
                        phase=('backward' if direction == 'backward'
                               else ('recompute' if is_recomputing() else 'forward')),
                        scope='Python launch helper entry, not runtime Device start')
            return original(*args, **kwargs)

        with (patch.object(routing, '_plan_replica_route', self.select),
              patch.object(routing, '_automatic_capacity_options', self.capacity_allowance),
              patch.object(native_routes, '_read_dynamic_route', readback),
              patch.object(routing, '_upload_route_metadata', metadata),
              patch.object(kernels, '_launch_forward_kernel',
                           lambda *a, **kw: submit(forward, 'forward', *a, **kw)),
              patch.object(kernels, '_launch_backward_kernel',
                           lambda *a, **kw: submit(backward, 'backward', *a, **kw))):
            yield self

    def facts(self) -> dict[str, Any]:
        """Expose the old decision solely as provenance, separate from the new execution identity."""
        return {'mode': 'exact_actual_plan_diagnostic_bypass', 'fixture_file': self.path.name,
                'fixture_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'actual_plan_sha256': self.digest, 'transfers': len(self.plan.transfers),
                'original_options': self.original_options,
                'origin_execution_signature': self.fixture['observed_execution_signature'],
                'guard_calls': self.guard_calls, 'accepted_routes': self.accepted_routes,
                'allowance_calls': self.allowance_calls, 'metadata_uploads': self.upload_calls,
                'kernel_submissions': self.kernel_submissions, 'actual_runtime_resources': self._runtime_resources,
                'host_stages': self._stages, 'host_stage_drops': self._stage_drops,
                'runtime_options': self._runtime_options,
                'scope': 'Fixed plan executor probe; not original planner wall or production admission acceptance'}
