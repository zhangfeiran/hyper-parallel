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
"""Untimed, caller-versioned residency opportunities on real unchanged transfers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
import sys
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_moe import function
from tests.torch.expert_parallel.replica_residency import ResidentWeight, ResidencyLedger


class ResidencyAudit:
    """Shadow incoming weight writes in one serial shared execution workspace."""

    def __init__(self, resources: Any) -> None:
        """Keep diagnostic metadata while the actual provider performs every write.

        Args:
            resources: Shared execution resources identifying the pool and heap generation.
        """
        self.resources = resources
        self.ledger = ResidencyLedger()
        self.records = []
        self.routes = {}
        self.current = None
        self.patch = None
        self.events = {}

    @contextmanager
    def invocation(self, layer: str, generation: int, name: str) -> Iterator[None]:
        """Provide explicit caller-owned versions for a controlled audit invocation.

        Args:
            layer: Stable layer identity, independent of a weight tensor's address.
            generation: Version advanced by the diagnostic's actual parameter updates.
            name: Forward or backward invocation label.
        """
        previous = self.current
        self.current = (layer, generation, name)
        try:
            yield
        finally:
            self.current = previous

    def _incoming(self, weights, route, backward):
        if backward:
            layer, generation, _ = self.routes[id(route)]
        else:
            layer, generation, _ = self.current
            self.routes[id(route)] = self.current
        home = route.plan.config.home_experts
        pending = []
        for transfer in route.plan.transfers:
            if transfer.target_rank != route.rank:
                continue
            for projection, weight in enumerate(weights):
                slot = transfer.target_slot - home
                key = (id(self.resources.workspace), slot, projection)
                layout = (str(weight.dtype), tuple(weight.shape[1:]), tuple(weight.stride()[1:]))
                identity = ResidentWeight(layer, transfer.logical_expert, generation, layout,
                                          self.resources.heap_manager.epoch)
                event = self.events.get(key)
                record = {"invocation": self.current[2], "layer": layer, "generation": generation,
                          "direction": "backward" if backward else "forward", "slot": slot,
                          "projection": projection, "expert": transfer.logical_expert,
                          "owner": transfer.owner_rank,
                          "payload_bytes": weight[transfer.owner_slot].numel() * weight.element_size(),
                          "reason": self.ledger.inspect(key, identity), "identity": asdict(identity),
                          "prior_completion_observed": event is not None and event.query(),
                          "completion_contract": "previous serial lease orders all writers and consumers",
                          "actual_data_copy": True}
                self.records.append(record)
                pending.append((key, identity, record))
        return pending

    def __enter__(self) -> ResidencyAudit:
        """Observe the real lease without modifying its transfer list or signals."""
        original = function.prefetch_weights

        @contextmanager
        def _prefetch(weights, route, **options):
            backward = options.get("backward", False)
            pending = self._incoming(weights, route, backward)
            context = original(weights, route, **options)
            successful = False
            try:
                pool = context.__enter__()
                try:
                    yield pool
                finally:
                    context.__exit__(*sys.exc_info())
                successful = True
            finally:
                if successful:
                    event = torch.npu.Event()
                    event.record(torch.npu.current_stream(weights[0].device))
                    for key, identity, record in pending:
                        self.ledger.commit(key, identity)
                        self.events[key] = event
                        record["lease_completed_ordering"] = True
                else:
                    self.ledger.invalidate(tuple(key for key, _, _ in pending))
                if backward:
                    self.routes.pop(id(route), None)

        self.patch = patch.object(function, "prefetch_weights", _prefetch)
        self.patch.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """Restore the original prefetch after the diagnostic invocations."""
        self.patch.__exit__(*args)


def _audit_forward(audit, module, values, ids, probabilities, layer, generation, name):
    with audit.invocation(layer, generation, name):
        return module(values, ids, probabilities)


def _audit_backward(audit, output, gradient, layer, generation, name):
    with audit.invocation(layer, generation, name):
        output.backward(gradient)


def audit_residency(modules: tuple, values: torch.Tensor, routes: dict,
                    probabilities: torch.Tensor, gradient: torch.Tensor) -> dict:
    """Measure potential hits across micro-batches, layer reuse and real weight updates.

    Args:
        modules: Two compatible, not-yet-executed modules sharing execution resources.
        values: Input leaf tensor used only by untimed audit invocations.
        routes: Benchmark routes including home_hot and rotating_hot.
        probabilities: Fixed router probabilities.
        gradient: Output cotangent matching the input shape.

    Returns:
        Per-write opportunities and globally checked scenario summaries. Every
        real weight copy remains enabled; ordered_hit is not a cache performance result.
    """
    first = modules[0]
    optimizers = [torch.optim.SGD(module.parameters(), lr=0.001) for module in modules]
    resources = first._get_execution_resources(values)
    scenarios = {}
    ids = routes["home_hot"][0]
    for _ in range(3):
        first(values, ids, probabilities).backward(gradient)
    torch.npu.synchronize()
    warm_capacity = resources.workspace.capacity_floor
    warm_heap_generation = resources.heap_manager.epoch
    with ResidencyAudit(resources) as audit:
        for name in ("same_version_microbatches", "cross_layer_shared_pool", "optimizer_update", "rotating_experts"):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            values.grad = None
            start = len(audit.records)
            _audit_scenario(name, audit, modules, optimizers, values, ids, routes, probabilities, gradient)
            torch.npu.synchronize()
            storage = resources.workspace.capacity_floor, resources.heap_manager.epoch
            if storage != (warm_capacity, warm_heap_generation):
                raise AssertionError(f"Residency scenario grew storage: scenario={name}, storage={storage}, "
                                     f"expected={(warm_capacity, warm_heap_generation)}")
            rows = audit.records[start:]
            reasons = Counter(row["reason"] for row in rows)
            scenarios[name] = {"reason_counts": dict(reasons),
                               "requested_weight_bytes": sum(row["payload_bytes"] for row in rows),
                               "potential_hit_bytes": sum(row["payload_bytes"] for row in rows
                                                          if row["reason"] == "ordered_hit")}
        records = audit.records
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, scenarios)
    combined = Counter()
    for rank in gathered:
        for scenario in rank.values():
            combined.update(scenario["reason_counts"])
    if any(combined[reason] == 0 for reason in ("ordered_hit", "layer_changed", "weight_updated")):
        raise AssertionError(f"Residency audit did not exercise required cases: reason_counts={dict(combined)}")
    return {"records": records, "scenarios": scenarios, "global_reason_counts": dict(combined),
            "warm_capacity": warm_capacity, "warm_heap_generation": warm_heap_generation,
            "scope": "Untimed shadow audit with explicit diagnostic caller generations; all actual copies enabled. "
                     "Ordered candidates rely on the existing serial lease, not a host-completed cache proof. "
                     "No bandwidth/latency saving or production weight-version integration is implemented."}


def _audit_scenario(name, audit, modules, optimizers, values, ids, routes, probabilities, gradient):
    first, second = modules
    if name == "optimizer_update":
        # The prior scenarios leave first's v0 in the shared slot. Update actual
        # owner parameters before advancing the diagnostic's authoritative version.
        warm = _audit_forward(audit, first, values, ids, probabilities, "layer-a", 0, "update-before-F")
        _audit_backward(audit, warm, gradient, "layer-a", 0, "update-before-B")
        optimizers[0].step()
        optimizers[0].zero_grad(set_to_none=True)
        output = _audit_forward(audit, first, values, ids, probabilities, "layer-a", 1, "update-after-F")
        _audit_backward(audit, output, gradient, "layer-a", 1, "update-after-B")
        return
    generation = 1 if name == "rotating_experts" else 0
    other = second if name == "cross_layer_shared_pool" else first
    other_layer = "layer-b" if other is second else "layer-a"
    other_ids = routes["rotating_hot"][1] if name == "rotating_experts" else ids
    left = _audit_forward(audit, first, values, ids, probabilities, "layer-a", generation, f"{name}-F_A")
    right = _audit_forward(audit, other, values, other_ids, probabilities,
                           other_layer, generation, f"{name}-F_B")
    _audit_backward(audit, right, gradient, other_layer, generation, f"{name}-B_B")
    _audit_backward(audit, left, gradient, "layer-a", generation, f"{name}-B_A")
