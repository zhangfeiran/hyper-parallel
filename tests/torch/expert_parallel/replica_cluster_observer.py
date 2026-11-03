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
"""Tensor-free lifecycle records around the real replica provider and saved context."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import math
from typing import Any, Iterator
import weakref
from unittest.mock import patch

import torch

from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.expert_parallel.hot_replica.residency import _ReplicaWeightCache
from hyper_parallel.core.expert_parallel.hot_replica.signal_transport import SignalReplicaTransport
from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.modules.mega_moe.function import _MegaMoeFunction
from tests.torch.expert_parallel.replica_cluster_lifecycle import readonly_view


class Ledger:
    """Record immutable facts and runtime copy calls outside performance timing."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.rows = []
        self.context = {'update': -1, 'micro': -1}
        self.active_layer = None
        self.weight_layers = {}
        self.active_provider = None
        self.phase = 'unknown'
        self.put_bytes = 0
        self.slot_writers = {}
        self.provider_bindings = {}

    def register(self, weights: tuple, layer: int) -> None:
        """Weak references protect observer identity from Python address reuse."""
        self.weight_layers[tuple(id(value) for value in weights)] = (
            tuple(weakref.ref(value) for value in weights), layer)

    def identify(self, weights: tuple) -> int | None:
        """Resolve the current invocation without retaining any model tensor."""
        found = self.weight_layers.get(tuple(id(value) for value in weights))
        if found is not None and all(ref() is value for ref, value in zip(found[0], weights)):
            return found[1]
        return self.active_layer

    @contextmanager
    def observe(self) -> Iterator[Ledger]:
        """Instrument the real cache and signal lease; control changes only normal-prefetch reuse."""
        original_prepare = _ReplicaWeightCache.prepare
        original_prefetch = SignalReplicaTransport._overlapped_prefetch
        original_lease = SignalReplicaTransport.lease
        original_put = shmem.put
        original_forward = _MegaMoeFunction.forward
        original_backward = _MegaMoeFunction.backward

        def forward(ctx: Any, *values: Any) -> Any:
            ctx._residency_ledger_layer = self.active_layer
            ctx._cluster_gradient_elements = sum(math.prod(value.shape[1:]) for value in values[1:3])
            workspace = values[5]
            provider = workspace.replica_provider
            if provider is not None:
                binding = (id(provider), id(workspace.heap_manager), workspace.heap_manager.epoch,
                           provider.storage.data_ptr(), provider.storage.numel())
                previous = self.provider_bindings.get(id(workspace))
                self.rows.append({**self.context, 'kind': 'provider_binding', 'layer': self.active_layer,
                                  'phase': 'recompute' if is_recomputing() else 'forward',
                                  'provider_id': binding[0], 'heap_manager_id': binding[1],
                                  'heap_epoch': binding[2], 'heap_storage_ptr': binding[3],
                                  'heap_storage_bytes': binding[4],
                                  'provider_changed': previous is not None and previous[0] != binding[0],
                                  'heap_changed': previous is not None and previous[1:3] != binding[1:3]})
                self.provider_bindings[id(workspace)] = binding
            return original_forward(ctx, *values)

        def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
            previous = self.active_layer
            self.active_layer = ctx._residency_ledger_layer
            try:
                result = original_backward(ctx, grad_output)
                route = getattr(ctx, 'replica_route', None)
                incoming = (sum(edge.owner_rank == route.rank for edge in route.plan.transfers)
                            if route is not None else 0)
                self.rows.append({**self.context, 'kind': 'backward_return_payload',
                                  'layer': self.active_layer, 'owner_incoming_edges': incoming,
                                  'fp32_payload_bytes': incoming * ctx._cluster_gradient_elements * 4,
                                  'scope': 'Executed route matrix payload; not measured network traffic'})
                return result
            finally:
                self.active_layer = previous

        def put(destination: torch.Tensor, source: torch.Tensor, peer: int, **kwargs: Any) -> Any:
            if source.dtype == torch.bfloat16:
                self.put_bytes += source.numel() * source.element_size()
            return original_put(destination, source, peer, **kwargs)

        def prepare(cache: Any, weights: tuple[torch.Tensor, ...], route: Any) -> Any:
            if not self.enabled:
                cache.clear()
            before = tuple(cache._entries.items())
            epoch = self.active_provider.epoch
            view = readonly_view(self.active_provider, weights, route)
            if tuple(cache._entries.items()) != before or self.active_provider.epoch != epoch:
                raise RuntimeError('Read-only residency query changed provider state')
            copies = original_prepare(cache, weights, route)
            expected = {(fact.target, fact.physical_slot + route.plan.config.home_experts)
                        for fact in view.facts if fact.status == 'valid'}
            if copies.reused != expected:
                raise RuntimeError('Read-only owner evidence disagrees with actual cache decision')
            facts = asdict(view)
            facts['unknown_remote_fact_count'] = sum(fact['status'] == 'unknown' for fact in facts['facts'])
            facts['facts'] = [fact for fact in facts['facts'] if fact['status'] != 'unknown']
            self.rows.append({**self.context, 'kind': 'prepare', 'layer': self.identify(weights),
                              'phase': self.phase, 'enabled': self.enabled, 'view': facts,
                              'control_forced_copy': not self.enabled})
            for edge in route.plan.transfers:
                key = id(self.active_provider), edge.target_rank, edge.target_slot - route.plan.config.home_experts
                writer = {'layer': self.identify(weights), 'owner': edge.owner_rank,
                          'expert': edge.owner_slot, **self.context, 'phase': self.phase}
                previous = self.slot_writers.get(key)
                record = {**self.context, 'kind': 'planned_slot_binding', 'layer': self.identify(weights),
                                  'phase': self.phase, 'provider_id': key[0], 'target': key[1],
                                  'physical_slot': key[2], 'observing_ep_rank': route.rank,
                                  'previous_writer': previous, 'writer': writer,
                                  'other_layer_overwrite': previous is not None
                                  and previous['layer'] != writer['layer'],
                                  'owner_or_expert_overwrite': previous is not None
                                  and (previous['owner'], previous['expert']) != (writer['owner'], writer['expert'])}
                if route.rank in (edge.owner_rank, edge.target_rank):
                    self.rows.append(record)
                self.slot_writers[key] = writer
            return copies

        @contextmanager
        def prefetch(provider: Any, weights: tuple, route: Any, **kwargs: Any) -> Iterator[Any]:
            previous = self.active_provider, self.phase
            self.active_provider = provider
            self.phase = 'backward' if kwargs.get('backward') else ('recompute' if is_recomputing() else 'forward')
            before = self.put_bytes
            try:
                with original_prefetch(provider, weights, route, **kwargs) as view:
                    self.rows.append({**self.context, 'kind': 'matrix_puts', 'layer': self.identify(weights),
                                      'phase': self.phase, 'weight_put_bytes': self.put_bytes - before,
                                      'provider_id': id(provider), 'protocol_epoch': provider.epoch,
                                      'counts': list(provider._last_weight_copy_counts)})
                    yield view
            finally:
                self.active_provider, self.phase = previous

        @contextmanager
        def lease(provider: Any, weights: tuple, route: Any, **kwargs: Any) -> Iterator[Any]:
            before = len(self.rows)
            with original_lease(provider, weights, route, **kwargs) as view:
                if kwargs.get('backward') and len(self.rows) == before:
                    self.rows.append({**self.context, 'kind': 'backward_saved_generation',
                                      'layer': self.identify(weights), 'phase': 'backward',
                                      'weight_put_bytes': 0, 'provider_id': id(provider),
                                      'protocol_epoch': provider.epoch})
                yield view

        with patch.object(_ReplicaWeightCache, 'prepare', prepare), \
                patch.object(SignalReplicaTransport, '_overlapped_prefetch', prefetch), \
                patch.object(SignalReplicaTransport, 'lease', lease), patch.object(shmem, 'put', put), \
                patch.object(_MegaMoeFunction, 'forward', staticmethod(forward)), \
                patch.object(_MegaMoeFunction, 'backward', staticmethod(backward)):
            yield self
