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
"""Bounded real-device metadata ownership checks, separate from performance."""

from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import torch

from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts, get_recompute_state
from hyper_parallel.core.expert_parallel.hot_replica import _metadata, routing
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel


class MetadataLedger:
    """Retain scalar facts only, without extending descriptor or graph lifetimes."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    @contextmanager
    def observe(self) -> Iterator[None]:
        """Count real uploads, cache hits and misses outside any performance claim."""
        lookup = _metadata._RecomputedRouteMetadata.lookup
        remember = _metadata._RecomputedRouteMetadata.remember
        upload = routing._async_device_bytes

        def find(cache: Any, plan: Any, rank: int, device: torch.device, kind: str) -> Any:
            result = lookup(cache, plan, rank, device, kind)
            self.rows.append(dict(kind='lookup', hit=result is not None, plan_identity=id(plan),
                                  rank=rank, descriptor_kind=kind, device=str(device)))
            return result

        def save(cache: Any, plan: Any, rank: int, device: torch.device, kind: str, values: tuple) -> None:
            remember(cache, plan, rank, device, kind, values)
            self.rows.append(dict(kind='remember', plan_identity=id(plan), rank=rank, descriptor_kind=kind,
                                  bytes=sum(value.numel() * value.element_size() for value in values)))

        def copy(data: bytes, device: torch.device) -> torch.Tensor:
            result = upload(data, device)
            self.rows.append(dict(kind='upload', bytes=len(data), device=str(device)))
            return result

        with (patch.object(_metadata._RecomputedRouteMetadata, 'lookup', find),
              patch.object(_metadata._RecomputedRouteMetadata, 'remember', save),
              patch.object(routing, '_async_device_bytes', copy)):
            yield


def device_metadata_gate(device: torch.device) -> dict[str, Any]:
    """Exercise actual NPU streams, guard misses and cleanup with owned integer views."""
    counts = ((2, 1, 1, 0), (2, 1, 0, 1))
    config, group = ExpertReplicaConfig(4, 2, 1), object()
    options = dict(target_load=4, capacity_limit=8, adaptive=True)
    ids_forward = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    ids_replay = torch.tensor([[2, 0], [1, 0]], dtype=torch.int64)
    if not torch.equal(torch.bincount(ids_forward.flatten(), minlength=4),
                       torch.bincount(ids_replay.flatten(), minlength=4)):
        raise ValueError('The token-order probe must preserve the exact histogram')
    source, consumer = torch.npu.Stream(), torch.npu.Stream()
    ledger = MetadataLedger()
    facts: dict[str, Any] = dict(nondefault_streams=True, same_counts_new_ids=False,
                                 registry_cleared=False, changed_counts_miss=False, changed_policy_miss=False,
                                 synthetic_cost_rejected_home=False)
    with patch.object(_metadata, '_metadata_reuse_enabled', return_value=True), ledger.observe():
        forward, replay = create_recompute_contexts()
        with torch.npu.stream(source), forward:
            plan, _ = routing._plan_replica_route(counts, config, group, 0, **options)
            original = routing._upload_route_metadata(plan, 0, device)
            actual_forward = routing._remap_replica_ids(ids_forward.to(device), plan, *original[:2])
            cache = get_recompute_state().get_resource(_metadata._RecomputedRouteMetadata, lambda: None)
        cpu_metadata = routing._upload_route_metadata(plan, 0, torch.device('cpu'))
        expected_forward = routing._remap_replica_ids(ids_forward, plan, *cpu_metadata[:2])
        expected_replay = routing._remap_replica_ids(ids_replay, plan, *cpu_metadata[:2])
        if torch.equal(expected_forward, expected_replay):
            raise ValueError('The token-order probe lacks a distinguishable remap counterexample')
        try:
            with torch.npu.stream(consumer), replay:
                same, reused = routing._plan_replica_route(counts, config, group, 0, **options)
                current = routing._upload_route_metadata(same, 0, device)
                if not reused or any(left is not right for left, right in zip(original, current)):
                    raise RuntimeError('Exact device metadata replay did not reuse owned views')
                actual_replay = routing._remap_replica_ids(ids_replay.to(device), same, *current[:2])
                raise RuntimeError('Bounded metadata cleanup probe')
        except RuntimeError as error:
            if str(error) != 'Bounded metadata cleanup probe':
                raise
        if cache.entries:
            raise RuntimeError('Interrupted replay retained its registry entries')
        del original, current
        pressure = torch.empty(16 * 1024 * 1024, device=device, dtype=torch.uint8).fill_(17)
        torch.npu.synchronize()
        observed_forward, observed_replay = actual_forward.cpu(), actual_replay.cpu()
        if (not torch.equal(observed_forward, expected_forward)
                or not torch.equal(observed_replay, expected_replay)
                or torch.equal(observed_forward, observed_replay)):
            raise RuntimeError('Cross-stream remap did not use the current per-token IDs')
        del pressure
        facts.update(same_counts_new_ids=True, registry_cleared=True)
        facts['token_order_counterexample'] = dict(ids_forward=ids_forward.tolist(), ids_replay=ids_replay.tolist(),
            expected_forward=expected_forward.tolist(), expected_replay=expected_replay.tolist(),
            observed_forward=observed_forward.tolist(), observed_replay=observed_replay.tolist())
        for change in ('counts', 'policy'):
            forward, replay = create_recompute_contexts()
            with forward:
                first, _ = routing._plan_replica_route(counts, config, group, 0, **options)
                first_metadata = routing._upload_route_metadata(first, 0, device)
            with replay:
                changed_counts = ((1, 2, 1, 0), counts[1]) if change == 'counts' else counts
                changed_options = options | ({'target_load': 3} if change == 'policy' else {})
                second, reused = routing._plan_replica_route(changed_counts, config, group, 0, **changed_options)
                second_metadata = routing._upload_route_metadata(second, 0, device)
                if reused or second_metadata[2] is first_metadata[2]:
                    raise RuntimeError('Changed inputs incorrectly reused device metadata')
            facts[f'changed_{change}_miss'] = True
        model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
            intermediate_size=8, ep_size=2, transport='p2p', forward_ms=((0, 0), (32, 32)),
            backward_ms=((0, 0), (32, 32)), weight_ms=1000000., gradient_ms=1000000.)
        forward, replay = create_recompute_contexts()
        producer = torch.tensor(counts, device=device, dtype=torch.int32)
        with forward:
            home, _ = routing._plan_replica_route(counts, config, group, 0, cost_model=model, **options)
            if home.transfers:
                raise RuntimeError('The explicit synthetic expensive-copy control did not reject copies')
            routing._home_route_counts(home, 0, producer)
        producer.zero_()
        with replay:
            same, reused = routing._plan_replica_route(counts, config, group, 0, cost_model=model, **options)
            owned = routing._home_route_counts(same, 0, producer)
            if not reused or owned.cpu().tolist() != [list(row) for row in counts]:
                raise RuntimeError('Compact-home metadata aliased the overwritten producer')
        facts['synthetic_cost_rejected_home'] = True
    torch.npu.synchronize()
    facts['observations'] = ledger.rows
    return facts
