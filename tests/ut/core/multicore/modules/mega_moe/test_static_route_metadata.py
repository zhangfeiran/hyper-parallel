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
"""Check exact CPU descriptors on the fixed-token path used by MXFP8 replicas."""

from dataclasses import replace
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hyper_parallel.core.expert_parallel.hot_replica import routing
from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.whole import build_whole_expert_plan
from hyper_parallel.core.multicore.modules.mega_moe import route as route_module
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from tests.common.mark_utils import arg_mark


FIELDS = ('dispatch_src_off', 'dispatch_target_off', 'dispatch_size', 'combine_src_off',
          'combine_target_off', 'combine_size', 'group_list')


def _make_spec(ep: int, rank: int, width: int, mode: str) -> MegaMoeSpec:
    """Bind only metadata geometry, with the fixed-token MX routing contract."""
    return MegaMoeSpec(local_num_tokens=4, hidden_size=4, intermediate_size=2,
                       num_experts=ep * width, top_k=2, initial_capacity_factor=None,
                       receive_capacity=10000, ep_size=ep, ep_group=None, rank_id=rank,
                       num_cube_cores=32, dispatch_mode=mode)


class TestStaticReplicaMetadata(unittest.TestCase):
    """Compare CPU packing with independent Torch prefixes and real complete plans."""

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_numpy_descriptors_match_every_rank_for_ragged_and_empty_sources(self) -> None:
        """Feature: Exact native descriptor packing.
        Description: Compare all ranks, both transports, zero rows and several guest-width geometries.
        Expectation: Every INT64 offset, INT32 size and group prefix matches the original Torch calculation.
        """
        generator = random.Random(71)
        for ep in (1, 2, 4, 16):
            for width in (2, 8):
                values = [[generator.randrange(20) for _ in range(ep * width)] for _ in range(ep)]
                values[0] = [0] * (ep * width)
                counts = torch.tensor(values, dtype=torch.int32)
                for mode in ('push', 'pull'):
                    for rank in range(ep):
                        spec = _make_spec(ep, rank, width, mode)
                        start = rank * width
                        expected = route_module._compute_route_metadata(
                            counts[rank], counts, counts[:, start:start + width], spec, 10000)
                        actual = route_module._numpy_route_metadata(np.array(values, dtype=np.int32),
                                                                   spec, 10000, 'cpu')
                        with self.subTest(ep=ep, width=width, mode=mode, rank=rank):
                            for name in FIELDS:
                                torch.testing.assert_close(getattr(actual, name), getattr(expected, name),
                                                           rtol=0, atol=0)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_fixed_complete_plan_reaches_host_packing_without_device_prefixes(self) -> None:
        """Feature: Fixed-token MX replica reachability.
        Description: Prepare both ranks from a real complete-K plan with automatic tokens disabled.
        Expectation: The actual public route skips gathering, device prefix construction and quota readbacks.
        """
        logical = ((4, 4, 0, 0),) * 2
        config = ExpertReplicaConfig(4, 2, 1)
        plan = build_whole_expert_plan(logical, config, capacity_limit=100, minimum_load_reduction=.4)
        self.assertTrue(plan.transfers)
        counts = torch.tensor(plan.dispatch_counts, dtype=torch.int32)
        ids = torch.tensor([[0, 1]] * 4, dtype=torch.int32)
        hidden = torch.ones(4, 4)
        for mode in ('push', 'pull'):
            for rank in range(2):
                spec = _make_spec(2, rank, config.slots_per_rank, mode)
                self.assertFalse(spec.dynamic_tokens)
                begin = rank * config.slots_per_rank
                expected = route_module._compute_route_metadata(
                    counts[rank], counts, counts[:, begin:begin + config.slots_per_rank], spec,
                    max(1, plan.destination_loads[rank]))
                slots, lengths, _ = routing._upload_route_metadata(plan, rank, torch.device('cpu'))
                physical = routing._remap_replica_ids(ids.to(torch.int64), plan, slots, lengths).to(ids.dtype)
                replica = routing.ReplicaRoute(plan, physical, counts, rank, None)
                permuted = hidden.repeat_interleave(2, dim=0), torch.arange(8, dtype=torch.int32)
                with (self.subTest(mode=mode, rank=rank),
                      patch.object(route_module, '_compute_route_metadata', side_effect=AssertionError('device scan')),
                      patch.object(route_module, '_finish_count_gather', side_effect=AssertionError('device slice')),
                      patch.object(route_module, '_start_count_gather', side_effect=AssertionError('extra gather')),
                      patch.object(route_module, '_permute_topk_input', return_value=permuted),
                      patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('extra count D2H'))):
                    workspace = None
                    if mode == 'pull':
                        # Pull ownership stays outside the descriptor conversion.
                        workspace = SimpleNamespace(in_use=True, source_buffer=permuted[0])
                    with patch.object(route_module, '_permute_topk_input_out', return_value=permuted):
                        actual = route_module.prepare_topk_route(
                            hidden, physical, torch.ones(4, 2), spec, counts[rank], workspace, replica)
                for name in FIELDS:
                    torch.testing.assert_close(getattr(actual.metadata, name), getattr(expected, name), rtol=0, atol=0)
                torch.testing.assert_close(actual.received_counts, counts[:, begin:begin + config.slots_per_rank])
                self.assertIs(actual.metadata.replica_route, replica)
                self.assertIsNone(actual.metadata.empty_experts)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_packed_views_own_independent_storage_when_a_later_plan_changes(self) -> None:
        """Feature: Retained graph descriptor lifetime.
        Description: Pack different source-locality plans while retaining all views from the first graph.
        Expectation: The first metadata remains unchanged and the changed source intervals get new storage.
        """
        config = ExpertReplicaConfig(4, 2, 1)
        first = build_whole_expert_plan(((4, 4, 0, 0),) * 2, config,
                                        capacity_limit=100, minimum_load_reduction=.4)
        second = build_whole_expert_plan(((3, 4, 0, 0), (5, 4, 0, 0)), config,
                                         capacity_limit=100, minimum_load_reduction=.4)
        spec = _make_spec(2, 0, config.slots_per_rank, 'push')
        routes = [routing.ReplicaRoute(plan, torch.empty(0, 2, dtype=torch.int32),
                                      torch.tensor(plan.dispatch_counts, dtype=torch.int32), 0, None)
                  for plan in (first, second)]
        before = route_module._cpu_plan_route_metadata(routes[0], spec, 100, 'cpu')
        saved = {name: getattr(before, name).clone() for name in FIELDS}
        after = route_module._cpu_plan_route_metadata(routes[1], spec, 100, 'cpu')
        for name in FIELDS:
            torch.testing.assert_close(getattr(before, name), saved[name], rtol=0, atol=0)
            self.assertNotEqual(getattr(before, name).data_ptr(), getattr(after, name).data_ptr())
        self.assertFalse(torch.equal(before.dispatch_size, after.dispatch_size))
        self.assertIs(first._dispatch_counts_bytes, first._dispatch_counts_bytes)
        self.assertNotEqual(first._dispatch_counts_bytes, second._dispatch_counts_bytes)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_home_fallback_skips_lazy_counts_and_mismatched_graph_is_rejected(self) -> None:
        """Feature: Home fallback and topology guards.
        Description: Pass a lazy home-only plan and an incompatible physical execution shape.
        Expectation: Home counts are never read; an active plan fails before upload if graph geometry changed.
        """
        config = ExpertReplicaConfig(4, 2, 1)
        lazy = routing._HomeOnlyPlan(config, object(), (8, 0))
        home = routing.ReplicaRoute(lazy, torch.empty(0, 2, dtype=torch.int32), torch.empty(0), 0, None,
                                    home_only=True)
        spec = _make_spec(2, 0, 3, 'push')
        with patch.object(route_module, '_upload_route_tensor', side_effect=AssertionError('unexpected upload')):
            self.assertIsNone(route_module._cpu_plan_route_metadata(home, spec, 100, 'cpu'))
            self.assertIsNone(route_module._cpu_plan_route_metadata(None, spec, 100, 'cpu'))
            plan = build_whole_expert_plan(((4, 4, 0, 0),) * 2, config,
                                           capacity_limit=100, minimum_load_reduction=.4)
            active = routing.ReplicaRoute(plan, torch.empty(0, 2, dtype=torch.int32), torch.empty(0), 0, None)
            with self.assertRaisesRegex(ValueError, 'topology'):
                route_module._cpu_plan_route_metadata(active, replace(spec, num_experts=8), 100, 'cpu')

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_peer_row_counts_and_gradient_initialization_contract_are_preserved(self) -> None:
        """Feature: Optional ingress metadata preservation.
        Description: Use both peer ingress modes with a complete plan containing an empty home slot.
        Expectation: Per-expert row counts match Torch and empty-gradient policy remains unchanged.
        """
        config = ExpertReplicaConfig(4, 2, 1)
        plan = build_whole_expert_plan(((4, 4, 0, 0),) * 2, config,
                                       capacity_limit=100, minimum_load_reduction=.4)
        counts = torch.tensor(plan.dispatch_counts, dtype=torch.int32)
        replica = routing.ReplicaRoute(plan, torch.empty(0, 2, dtype=torch.int32), counts, 0, None)
        for mode in ('peer_overlap', 'peer_token'):
            spec = replace(_make_spec(2, 0, 3, 'push'), forward_ingress_mode=mode)
            actual = route_module._cpu_plan_route_metadata(replica, spec, 100, 'cpu')
            torch.testing.assert_close(actual.expert_row_counts, counts[:, :3].sum(0, dtype=torch.int32))
            self.assertIsNone(actual.empty_experts)
