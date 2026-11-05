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
"""MX task namespace, physical-root refresh and projected replica contracts."""

from contextlib import contextmanager
from collections.abc import Iterator
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tests.common.mark_utils import arg_mark

from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_parallel.core.multicore.modules.mega_moe.mxfp8_module import MXFP8MegaMoeExperts, _mx_home_spec
from hyper_parallel.core.multicore.modules.mega_moe.mxfp8_replica import materialize_replica_weights
from hyper_parallel.core.multicore.modules.mega_moe.mxfp8_weights import (
    MXFP8ExpertWeightBinding,
    MXFP8PhysicalWeight,
)
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import TaskType, mega_moe_event_capacity
from hyper_parallel.core.multicore.scheduler.mxfp8 import MX_GMM_TASK_TYPE, MxGmmDescriptor


def _physical(experts: int, rows: int, columns: int) -> MXFP8PhysicalWeight:
    data = torch.arange(experts * rows * columns, dtype=torch.int64).to(torch.uint8)
    return MXFP8PhysicalWeight(
        (experts, rows, columns),
        data.view(torch.float8_e4m3fn).reshape(experts, rows, columns),
        torch.full((experts, rows // 64, columns, 2), 123, dtype=torch.uint8),
        data.flip(0).view(torch.float8_e4m3fn).reshape(experts, rows, columns),
        torch.full((experts, rows, columns // 64, 2), 231, dtype=torch.uint8),
    )


class TestMXFP8Integration(unittest.TestCase):
    """Protect the combined ABI and exact physical payloads without NPU execution."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_mx_task_namespace_does_not_alias_bf16_ingress(self) -> None:
        """Feature: MX task ABI.
        Description: Build EP4/64 B0/B2 task graphs.
        Expectation: Unique MX tasks and FP32 TN outputs fit the workspace.
        """
        self.assertNotIn(MX_GMM_TASK_TYPE, set(TaskType))
        for ep, logical in ((4, 24), (64, 384)):
            for budget in (0, 2):
                with self.subTest(ep=ep, budget=budget):
                    experts = logical + ep * budget
                    spec = MegaMoeSpec(
                        4096,
                        5120,
                        1792,
                        experts,
                        8,
                        1.25,
                        40960,
                        ep,
                        None,
                        0,
                        32,
                        compute_stages=6,
                        mx_vector_policy="window2",
                    )
                    for forward in (True, False):
                        runtime = build_mxfp8_runtime(spec, forward=forward)
                        capacity = mega_moe_event_capacity(experts, ep, compute_stages=6)
                        self.assertLessEqual(runtime.config.event_capacity, capacity)
                        records = [
                            MxGmmDescriptor.from_buffer_copy(runtime.gmm_descriptors, offset)
                            for offset in range(0, len(runtime.gmm_descriptors), 128)
                        ]
                        tn = [record for record in records if record.layout == 2]
                        self.assertEqual(bool(tn), not forward)
                        self.assertTrue(all(record.output_dtype == 1 for record in tn))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_forward_root_is_resolved_again_after_backward_regather(self) -> None:
        """Feature: FSDP current roots.
        Description: Replace physical root axes between phases.
        Expectation: Backward consumes refreshed storage and requires both axes.
        """
        physical = _physical(2, 64, 128)
        root = torch.empty(128, 128, requires_grad=True)
        root.col_data, root.col_scale = physical.col_data, physical.col_scale
        root.row_data = root.row_scale = None
        root.fsdp_forward_direction = "col"
        binding = MXFP8ExpertWeightBinding(root, physical.logical_shape)
        first = binding.resolve("forward")
        self.assertIsNone(first.row_data)
        with self.assertRaisesRegex(ValueError, "missing required row"):
            binding.resolve("backward")
        refreshed = _physical(2, 64, 128)
        root.col_data, root.col_scale = refreshed.col_data, refreshed.col_scale
        root.row_data, root.row_scale = refreshed.row_data, refreshed.row_scale
        backward = binding.resolve("backward")
        self.assertEqual(backward.col_data.data_ptr(), refreshed.col_data.data_ptr())
        self.assertNotEqual(backward.col_data.data_ptr(), first.col_data.data_ptr())
        self.assertEqual(backward.row_scale.data_ptr(), refreshed.row_scale.data_ptr())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_replica_byte_copy_preserves_both_axes_and_scale_bits(self) -> None:
        """Feature: MXFP8 replicas.
        Description: Append exact guest data and scale bytes.
        Expectation: Both axes preserve every byte and inactive slots are zero.
        """
        owner, guest = _physical(2, 64, 128), _physical(1, 64, 128)
        config = ExpertReplicaConfig(4, 2, 2)
        transfer = SimpleNamespace(target_rank=1, target_slot=2)
        route = SimpleNamespace(plan=SimpleNamespace(config=config, transfers=(transfer,)), rank=1, home_only=False)
        for backward in (False, True):
            with self.subTest(backward=backward):
                fields = ("col_data", "col_scale", "row_data", "row_scale") if backward else ("col_data", "col_scale")

                @contextmanager
                def prefetch(aliases: tuple, current_route: object) -> Iterator[SimpleNamespace]:
                    """Provide exact guest payload bytes without a distributed backend."""
                    self.assertIs(current_route, route)
                    values = []
                    for field in fields:
                        payload = getattr(guest, field).view(torch.uint8).reshape(1, -1)
                        values.append(torch.cat((payload, torch.full_like(payload, 77)), dim=0))
                    yield SimpleNamespace(weights=tuple(values))

                path = "hyper_parallel.core.multicore.modules.mega_moe.mxfp8_replica.prefetch_weights"
                with mock.patch(path, side_effect=prefetch):
                    (actual,) = materialize_replica_weights((owner,), route, backward=backward)
                self.assertEqual(actual.logical_shape, (4, 64, 128))
                for field in fields:
                    value = getattr(actual, field).view(torch.uint8)
                    self.assertTrue(torch.equal(value[:2], getattr(owner, field).view(torch.uint8)))
                    self.assertTrue(torch.equal(value[2:3], getattr(guest, field).view(torch.uint8)))
                    self.assertEqual(int(value[3].count_nonzero()), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_constructor_keeps_logical_and_physical_expert_counts_distinct(self) -> None:
        """Feature: MXFP8 B2.
        Description: Construct the EP64 E384 executor.
        Expectation: Six owner roots and eight physical slots remain distinct.
        """
        module = MXFP8MegaMoeExperts(
            local_num_tokens=4096,
            hidden_size=5120,
            intermediate_size=1792,
            num_experts=384,
            top_k=8,
            ep_size=64,
            replica_slots_per_rank=2,
        )
        self.assertEqual(module.local_experts, 6)
        self.assertEqual(module._resource_group.specification["num_experts"], 512)
        self.assertFalse(module._cost_model_enabled)
        self.assertEqual(module.dispatch_split, 512)
        module.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_home_only_replica_plan_borrows_owner_axes(self) -> None:
        """Feature: MXFP8 adaptive B0 path.
        Description: Reject all guest materialization when the planner retains home ownership.
        Expectation: EP64 B2 keeps six compute experts without copying or appending axes.
        """
        spec = MegaMoeSpec(
            4096,
            5120,
            1792,
            512,
            8,
            1.25,
            40960,
            64,
            None,
            0,
            32,
            logical_num_experts=384,
            replica_slots_per_rank=2,
            compute_stages=6,
        )
        home = _mx_home_spec(spec)
        self.assertEqual(home.local_experts, 6)
        self.assertEqual(home.num_experts, 384)
        weights = (_physical(2, 64, 128),)
        route = SimpleNamespace(home_only=True)
        self.assertIs(materialize_replica_weights(weights, route, backward=False), weights)
        self.assertIs(materialize_replica_weights(weights, route, backward=True), weights)
