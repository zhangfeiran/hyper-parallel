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
"""Real HCCL group mappings and AIV planner residency acceptance."""

from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import
from torch.utils._python_dispatch import TorchDispatchMode

from hyper_parallel.core.expert_parallel.hot_replica import ExpertReplicaConfig, build_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.device import build_device_expert_replica_plan
from hyper_parallel.core.expert_parallel.hot_replica.routing import prepare_replica_route
from hyper_parallel.core.expert_parallel.hot_replica.transport import prefetch_weights, return_gradients


class _NoReadback(TorchDispatchMode):
    """Reject scalar extraction or CPU copies inside device solving."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if "_local_scalar_dense" in name or (name == "aten._to_copy.default" and
                                            str((kwargs or {}).get("device", "")) == "cpu"):
            raise AssertionError(f"Device solver readback: {name}")
        return func(*args, **(kwargs or {}))


def test_device_replica_boundaries_npu() -> None:
    """Compare retained cross-stream AIV outputs with the shared CPU policy."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    retained = []
    for budget in (0, 1, 6, 25):
        config = ExpertReplicaConfig(24, 4, budget)
        for minimum, target in ((0, None), (4096, None), (4096, 10000)):
            counts = [[128 if expert < 2 else 0 for expert in range(24)] for _ in range(4)]
            tensor = torch.tensor(counts, device="npu", dtype=torch.int32)
            stream = streams[len(retained) % 2]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream), _NoReadback():
                plan = build_device_expert_replica_plan(tensor, config, minimum_replica_rows=minimum,
                                                         target_load=target)
            torch.npu.current_stream().wait_stream(stream)
            oracle = build_expert_replica_plan(counts, budget, minimum_replica_rows=minimum, target_load=target)
            retained.append((plan, oracle))
    for plan, oracle in reversed(retained):
        summary = plan.host_summary()
        if summary.slot_to_logical != oracle.slot_to_logical or summary.destination_counts != oracle.destination_counts:
            raise AssertionError("AIV plan differs from CPU policy")
        torch.testing.assert_close(plan.dispatch_counts.cpu(), torch.tensor(oracle.dispatch_counts, dtype=torch.int32))
        with _NoReadback():
            order, _ = plan.source_runs(0)
        logical = plan.slot_to_logical.flatten()
        expected = torch.where(logical >= 0, logical, plan.config.num_experts).float().argsort(stable=True)
        torch.testing.assert_close(order, expected)
        torch.testing.assert_close(plan.source_counts, plan.dispatch_counts[:, expected].to(torch.int64))
    if len({plan.control.data_ptr() for plan, _ in retained}) != len(retained):
        raise AssertionError("Live plans alias their control storage")
    print(f"DEVICE_STREAM_PARITY_OK retained={len(retained)}", flush=True)


def _group_roundtrip(group):
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    config = ExpertReplicaConfig(2 * size, size, 1)
    weights = (torch.full((2, 32, 64), rank + 1, device=device, dtype=torch.bfloat16),
               torch.full((2, 32, 32), rank + 1, device=device, dtype=torch.bfloat16))
    records = []
    for hot in (False, True):
        ids = torch.zeros((128, 1), device=device, dtype=torch.int64) if hot else torch.arange(
            128, device=device).remainder(config.num_experts).reshape(-1, 1)
        route = prepare_replica_route(ids, config, group)
        if bool(route.plan.transfers) != hot:
            raise AssertionError("Expected balanced to hot transfer transition")
        with prefetch_weights(weights, route, backward=True) as pool:
            for transfer in route.plan.transfers:
                if transfer.target_rank == rank:
                    for weight in pool.weights:
                        torch.testing.assert_close(weight[transfer.target_slot - 2],
                                                   torch.full_like(weight[0], transfer.owner_rank + 1))
            for gradient in pool.gradients:
                gradient.fill_(rank + 1)
            home = tuple(torch.ones_like(weight, dtype=torch.float32) for weight in weights)
            actual = return_gradients(home, route, pool.gradients)
        expected = [torch.ones_like(value) for value in actual]
        for transfer in route.plan.transfers:
            if transfer.owner_rank == rank:
                for value in expected:
                    value[transfer.owner_slot].add_(transfer.target_rank + 1)
        for result, oracle in zip(actual, expected):
            torch.testing.assert_close(result, oracle, rtol=0, atol=0)
        records.append((route.plan, actual))
    return records


def test_replica_groups_npu() -> None:
    """Exercise None, WORLD and noncontiguous subgroup weight/gradient traffic."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        implicit, explicit = _group_roundtrip(None), _group_roundtrip(dist.group.WORLD)
        for (left_plan, left), (right_plan, right) in zip(implicit, explicit):
            if left_plan != right_plan:
                raise AssertionError("None and WORLD plans differ")
            for result, oracle in zip(left, right):
                torch.testing.assert_close(result, oracle, rtol=0, atol=0)
        group = dist.new_group([0, 2])
        if dist.get_rank() in (0, 2):
            _group_roundtrip(group)
            dist.destroy_process_group(group)
        dist.barrier()
        print("DEFAULT_AND_SUBGROUP_ROUNDTRIP_OK", flush=True)
    finally:
        dist.destroy_process_group()
