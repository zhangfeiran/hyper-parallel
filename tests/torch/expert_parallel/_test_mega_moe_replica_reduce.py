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
"""Distributed checks for bounded hot-expert gradient receives, without kernels."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_moe.placement import (
    ExpertPlacementPlan,
    build_constrained_expert_placement,
)
from hyper_parallel.core.multicore.modules.mega_moe.replica import ExpertReplicaManager
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec


def _fan_in_plan(world: int, local_experts: int = 6) -> ExpertPlacementPlan:
    """Replicate every rank-zero expert onto every other rank."""
    counts = ((world,) * local_experts + (0,) * ((world - 1) * local_experts),) * world
    slots = tuple(
        tuple(range(rank * local_experts, (rank + 1) * local_experts))
        + (tuple(range(local_experts)) if rank else (-1,) * local_experts)
        for rank in range(world)
    )
    row = [0] * (world * local_experts * 2)
    for rank in range(world):
        for expert in range(local_experts):
            row[rank * local_experts * 2 + (local_experts if rank else 0) + expert] = 1
    return ExpertPlacementPlan(counts, local_experts, slots, (tuple(row),) * world)


def _spec(plan: ExpertPlacementPlan, rank: int, hidden: int, intermediate: int) -> MegaMoeSpec:
    """Bind the communication manager without allocating a kernel descriptor."""
    return MegaMoeSpec(
        local_num_tokens=128, hidden_size=hidden, intermediate_size=intermediate,
        num_experts=plan.logical_num_experts, top_k=1, expert_capacity_factor=1.5,
        receive_capacity=192, ep_size=plan.ep_size, ep_group=dist.group.WORLD,
        rank_id=rank, num_cube_cores=24, replica_slots_per_rank=plan.replica_slots_per_rank,
    )


def _partial(rank: int, world: int, expert: int, dtype: torch.dtype, step: int) -> float:
    """Create cancellation that detects rounding between receive batches."""
    if expert < 0:
        return 0.0
    if expert == 0 and dtype != torch.float32:
        large = 256 if dtype == torch.bfloat16 else 2048
        return float(large if rank == 0 else -large if rank == world - 1 else 1)
    return float((rank + 1) * (expert + 1) + step)


def _check_reduction(
    plan: ExpertPlacementPlan, device: torch.device, dtype: torch.dtype, window: int, separate: bool,
) -> None:
    """Compare each returned home expert with a logical-expert sum oracle."""
    rank = dist.get_rank()
    manager = ExpertReplicaManager(_spec(plan, rank, 64, 32), gradient_receive_slots=window)
    for step in range(2):
        values = [_partial(rank, plan.ep_size, expert, dtype, step)
                  for expert in plan.slot_to_logical_expert[rank]]
        physical = tuple(
            torch.tensor(values, device=device, dtype=dtype)[:, None, None].expand(-1, *shape).clone()
            for shape in ((64, 64), (32, 64))
        )
        local = plan.local_home_experts
        gradients = tuple(value[:local].clone() for value in physical) if separate else physical
        replicas = tuple(value[local:] for value in physical) if separate else None
        actual = manager.reduce_replica_gradients(plan, *gradients, replica_gradients=replicas)
        expected = [
            sum(float(torch.tensor(_partial(peer, plan.ep_size, expert, dtype, step), dtype=dtype))
                for peer, slots in enumerate(plan.slot_to_logical_expert) if expert in slots)
            for expert in range(rank * local, (rank + 1) * local)
        ]
        for home in actual:
            reference = torch.tensor(expected, dtype=dtype)[:, None, None].expand(home.shape)
            torch.testing.assert_close(home.cpu(), reference, atol=0, rtol=0)
        # A rank with no incoming copies must still finish every outgoing send
        # before its replica storage can be reused by the next backward.
        for tensor in physical:
            tensor.zero_()
        dist.barrier()


def test_replica_gradient_receive_windows() -> None:
    """Verify many-to-one, chained and empty plans with real P2P traffic."""
    backend = os.getenv("HP_MEGA_MOE_REDUCE_BACKEND", "hccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{local_rank}" if backend == "hccl" else "cpu")
    if device.type == "npu":
        torch.npu.set_device(device)
    dist.init_process_group(backend, timeout=timedelta(seconds=180))
    try:
        world = dist.get_world_size()
        if world < 3:
            raise ValueError("gradient-window regression needs at least three ranks.")
        dist.all_reduce(torch.ones(1, device=device))
        fan_in = _fan_in_plan(world)
        row = tuple(value for count in ([0, 3, 3] + [2] * (world - 3)) for value in (count, 0))
        chain = build_constrained_expert_placement((row,) * world, 1)
        empty = build_constrained_expert_placement(((1,) * (world * 2),) * world, 1)
        for plan in (fan_in, chain, empty):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                for window in (1, 2, 4, 8):
                    for separate in (False, True):
                        _check_reduction(plan, device, dtype, window, separate)
        if device.type == "npu":
            with torch.npu.stream(torch.npu.Stream(device=device)):
                _check_reduction(fan_in, device, torch.float32, 2, True)
            torch.npu.synchronize()
        if dist.get_rank() == 0:
            print(f"REPLICA_GRADIENT_WINDOWS_PASS backend={backend} ranks={world}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_replica_gradient_receive_windows()
