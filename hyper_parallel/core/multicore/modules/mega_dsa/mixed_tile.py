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
"""Experimental SFA mixed-group schedule and explicit device-evidence validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

_MIXED_MAGIC = 0x48504453414D4958
_MIXED_VERSION = 1
_PHYSICAL_GROUPS = 20
_TRACE_WORDS = 64
_MEMBER_WORDS = 16


@dataclass(frozen=True)
class MixedSfaSchedule:
    """Schedule complete logical SFA partitions on reusable 1 AIC / 2 AIV groups.

    This is an experimental forward-only CP1 probe. Idle physical groups are
    reserved but do not yet run communication progress. The schedule neither
    claims a fused DSA backend nor installs an autograd fallback.

    Args:
        compute_groups: Number of physical groups, from 1 through 19. One
            physical group is always reserved outside the compute team.
        rounds: Repeat the logical partition traversal from 1 through 64.
            Repeats exercise scratch/event reuse and overwrite identical outputs.
    """

    compute_groups: int
    rounds: int = 1

    def __post_init__(self) -> None:
        """Reject configurations that could lose workers or exhaust reserved groups."""
        for name, value, upper in (("compute_groups", self.compute_groups, 19), ("rounds", self.rounds, 64)):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in [1,{upper}]")

    def runtime_config(self, device: torch.device | str) -> torch.Tensor:
        """Create versioned device metadata outside an invocation's hot path."""
        return torch.tensor((_MIXED_MAGIC, _MIXED_VERSION, self.compute_groups, self.rounds),
                            dtype=torch.int64, device=device)

    def new_trace(self, device: torch.device | str) -> torch.Tensor:
        """Allocate zeroed, independent cache lines for all three members of each group."""
        return torch.zeros((_PHYSICAL_GROUPS, _TRACE_WORDS), dtype=torch.int64, device=device)


def validate_mixed_trace(trace: torch.Tensor, schedule: MixedSfaSchedule) -> dict:
    """Validate an explicit CPU snapshot of every member's completed logical tasks.

    Args:
        trace: Explicit CPU int64 snapshot shaped [20,64]. This function never
            transfers device tensors implicitly. Cube and both vectors own
            separate cache lines and independently record count, sum and last ID.
        schedule: The host-declared schedule for this completed invocation.

    Returns:
        Total logical tasks and the number of participating physical groups.

    Raises:
        ValueError: An invalid snapshot, missing or disagreeing member evidence,
            an unexpected reserved-group write or an incorrect final ticket.
    """
    if trace.device.type != "cpu" or trace.dtype != torch.int64 or trace.shape != (20, 64):
        raise ValueError("mixed trace validation requires an explicit CPU int64 [20,64] snapshot")
    values = trace.tolist()
    for group, row in enumerate(values):
        if group >= schedule.compute_groups:
            if any(row):
                raise ValueError(f"reserved physical group {group} wrote compute evidence")
            continue
        tasks = tuple(range(group, _PHYSICAL_GROUPS, schedule.compute_groups))
        expected = (len(tasks) * schedule.rounds, sum(task + 1 for task in tasks) * schedule.rounds, tasks[-1])
        if row[0] != tasks[-1]:
            raise ValueError(f"physical group {group} did not publish the final logical ticket")
        for member in range(3):
            offset = member * _MEMBER_WORDS
            if tuple(row[offset + 1:offset + 4]) != expected:
                raise ValueError(f"physical group {group}, member {member}: "
                                 f"record={tuple(row[offset + 1:offset + 4])}, expected={expected}")
    return {"logical_tasks": _PHYSICAL_GROUPS * schedule.rounds,
            "compute_groups": schedule.compute_groups, "members_per_group": 3}
