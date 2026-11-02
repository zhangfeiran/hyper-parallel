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
"""Small runtime patches that omit only unmapped guest suffix computation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import struct
from typing import Any
import weakref

import torch

from hyper_parallel.core.expert_parallel.hot_replica._metadata import _copy_device_bytes
from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC, TaskType


UNMAPPED_GUEST_SUFFIX_SCHEDULE = 'fixed_queue_unmapped_guest_suffix_v1'


@dataclass(frozen=True)
class GuestTaskPrefix:
    """Queue and trigger patches retaining all physical descriptors and strides."""

    home_experts: int
    active_experts: int
    local_experts: int
    patches: tuple[tuple[int, bytes], ...]
    dropped_cube: int
    dropped_vector: int


def _whole_round_suffixes(queue: list[int], experts: dict[int, int], active: int,
                          workers: int) -> tuple[list[int], list[int]]:
    """Keep worker ownership by deleting only whole rounds from inactive runs."""
    kept, removed = [], []
    position = 0
    while position < len(queue):
        end = position
        while end < len(queue) and experts.get(queue[end], -1) >= active:
            end += 1
        if end == position:
            kept.append(queue[position])
            position += 1
            continue
        count = (end - position) // workers * workers
        removed.extend(queue[position:position + count])
        kept.extend(queue[position + count:end])
        position = end
    return kept, removed


def _encode(values: Sequence[int]) -> bytes:
    """Encode native little-endian int32 arrays without changing their storage size."""
    return struct.pack(f'<{len(values)}i', *values)


def build_guest_task_prefixes(config: RuntimeConfigC, *, home_experts: int) -> tuple[GuestTaskPrefix, ...]:
    """Precompile bounded CPU patches for every possible mapped guest prefix.

    Communication descriptors and their queues remain untouched except for
    positions shifted by complete worker rounds. Partial rounds remain as harmless
    empty tasks so retained work stays on the same physical worker.

    Args:
        config: Final physical runtime, including any independent producer fences.
        home_experts: Number of owned expert slots preceding the guest pool.

    Returns:
        One immutable patch per active prefix, including the complete restoration
        image. Unknown queue geometries retain ordinary execution with no patches.
    """
    local = int(config.dynamic_data.dynamic_group_size)
    if type(home_experts) is not int or not 0 < home_experts < local:
        raise ValueError('home_experts must be positive and smaller than the physical local expert count')
    workers = int(config.num_workers) // 2
    if workers < 1 or config.num_workers % 2 or config.task_index_num[2]:
        return ()
    queues = [list(config.cube_task_indices[:config.task_index_num[0]]),
              list(config.vector_task_indices[:config.task_index_num[1]])]
    experts = {}
    trigger_events = {}
    for task_id in (*queues[0], *queues[1]):
        task = config.all_tasks[task_id]
        if task.task_type not in (TaskType.TASK_GROUPED_MATMUL, TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
            continue
        if task.task_split_num < local or task.task_split_num % local:
            return ()
        experts[task_id] = int(task.task_index) // (int(task.task_split_num) // local)
        trigger_events[task_id] = (int(task.trigger_event),)
        if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1:
            trigger_events[task_id] += (int(task.extra_value_1),)
    layout = type(config)
    prefixes = []
    for active in range(home_experts, local + 1):
        selected = [_whole_round_suffixes(queue, experts, active, workers) for queue in queues]
        reductions = Counter(event for _, removed in selected for task in removed for event in trigger_events[task])
        triggers = list(config.all_event_num_triggers)
        for event, count in reductions.items():
            if not 0 <= event < len(triggers) or triggers[event] < count:
                return ()
            triggers[event] -= count
        counts = list(config.task_index_num)
        counts[:2] = [len(kept) for kept, _ in selected]
        queue_bytes = []
        for original, (kept, _) in zip(queues, selected):
            queue_bytes.append(_encode([*kept, *original[len(kept):]]))
        patches = (
            (layout.all_event_num_triggers.offset, _encode(triggers)),
            (layout.task_index_num.offset, _encode(counts)),
            (layout.cube_task_indices.offset, queue_bytes[0]),
            (layout.vector_task_indices.offset, queue_bytes[1]),
        )
        prefixes.append(GuestTaskPrefix(home_experts, active, local, patches,
                                        len(selected[0][1]), len(selected[1][1])))
    return tuple(prefixes)


def select_guest_task_prefix(prefixes: tuple[GuestTaskPrefix, ...],
                             slot_to_logical: Sequence[int] | None) -> GuestTaskPrefix | None:
    """Retain holes and mapped zero-row guests; remove only a proven unmapped suffix.

    Args:
        prefixes: Physical runtime patches compiled before steady-state execution.
        slot_to_logical: The saved invocation's local physical mapping, or no pruning.

    Returns:
        The smallest prefix containing every mapped slot. The full prefix restores
        a reused runtime when no immutable route mapping is available.
    """
    if not prefixes:
        return None
    full = prefixes[-1]
    if slot_to_logical is None:
        return full
    if len(slot_to_logical) != full.local_experts:
        raise ValueError('Replica mapping does not match the physical runtime expert count')
    if any(type(value) is not int or value < -1 for value in slot_to_logical):
        raise ValueError('Replica physical mapping must contain logical IDs or -1')
    active = full.home_experts
    for slot in range(full.home_experts, full.local_experts):
        if slot_to_logical[slot] >= 0:
            active = slot + 1
    return prefixes[active - full.home_experts]


def apply_guest_task_prefix(runtime: torch.Tensor, prefix: GuestTaskPrefix | None, *,
                            base: torch.Tensor, cache: dict[int, tuple[Any, int]] | None = None) -> torch.Tensor:
    """Update only an exclusively leased runtime image, never the saved plan body.

    Args:
        runtime: Writable split runtime ordered by the existing workspace lease.
        prefix: Saved route's immutable queue variant, including full restoration.
        base: Original runtime identity whose body remains immutable.
        cache: Workspace-local weak identity and prefix of the last applied image.

    Returns:
        The same device image, with small current-stream metadata copies only when
        its body variant changes. No extra device runtime allocation is retained.
    """
    if prefix is None:
        return runtime
    if runtime is base:
        raise ValueError('Guest task patches require an exclusive runtime image')
    previous = None if cache is None else cache.get(id(base))
    if previous is not None and previous[0]() is runtime and previous[1] == prefix.active_experts:
        return runtime
    for offset, payload in prefix.patches:
        if offset < 0 or offset + len(payload) > base.numel():
            raise ValueError('Guest task patch exceeds the physical runtime body')
    # A failed submission can leave a partially rewritten body; never advertise its old prefix.
    if cache is not None:
        cache.pop(id(base), None)
    for offset, payload in prefix.patches:
        _copy_device_bytes(payload, runtime[offset:offset + len(payload)])
    if cache is not None:
        cache[id(base)] = (weakref.ref(runtime), prefix.active_experts)
    return runtime
