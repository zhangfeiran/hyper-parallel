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
"""Storage lifetimes for routed MegaMoe, excluding unspecified model state.

This is a named-allocation estimate. The NPU allocator's reserved high-water
mark and driver/HCCL allocations cannot be inferred from tensor shapes alone.
They are separate explicit budgets; SHMEM views never add to the heap twice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .cost import Options, aligned
from .model import Training
from .workload import Shape

MIB = 2**20
GIB = 2**30


@dataclass(frozen=True)
class Storage:
    """One unique storage, alive on [begin, end); aliases share its name."""

    name: str
    size: int
    begin: int
    end: int
    scope: str = "allocator"


def peak(storages: list[Storage]) -> dict[str, Any]:
    """Compute a live-storage peak, releasing before allocation at equal times."""
    if any(item.size < 0 or item.begin >= item.end for item in storages):
        raise ValueError("storage sizes must be nonnegative and lifetimes nonempty")
    if len({item.name for item in storages}) != len(storages):
        raise ValueError("storage names must be unique; express aliases through a single lifetime")
    best = {"bytes": 0, "time": 0, "live": []}
    for time in sorted({item.begin for item in storages}):
        live = [item for item in storages if item.begin <= time < item.end]
        size = sum(item.size for item in live)
        if size > best["bytes"]:
            best = {
                "bytes": size,
                "time": time,
                "live": [asdict(item) for item in live],
            }
    return best


def receive_widths(plan: Any, cap: int) -> list[int]:
    """Mirror peer-rounded FP32 windows, returning each owner's live receives."""
    owners = [[] for _ in range(plan.ep_size)]
    for transfer in plan.replica_transfers:
        owners[transfer.owner_rank].append(transfer.target_rank)
    widths = []
    for peers in owners:
        distinct = len(set(peers))
        width = cap - cap % distinct if 0 < distinct <= cap else cap
        widths.append(min(len(peers), width))
    return widths


def memory_ledger(
    shape: Shape, plan: Any, training: Training, options: Options, *, configured_slots: int | None = None
) -> dict[str, Any]:
    """Predict warmed phase peaks for default forward-permute/backward-both.

    FP32-main AdamW stores BF16 weights plus FP32 masters/moments. BF16 AdamW
    stores moments in parameter dtype. Checkpoint boundaries contain one hidden
    tensor per layer. Additional caller outputs, route networks, attention and allocator
    slack must be included in external_gib/reserve_gib by a full-model caller.
    """
    ranks, layers = plan.ep_size, training.layers
    slots, home = plan.replica_slots_per_rank, shape.experts // ranks
    configured_slots = slots if configured_slots is None else configured_slots
    allocated_slots = max(slots, options.previous_replica_slots)
    rows = shape.tokens_per_rank * shape.top_k
    maximum = max(plan.destination_loads)
    multi = maximum > shape.capacity
    rolling = shape.capacity if multi else aligned(maximum, 128)
    retained = maximum if multi else rolling
    elements = sum(shape.weight_elements)
    home_elements = home * elements
    grad_bytes = 4 if slots else shape.element_bytes
    replicas = layers if training.pool == "per-layer" else 1
    event_bytes = (
        4096
        + (ranks + 1) * 64
        + 2 * (shape.experts + ranks * configured_slots) * aligned(shape.tokens_per_rank, 128) // 128 * 64
    )
    heap = aligned((shape.capacity + rows) * shape.row_bytes + 2 * event_bytes + 4 * 511, 64 * MIB)
    scratch = (home + slots) * elements * grad_bytes if multi and not options.atomic_dw else 0
    scratch = max(scratch, options.previous_scratch_bytes)
    persistent = {
        "none": shape.element_bytes,
        "bf16-adamw": 3 * shape.element_bytes,
        "fp32-main-adamw": shape.element_bytes + 12,
    }[options.optimizer]
    end = 10 * layers + 10
    optimizer_time = 8 * layers + 1
    shared = [
        Storage("audited_static_plan_tensors", options.plan_storage_bytes, 0, end),
        Storage("shmem_heap_including_views", heap, 0, end, "external_shmem"),
        Storage("gmm_swiglu_and_grow_only_dw_scratch", 48 * MIB + 4096 + scratch, 0, end),
        Storage(
            "home_parameters_and_warmed_optimizer_states",
            layers * home_elements * persistent,
            0,
            end,
        ),
        Storage(
            "replica_weight_and_fp32_gradient_pool",
            replicas * allocated_slots * elements * (shape.element_bytes + 4),
            0,
            end,
        ),
    ]
    if options.external_gib:
        shared.append(
            Storage(
                "caller_supplied_other_live_hbm",
                int(options.external_gib * GIB),
                0,
                end,
                "external",
            )
        )
    shared.append(Storage("upstream_gradient", shape.tokens_per_rank * shape.row_bytes, 0, optimizer_time))
    for layer in range(layers):
        fwd = layer * 2 + 1
        bwd = 2 * layers + (layers - 1 - layer) * 6 + 1
        shared.append(
            Storage(
                f"layer{layer}/checkpoint_input",
                shape.tokens_per_rank * shape.row_bytes,
                0,
                bwd + 5,
            )
        )
        saved_start = bwd if training.checkpoint else fwd
        saved = retained * 2 * shape.intermediate * shape.element_bytes
        saved += rolling * shape.intermediate * shape.element_bytes + rows * shape.row_bytes
        if not multi and not options.inplace_backward:
            saved += rolling * shape.row_bytes
        shared.append(
            Storage(
                f"layer{layer}/saved_up_activation_output_and_input",
                saved,
                saved_start,
                bwd + 5,
            )
        )
        # Original checkpoint forward still allocates a complete live forward set.
        if training.checkpoint:
            forward_bytes = retained * 2 * shape.intermediate * shape.element_bytes
            forward_bytes += rolling * (shape.intermediate * shape.element_bytes + shape.row_bytes)
            forward_bytes += rows * shape.row_bytes
            shared.append(Storage(f"layer{layer}/original_forward", forward_bytes, fwd, fwd + 1))
        # Recomputed down-projection is dead before backward when input is replayed.
        if multi or options.inplace_backward:
            shared.append(
                Storage(
                    f"layer{layer}/recompute_down_proj",
                    rolling * shape.row_bytes,
                    saved_start,
                    saved_start + 1,
                )
            )
        intermediates = rolling * 3 * shape.intermediate * shape.element_bytes
        if not options.inplace_backward:
            intermediates += rolling * shape.row_bytes
        shared.append(Storage(f"layer{layer}/backward_intermediates", intermediates, bwd + 1, bwd + 5))
        parameter_grad_end = optimizer_time + 2 * layer + 1
        # With matching dtype, AccumulateGrad can own the returned storage.
        # FP32 replica reduction requires a separate cast for BF16 parameters.
        if grad_bytes == shape.element_bytes:
            shared.append(
                Storage(
                    f"layer{layer}/home_gradient_and_parameter_grad",
                    home_elements * grad_bytes,
                    bwd + 1,
                    parameter_grad_end,
                )
            )
        else:
            shared.append(Storage(f"layer{layer}/kernel_home_gradient", home_elements * grad_bytes, bwd + 1, bwd + 5))
            shared.append(
                Storage(
                    f"layer{layer}/parameter_gradient", home_elements * shape.element_bytes, bwd + 4, parameter_grad_end
                )
            )
        shared.append(Storage(f"layer{layer}/caller_output", shape.tokens_per_rank * shape.row_bytes, fwd, bwd + 5))
        shared.append(
            Storage(f"layer{layer}/input_gradient", shape.tokens_per_rank * shape.row_bytes, bwd + 3, optimizer_time)
        )
        if options.optimizer == "fp32-main-adamw":
            shared.append(
                Storage(f"layer{layer}/fp32_main_gradient", home_elements * 4, optimizer_time + 2 * layer, end)
            )
    per_rank = []
    largest = {"bytes": -1}
    widths = receive_widths(plan, options.gradient_receive_slots)
    peaks_by_width = {}
    for rank in range(ranks):
        width = widths[rank]
        # All other lifetimes use the same global capacities. Owners with the
        # same receive width therefore have identical live-storage schedules.
        if width not in peaks_by_width:
            storages = list(shared)
            for layer in range(layers):
                bwd = 2 * layers + (layers - 1 - layer) * 6 + 1
                storages.append(
                    Storage(
                        f"layer{layer}/owner_gradient_receive_window",
                        width * elements * grad_bytes,
                        bwd + 2,
                        bwd + 4,
                    )
                )
            peaks_by_width[width] = peak(storages)
        result = peaks_by_width[width]
        per_rank.append(
            {
                "rank": rank,
                "peak_bytes": result["bytes"],
                "phase_time": result["time"],
                "receive_bytes": widths[rank] * elements * grad_bytes,
            }
        )
        if result["bytes"] > largest["bytes"]:
            largest = {**result, "rank": rank}
    named = largest["bytes"]
    budget = int(options.device_gib * GIB)
    reserve = int(options.reserve_gib * GIB)
    return {
        "scope": "named routed-MoE storages; default permute/both; warmed optimizer; shared workspace",
        "named_peak_bytes": named,
        "headroom_after_reserve_bytes": budget - named - reserve,
        "fits_budget_with_reserve": named + reserve <= budget,
        "reserve_bytes": reserve,
        "shmem_heap_bytes": heap,
        "retained_rows": retained,
        "rolling_rows": rolling,
        "max_owner_receive_bytes": max(row["receive_bytes"] for row in per_rank),
        "peak_live_storages": largest,
        "per_rank": per_rank,
        "unmodeled": [
            "allocator reserved/fragmentation",
            "driver and HCCL",
            "unlisted native op scratch",
            "attention/router/shared experts/embeddings",
            "additional caller outputs or retained graphs",
        ]
        + ([] if options.plan_storage_bytes else ["serialized task configs and tiling tensors; supply a plan audit"]),
    }
