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
"""
Runtime configuration: topology parameters, C++ ABI structs, and constants.

Combines:
- TaskSplitValue / init_task_split_value  — TP/EP/seq topology + per-rank counters
- ctypes.Structure mirrors of runtime_head.hpp — passed as binary blobs to C++ kernels
"""
import ctypes
import os
import struct
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

# ── Constants (match runtime_head.hpp exactly) ────────────────────────────────
MAX_TENSOR_DIMS      = 4
MAX_INPUTS_PER_TASK  = 4
MAX_OUTPUTS_PER_TASK = 4

MAX_TASK_NUM         = 256 * 100
MAX_EVENT_NUM        = 1024
PHASE_BARRIER_EVENT  = MAX_EVENT_NUM - 2
NUM_WORKERS_VECTOR   = 48
NUM_WORKERS_CUBE     = 24
QUEUE_CAPACITY       = 100
TASK_TYPE_INDEX_NUM  = 256 * 100
MAX_GROUP_LIST       = 512
ATOMIC_ADD_VALUE_LEN = 8
READY_HANDSHAKE_MAGIC   = 0x4D4D5231
READY_HANDSHAKE_VERSION = 1
READY_HANDSHAKE_COPY_ACK_VERSION = 2
READY_HANDSHAKE_DESTINATION_FENCE_VERSION = 3
EVENT_COUNTER_BYTES     = MAX_EVENT_NUM * ctypes.sizeof(ctypes.c_int32)
READY_CACHE_LINE_BYTES  = 64
ROUTE_ARRIVAL_STRIDE_BYTES = READY_CACHE_LINE_BYTES
ROUTE_COPY_ACK_FLAG = 1 << 24
ROUTE_ARRIVAL_WAIT_FLAG = 1 << 25
ROUTE_COMPUTE_ACK_FLAG = 1 << 26
ROUTE_EXPERT_ACK_FLAG = 1 << 27
RETAINED_DW_SCRATCH_FLAG = 1 << 28
ATOMIC_DW_ACCUMULATION_FLAG = 1 << 29
RUNTIME_TASK_CAPACITY_MAGIC = 0x80000000
RUNTIME_TASK_CAPACITY_MASK = RUNTIME_TASK_CAPACITY_MAGIC - 1
RUNTIME_TASK_CAPACITY_ALIGNMENT = 16
RUNTIME_EXTENDED_MAGIC = 0x40000000
RUNTIME_EXTENDED_CAPACITY_MASK = RUNTIME_EXTENDED_MAGIC - 1
RUNTIME_FORMAT_VERSION = 2
COMPACT_TASK_DESC_SIZE = 128


def compact_runtime_enabled() -> bool:
    """Select the validated tensor-table format for managed online plans."""
    name = "HP_MEGA_MOE_COMPACT_RUNTIME"
    value = os.getenv(name, "1")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}.")
    return value == "1"


# ── Enums ─────────────────────────────────────────────────────────────────────
class TaskAiCoreType(IntEnum):
    TASK_AICORE_INVALID = 0
    TASK_AICORE_CUBE    = 1
    TASK_AICORE_VECTOR  = 2
    TASK_AICORE_MIX     = 3


class TaskType(IntEnum):
    TASK_TERMINATE            = 0
    TASK_BEGIN_TASK_GRAPH     = 10
    TASK_ADD_CUSTOM           = 101
    TASK_SWI_GLU              = 102
    TASK_MATMUL               = 103
    TASK_GROUPED_MATMUL       = 104
    TASK_SHMEM_PUT_MEM_SIGNAL = 105
    TASK_SWI_GLU_GRAD         = 106
    TASK_UNPERMUTE_REDUCE_GRAD = 110
    TASK_TOPK_WEIGHT_GRAD     = 111
    TASK_WEIGHTED_UNPERMUTE   = 112
    TASK_LOCAL_COPY           = 113
    TASK_LOCAL_ACCUMULATE     = 114
    TASK_PHASE_BARRIER        = 115


class EventType(IntEnum):
    EVENT_EMPTY                  = 900
    EVENT_LAUNCH_TASKS           = 901
    EVENT_LAUNCH_MASSIVE_TASKS   = 902
    EVENT_LAUNCH_DEPENDENT_TASKS = 903
    EVENT_END_OF_TASK_GRAPH      = 910
    EVENT_TERMINATION            = 911
    EVENT_INVALID                = 999


class DynamicType(IntEnum):
    DYNAMIC_EMPTY        = 0
    DYNAMIC_DSV3_MOE = 101


# ── ctypes Structures (mirror runtime_head.hpp) ───────────────────────────────

class TensorDescC(ctypes.Structure):
    _fields_ = [
        ("tensor_type",     ctypes.c_uint32),
        ("num_dims",        ctypes.c_uint32),
        ("dim",             ctypes.c_uint32 * MAX_TENSOR_DIMS),
        ("stride",          ctypes.c_uint32 * MAX_TENSOR_DIMS),
        ("data_type",       ctypes.c_uint32),
        ("input_position",  ctypes.c_uint32),
        ("base_ptr_offset", ctypes.c_uint32),
        ("transpose_flag",  ctypes.c_uint32),
        ("dynamic_shape",   ctypes.c_uint32),
        ("dynamic_dim",     ctypes.c_uint32),
    ]


class TaskDescC(ctypes.Structure):
    _fields_ = [
        ("task_type",            ctypes.c_uint32),
        ("task_aicore_type",     ctypes.c_uint32),
        ("num_inputs",           ctypes.c_uint32),
        ("num_outputs",          ctypes.c_uint32),
        ("trigger_event",        ctypes.c_uint32),
        ("dependent_event",      ctypes.c_uint32),
        ("inputs",               TensorDescC * MAX_INPUTS_PER_TASK),
        ("outputs",              TensorDescC * MAX_OUTPUTS_PER_TASK),
        ("tiling_data_position", ctypes.c_uint32),
        ("tiling_data_offset",   ctypes.c_uint32),
        ("task_index",           ctypes.c_uint32),
        ("task_split_num",       ctypes.c_uint32),
        ("task_split_value",     ctypes.c_uint32),
        ("extra_value_0",        ctypes.c_uint32),
        ("extra_value_1",        ctypes.c_uint32),
        ("extra_value_2",        ctypes.c_uint32),
        ("extra_value_3",        ctypes.c_uint32),
        ("extra_value_4",        ctypes.c_uint32),
    ]


class EventDescC(ctypes.Structure):
    _fields_ = [
        ("event_type",    ctypes.c_uint32),
        ("num_triggers",  ctypes.c_uint32),
        ("first_task_id", ctypes.c_uint32),
        ("last_task_id",  ctypes.c_uint32),
    ]


class DynamicDataC(ctypes.Structure):
    _fields_ = [
        ("dynamic_type",           ctypes.c_uint32),
        ("dynamic_input_position", ctypes.c_uint32),
        ("dynamic_group_size",     ctypes.c_uint32),
        ("dynamic_max_seq_len",    ctypes.c_int32),
    ]


class ReadyHandshakeMetaC(ctypes.Structure):
    """Device metadata for the cross-rank workspace-ready handshake."""

    _fields_ = [
        ("magic",       ctypes.c_uint32),
        ("version",     ctypes.c_uint32),
        ("ready_event", ctypes.c_uint32),
    ]


class RuntimeConfigC(ctypes.Structure):
    # Keep the full logical-capacity mirror while graph builders populate it.
    # serialize_runtime_config() removes unused task and task-index tails before
    # the online runtime copies the descriptor to device HBM.
    _fields_ = [
        ("task_num",                  ctypes.c_uint32),
        ("num_workers",               ctypes.c_uint32),
        ("queue_capacity",            ctypes.c_uint32),
        ("config_extra_value",        ctypes.c_uint32),
        ("all_event_num_triggers",    ctypes.c_int32   * MAX_EVENT_NUM),
        ("all_tasks",                 TaskDescC         * MAX_TASK_NUM),
        ("all_events",                EventDescC        * MAX_EVENT_NUM),
        ("task_index_num",            ctypes.c_int32   * 4),
        ("cube_task_indices",         ctypes.c_int32   * TASK_TYPE_INDEX_NUM),
        ("vector_task_indices",       ctypes.c_int32   * TASK_TYPE_INDEX_NUM),
        ("mix_task_indices",          ctypes.c_int32   * TASK_TYPE_INDEX_NUM),
        ("dynamic_data",              DynamicDataC),
        ("grouped_matmul_group_list", ctypes.c_int64   * MAX_GROUP_LIST),
        ("atomic_add_values",         ctypes.c_int32   * ATOMIC_ADD_VALUE_LEN),
        ("ready_handshake_meta",      ReadyHandshakeMetaC),
    ]


TENSOR_DESC_SIZE = ctypes.sizeof(TensorDescC)
TASK_DESC_SIZE = ctypes.sizeof(TaskDescC)
assert TENSOR_DESC_SIZE == 64
assert TASK_DESC_SIZE == 576

RUNTIME_CONFIG_FIXED_BYTES = (
    ctypes.sizeof(RuntimeConfigC)
    - MAX_TASK_NUM * (TASK_DESC_SIZE + 3 * ctypes.sizeof(ctypes.c_int32))
)
RUNTIME_CONFIG_BYTES_PER_TASK_CAPACITY = (
    TASK_DESC_SIZE + 3 * ctypes.sizeof(ctypes.c_int32)
)


def mega_moe_event_capacity(num_experts: int, ep_size: int) -> int:
    """Reserve all backward replay events and a separate barrier cache line.

    The largest graph uses ``2 * E + 9 * local_E + 5`` ordinary events,
    followed by readiness/route events up to that value plus five. Phase and
    wave barriers occupy the final cache line, isolated from op atomics.
    ``num_experts`` includes physical replica slots.
    """
    if ep_size < 1 or num_experts < 1 or num_experts % ep_size:
        raise ValueError("expert count must be positive and divisible by EP size")
    ordinary = 2 * num_experts + 9 * (num_experts // ep_size) + 11
    return max(MAX_EVENT_NUM, (ordinary + 15) // 16 * 16 + 16)


@lru_cache(maxsize=16)
def _runtime_config_type(task_capacity: int, event_capacity: int) -> type:
    """Keep the legacy host mirror for small graphs; grow only when needed."""
    if task_capacity <= MAX_TASK_NUM and event_capacity == MAX_EVENT_NUM:
        return RuntimeConfigC
    replacements = {
        "all_event_num_triggers": ctypes.c_int32 * event_capacity,
        "all_tasks": TaskDescC * task_capacity,
        "all_events": EventDescC * event_capacity,
        "cube_task_indices": ctypes.c_int32 * task_capacity,
        "vector_task_indices": ctypes.c_int32 * task_capacity,
        "mix_task_indices": ctypes.c_int32 * task_capacity,
    }
    return type(
        "SizedRuntimeConfigC", (ctypes.Structure,),
        {"_fields_": [(name, replacements.get(name, kind)) for name, kind in RuntimeConfigC._fields_]},
    )


def allocate_runtime_config(task_capacity: int, event_capacity: int = MAX_EVENT_NUM) -> RuntimeConfigC:
    """Allocate a host graph image with sufficient task, queue and event slots."""
    if task_capacity < 0 or event_capacity < MAX_EVENT_NUM or event_capacity % 16:
        raise ValueError("invalid runtime allocation capacity")
    capacity = (task_capacity + 15) // 16 * 16
    # Device byte offsets are uint32, even for the expanded dense reference ABI.
    runtime_config_serialized_size(capacity, event_capacity=event_capacity)
    return _runtime_config_type(capacity, event_capacity)()


def runtime_config_task_capacity(cfg: RuntimeConfigC) -> int:
    """Return the aligned wire capacity required by populated task queues.

    Args:
        cfg: Populated runtime configuration.

    Returns:
        Aligned number of serialized task slots.
    """
    counts = [int(cfg.task_index_num[index]) for index in range(3)]
    queues = (cfg.cube_task_indices, cfg.vector_task_indices, cfg.mix_task_indices)
    for count, indices in zip(counts, queues):
        if count < 0 or count > len(indices):
            raise ValueError(f"invalid runtime task-index count {count}")

    maximum_task_id = -1
    for indices, count in zip(
        queues,
        counts,
    ):
        for index in range(count):
            task_id = int(indices[index])
            if task_id < 0 or task_id >= len(cfg.all_tasks):
                raise ValueError(f"invalid runtime task id {task_id}")
            maximum_task_id = max(maximum_task_id, task_id)

    required = max(int(cfg.task_num), *counts, maximum_task_id + 1)
    if required > len(cfg.all_tasks):
        raise ValueError(
            f"runtime requires {required} task slots, allocated {len(cfg.all_tasks)}"
        )
    alignment = RUNTIME_TASK_CAPACITY_ALIGNMENT
    return (required + alignment - 1) // alignment * alignment


def runtime_config_serialized_size(
    task_capacity: int, *, event_capacity: int = MAX_EVENT_NUM,
    compact: bool = False, tensor_count: int = 0,
) -> int:
    """Return the dynamic wire size for an aligned full task capacity.

    Args:
        task_capacity: Aligned number of serialized task slots.
        event_capacity: Cache-line-aligned event slots, shared by all plans.
        compact: Whether tasks reference an interned tensor table.
        tensor_count: Number of 64-byte tensor templates in the table.

    Returns:
        Runtime descriptor size in bytes.
    """
    if (
        task_capacity < 0
        or task_capacity > RUNTIME_EXTENDED_CAPACITY_MASK
        or task_capacity % RUNTIME_TASK_CAPACITY_ALIGNMENT
        or event_capacity < MAX_EVENT_NUM or event_capacity % 16
        or tensor_count < 0 or (tensor_count and not compact)
    ):
        raise ValueError(f"invalid runtime task capacity {task_capacity}")
    extended = compact or event_capacity != MAX_EVENT_NUM or task_capacity > MAX_TASK_NUM
    size = (
        RUNTIME_CONFIG_FIXED_BYTES
        + (event_capacity - MAX_EVENT_NUM) * 20
        + (48 if extended else 0)
        + task_capacity * ((COMPACT_TASK_DESC_SIZE if compact else TASK_DESC_SIZE) + 12)
    )
    if extended:
        size = (size + 63) // 64 * 64
    size += tensor_count * TENSOR_DESC_SIZE
    if size >= 1 << 32:
        raise ValueError("runtime descriptor exceeds uint32 device byte offsets")
    return size


def _compact_tasks(raw: bytes, capacity: int) -> tuple[bytes, bytes]:
    """Intern tensors without their per-tile base offset; retain every field."""
    tasks = bytearray(capacity * COMPACT_TASK_DESC_SIZE)
    templates = {}
    for index in range(capacity):
        source = index * TASK_DESC_SIZE
        target = index * COMPACT_TASK_DESC_SIZE
        tasks[target:target + 24] = raw[source:source + 24]
        tasks[target + 88:target + 128] = raw[source + 536:source + 576]
        for tensor_index in range(8):
            tensor_start = source + 24 + tensor_index * TENSOR_DESC_SIZE
            tensor = raw[tensor_start:tensor_start + TENSOR_DESC_SIZE]
            normalized = tensor[:48] + bytes(4) + tensor[52:]
            template_id = templates.setdefault(normalized, len(templates))
            offset = struct.unpack_from("<I", tensor, 48)[0]
            struct.pack_into("<II", tasks, target + 24 + tensor_index * 8, template_id, offset)
    return bytes(tasks), b"".join(templates)


def serialize_runtime_config(cfg: RuntimeConfigC, *, compact: bool = False) -> bytes:
    """Serialize only populated task and task-index prefixes.

    Bit 31 marks dynamic task capacity; bit 30 extends the header to 64 bytes
    with event capacity, tensor count, task stride and format version. Small dense
    images retain the previous wire ABI. Compact records reference a tensor
    table after the aligned metadata tail; split-weight pointers follow it.

    Args:
        cfg: Populated runtime configuration.
        compact: Share tensor templates between 128-byte task records.

    Returns:
        Compact runtime descriptor bytes.
    """
    capacity = runtime_config_task_capacity(cfg)
    raw = bytes(cfg)
    layout = type(cfg)
    event_capacity = len(cfg.all_event_num_triggers)
    extended = compact or event_capacity != MAX_EVENT_NUM or capacity > MAX_TASK_NUM
    task_bytes = raw[layout.all_tasks.offset:layout.all_tasks.offset + capacity * TASK_DESC_SIZE]
    templates = b""
    if compact:
        task_bytes, templates = _compact_tasks(task_bytes, capacity)
    tensor_count = len(templates) // TENSOR_DESC_SIZE
    prefix = bytearray(raw[:16])
    encoded_capacity = RUNTIME_TASK_CAPACITY_MAGIC | capacity
    if extended:
        encoded_capacity |= RUNTIME_EXTENDED_MAGIC
    queue_offset = layout.queue_capacity.offset
    prefix[queue_offset:queue_offset + 4] = encoded_capacity.to_bytes(4, "little")
    if extended:
        prefix.extend(struct.pack(
            "<4I", event_capacity, tensor_count,
            COMPACT_TASK_DESC_SIZE if compact else TASK_DESC_SIZE, RUNTIME_FORMAT_VERSION,
        ))
        prefix.extend(bytes(32))
    prefix.extend(raw[layout.all_event_num_triggers.offset:layout.all_tasks.offset])

    int32_size = ctypes.sizeof(ctypes.c_int32)
    data = b"".join(
        (
            bytes(prefix),
            task_bytes,
            raw[layout.all_events.offset:layout.task_index_num.offset],
            raw[layout.task_index_num.offset:layout.cube_task_indices.offset],
            raw[
                layout.cube_task_indices.offset:
                layout.cube_task_indices.offset + capacity * int32_size
            ],
            raw[
                layout.vector_task_indices.offset:
                layout.vector_task_indices.offset + capacity * int32_size
            ],
            raw[
                layout.mix_task_indices.offset:
                layout.mix_task_indices.offset + capacity * int32_size
            ],
            raw[layout.dynamic_data.offset:],
        )
    )
    if extended:
        data += bytes((-len(data)) % 64) + templates
    expected = runtime_config_serialized_size(
        capacity, event_capacity=event_capacity, compact=compact, tensor_count=tensor_count,
    )
    if len(data) != expected:
        raise RuntimeError(
            f"runtime config serialization produced {len(data)} bytes, expected {expected}"
        )
    return data


class TilingDataC(ctypes.Structure):
    """Tiling data for add_custom ops (not GMM)."""
    _fields_ = [
        ("smallCoreDataNum",     ctypes.c_uint64),
        ("bigCoreDataNum",       ctypes.c_uint64),
        ("ubPartDataNum",        ctypes.c_uint64),
        ("smallCoreTailDataNum", ctypes.c_uint64),
        ("bigCoreTailDataNum",   ctypes.c_uint64),
        ("smallCoreLoopNum",     ctypes.c_uint64),
        ("bigCoreLoopNum",       ctypes.c_uint64),
        ("tailBlockNum",         ctypes.c_uint64),
    ]


class SwiGluTilingDataC(ctypes.Structure):
    _fields_ = [
        ("is32BAligned",         ctypes.c_uint32),
        ("isDoubleBuffer",       ctypes.c_uint32),
        ("rowLen",               ctypes.c_uint64),
        ("colLen",               ctypes.c_uint64),
        ("baseRowLen",           ctypes.c_uint32),
        ("baseColLen",           ctypes.c_uint32),
        ("activateLeft",         ctypes.c_uint32),
        ("biasIsEmpty",          ctypes.c_uint32),
        ("quantScaleIsEmpty",    ctypes.c_uint32),
        ("activateScaleIsEmpty", ctypes.c_uint32),
        ("swiColLen",            ctypes.c_uint64),
        ("perRowLen",            ctypes.c_uint64),
        ("modRowLen",            ctypes.c_uint64),
        ("usedCoreNum",          ctypes.c_uint32),
    ]


# ── Topology + runtime counters ───────────────────────────────────────────────

@dataclass
class TaskSplitValue:
    """
    Hardware topology parameters + per-rank runtime counters.

    Topology fields are user-supplied; derived properties compute
    sequence/expert partition sizes.  Runtime counters are reset by
    init_task_split_value() before each rank's fill loop.
    """
    # ── User inputs ───────────────────────────────────────────────────────────
    tp:             int = 4
    ep:             int = 4
    seq_size:       int = 8192
    all_expert_num: int = 32
    top_k:          int = 8
    extra_event_num: int = 0
    event_capacity: int = 0

    @property
    def runtime_event_capacity(self) -> int:
        """Return the common event layout for forward and both backward graphs."""
        return max(self.event_capacity, mega_moe_event_capacity(self.all_expert_num, self.ep))

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def single_rank_expert_num(self) -> int:
        """Number of experts assigned to a single rank."""
        return self.all_expert_num // self.ep

    @property
    def seq_all(self) -> int:
        """Total sequence length after accounting for EP and TP parallelism."""
        return (self.seq_size * self.ep * self.top_k) // self.tp

    @property
    def per_expert_seq(self) -> int:
        """Sequence length per expert (across all EP ranks)."""
        return self.seq_all // self.top_k

    @property
    def per_rank_seq(self) -> int:
        """Sequence length assigned to each rank."""
        return self.seq_all // self.ep

    @property
    def per_expert_seq_to_other(self) -> int:
        """Sequence length per expert when sending to other ranks."""
        return self.seq_all // (self.ep * self.top_k)

    @property
    def all_event_num(self) -> int:
        """Total number of events needed for the schedule."""
        e = self.single_rank_expert_num
        return 1 + self.all_expert_num + e + e + e + self.extra_event_num

    # ── Runtime counters (reset by init_task_split_value per rank) ────────────
    rank_id:              int = 0
    pre_pre_event_num:    int = 0
    pre_event_num:        int = 0
    pre_task_num:         int = 0
    pre_cube_task_num:    int = 0
    pre_vector_task_num:  int = 0
    pre_mix_task_num:     int = 0


def init_task_split_value(tsv: TaskSplitValue) -> None:
    """Reset per-rank runtime counters to zero.

    Args:
        tsv: Mutable scheduler counters to reset.
    """
    tsv.pre_pre_event_num   = 0
    tsv.pre_event_num       = 0
    tsv.pre_task_num        = 0
    tsv.pre_cube_task_num   = 0
    tsv.pre_vector_task_num = 0
    tsv.pre_mix_task_num    = 0


def configure_ready_handshake(
    cfg: RuntimeConfigC,
    tsv: TaskSplitValue,
    *,
    version: int = READY_HANDSHAKE_VERSION,
) -> None:
    """Enable the device-side workspace-ready fence for multi-rank EP.

    Args:
        cfg: Runtime configuration receiving handshake metadata.
        tsv: Scheduler dimensions used to allocate the ready event.
        version: Ready-handshake protocol version selected by the graph.
    """
    if tsv.ep == 1:
        return
    ready_event = tsv.all_event_num + 3
    if ready_event + 2 >= len(cfg.all_event_num_triggers) - 16:
        raise ValueError("ready/route events overlap the reserved barrier cache line.")
    cfg.ready_handshake_meta.magic = READY_HANDSHAKE_MAGIC
    if version not in {
        READY_HANDSHAKE_VERSION,
        READY_HANDSHAKE_COPY_ACK_VERSION,
        READY_HANDSHAKE_DESTINATION_FENCE_VERSION,
    }:
        raise ValueError(f"unsupported ready handshake version {version}.")
    cfg.ready_handshake_meta.version = version
    cfg.ready_handshake_meta.ready_event = ready_event
    cfg.all_event_num_triggers[ready_event] = 1
    cfg.config_extra_value = READY_HANDSHAKE_MAGIC


def event_workspace_bytes(ep_size: int, event_capacity: int = MAX_EVENT_NUM) -> int:
    """Return event-counter plus persistent ready-state storage.

    Args:
        ep_size: Number of ranks in the expert-parallel group.
        event_capacity: Number of event counters in the runtime descriptor.

    Returns:
        Required event workspace size in bytes.
    """
    if ep_size <= 1:
        return event_capacity * 4
    return event_capacity * 4 + (ep_size + 1) * READY_CACHE_LINE_BYTES
