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
"""Binary contracts between host serialization and the device's actual decoder."""

import ctypes
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import (
    build_multi_wave_backward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank as build_forward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_route_forward_graph,
)
from hyper_parallel.core.multicore.scheduler.config import (
    RuntimeConfigC,
    TaskDescC,
    TaskSplitValue,
    allocate_runtime_config,
    mega_moe_event_capacity,
    runtime_config_serialized_size,
    runtime_config_task_capacity,
    serialize_runtime_config,
)

_BRIDGE = r"""
#include <cstdint>
#include <stdexcept>
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t *;
namespace AscendC {
enum class HardEvent { DUMMY };
inline void Trap() { throw std::runtime_error("invalid runtime format"); }
template <HardEvent event> void SetFlag(uint32_t) {}
template <HardEvent event> void WaitFlag(uint32_t) {}
}
struct Pipe { uint32_t FetchEventID(AscendC::HardEvent) { return 0; } };
inline Pipe *GetTPipePtr() { static Pipe pipe; return &pipe; }
#include "runtime_config.hpp"
static_assert(sizeof(TaskDesc) == 576);
static_assert(sizeof(TensorDesc) == 64);
extern "C" int validate(uint8_t *wire) {
  try { validateRuntimeFormat(wire); return 0; }
  catch (const std::runtime_error &) { return -1; }
}
extern "C" void decode(uint8_t *wire, TaskDesc *tasks, uint32_t count) {
  for (uint32_t index = 0; index < count; ++index) getTaskDesc(wire, tasks + index, index);
}
extern "C" void offsets(uint8_t *wire, uint32_t *values) {
  values[0] = getRuntimeTaskCapacity(wire);
  values[1] = getRuntimeEventCapacity(wire);
  values[2] = getAllEventNumTriggersOffset(wire);
  values[3] = getAllEventsOffset(wire);
  values[4] = getTaskIndexNumOffset(wire);
  values[5] = getCubeTaskIndexsOffset(wire);
  values[6] = getVectorTaskIndexsOffset(wire);
  values[7] = getMixTaskIndexsOffset(wire);
  values[8] = getDynamicDataOffset(wire);
  values[9] = getReadyHandshakeMetaOffset(wire);
  values[10] = getRuntimeTensorTableOffset(wire);
  values[11] = getRuntimeTensorCount(wire);
}
extern "C" void replicas(uint8_t *wire, uint64_t *values) {
  GM_ADDR bases[4];
  getReplicaMatrixBases(wire, bases);
  for (uint32_t index = 0; index < 4; ++index) values[index] = reinterpret_cast<uint64_t>(bases[index]);
}
"""


class TestRuntimeSerialization(unittest.TestCase):
    """Exercise the shipping C++ decoder without CANN or device allocation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compile and load the decoder before releasing temporary build files."""
        compiler = shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("host C++ compiler required for binary ABI verification")
        with tempfile.TemporaryDirectory(prefix="megamoe-runtime-abi-") as temporary:
            directory = Path(temporary)
            source = directory / "bridge.cpp"
            source.write_text(_BRIDGE)
            root = Path(__file__).resolve().parents[5]
            library = directory / "bridge.so"
            subprocess.run(
                [compiler, "-std=c++17", "-O2", "-shared", "-fPIC", str(source),
                 "-I", str(root / "hyper_parallel/core/multicore/ops/runtime"), "-o", str(library)],
                check=True, capture_output=True, text=True,
            )
            cls.bridge = ctypes.CDLL(str(library))
        cls.bridge.decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        cls.bridge.offsets.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls.bridge.replicas.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls.bridge.validate.argtypes = [ctypes.c_void_p]
        cls.bridge.validate.restype = ctypes.c_int

    def assert_wire_matches(self, config: RuntimeConfigC, *, compact: bool) -> bytes:
        """Compare every tensor/task word, all queue entries and metadata tails."""
        wire = serialize_runtime_config(config, compact=compact)
        capacity = runtime_config_task_capacity(config)
        buffer = ctypes.create_string_buffer(wire)
        decoded = (TaskDescC * capacity)()
        self.assertEqual(self.bridge.validate(buffer), 0)
        self.bridge.decode(buffer, decoded, capacity)
        self.assertEqual(bytes(decoded), bytes(config.all_tasks)[:capacity * 576])
        offsets = (ctypes.c_uint32 * 12)()
        self.bridge.offsets(buffer, offsets)
        self.assertEqual((offsets[0], offsets[1]), (capacity, len(config.all_events)))
        for field, offset in (
            (config.all_event_num_triggers, offsets[2]), (config.all_events, offsets[3]),
            (config.task_index_num, offsets[4]), (config.dynamic_data, offsets[8]),
            (config.ready_handshake_meta, offsets[9]),
        ):
            self.assertEqual(wire[offset:offset + len(bytes(field))], bytes(field))
        for field, offset in zip(
            (config.cube_task_indices, config.vector_task_indices, config.mix_task_indices), offsets[5:8],
        ):
            self.assertEqual(wire[offset:offset + capacity * 4], bytes(field)[:capacity * 4])
        self.assertEqual(offsets[10] + offsets[11] * 64, len(wire))
        pointers = (0x102030405060, 0xFFFFFFFF12345678, 0, 0x9876543210)
        trailer = struct.pack("<II4Q", 0x53505754, 1, *pointers)
        decoded_pointers = (ctypes.c_uint64 * 4)()
        self.bridge.replicas(ctypes.create_string_buffer(wire + trailer), decoded_pointers)
        self.assertEqual(tuple(decoded_pointers), pointers)
        return wire

    def test_compact_and_legacy_preserve_all_descriptor_fields(self) -> None:
        """
        Feature: tensor-table runtime ABI
        Description: round trip every field, varying offsets and unused tensor slots
        Expectation: the shipping C++ decoder reproduces all 576 task bytes exactly
        """
        config = RuntimeConfigC()
        config.task_num = 32
        config.task_index_num[0] = 33
        for index in range(33):
            config.cube_task_indices[index] = 32 - index
            task = TaskDescC.from_buffer_copy(struct.pack("<144I", *range(144)))
            for tensor in (*task.inputs, *task.outputs):
                tensor.base_ptr_offset = (index * 12345 + tensor.base_ptr_offset) % (1 << 32)
            config.all_tasks[index] = task
        config.all_tasks[32].extra_value_4 = 0xFFFFFFFF
        dense = self.assert_wire_matches(config, compact=False)
        compact = self.assert_wire_matches(config, compact=True)
        self.assertLess(len(compact), len(dense))
        self.assertEqual(len(dense), runtime_config_serialized_size(48))

    def test_unmarked_fixed_capacity_image_remains_readable(self) -> None:
        """
        Feature: persisted legacy runtime ABI
        Description: decode a full unmarked host structure without the new serializer
        Expectation: original task offsets and fixed array capacities remain valid
        """
        config = RuntimeConfigC()
        config.queue_capacity = 100
        config.all_tasks[1].extra_value_4 = 0xFEDCBA98
        decoded = (TaskDescC * 2)()
        self.bridge.decode(ctypes.create_string_buffer(bytes(config)), decoded, 2)
        self.assertEqual(bytes(decoded), bytes(config.all_tasks)[:1152])

    def test_expanded_tasks_events_and_terminal_reference(self) -> None:
        """
        Feature: graph-sized allocation
        Description: reference a termination task above the old 25600 bound
        Expectation: expanded dense and compact images preserve queues and the new event layout
        """
        config = allocate_runtime_config(25633, 1168)
        config.task_num = 25632
        config.task_index_num[1] = 1
        config.vector_task_indices[0] = 25632
        config.all_tasks[25632].extra_value_0 = 123
        config.all_event_num_triggers[1100] = 73
        config.ready_handshake_meta.ready_event = 1100
        for compact in (False, True):
            with self.subTest(compact=compact):
                self.assert_wire_matches(config, compact=compact)

    def test_real_e384_graphs_decode_identically(self) -> None:
        """
        Feature: E384 forward and multi-wave backward descriptors
        Description: build more than 25600 tasks and more than 1024 events
        Expectation: compact decoding preserves real task order and all event dependencies
        """
        for ep, tokens, backward, optimized in (
            (64, 4096, False, False), (8, 512, True, False), (8, 512, True, True),
        ):
            with self.subTest(ep=ep, backward=backward, optimized=optimized):
                values = TaskSplitValue(tp=1, ep=ep, seq_size=tokens, all_expert_num=384)
                kwargs = {"hidden_size": 512, "intermediate_size": 128}
                if backward:
                    graph = build_multi_wave_backward_graph(
                        values, integrated_permute=optimized, inplace_wave_backward=optimized,
                        tile_phase_ack=optimized, atomic_dw_accumulation=optimized,
                        destination_wave_fence=optimized, **kwargs,
                    )
                else:
                    graph = build_route_forward_graph(values, **kwargs)
                graph.propagate_splits(values)
                if backward:
                    config = build_backward_config(graph, values, ep - 1, kernel_multi_wave=True)
                    self.assertGreater(len(config.all_events), 1024)
                else:
                    config = build_forward_config(graph, values, ep - 1)
                    self.assertGreater(config.task_num, 25600)
                self.assert_wire_matches(config, compact=True)

    def test_capacity_validation_and_barrier_separation(self) -> None:
        """
        Feature: descriptor address and event capacity bounds
        Description: reject invalid counts, offsets and incompatible wire versions
        Expectation: invalid layouts fail before execution and barrier cache lines remain isolated
        """
        for ep in (8, 16, 32, 64):
            for slots in (0, 1, 3):
                experts = 384 + ep * slots
                highest_event = 2 * experts + 9 * (experts // ep) + 10
                self.assertLess(highest_event, mega_moe_event_capacity(experts, ep) - 16)
        for capacity in (-1, 17, 1 << 28):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                runtime_config_serialized_size(capacity)
        config = RuntimeConfigC()
        config.task_index_num[0] = 1
        config.cube_task_indices[0] = -1
        with self.assertRaises(ValueError):
            serialize_runtime_config(config, compact=True)
        wire = bytearray(serialize_runtime_config(RuntimeConfigC(), compact=True))
        struct.pack_into("<I", wire, 28, 100)
        self.assertEqual(self.bridge.validate(ctypes.create_string_buffer(bytes(wire))), -1)
