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
"""Host checks for the backward peer-buffer reuse fence and its schedule."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import build_config_for_rank
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import (
    build_multi_wave_backward_graph,
    build_route_backward_graph,
)
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue


_ROOT = Path(__file__).resolve().parents[5]
_WORKER = _ROOT / "hyper_parallel/core/multicore/ops/runtime/worker_kernel.h"


def _method(source: str, name: str) -> str:
    """Extract the actual inline method for host compilation with primitive stubs."""
    start = source.index(f"  __aicore__ inline void {name}(")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


class TestWorkerKernel(unittest.TestCase):
    """Keep source-local completion separate from peer-safe buffer reuse."""

    def test_reuse_fence_runs_after_readers_and_before_remote_writes(self) -> None:
        """The first Combine must fence peers after local gate/dTopK dependencies."""
        body = _method(_WORKER.read_text(encoding="utf-8"), "ExecuteTask")
        local_reader = body.index("WaitForDependency(task_desc.extra_value_2)")
        peer_fence = body.index("SynchronizeCombineReuse(task_desc)")
        remote_write = body.index("static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc)")
        self.assertLess(local_reader, peer_fence)
        self.assertLess(peer_fence, remote_write)

    def test_reuse_fence_keeps_readiness_counters_and_runs_only_once(self) -> None:
        """Compile the real helper on CPU, substituting only device primitives."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required for the inline helper test.")
        method = _method(_WORKER.read_text(encoding="utf-8"), "SynchronizeCombineReuse")
        source = r"""
#include <cassert>
#include <cstdint>
#define __aicore__
constexpr uint32_t EVENT_INVALID_ID = 0xFFFFFFFF;
struct TensorDesc { uint32_t input_position = 0; };
struct TaskDesc { TensorDesc outputs[1]; uint32_t extra_value_2 = EVENT_INVALID_ID; };
struct ReadyHandshakeMeta { uint32_t ready_event = 100; };
struct Harness {
  bool combine_reuse_synchronized_ = false, has_ready_handshake_ = true, multi = false;
  uint32_t wave_index_ = 0, worker_id_ = 1;
  ReadyHandshakeMeta ready_handshake_meta_;
  int publishes = 0, waits = 0;
  bool IsMultiWave() { return multi; }
  void PublishDeviceReady(const ReadyHandshakeMeta &meta) {
    assert(meta.ready_event == 101); ++publishes;
  }
  void WaitForEventCount(uint32_t event, int32_t needed) {
    assert(event == 101 && needed == 1); ++waits;
  }
""" + method + r"""
};
int main() {
  TaskDesc dispatch_reuse;
  Harness leader;
  leader.SynchronizeCombineReuse(dispatch_reuse);
  leader.SynchronizeCombineReuse(dispatch_reuse);
  assert(leader.publishes == 1 && leader.waits == 1);
  assert(leader.ready_handshake_meta_.ready_event == 100);
  Harness follower; follower.worker_id_ = 3;
  follower.SynchronizeCombineReuse(dispatch_reuse);
  assert(follower.publishes == 0 && follower.waits == 1);
  TaskDesc expert_output_reuse; expert_output_reuse.outputs[0].input_position = 13;
  Harness legacy; legacy.SynchronizeCombineReuse(expert_output_reuse);
  assert(legacy.publishes == 0 && legacy.waits == 0);
  expert_output_reuse.extra_value_2 = 90;
  Harness multi; multi.multi = true;
  multi.SynchronizeCombineReuse(expert_output_reuse);
  assert(multi.publishes == 1 && multi.waits == 1);
  Harness later; later.multi = true; later.wave_index_ = 1;
  later.SynchronizeCombineReuse(expert_output_reuse);
  assert(later.publishes == 0 && later.waits == 0);
  Harness local; local.has_ready_handshake_ = false;
  local.SynchronizeCombineReuse(dispatch_reuse);
  assert(local.publishes == 0 && local.waits == 0);
}
"""
        with tempfile.TemporaryDirectory() as directory:
            executable = str(Path(directory) / "reuse_fence")
            subprocess.run(
                [compiler, "-std=c++17", "-x", "c++", "-", "-o", executable],
                input=source, text=True, capture_output=True, check=True,
            )
            subprocess.run([executable], check=True, capture_output=True)

    def test_gate_completion_implies_all_single_wave_dy_readers_finished(self) -> None:
        """Every Cube lane drains act-grad/dW2 before any gate can release Combine."""
        for experts in (96, 112):
            with self.subTest(experts=experts):
                values = TaskSplitValue(tp=1, ep=16, seq_size=4096, all_expert_num=experts, top_k=8)
                graph = build_route_backward_graph(values)
                graph.propagate_splits(values)
                runtime = build_config_for_rank(graph, values, rank_id=13)
                tasks = [runtime.all_tasks[i] for i in runtime.cube_task_indices[: runtime.task_index_num[0]]]
                readers = [i for i, task in enumerate(tasks) if task.outputs[0].input_position in {6, 8}]
                gates = [i for i, task in enumerate(tasks) if task.outputs[0].input_position == 12]
                self.assertLess(max(readers), min(gates))

    def test_reuse_release_event_is_not_an_operator_event(self) -> None:
        """The extra local release counter cannot satisfy an existing graph dependency."""
        for multi in (False, True):
            with self.subTest(multi=multi):
                values = TaskSplitValue(tp=1, ep=16, seq_size=4096, all_expert_num=96, top_k=8)
                builder = build_multi_wave_backward_graph if multi else build_route_backward_graph
                graph = builder(values, integrated_permute=True)
                graph.propagate_splits(values)
                runtime = build_config_for_rank(graph, values, rank_id=13, kernel_multi_wave=multi)
                release_event = runtime.ready_handshake_meta.ready_event + 1
                events = {
                    event for task in runtime.all_tasks[: runtime.task_num]
                    for event in (task.dependent_event, task.trigger_event)
                }
                self.assertNotIn(release_event, events)
                self.assertEqual(runtime.all_event_num_triggers[release_event], 0)
