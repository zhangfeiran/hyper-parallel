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
"""Compile the device credit arithmetic and check its actual task partitions."""

from collections import defaultdict
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import TaskType


_ROOT = Path(__file__).resolve().parents[4]


class TestEmptyTailCredit(unittest.TestCase):
    """Retain all payload completions while compressing only empty contributions."""

    def test_all_counts_and_partial_tiles_preserve_the_completion_threshold(self) -> None:
        """Withhold any active completion after crediting every empty tile first."""
        source = r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include "runtime/empty_tail_credit.h"
int main() {
  for (int64_t tiles : {1, 2, 8, 32, 256}) {
    for (int64_t hidden : {1, 5120}) {
      const int64_t tile_elements = 128 * hidden;
      for (int64_t rows = 0; rows <= tiles * 128; ++rows) {
        int32_t signals = 0;
        int32_t empty_credits = 0;
        int32_t active = 0;
        int32_t empty_signals = 0;
        for (int64_t index = 0; index < tiles; ++index) {
          if (index * 128 < rows) {
            ++active;
            ++signals;
          } else {
            const int32_t credit = MulticoreRuntime::EmptyTailCompletionCredit(
                index, tiles, rows * hidden, tile_elements);
            assert(credit >= 0 && credit <= tiles);
            signals += credit;
            empty_credits += credit;
            empty_signals += credit != 0;
          }
        }
        assert(signals == tiles);
        assert(empty_signals == (active < tiles ? 1 : 0));
        for (int32_t completed = 0; completed < active; ++completed) {
          assert(completed + empty_credits < tiles);
        }
        assert(active + empty_credits == tiles);
      }
    }
  }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "empty-tail-credit"
            built = subprocess.run(
                ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 f"-I{_ROOT / 'hyper_parallel/core/multicore/ops'}", "-x", "c++", "-", "-o", str(executable)],
                input=source, capture_output=True, text=True, check=False,
            )
            self.assertEqual(built.returncode, 0, f"expected=0 actual={built.returncode}: {built.stderr}")
            result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")

    def test_real_forward_backward_descriptors_partition_each_signal_exactly_once(self) -> None:
        """EP4 and EP64 share one event and size across each complete tile partition."""
        for ep, experts in ((4, 32), (64, 384)):
            spec = MegaMoeSpec(
                local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                num_experts=experts, top_k=8, receive_capacity=32768, initial_capacity_factor=None,
                ep_size=ep, ep_group=None, rank_id=0, num_cube_cores=32,
                dispatch_mode="pull", forward_ingress_mode="indexed",
            )
            with patch.dict("os.environ", {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
                                         "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "32"}):
                _, forward, _, backward = _build_runtime_artifacts(spec)
            for direction, config in (("forward", forward), ("backward", backward)):
                with self.subTest(ep=ep, direction=direction):
                    groups = defaultdict(list)
                    for index in range(config.task_num):
                        task = config.all_tasks[index]
                        if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                            tiles = task.task_split_num // experts
                            groups[task.task_index // tiles].append(task)
                    self.assertEqual(len(groups), experts, f"actual={len(groups)} expected={experts}")
                    for tasks in groups.values():
                        tiles = tasks[0].task_split_num // experts
                        indices = sorted(task.task_index % tiles for task in tasks)
                        self.assertEqual(indices, list(range(tiles)), f"actual={indices} expected={list(range(tiles))}")
                        signatures = {
                            (task.trigger_event, task.inputs[3].input_position, task.inputs[3].base_ptr_offset,
                             task.inputs[3].data_type, task.task_split_value)
                            for task in tasks
                        }
                        self.assertEqual(len(signatures), 1, f"actual={len(signatures)} expected=1")
