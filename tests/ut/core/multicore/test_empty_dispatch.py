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
"""Validate descriptor scope and receiver-side readiness of empty Push credits."""

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from hyper_parallel.core.multicore.modules.mega_moe.empty_dispatch import EMPTY_DISPATCH_MARKER
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import TaskType


class TestEmptyDispatch(unittest.TestCase):
    """Only owner-only Push dispatch participates in the experiment."""

    def test_only_dispatch_tasks_change_with_rotated_peer_queues(self) -> None:
        """RATR must retain every source-to-destination tile, dependency and trigger threshold."""
        for mode, slots in (('push', 0), ('push', 2), ('pull', 0)):
            for rank in (0, 3):
                with self.subTest(mode=mode, slots=slots, rank=rank):
                    spec = MegaMoeSpec(
                        local_num_tokens=512, hidden_size=5120, intermediate_size=1792,
                        num_experts=24 + 4 * slots, top_k=8, initial_capacity_factor=1.5,
                        receive_capacity=6144, ep_size=4, ep_group=None, rank_id=rank,
                        num_cube_cores=24, logical_num_experts=24, replica_slots_per_rank=slots,
                        dispatch_mode=mode, dispatch_split=128, combine_split=512,
                    )
                    _, fwd, _, bwd = _build_runtime_artifacts(spec)
                    _, new_fwd, _, new_bwd = _build_runtime_artifacts(replace(spec, _batch_empty_dispatch=True))
                    for original, candidate in ((fwd, new_fwd), (bwd, new_bwd)):
                        grouped: dict[int, list[int]] = {}
                        for task_id in candidate.vector_task_indices[:candidate.task_index_num[1]]:
                            task = candidate.all_tasks[task_id]
                            if (mode == 'push' and not slots and task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
                                    and task.inputs[3].input_position == 4):
                                self.assertEqual(task.extra_value_0, EMPTY_DISPATCH_MARKER)
                                self.assertEqual(task.inputs[3].base_ptr_offset, task.task_index // 4)
                                grouped.setdefault(int(task.trigger_event), []).append(int(task.task_index % 4))
                                task.extra_value_0 = original.all_tasks[task_id].extra_value_0
                        if mode == 'push' and not slots:
                            self.assertEqual(len(grouped), spec.num_experts)
                            for event, tiles in grouped.items():
                                self.assertEqual(sorted(tiles), [0, 1, 2, 3])
                                self.assertEqual(candidate.all_event_num_triggers[event], 4 * spec.ep_size)
                        self.assertEqual(bytes(original), bytes(candidate))

    def test_device_credit_does_not_release_early(self) -> None:
        """Interleave sender tiles independently and require exact release in both modes."""
        compiler = shutil.which('g++')
        if compiler is None:
            self.skipTest('The empty dispatch test requires g++')
        root = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / 'check'
            subprocess.run(
                [compiler, '-std=c++17', '-O2', '-I', str(root),
                 str(Path(__file__).with_name('empty_dispatch.cpp')), '-o', str(binary)],
                check=True,
            )
            subprocess.run([str(binary)], check=True)
