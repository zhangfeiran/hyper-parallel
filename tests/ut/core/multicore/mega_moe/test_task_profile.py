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
"""Bounded diagnostic ABI, stale-buffer rejection and stream-safe growth."""

import os
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import task_profile
from hyper_parallel.core.multicore.modules.mega_moe.workspace import MegaMoeWorkspace


class TestTaskProfile(unittest.TestCase):
    """Test storage contracts with CPU tensors, without mocking recorded work."""

    def test_opt_in_and_reset_protect_legacy_tail(self) -> None:
        """A grown arena loses its marker on the next ordinary profiled launch."""
        workspace = MegaMoeWorkspace(symmetric_memory=Mock(), shared=True)
        workspace.gmm_workspace = torch.empty(64, dtype=torch.uint8)
        with patch.dict(os.environ, {"HP_MEGA_MOE_TASK_PROFILE": "1"}):
            arena = workspace.prepare_forward_cycle_profile(enabled=True)
        self.assertEqual(arena.numel(), task_profile.BASE_BYTES + task_profile.ARENA_BYTES)
        self.assertEqual(arena[task_profile.MARKER_OFFSET:task_profile.MARKER_OFFSET + 8].view(torch.int64).item(),
                         task_profile.MARKER)
        with patch.dict(os.environ, {"HP_MEGA_MOE_TASK_PROFILE": "0"}):
            workspace.prepare_forward_cycle_profile(enabled=True)
        with self.assertRaisesRegex(RuntimeError, "Last launch"):
            workspace.collect_task_profile(24, forward=True)

    def test_decode_rejects_missing_headers_and_clock_reversal(self) -> None:
        """ABI mismatches fail instead of being interpreted as timing data."""
        values = [[[0] * task_profile.RECORD_VALUES for _ in range(task_profile.CAPACITY + 1)]
                  for _ in range(task_profile.WORKERS)]
        with self.assertRaises(ValueError):
            task_profile.decode(values, 1)
        for worker in values:
            worker[0][:5] = [1, 1, 1, 100, 200]
            worker[1][7:11] = [100, 120, 180, 190]
        decoded = task_profile.decode(values, 1)
        self.assertEqual(len(decoded["workers"]), 2)
        self.assertEqual(decoded["workers"][0]["records"][0]["compute_end_cycle"], 180)
        values[0][1][9] = 90
        with self.assertRaisesRegex(ValueError, "nonmonotonic"):
            task_profile.decode(values, 1)

    def test_disabled_launch_never_grows_storage(self) -> None:
        """Setting the detail flag alone does not instrument production kernels."""
        workspace = MegaMoeWorkspace(symmetric_memory=Mock(), shared=True)
        workspace.gmm_workspace = torch.empty(64, dtype=torch.uint8)
        with patch.dict(os.environ, {"HP_MEGA_MOE_TASK_PROFILE": "1"}):
            result = workspace.prepare_forward_cycle_profile(enabled=False)
        self.assertIs(result, workspace.gmm_workspace)
        self.assertIsNone(workspace.forward_cycle_profile)
