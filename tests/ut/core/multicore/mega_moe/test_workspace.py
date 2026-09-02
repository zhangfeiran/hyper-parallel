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
"""Unit tests for bounded MegaMoe workspace sizing."""

import unittest

from hyper_parallel.core.multicore.platform.torch.mega_moe.config import (
    _resolve_receive_capacity,
)
from hyper_parallel.core.multicore.platform.torch.mega_moe.workspace import (
    _EVENT_COUNTER_BYTES,
    _WORKSPACE_ALIGNMENT,
    _spec_workspace_bytes,
)


class TestMegaMoeWorkspaceSizing(unittest.TestCase):
    """Validate SHMEM planning without allocating accelerator memory."""

    def test_default_capacity_uses_bounded_headroom(self):
        """Plan 1.5x average capacity instead of the lossless EP maximum."""
        specification = {
            "local_num_tokens": 128,
            "hidden_size": 16,
            "num_experts": 8,
            "top_k": 2,
            "expert_capacity_factor": 1.5,
            "ep_size": 8,
        }
        routed_slots = 256
        expected_capacity = 384
        element_size = 2
        expected_bytes = (
            (expected_capacity + routed_slots)
            * specification["hidden_size"]
            * element_size
            + 2 * _EVENT_COUNTER_BYTES
            + 4 * (_WORKSPACE_ALIGNMENT - 1)
        )

        actual_capacity = _resolve_receive_capacity(
            specification["expert_capacity_factor"], routed_slots
        )
        actual_bytes = _spec_workspace_bytes(specification, element_size)

        self.assertEqual(actual_capacity, expected_capacity)
        self.assertEqual(actual_bytes, expected_bytes)

    def test_custom_capacity_factor_changes_workspace_ceiling(self):
        """Apply a caller-provided multiplier before communication alignment."""
        specification = {
            "local_num_tokens": 128,
            "hidden_size": 16,
            "num_experts": 8,
            "top_k": 2,
            "expert_capacity_factor": 2.0,
            "ep_size": 8,
        }

        actual_capacity = _resolve_receive_capacity(
            specification["expert_capacity_factor"], 256
        )

        self.assertEqual(actual_capacity, 512)


if __name__ == "__main__":
    unittest.main()
