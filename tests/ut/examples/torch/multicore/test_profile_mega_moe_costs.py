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
"""Keep every profile-off HBM window ahead of persistent diagnostic allocations."""

import importlib.util
import json
import sys
import tempfile
import unittest
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch


@dataclass
class _Shape:
    """Minimal immutable values serialized by the diagnostic worker."""

    local_num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    ep_size: int


class TestCostProfileIsolation(unittest.TestCase):
    """A task arena retained after route one must never enter route two's measured HBM."""

    def test_all_routes_measure_before_any_diagnostic_allocation(self) -> None:
        """Persistent diagnostic storage never enters a subsequent measurement."""
        worker = MagicMock()
        worker._RANK = 0
        worker.dist.get_world_size.return_value = 2
        worker._MoeShape.side_effect = _Shape
        worker._make_data.return_value = (None, None)
        worker._new_layers.return_value = (MagicMock(), MagicMock())
        worker._make_dirichlet_route.return_value = (1, 2, 3)
        worker._summarize_performance_route.return_value = {"expected_wave_count": 1}
        worker._timed_fwd_bwd.return_value = 1.25
        package = ModuleType("tests.torch.expert_parallel")
        package._test_mega_moe = worker
        path = Path(__file__).resolve().parents[5] / "examples/torch/multicore/profile_mega_moe_costs.py"
        spec = importlib.util.spec_from_file_location("cost_profile_fixture", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"tests.torch.expert_parallel": package}):
            spec.loader.exec_module(module)
        events = []
        allocated = False

        def memory(*_args: object) -> dict:
            """Model an allocator peak that must exclude persistent diagnostic arenas."""
            self.assertFalse(allocated, "diagnostic allocation leaked into a later measured route")
            events.append("memory")
            return {"peak_bytes": 100}

        def scopes(_torch: object, call: Callable[[], object], **_kwargs: object) -> list:
            """Retain diagnostic storage until this worker exits, as the native arena does."""
            nonlocal allocated
            allocated = True
            events.append("diagnostic")
            call()
            return []

        worker._collect_memory_result.side_effect = memory
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "argv", ["cost-worker", "--output", directory, "--pairs", "1:1,0.1:2",
                                       "--warmup", "0", "--measured", "1", "--profile-scopes", "--skip-task-profile"]),
            patch.object(module, "_identity", return_value={"fixture": True}),
            patch.object(module, "_counts", return_value=[[1, 2]]),
            patch.object(module, "new_capture_id", side_effect=["scopes:first", "scopes:second"]),
            patch.object(module.ProfileContext, "observe", return_value=nullcontext()),
            patch.object(module, "profile_scoped_events", side_effect=scopes),
            patch.object(module.importlib, "import_module", return_value=object()),
        ):
            module.main()
            result = json.loads((Path(directory) / "rank0.json").read_text(encoding="utf-8"))
        self.assertEqual(events, ["memory", "memory", "diagnostic", "diagnostic"])
        self.assertEqual([item["memory"]["peak_bytes"] for item in result["results"]], [100, 100])
