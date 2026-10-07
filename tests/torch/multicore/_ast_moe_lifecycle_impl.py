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
"""AST MoE shared-workspace, stream, checkpoint and profiling acceptance."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import profiler as multicore_profiler
from hyper_parallel.core.multicore.runtime.moe_native import verify_moe_native
from tests.torch.multicore import _test_mega_moe as legacy
from tests.torch.multicore import _test_mega_moe_ready as ready
from tests.torch.multicore import _test_mega_moe_resources as resources
from tests.torch.multicore._ast_moe_impl import _AstExperts


def _profile_case(root):
    shape = legacy.MoeShape()
    layer = _AstExperts(local_num_tokens=128, hidden_size=shape.hidden_size,
                        intermediate_size=shape.intermediate_size, num_experts=shape.num_experts,
                        top_k=2, ep_size=2, ep_group=dist.group.WORLD,
                        dispatch_mode=os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")).to(
                            device=legacy.DEVICE, dtype=torch.bfloat16)
    hidden, gradient = legacy.make_data(shape)
    ids, probabilities, counts = legacy.make_balanced_route(shape)
    try:
        legacy.run_layer(layer, hidden, ids, probabilities, counts, gradient)
        with multicore_profiler.mega_kernel_profile(
                schedule=multicore_profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                detailed_task_names=True) as profiler:
            legacy.run_layer(layer, hidden, ids, probabilities, counts, gradient)
            profiler.step()
        trace = profiler.export_chrome_trace(root / f"profile-rank{dist.get_rank()}.json")
        metadata = trace["megaKernelCycleTrace"]
        events = [event for event in trace["traceEvents"] if event.get("cat") == "MegaKernelInternal"]
        directions = {event["args"]["direction"] for event in events}
        if directions != {"forward", "backward"} or metadata["droppedRecordCount"] != 0 or not events:
            raise AssertionError(f"Incomplete AST forward/backward trace: {metadata}, {directions}")
        return metadata
    finally:
        layer.close()


def run_acceptance(result_dir: str) -> None:
    """Exercise original lifecycle and profiling through AST module factories.

    Args:
        result_dir: Directory receiving per-rank evidence and device cycle traces.
    """
    root = Path(result_dir)
    root.mkdir(parents=True, exist_ok=True)
    with patch.object(resources, "MegaMoeExperts", _AstExperts), patch.object(ready, "MegaMoeExperts", _AstExperts):
        cases = [ready._run_ready_case(replay) for replay in (False, True)]
    data = {"rank": dist.get_rank(), "mode": os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push"),
            "build_fingerprint": verify_moe_native()["build_fingerprint"],
            "lifecycle": cases, "profile": _profile_case(root)}
    (root / f"lifecycle-rank{dist.get_rank()}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if dist.get_rank() == 0:
        print(json.dumps(data), flush=True)


def test_ast_moe_lifecycle() -> None:
    """Validate streams, shared resources, checkpoint replay and device records."""
    run_acceptance(os.environ["HP_AST_MOE_RESULTS"])


if __name__ == "__main__":
    try:
        run_acceptance(os.environ["HP_AST_MOE_RESULTS"])
    finally:
        dist.destroy_process_group()
