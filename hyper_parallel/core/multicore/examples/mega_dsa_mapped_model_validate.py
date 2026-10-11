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
"""Real BF16 attention CP subgroups on one encompassing root against independent stock models."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_fused_model_cp_validate import _case, model_fixture
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import _loaded_library_identity
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import collective_validation_check
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import FusedDsaCpForwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import DsaWorkspaceSpec, MegaDsaWorkspace
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _fixture(report, groups, partitioned, pattern):
    root = SharedShmemRoot(torch.device("npu", int(os.environ.get("LOCAL_RANK", "0"))), root_group=dist.group.WORLD)
    group_id = int(partitioned and dist.get_rank() in groups[1][0])
    members, group = groups[group_id]
    workspace = MegaDsaWorkspace(root, "mapped-model", DsaWorkspaceSpec(13, 13, 13, 2), cp_group=group)
    scenario = {"partitioned": partitioned, "pattern": pattern, "participates": workspace.participates,
                "members": members, "cases": []}
    report["scenarios"].append(scenario)
    try:
        workspace.bind()
        def _work():
            if not workspace.participates:
                return {"participates": False}
            template = model_fixture()
            if group_id:
                with torch.no_grad():
                    template.q_b_proj.weight.mul_(1.125)
                    template.indexer.wq_b.weight.mul_(1.125)
            meta = _metadata((3, 10), pattern, len(members), dist.get_rank(group))
            meta = replace(meta, q_global_ids=meta.kv_global_ids, cp_ranks=members, root_pes=workspace.root_pes,
                           heap_generation=root.generation)
            backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=32,
                                             attention_scale=template.scaling, schedule=MixedSfaSchedule(7))
            cases = [("joint", replay) for replay in ("plain", "retained", "checkpoint")]
            cases += [(mode, "plain") for mode in ("kl_only", "freeze", "zero", "eval", "aux_zero")]
            for mode, replay in cases:
                collective_validation_check(partial(_case, scenario, workspace, backend, meta, template, mode, replay),
                                            group=group)
            if not all(case["passed"] for case in scenario["cases"]):
                raise RuntimeError("mapped subgroup model numerical mismatches recorded in scenario cases")
            return {"participates": True}
        collective_validation_check(_work)
        scenario.update(status="passed", root_generation=root.generation)
    finally:
        workspace.close()
        root.close()


def main() -> None:
    """Keep all subgroup model evidence and root-wide failure/close agreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "running", "scenarios": [], "scope": "mapped real HP BF16 attention versus stock",
              "full_lm_model_acceptance": False, "independent_fp32_model_acceptance": False,
              "cp_ep_coexistence_acceptance": False, "performance_acceptance": False}
    try:
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("hccl", timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError("mapped real model validator requires four root members")
        report.update(rank=dist.get_rank(), cp_size=2, root_size=4)
        groups = tuple((members, dist.new_group(list(members))) for members in ((1, 3), (0, 2)))
        for partitioned in (False, True):
            for pattern in ("strided", "empty"):
                _fixture(report, groups, partitioned, pattern)
        accepted = all(case["passed"] for scenario in report["scenarios"] for case in scenario["cases"])
        report.update(status="passed" if accepted else "numerical_mismatch",
                      loaded_library_sha256=_loaded_library_identity())
        if not accepted:
            raise RuntimeError("mapped real model mismatches recorded; inspect per-field stock comparisons")
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        if report["status"] == "running":
            report["status"] = "error"
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
