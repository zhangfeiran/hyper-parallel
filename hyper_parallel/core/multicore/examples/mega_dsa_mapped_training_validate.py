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
"""Mapped CP training VJPs and replay against a masked global Native reference."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from functools import partial
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples import mega_dsa_local_training_validate as validation
from hyper_parallel.core.multicore.examples.mega_dsa_local_training_oracle import ProjectionGeometry, projection_fixture
from hyper_parallel.core.multicore.examples.mega_dsa_local_training_subset_validate import _check_native
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import collective_validation_check
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import FusedDsaCpForwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import DsaWorkspaceSpec, MegaDsaWorkspace
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _cases(smoke):
    if smoke:
        return (("lm_plus_kl", .3, 1, "interior", "plain"),
                ("kl_only", .3, 2, "first_sequence", "retained"),
                ("lm_plus_kl", .3, 2, "owner", "checkpoint"),
                ("lm_only", .3, 1, "owner", "retained"),
                ("lm_plus_kl", 0., 2, "owner", "plain"),
                ("kl_only", 0., 1, "interior", "checkpoint"),
                ("lm_plus_kl", .3, 1, "empty", "checkpoint"),
                ("kl_only", .3, 1, "empty", "retained"))
    return product(("lm_only", "kl_only", "lm_plus_kl"), (0., .3), (1, 2),
                   ("owner", "interior", "first_sequence", "empty"), ("plain", "retained", "checkpoint"))


def _active(report, workspace, members, group, heads, dtype, pattern, smoke):
    meta = validation._metadata((3, 10), pattern, len(members), dist.get_rank(group))
    meta = replace(meta, cp_ranks=members, root_pes=workspace.root_pes, heap_generation=workspace.root.generation)
    backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=heads,
                                     attention_scale=validation._SCALE, schedule=MixedSfaSchedule(7))
    geometry = ProjectionGeometry(heads=heads)
    fixture = projection_fixture(13, geometry, seed=20261012+meta.root_pes[0])

    def _check(*args, **kwargs):
        proof = collective_validation_check(partial(_check_native, *args, **kwargs), group=group)
        if not proof["passed"]:
            raise RuntimeError(f"mapped native training VJP mismatch: {proof}")
        return proof
    with patch.object(validation, "_check", side_effect=_check):
        for mode, coefficient, divisor, tile, replay in _cases(smoke):
            stage = {"members": members, "root_pes": meta.root_pes, "cp_rank": meta.cp_rank,
                     "heads": heads, "weight_dtype": str(dtype), "pattern": pattern, "mode": mode,
                     "coefficient": coefficient, "auxiliary": -2.5, "reducer_divisor": divisor,
                     "tile": tile, "replay": replay}
            report["stage"] = stage
            evidence = validation._step(workspace, backend, meta, geometry, fixture, tile_mode=tile,
                                        weight_dtype=dtype, mode=mode, coefficient=coefficient, auxiliary=-2.5,
                                        divisor=divisor, replay=replay, fused_kl=True,
                                        sparse_main_return=True, sparse_index_return=True, sgd_step=True)
            report["cases"].append({**stage, **evidence})


def _fixture(report, groups, partitioned, heads, dtype, pattern, smoke):
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    group_id = int(partitioned and dist.get_rank() in groups[1][0])
    members, group = groups[group_id]
    workspace = MegaDsaWorkspace(root, "mapped-training", DsaWorkspaceSpec(13, 13, 13, 2), cp_group=group)
    scenario = {"partitioned": partitioned, "participates": workspace.participates,
                "heads": heads, "weight_dtype": str(dtype), "pattern": pattern, "cases": []}
    report["scenarios"].append(scenario)
    try:
        workspace.bind()
        def _work():
            if workspace.participates:
                _active(scenario, workspace, members, group, heads, dtype, pattern, smoke)
            return {"participates": workspace.participates}
        collective_validation_check(_work)
        scenario.update(status="passed", root_generation=root.generation)
    finally:
        workspace.close()
        root.close()


def main() -> None:
    """Persist CP subgroup VJP evidence without reusing complete-query FP64 calibration budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "scenarios": [], "scope": "mapped fixed-value Native projection VJP",
              "full_model_acceptance": False, "bf16_projection_composition_acceptance": False,
              "performance_acceptance": False, "sgd_acceptance": False}
    try:
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("hccl", timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError("mapped training validator requires four root members")
        report.update(rank=dist.get_rank(), cp_size=2, root_size=4)
        groups = tuple((members, dist.new_group(list(members))) for members in ((1, 3), (0, 2)))
        for partitioned in (False, True):
            fixtures = [(32, torch.float32, "strided")] if args.smoke else product(
                (32, 64), (torch.bfloat16, torch.float32), ("contiguous", "strided", "zigzag", "empty"))
            for heads, dtype, pattern in fixtures:
                _fixture(report, groups, partitioned, heads, dtype, pattern, args.smoke)
        report.update(status="passed", sgd_acceptance=True,
                      loaded_library_sha256=validation._loaded_library_identity())
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
