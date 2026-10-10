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
"""Complete-query CP projection matrix with an independently validated primitive reference profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples import (
    mega_dsa_local_training_validate as validation,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_training_oracle import (
    ProjectionGeometry,
    projection_fixture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.projection_precision import (
    validate_loaded_projection_reference,
    validate_projection_reference_profile,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _fixture(report, heads, dtype, pattern, budgets, fused_kl, sparse_main_return, sparse_index_return):
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    rank, size = dist.get_rank(), dist.get_world_size()
    meta = validation._metadata((3, 10), pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "training-profile-owner", DsaWorkspaceSpec(capacity, 13, capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=heads,
                                     attention_scale=validation._SCALE, schedule=MixedSfaSchedule(7))
    geometry = ProjectionGeometry(heads=heads)
    fixture = projection_fixture(13, geometry)
    try:
        for mode in ("lm_only", "kl_only", "lm_plus_kl"):
            for divisor in sorted({1, size}):
                for replay in ("plain", "retained", "checkpoint"):
                    stage = {"heads": heads, "weight_dtype": str(dtype), "pattern": pattern, "mode": mode,
                             "coefficient": .3, "auxiliary": -2.5, "reducer_divisor": divisor,
                             "tile": "owner", "replay": replay}
                    report["stage"] = stage
                    evidence = validation._step(workspace, backend, meta, geometry, fixture, tile_mode="owner",
                                                weight_dtype=dtype, mode=mode, coefficient=.3, auxiliary=-2.5,
                                                divisor=divisor, replay=replay, precision_budgets=budgets,
                                                fused_kl=fused_kl, sparse_main_return=sparse_main_return,
                                                sparse_index_return=sparse_index_return)
                    report["cases"].append({**stage, **evidence})
                    if not evidence["passed"]:
                        raise RuntimeError(f"calibrated complete-query projection matrix failed: {evidence}")
    finally:
        workspace.close()
        root.close()


def run_validation(report: dict, reference_profile: Path, policy_sha256: str, *, fused_kl: bool = False,
                   sparse_main_return: bool = False, sparse_index_return: bool = False) -> None:
    """Check sum/average reducers, empty owners and replay without applying the profile to subsets."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    report.update(rank=dist.get_rank(), cp_size=dist.get_world_size(), cases=[], full_model_acceptance=False,
                  bf16_projection_composition_acceptance=False, scope="complete-query CP projection VJP matrix",
                  fused_kl=fused_kl, sparse_main_return=sparse_main_return, sparse_index_return=sparse_index_return)
    try:
        profile = json.loads(reference_profile.read_text())
        libraries = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                     for name in profile["loaded_library_sha256"]}
        budgets = validate_projection_reference_profile(
            profile, expected_versions={"torch_version": torch.__version__,
                                        "torch_npu_version": torch_npu.__version__},
            expected_library_hashes=libraries, expected_policy_sha256=policy_sha256)
        report.update(reference_profile_sha256=hashlib.sha256(reference_profile.read_bytes()).hexdigest(),
                      reference_policy_sha256=policy_sha256)
        for heads in (32, 64):
            for dtype in (torch.bfloat16, torch.float32):
                for pattern in ("contiguous", "strided", "zigzag", "empty"):
                    _fixture(report, heads, dtype, pattern, budgets, fused_kl, sparse_main_return, sparse_index_return)
        loaded = validation._loaded_library_identity()
        validate_loaded_projection_reference(profile, loaded)
        report.update(status="passed", loaded_library_sha256=loaded)
    finally:
        dist.destroy_process_group()


def main() -> None:
    """Persist all original pointwise failures alongside the independent calibrated gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--profile-policy-sha256", required=True)
    parser.add_argument("--fused-kl", action="store_true", help="Use one native LI/pull/SFA/KL forward graph")
    parser.add_argument("--sparse-main-return", action="store_true", help="Omit zero merged main-gradient rows")
    parser.add_argument("--sparse-index-return", action="store_true",
                        help="Return index-K FP32 rows through device inboxes")
    args = parser.parse_args()
    report = {"status": "running", "full_model_acceptance": False}
    try:
        run_validation(report, args.reference_profile, args.profile_policy_sha256, fused_kl=args.fused_kl,
                       sparse_main_return=args.sparse_main_return, sparse_index_return=args.sparse_index_return)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
