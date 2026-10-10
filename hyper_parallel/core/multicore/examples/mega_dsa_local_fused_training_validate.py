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
"""Single-kernel local LI/pull/SFA/KL validation against separately submitted native phases."""

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

from hyper_parallel.core.multicore.examples import (
    mega_dsa_local_kl_validate as baseline,
)
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import (
    collective_validation_check,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_kl import local_kl_gradients
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_kl import (
    mixed_kl_native_key_cast,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _check(full, fused, separate, raw, backend):
    for actual, expected in zip((fused.output, fused.global_indices, fused.values, fused.maximum, fused.denominator),
                               (separate.output, separate.global_indices, separate.values,
                                separate.maximum, separate.denominator)):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    metrics = [baseline._metrics(actual, expected) for actual, expected in
               zip((*fused.kl_gradients, fused.kl_loss, fused.kl_key_partials),
                   (*raw.gradients, raw.loss, raw.key_partials))]
    if not all(metric["pointwise_pass"] for metric in metrics):
        raise RuntimeError(f"fused local KL differs from separate native calls: {metrics}")
    expected = baseline._oracle(full, fused)
    independent = [baseline._metrics(actual, oracle) for actual, oracle in
                   zip((*fused.kl_gradients, fused.kl_loss, fused.kl_key_partials), (*expected, expected[1]))]
    if not all(metric["relative_l2"] <= .02 or metric["max_abs"] <= 2e-5 for metric in independent):
        raise RuntimeError(f"fused local KL independent oracle mismatch: {independent}")
    if fused.output.shape[0]:
        torch.testing.assert_close(mixed_kl_native_key_cast(fused.kl_key_partials.cpu()).view(torch.int16),
                                   fused.kl_gradients[1].cpu().view(torch.int16), rtol=0, atol=0)
    else:
        torch.testing.assert_close(fused.kl_key_partials.cpu(), torch.zeros_like(fused.kl_key_partials.cpu()),
                                   rtol=0, atol=0)
    selected = tuple(value.cpu() for value in (*fused.selected_requests, fused.saved.indices))
    traces = backend.validate_trace(fused, tuple(trace.cpu() for trace in fused.phase_traces),
                                     fused.transport_trace.cpu(), require_overlap=False, selected_snapshots=selected)
    return {"separate_native": metrics, "independent_oracle": independent, "trace": traces,
            "query_rows": fused.output.shape[0], "native_launches": 1, "device_phases": 9}


def _fixture(report, heads, dtype, pattern, smoke):
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    rank, size = dist.get_rank(), dist.get_world_size()
    meta = baseline._metadata((3, 10), pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "local-fused-training", DsaWorkspaceSpec(capacity, 13, capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    cpu, _ = baseline._inputs(heads, dtype, (3, 10))
    if dtype == torch.float32:
        cpu = (*cpu[:6], cpu[6]*1.003+.000013)
    full = tuple(tensor.to(device) for tensor in cpu)
    try:
        for groups in ((7,) if smoke else (1, 7, 19)):
            backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=heads,
                                             attention_scale=baseline._SCALE, schedule=MixedSfaSchedule(groups))
            for mode in ("owner", "interior", "first_sequence", "empty"):
                tile = baseline._tile(meta, mode)
                layout = backend.prepare_local_queries(tile.q_global_ids)
                main_states, index = baseline._local_states(tile, full[:4], (full[4], full[5][:, None], full[6]))
                for repeat in range(2):
                    report["stage"] = {"heads": heads, "dtype": str(dtype), "pattern": pattern,
                                       "groups": groups, "tile": mode, "repeat": repeat}
                    invocation = workspace.prepare(replace(meta, layer=repeat, microbatch=repeat,
                                                           invocation=workspace.transport_epoch+1))
                    fused = backend.forward_local(invocation, main_states, index, query_layout=layout, selected_kl=True)
                    other = workspace.prepare(replace(meta, layer=repeat+3, microbatch=repeat+3,
                                                      invocation=workspace.transport_epoch+1))
                    separate = backend.forward_local(other, main_states, index, query_layout=layout)
                    raw = local_kl_gradients(separate)
                    torch.npu.synchronize()
                    evidence = collective_validation_check(partial(_check, full, fused, separate, raw, backend))
                    report["cases"].append({**report["stage"], **evidence})
    finally:
        workspace.close()
        root.close()


def run_validation(report: dict, *, smoke: bool = False) -> None:
    """Keep single-kernel raw correctness separate from complete training and performance acceptance."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    report.update(rank=dist.get_rank(), cp_size=dist.get_world_size(), cases=[], full_model_acceptance=False)
    fixtures = [(32, torch.float32, "strided")] if smoke else [
        (heads, dtype, pattern) for heads in (32, 64) for dtype in (torch.bfloat16, torch.float32)
        for pattern in ("contiguous", "strided", "empty")]
    try:
        for heads, dtype, pattern in fixtures:
            _fixture(report, heads, dtype, pattern, smoke)
        report.update(status="passed", loaded_library_sha256=baseline._loaded_library_identity())
    finally:
        dist.destroy_process_group()


def main() -> None:
    """Archive rank-local failures and complete nine-phase trace evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "full_model_acceptance": False}
    try:
        run_validation(report, smoke=args.smoke)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
