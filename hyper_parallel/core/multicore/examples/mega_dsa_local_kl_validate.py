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
"""Native local-query KL phase, precision and retained-invocation acceptance."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

import hyper_parallel
from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _local_states, _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import _loaded_library_identity
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_validate import _SCALE, _tile
from hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate import _inputs, _metrics
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import collective_validation_check
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout, CannDsaStats
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import FusedDsaCpForwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.local_kl import local_kl_gradients, validate_local_kl_trace
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_kl import _submit_mixed_kl, mixed_kl_native_key_cast
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.reference import selected_kl_reference
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import DsaWorkspaceSpec, MegaDsaWorkspace
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _oracle(full, forward):
    meta = forward.saved.batch_meta
    rows = list(meta.q_global_ids)
    iq, ik, weight = (full[field].cpu().float().detach().requires_grad_() for field in (4, 5, 6))
    main_states = tuple(full[field].cpu().float() for field in range(4))
    if not rows:
        return (torch.zeros_like(iq[rows]), torch.zeros_like(ik), torch.zeros_like(weight[rows]), torch.zeros(()))
    inputs = (iq[rows], ik, weight[rows])
    main_states = (main_states[0][rows], main_states[1], main_states[2][rows], main_states[3])
    packed = replace(meta, kv_global_ids=tuple(range(meta.global_valid_queries)))
    loss = selected_kl_reference(*inputs, *main_states, forward.global_indices[:, 0].cpu(), packed,
                                 attention_scale=_SCALE,
                                 normalization=DsaLossNormalization(meta.global_valid_queries,
                                                                    meta.global_valid_queries))
    return (*torch.autograd.grad(loss, inputs), loss.detach())


def _baseline(full, forward, schedule):
    saved = forward.saved
    rows = list(saved.local_query_layout.global_ids)
    tokens = full[0].shape[0]
    layout = CannDsaLayout(DsaBatchMeta.packed(tuple(end-start for start, end in
                                                    zip(saved.batch_meta.global_cu_seqlens,
                                                        saved.batch_meta.global_cu_seqlens[1:]))), full[0].device)
    actual = [tensor.detach().clone() for tensor in full]
    for field in (0, 2, 4, 6):
        actual[field].zero_()
        actual[field][rows] = full[field][rows]
    indices = torch.ops.npu.npu_lightning_indexer(
        full[4], full[5][:, None], full[6], actual_seq_lengths_query=layout.length_tensor,
        actual_seq_lengths_key=layout.length_tensor, layout_query="TND", layout_key="TND",
        sparse_count=2048, sparse_mode=3, return_value=True)[0]
    indices[rows] = saved.indices
    # Unowned teacher and predictor both become uniform, so their KL and key VJP are zero.
    maximum = torch.zeros(1, tokens, full[0].shape[1], device=full[0].device)
    denominator = torch.ones_like(maximum)
    for token in range(tokens):
        denominator[:, token] = saved.batch_meta.sequence_position(token)[1]+1
    maximum[:, rows] = saved.forward[1]
    denominator[:, rows] = saved.forward[2]
    result = _submit_mixed_kl(tuple(actual[:4]), tuple(actual[4:]), indices, layout.cumulative_lengths,
                              CannDsaStats(maximum, denominator), schedule.runtime_config(layout.device), _SCALE)
    local = saved.local_query_layout
    return (local.restore(result.gradients[0][rows]), result.gradients[1],
            local.restore(result.gradients[2][rows]), result.loss), result


def _check(full, forward, result, schedule):
    oracle = _oracle(full, forward)
    actual = (*result.gradients, result.loss)
    independent = [_metrics(value, expected) for value, expected in zip(actual, oracle)]
    key_metrics = _metrics(result.key_partials, oracle[1])
    if not all(item["relative_l2"] <= .02 or item["max_abs"] <= 2e-5 for item in (*independent, key_metrics)):
        raise RuntimeError(f"local KL independent oracle mismatch: {independent}; key FP32={key_metrics}")
    torch.testing.assert_close(mixed_kl_native_key_cast(result.key_partials.cpu()),
                               result.gradients[1].cpu(), rtol=0, atol=0)
    baseline, owned = _baseline(full, forward, schedule)
    native = [_metrics(value, expected) for value, expected in zip(actual, baseline)]
    if not all(item["pointwise_pass"] for item in native):
        raise RuntimeError(f"local KL zero-contribution global baseline mismatch: {native}")
    del owned
    evidence = validate_local_kl_trace(result, tuple(trace.cpu() for trace in result.traces), schedule)
    return {"oracle": independent, "fp32_key_oracle": key_metrics, "native": native, "trace": evidence,
            "key_cast_bits_exact": True, "queries": len(forward.saved.batch_meta.q_global_ids),
            "strict_causal_bound_rows": sum(exact < conservative for exact, conservative in
                                            forward.saved.local_query_layout.causal_bounds())}


def _fixture(report, lengths, heads, dtype, pattern, *, smoke):
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    rank, size = dist.get_rank(), dist.get_world_size()
    meta = _metadata(lengths, pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "dsa-local-kl", DsaWorkspaceSpec(capacity, sum(lengths), capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    cpu, _ = _inputs(heads, dtype, lengths)
    if dtype == torch.float32:
        cpu = (*cpu[:6], cpu[6]*1.003+.000013)
    full = tuple(tensor.to(device) for tensor in cpu)
    index = (full[4], full[5][:, None], full[6])
    try:
        for mode in ("owner", "interior", "first_sequence", "empty"):
            tile = _tile(meta, mode)
            for groups in ((7,) if smoke else (1, 7, 19)):
                schedule = MixedSfaSchedule(groups)
                backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=heads,
                                                 attention_scale=_SCALE, schedule=schedule)
                local = backend.prepare_local_queries(tile.q_global_ids)
                pending = []
                for repeat in range(2):
                    states = (full[0], (full[1]+repeat*.015625).bfloat16(), full[2], full[3])
                    local_main, local_index = _local_states(tile, states, index)
                    invocation = workspace.prepare(replace(meta, layer=repeat, microbatch=repeat,
                                                           invocation=workspace.transport_epoch+1))
                    forward = backend.forward_local(invocation, local_main, local_index, query_layout=local)
                    raw = local_kl_gradients(forward)
                    pending.append((forward, raw, (*states, *full[4:])))
                torch.npu.synchronize()
                for repeat, (forward, raw, states) in enumerate(pending):
                    report["stage"] = {"lengths": lengths, "heads": heads, "dtype": str(dtype),
                                       "pattern": pattern, "tile": mode, "groups": groups, "repeat": repeat}
                    evidence = collective_validation_check(lambda: _check(states, forward, raw, schedule))
                    report["cases"].append({**report["stage"], **evidence})
    finally:
        workspace.close()
        root.close()


def run_validation(report: dict, *, smoke: bool = False) -> None:
    """Check actual local forward plus raw KL on CP1/2/4; training scaling remains a separate gate."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    report.update(rank=dist.get_rank(), cp_size=dist.get_world_size(), cases=[], package_path=hyper_parallel.__file__)
    fixtures = [(32, torch.bfloat16, "strided")] if smoke else [
        (heads, dtype, pattern) for heads in (32, 64) for dtype in (torch.bfloat16, torch.float32)
        for pattern in ("contiguous", "strided", "empty")]
    try:
        for heads, dtype, pattern in fixtures:
            _fixture(report, (3, 10), heads, dtype, pattern, smoke=smoke)
        report.update(status="passed", loaded_library_sha256=_loaded_library_identity())
    finally:
        dist.destroy_process_group()


def main() -> None:
    """Persist rank-local success or failure without treating primitive acceptance as full training acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "scope": "raw native local-query selected KL",
              "lm_plus_kl_acceptance": False, "full_model_acceptance": False}
    try:
        run_validation(report, smoke=args.smoke)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        (args.output/f"rank{rank}.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
