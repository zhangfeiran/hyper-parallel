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
"""External Top-K CP attention, fixed-selection backward and owner-return validation."""

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
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # Registers the NPU backend.
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_backward_validate import (
    _cotangent,
    _owner_oracle,
    _reference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import (
    _SCALE,
    _metadata,
)
from hyper_parallel.core.multicore.examples.mega_dsa_fused_forward_validate import (
    _main_states,
)
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_tile_validate import (
    _reference as forward_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.module import MegaDsaCore
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    sparse_attention_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _selection(meta, fixture):
    selected = torch.full((meta.global_valid_queries, 2048), -1, dtype=torch.int32)
    for query in range(meta.global_valid_queries):
        sequence, _ = meta.sequence_position(query)
        start = meta.global_cu_seqlens[sequence]
        candidates = list(range(start, query + 1))
        if fixture == "complete":
            candidates = candidates[-2048:]
            selected[query, :len(candidates)] = torch.tensor(candidates, dtype=torch.int32)
            continue
        if fixture == "all_empty":
            candidates = []
        elif fixture == "owner_zero":
            candidates = [token for token in candidates if meta.token_owners[token] == 0]
        else:
            # Holes, future and foreign-sequence candidates exercise stable legal-set compaction.
            candidates = candidates[::2][-1020:]
            candidates += [token for token in (query + 1, 0) if token < meta.global_valid_queries
                           and token not in candidates and (token > query or token < start)]
        if candidates:
            selected[query, 1:2 * len(candidates):2] = torch.tensor(candidates, dtype=torch.int32)
    return selected


def _local_main(meta, full):
    query, key = list(meta.q_global_ids), list(meta.kv_global_ids)
    return tuple(tensor[rows].contiguous() for tensor, rows in zip(full, (query, key, query, key)))


def _metrics(actual, expected):
    delta = actual.double() - expected.double()
    return {"max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            "relative_l2": float(delta.norm() / expected.double().norm().clamp_min(1e-30)),
            "existing_pointwise_pass": bool(torch.allclose(actual.float(), expected.float(), rtol=.02, atol=2e-5))}


def _oracle_measurements(forward, backward, full, selected, meta, repeat):
    if sum(end-start for start, end in zip(meta.global_cu_seqlens, meta.global_cu_seqlens[1:])) >= 32:
        return {}
    cpu = tuple(tensor.detach().float().cpu().requires_grad_() for tensor in full)
    packed = DsaBatchMeta.packed(tuple(end-start for start, end in zip(meta.global_cu_seqlens,
                                                                     meta.global_cu_seqlens[1:])))
    oracle, _ = sparse_attention_reference(*cpu, selected, packed, attention_scale=_SCALE)
    query, key = list(meta.q_global_ids), list(meta.kv_global_ids)
    measurement = {"oracle_forward": _metrics(forward.output.float().cpu(), oracle.detach()[query])}
    if backward is None:
        return measurement
    generator = torch.Generator().manual_seed(20261009 + full[0].shape[1] + repeat)
    cotangent = (torch.randn(full[0].shape, generator=generator)*.1).bfloat16()
    weights = torch.tensor([owner+1 for owner in meta.token_owners], dtype=torch.bfloat16)[:, None, None]
    cotangent = (cotangent * weights).bfloat16().float()
    gradients = torch.autograd.grad(oracle, cpu, grad_outputs=cotangent)
    measurement["oracle_gradients"] = [_metrics(actual.float().cpu(), expected[rows])
                                       for actual, expected, rows in zip(backward.gradients, gradients,
                                                                         (query, key, query, key))]
    return measurement


def _backward_proof(backward, core, forward, cotangent):
    if backward is None:
        return {}
    return {"backward_trace": core.backward_backend.validate_trace(
                backward, tuple(value.cpu() for value in backward.phase_traces), backward.transport_trace.cpu()),
            **_owner_oracle(backward, core.backward_backend),
            **_reference(backward, forward, core.backward_backend, cotangent)}


def _run_fixture(report, lengths, heads, pattern, fixture, device, *, smoke, forward_only):
    meta = _metadata(lengths, pattern, dist.get_world_size(), dist.get_rank())
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(dist.get_world_size())))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "dsa-external-core", DsaWorkspaceSpec(capacity, sum(lengths), capacity,
                                                                          dist.get_world_size()))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    selected = _selection(meta, fixture)
    try:
        for groups in ((7,) if smoke else (1, 2, 7, 19)):
            prepared = workspace.prepare(meta)
            core = MegaDsaCore(workspace, prepared, heads=heads, attention_scale=_SCALE,
                               schedule=MixedSfaSchedule(groups))
            admitted = core.prepare_selection(selected[list(meta.q_global_ids)])
            if not forward_only and not admitted.backward_ready:
                raise NotImplementedError("padded external backward awaits the native selected-count fix")
            for repeat in range(1 if smoke else 2):
                report["stage"] = {"lengths": lengths, "heads": heads, "pattern": pattern,
                                   "selection": fixture, "groups": groups, "repeat": repeat}
                full = _main_states(sum(lengths), heads, device)
                full = (full[0], (full[1] + repeat * .015625).bfloat16(), full[2], full[3])
                local = _local_main(meta, full)
                invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch + 1,
                                                       layer=repeat % 2, microbatch=repeat))
                forward = core.raw_forward(invocation, local, admitted)
                cotangent = _cotangent(meta, heads, repeat, device)
                backward = None if forward_only else core.backward_backend.backward(forward, cotangent)
                native = core.backend.native_layout.global_to_sequence_indices(selected.to(device))
                baseline = forward_reference(full, native, core.backend.native_layout)
                torch.npu.synchronize()
                rows = core.backend.layout.local_query_ids
                for actual, expected in zip((forward.output, forward.maximum, forward.denominator), baseline):
                    axis = 0 if expected.ndim == 3 and expected.shape[0] != 1 else 1
                    torch.testing.assert_close(actual.cpu(), expected.index_select(axis, rows).cpu(), rtol=0, atol=0)
                trace = core.validate_trace(forward, tuple(value.cpu() for value in forward.phase_traces),
                                            forward.transport_trace.cpu())
                proof = _backward_proof(backward, core, forward, cotangent)
                oracle = _oracle_measurements(forward, backward, full, selected, meta, repeat)
                report["cases"].append({**report["stage"], "forward_stock_exact": True,
                                        "forward_trace": trace, "backward_executed": backward is not None,
                                        **proof, **oracle})
    finally:
        workspace.close()
        root.close()


def _lifecycle(report, device, pattern, fixture):
    meta = _metadata((3, 10), pattern, dist.get_world_size(), dist.get_rank())
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(dist.get_world_size())))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "dsa-core-lifecycle", DsaWorkspaceSpec(capacity, 13, capacity,
                                                                           dist.get_world_size()))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    try:
        prepared = workspace.prepare(meta)
        core = MegaDsaCore(workspace, prepared, heads=32, attention_scale=_SCALE, schedule=MixedSfaSchedule(7))
        selection = core.prepare_selection(_selection(meta, fixture)[list(meta.q_global_ids)])
        if not selection.backward_ready:
            raise NotImplementedError("padded external lifecycle awaits the native selected-count fix")
        local = _local_main(meta, _main_states(13, 32, device))
        raw = core.raw_forward(prepared, local, selection)
        cotangent = _cotangent(meta, 32, 0, device)
        ready = torch.npu.Event()
        ready.record()
        first_raw = core.backward_backend.backward(raw, cotangent)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            ready.wait()
            second_raw = core.backward_backend.backward(raw, cotangent)
        main_inputs = tuple(tensor.clone().requires_grad_() for tensor in local)
        out, stats = core(*main_inputs, selection, meta)
        if stats.maximum.requires_grad or stats.denominator.requires_grad:
            raise RuntimeError("core native statistics unexpectedly have a gradient path")
        later = workspace.prepare(replace(meta, layer=3, microbatch=4, invocation=workspace.transport_epoch + 1))
        core.raw_forward(later, tuple((value.detach()+.0625).bfloat16() for value in local), selection)
        first = torch.autograd.grad(out, main_inputs, grad_outputs=cotangent, retain_graph=True)
        second = torch.autograd.grad(out, main_inputs, grad_outputs=cotangent)
        checkpoint_main = tuple(tensor.clone().requires_grad_() for tensor in local)
        recomputed = checkpoint(lambda *values: core(*values, selection, meta)[0],
                                *checkpoint_main, use_reentrant=False)
        third = torch.autograd.grad(recomputed, checkpoint_main, grad_outputs=cotangent)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), raw.output.cpu(), rtol=0, atol=0)
        for actual in (second_raw.gradients, first, second, third):
            for value, expected in zip(actual, first_raw.gradients):
                torch.testing.assert_close(value.cpu(), expected.cpu(), rtol=.02, atol=2e-5)
        _owner_oracle(first_raw, core.backward_backend)
        _owner_oracle(second_raw, core.backward_backend)
        report.setdefault("lifecycle", []).append(
            {"pattern": pattern, "selection": fixture, "retained_graph": True, "delayed_after_republication": True,
             "non_reentrant_checkpoint": True, "cross_stream_backward": True,
             "four_main_gradients": True, "statistics_nondifferentiable": True})
    finally:
        workspace.close()
        root.close()


def run_validation(report: dict, output_dir: Path, *, smoke: bool, long_history: bool,
                   forward_only: bool = True, selection_fixture: str | None = None) -> None:
    """Run native CP main attention without executing an indexer or KL operator."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    if smoke:
        torch.npu.set_op_timeout_ms(30000)
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    report.update(rank=dist.get_rank(), cp_size=dist.get_world_size(), cases=[])
    fixtures = [((3, 10), 32, "contiguous", "holes")]
    if not smoke:
        fixtures.extend(((3, 10), heads, pattern, "holes") for heads in (32, 64)
                        for pattern in ("strided", "zigzag", "empty"))
        fixtures.extend(((3, 10), 32, "strided", fixture) for fixture in ("all_empty", "owner_zero"))
    if long_history:
        fixtures.extend(((2176, 2240), heads, "contiguous", "holes") for heads in (32, 64))
    if selection_fixture is not None:
        fixtures = list(dict.fromkeys((*fixture[:3], selection_fixture) for fixture in fixtures))
    report.update(forward_only=forward_only, selection_fixture=selection_fixture)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fixture in fixtures:
        _run_fixture(report, *fixture, device, smoke=smoke, forward_only=forward_only)
        (output_dir / f"rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    if not smoke and not forward_only:
        for pattern in ("zigzag", "empty"):
            _lifecycle(report, device, pattern, selection_fixture or "holes")
    report.update(status="passed", stage="complete", case_count=len(report["cases"]))
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Persist exact per-rank native proof and explicitly separate FP32 oracle measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--long-history", action="store_true")
    parser.add_argument("--selection-fixture", choices=("complete", "holes", "all_empty", "owner_zero"))
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--forward-only", dest="forward_only", action="store_true")
    execution.add_argument("--backward", dest="forward_only", action="store_false")
    parser.set_defaults(forward_only=True)
    args = parser.parse_args()
    report = {"status": "running", "scope": "external Top-K CP core forward/backward",
              "full_model_acceptance": False, "selected_kl": False, "oracle_acceptance": "measurement only"}
    try:
        run_validation(report, args.output_dir, smoke=args.smoke, long_history=args.long_history,
                       forward_only=args.forward_only, selection_fixture=args.selection_fixture)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        (args.output_dir / f"rank{rank}.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
