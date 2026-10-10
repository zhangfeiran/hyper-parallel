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
"""Actual local-query CP LI, selected main-KV pull and SFA with explicit causal positions."""

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
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # Registers the NPU backend.

import hyper_parallel
from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import (
    _local_states, _metadata,
)
from hyper_parallel.core.multicore.examples.mega_dsa_fused_forward_validate import _main_states
from hyper_parallel.core.multicore.examples.mega_dsa_local_indexer_validate import _check_indices
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_indexer_validate import _fixture
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_tile_validate import _reference
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import FusedDsaCpForwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.reference import sparse_attention_reference
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import DsaWorkspaceSpec, MegaDsaWorkspace
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot

_SCALE = 192**-0.5


def _tile(meta, mode):
    ids = meta.q_global_ids
    if mode == "interior":
        ids = tuple(token for token in ids if 0 < meta.sequence_position(token)[1]
                    < (meta.global_cu_seqlens[meta.sequence_position(token)[0] + 1]
                       - meta.global_cu_seqlens[meta.sequence_position(token)[0]]) * 3 // 4)
    elif mode == "first_sequence":
        ids = tuple(token for token in ids if meta.sequence_position(token)[0] == 0)
    elif mode == "empty":
        ids = ()
    return replace(meta, q_global_ids=ids)


def _oracle(result, main_states, meta):
    """Evaluate every local query against independent FP32 selected attention in bounded chunks."""
    ids = meta.q_global_ids
    key, key_rope = main_states[1].cpu(), main_states[3].cpu()
    query, query_rope = main_states[0].cpu()[list(ids)], main_states[2].cpu()[list(ids)]
    indices = result.global_indices[:, 0].cpu()
    actual = result.output.cpu().float()
    total_square, delta_square, maximum = 0., 0., 0.
    for start in range(0, len(ids), 4):
        end = min(start + 4, len(ids))
        oracle_meta = replace(meta, q_global_ids=ids[start:end], kv_global_ids=tuple(range(meta.global_valid_queries)))
        expected, _stats = sparse_attention_reference(
            query[start:end], key, query_rope[start:end], key_rope, indices[start:end], oracle_meta,
            attention_scale=_SCALE)
        delta = actual[start:end] - expected
        maximum = max(maximum, float(delta.abs().max()))
        delta_square += float(delta.square().sum())
        total_square += float(expected.square().sum())
    relative = (delta_square / max(total_square, 1e-30))**0.5
    if relative > .02:
        raise RuntimeError(f"local SFA independent FP32 relative L2 {relative} exceeds .02")
    return {"all_query_rows": len(ids), "max_abs": maximum, "relative_l2": relative,
            "relative_l2_limit": .02, "query_chunk": 4}


def _check(result, backend, main_states, index, cpu_index, stock, layout, meta, fixture):
    local = backend.local_query_layout
    raw = result.saved.indices.cpu()
    rows = list(local.global_ids)
    stock_indices, stock_values = (tensor.cpu() for tensor in stock)
    torch.testing.assert_close(raw.sort(-1).values, stock_indices[rows].sort(-1).values, rtol=0, atol=0)
    torch.testing.assert_close(result.values.cpu().view(torch.int16),
                               stock_values[list(meta.q_global_ids)].view(torch.int16), rtol=0, atol=0)
    _check_indices(raw[:, 0], replace(meta, q_global_ids=local.global_ids), fixture, cpu_index)
    aligned = stock_indices.clone()
    aligned[rows] = raw
    baseline = _reference(main_states, aligned.to(layout.device), layout)
    for field, (actual, expected) in enumerate(zip((result.output, result.maximum, result.denominator), baseline)):
        expected = expected.cpu()
        expected = expected[list(meta.q_global_ids)] if field == 0 else expected[:, list(meta.q_global_ids)]
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    snapshots = (*[value.cpu() for value in result.selected_requests], raw)
    try:
        evidence = backend.validate_trace(result, tuple(trace.cpu() for trace in result.phase_traces),
                                          result.transport_trace.cpu(), require_overlap=False,
                                          selected_snapshots=snapshots)
    except ValueError as error:
        raise ValueError(f"{error}; epoch={result.epoch}, transport={result.transport_trace.cpu().tolist()}, "
                         f"counts={snapshots[1].tolist()}") from error
    membership = snapshots[2][:meta.global_valid_queries] > 0
    torch.testing.assert_close(result.global_keys[0].cpu().view(torch.int16), index[1].cpu().view(torch.int16),
                               rtol=0, atol=0)
    for field_index, (actual, expected) in enumerate(zip(result.global_keys[1:],
                                                        (main_states[1][:, None], main_states[3][:, None]))):
        actual, expected = actual.cpu(), expected.cpu()
        torch.testing.assert_close(actual[membership].view(torch.int16), expected[membership].view(torch.int16),
                                   rtol=0, atol=0)
        if not bool(torch.isnan(actual[~membership]).all()):
            cold = actual[~membership]
            bad = (~torch.isnan(cold)).nonzero()
            details = {"field": field_index, "bad_elements": len(bad), "first_offsets": bad[:8].tolist(),
                       "first_values": [float(cold[tuple(offset)]) for offset in bad[:8].tolist()],
                       "epoch": result.epoch, "counts": snapshots[1].tolist(),
                       "transport": result.transport_trace.cpu().tolist()}
            raise RuntimeError(f"local selected pull overwrote unrequested poisoned main-KV: {details}")
    result.saved.validate_versions()
    return {"trace": evidence, "fp32_oracle": _oracle(result, main_states, meta),
            "selected_sets_and_values_stock_exact": True, "slot_aligned_sfa_stock_exact": True,
            "complete_index_key_bits_exact": True, "selected_main_key_bits_exact": True,
            "unrequested_main_rows_poisoned": int((~membership).sum()),
            "local_queries": len(rows), "global_queries": meta.global_valid_queries,
            "query_cumulative_lengths": local.cumulative_queries, "query_collectives": 0,
            "strict_causal_bound_rows": sum(exact < conservative for exact, conservative in local.causal_bounds())}


def _run_fixture(report, lengths, heads, dtype, pattern, device, *, smoke):
    rank, size = dist.get_rank(), dist.get_world_size()
    meta = _metadata(lengths, pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "dsa-local-cp", DsaWorkspaceSpec(capacity, sum(lengths), capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    fixture = "signed_decreasing" if max(lengths) > 2048 else "random_signed"
    _, layout, cpu_index, index = _fixture(lengths, fixture, device)
    if dtype == torch.float32:
        cpu_index = (*cpu_index[:2], cpu_index[2].float() * 1.003 + .000013)
        index = (*index[:2], cpu_index[2].to(device))
    main_states = _main_states(sum(lengths), heads, device)
    streams = (torch.npu.current_stream(), torch.npu.Stream())
    try:
        for mode in (("owner", "empty") if smoke else ("owner", "interior", "first_sequence", "empty")):
            tile_meta = _tile(meta, mode)
            for groups in ((7,) if smoke or max(lengths) > 2048 else (1, 7, 19)):
                backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=heads,
                                                 attention_scale=_SCALE, schedule=MixedSfaSchedule(groups))
                backend.prepare_local_queries(tile_meta.q_global_ids)
                pending = []
                for repeat in range(2):
                    report["stage"] = {"lengths": lengths, "heads": heads, "dtype": str(dtype),
                                       "pattern": pattern, "tile": mode, "groups": groups, "repeat": repeat}
                    states = (main_states[0], (main_states[1] + repeat * .015625).bfloat16(), main_states[2],
                              (main_states[3] * (1 if repeat == 0 else -1)).bfloat16())
                    local_main, local_index = _local_states(tile_meta, states, index)
                    invocation = workspace.prepare(replace(meta, layer=repeat, microbatch=repeat,
                                                           invocation=workspace.transport_epoch + 1))
                    streams[repeat].wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(streams[repeat]):
                        result = backend.forward_local(invocation, local_main, local_index)
                    pending.append((dict(report["stage"]), result, states))
                torch.npu.synchronize()
                stock = torch.ops.npu.npu_lightning_indexer(
                    *index, actual_seq_lengths_query=layout.length_tensor,
                    actual_seq_lengths_key=layout.length_tensor, layout_query="TND", layout_key="TND",
                    sparse_count=2048, sparse_mode=3, return_value=True)
                for stage, result, states in pending:
                    report["stage"] = stage
                    report["cases"].append({**stage, **_check(result, backend, states, index, cpu_index, stock,
                                                            layout, tile_meta, fixture),
                                            "cross_stream_submissions_before_observation": True})
    finally:
        workspace.close()
        root.close()
    if shmem.debug_state()["reference_count"] != 0:
        raise RuntimeError("local fused CP root did not close cleanly")


def run_validation(report: dict, output_dir: Path, *, smoke: bool = False, long_history: bool = False) -> None:
    """Validate CP1/2/4 real owner publication and six-stage closure for local and empty tiles."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    report.update(rank=rank, cp_size=dist.get_world_size(), cases=[], package_path=hyper_parallel.__file__,
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fixtures = [((3, 10), 32, torch.bfloat16, "contiguous")]
    if not smoke:
        fixtures = [((3, 10), heads, dtype, pattern) for heads in (32, 64)
                    for dtype in (torch.bfloat16, torch.float32) for pattern in ("contiguous", "strided", "empty")]
    if long_history:
        fixtures.extend(((3, 2113), heads, dtype, "strided") for heads in (32, 64)
                        for dtype in (torch.bfloat16, torch.float32))
    output_dir.mkdir(parents=True, exist_ok=True)
    for lengths, heads, dtype, pattern in fixtures:
        _run_fixture(report, lengths, heads, dtype, pattern, torch.device(f"npu:{local_rank}"), smoke=smoke)
        (output_dir / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    loaded = {line.split()[-1] for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
              if any(name in line for name in
                     ("libcust_opapi.so", "libcust_opmaster_rt2.0.so", "libhyper_parallel_mega_moe_torch.so"))}
    report.update(status="passed", stage="complete", case_count=len(report["cases"]),
                  loaded_library_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in loaded})
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Persist per-rank correctness and transport results without claiming training acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--long-history", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "scope": "local-query CP LI plus selected main-KV pull and SFA",
              "backward": False, "kl": False, "performance_measurement": False}
    try:
        run_validation(report, args.output_dir, smoke=args.smoke, long_history=args.long_history)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"rank{os.environ.get('RANK', '0')}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
