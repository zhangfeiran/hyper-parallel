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
"""Real CP1/2/4 device KV pull and fused LI/SFA correctness and progress evidence."""

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

from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.examples.mega_dsa_fused_forward_validate import (
    _main_states,
)
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_indexer_validate import (
    _fixture as indexer_fixture,
)
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_tile_validate import (
    _reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot

_SCALE = 192**-0.5


def _metadata(lengths: tuple, pattern: str, size: int, rank: int) -> DsaBatchMeta:
    total = sum(lengths)
    if pattern == "contiguous":
        owners = tuple(token * size // total for token in range(total))
    elif pattern == "strided":
        owners = tuple(token % size for token in range(total))
    elif pattern == "empty":
        owners = (0,) * total
    else:
        owners = tuple(min(token, total - 1 - token) % size for token in range(total))
    ids = tuple(tuple(token for token in reversed(range(total)) if owners[token] == peer) for peer in range(size))
    offsets = tuple(ids[peer].index(token) for token, peer in enumerate(owners))
    queries = tuple(reversed(ids[rank]))
    queries = queries[1:] + queries[:1]
    cu = [0]
    for count in lengths:
        cu.append(cu[-1] + count)
    return DsaBatchMeta(tuple(cu), queries, ids[rank], owners, offsets, cp_ranks=tuple(range(size)),
                        root_pes=tuple(range(size)), cp_rank=rank, layout_id=pattern)


def _full_states(lengths: tuple, heads: int, fixture: str, repeat: int, device: torch.device) -> tuple:
    _, layout, _, index = indexer_fixture(lengths, fixture, device)
    main_states = _main_states(sum(lengths), heads, device)
    # Different publication values expose stale ready/ACK epochs even when the same arena is reused.
    index = (index[0], (index[1] * (1 + repeat * 0.125)).bfloat16(), index[2])
    main_states = (main_states[0], (main_states[1] + repeat * 0.015625).bfloat16(), main_states[2],
            (main_states[3] * (1 if repeat % 2 == 0 else -1)).bfloat16())
    selection = torch.ops.npu.npu_lightning_indexer(
        *index, actual_seq_lengths_query=layout.length_tensor, actual_seq_lengths_key=layout.length_tensor,
        layout_query="TND", layout_key="TND", sparse_count=2048, sparse_mode=3, return_value=True)
    expected = (*selection, *_reference(main_states, selection[0], layout))
    return layout, main_states, index, tuple(tensor.cpu() for tensor in expected)


def _local_states(meta: DsaBatchMeta, main_states: tuple, index: tuple) -> tuple:
    queries, keys = list(meta.q_global_ids), list(meta.kv_global_ids)
    local_main = (main_states[0][queries].contiguous(), main_states[1][keys].contiguous(),
                  main_states[2][queries].contiguous(), main_states[3][keys].contiguous())
    local_index = (index[0][queries].contiguous(), index[1][keys, 0].contiguous(), index[2][queries].contiguous())
    return local_main, local_index


def _assert_outputs(result, baseline: tuple, layout: CannDsaLayout, meta: DsaBatchMeta,
                    main_states: tuple, index: tuple) -> None:
    rows = list(meta.q_global_ids)
    actual = (result.global_indices, result.values, result.output, result.maximum, result.denominator)
    native = baseline[0].to(layout.device)
    global_ids = layout.sequence_to_global_indices(native).cpu()
    expected = (global_ids[rows], baseline[1][rows], baseline[2][rows], baseline[3][:, rows], baseline[4][:, rows])
    for tensor, reference in zip(actual, expected):
        torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)
    torch.testing.assert_close(result.values.cpu().view(torch.int16), baseline[1][rows].view(torch.int16),
                               rtol=0, atol=0)
    for tensor, reference in zip(result.global_keys, (index[1], main_states[1][:, None], main_states[3][:, None])):
        torch.testing.assert_close(tensor.cpu().view(torch.int16), reference.cpu().view(torch.int16), rtol=0, atol=0)


def _run_fixture(report: dict, lengths: tuple, heads: int, pattern: str, fixture: str, device: torch.device) -> None:
    rank, size = dist.get_rank(), dist.get_world_size()
    meta = _metadata(lengths, pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "dsa-fused-cp", DsaWorkspaceSpec(capacity, sum(lengths), capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    streams = (torch.npu.current_stream(), torch.npu.Stream(), torch.npu.Stream())
    try:
        for groups in (1, 2, 7, 19):
            invocation = workspace.prepare(meta)
            backend = FusedDsaCpForwardProbe(workspace, invocation, heads=heads,
                                             attention_scale=_SCALE, schedule=MixedSfaSchedule(groups))
            prepared = [_full_states(lengths, heads, fixture, repeat, device) for repeat in range(3)]
            local_states = [_local_states(meta, item[1], item[2]) for item in prepared]
            invocations = [workspace.prepare(replace(meta, invocation=workspace.transport_epoch + repeat + 1,
                                                     layer=repeat % 2, microbatch=repeat)) for repeat in range(3)]
            pending = []
            for repeat in range(3):
                report["stage"] = {"lengths": lengths, "heads": heads, "pattern": pattern,
                                   "fixture": fixture, "groups": groups, "repeat": repeat}
                layout, main_states, index, baseline = prepared[repeat]
                local_main, local_index = local_states[repeat]
                invocation = invocations[repeat]
                stream = streams[repeat]
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    result = backend.forward(invocation, local_main, local_index)
                pending.append((dict(report["stage"]), result, layout, main_states, index, baseline))
            # Observe only after all streams have submitted; arena reuse must be ordered by the lease's events.
            torch.npu.synchronize()
            for stage, result, layout, main_states, index, baseline in pending:
                report["stage"] = stage
                _assert_outputs(result, baseline, layout, meta, main_states, index)
                evidence = backend.validate_trace(result, tuple(trace.cpu() for trace in result.phase_traces),
                                                  result.transport_trace.cpu(), require_overlap=max(lengths) > 2048)
                report["cases"].append({**report["stage"], "trace": evidence,
                                        "five_outputs_stock_exact": True, "all_global_kv_storage_bits_exact": True,
                                        "indexer_values_storage_bits_exact": True,
                                        "cross_stream_submission_before_observation": True,
                                        "owner_counts": backend.layout.counts})
    finally:
        workspace.close()
        root.close()
    if shmem.debug_state()["reference_count"] != 0:
        raise RuntimeError("fused CP root did not close cleanly")


def run_validation(report: dict, output_dir: Path, *, long_history: bool, smoke: bool = False) -> None:
    """Verify actual remote payloads, packed ownership, phase closure and serialized cross-stream reuse."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    if "910B3" not in torch.npu.get_device_name(local_rank):
        raise RuntimeError("fused CP initially requires verified Ascend 910B3 hardware")
    if smoke:
        torch.npu.set_op_timeout_ms(30000)
        report["diagnostic_kernel_timeout_ms"] = 30000
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    report.update(rank=dist.get_rank(), cp_size=dist.get_world_size(), cases=[],
                  device=torch.npu.get_device_name(local_rank))
    fixtures = [((3, 10), 32, "contiguous", "random_signed")]
    if not smoke:
        fixtures.extend(((3, 10), heads, pattern, "random_signed_fp32_weights")
                        for heads in (32, 64) for pattern in ("strided", "zigzag", "empty"))
    if long_history:
        fixtures.extend(((2176, 2240), heads, "contiguous", "random_signed_fp32_weights") for heads in (32, 64))
    output_dir.mkdir(parents=True, exist_ok=True)
    for lengths, heads, pattern, fixture in fixtures:
        _run_fixture(report, lengths, heads, pattern, fixture, device)
        (output_dir / f"rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report.update(status="passed", stage="complete", case_count=len(report["cases"]))
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Persist terminal results and the failing stage for each CP member."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--long-history", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    report = {"status": "running", "scope": "native CP device KV pull plus fused LI/SFA forward",
              "backward": False, "selected_kl": False, "performance_measurement": False}
    try:
        run_validation(report, args.output_dir, long_history=args.long_history, smoke=args.smoke)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
