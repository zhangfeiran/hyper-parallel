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
"""Multi-card compact local-query LI verification with explicit global causal positions."""

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
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # Registers the NPU backend.

import hyper_parallel
from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_indexer_validate import _fixture
from hyper_parallel.core.multicore.modules.mega_dsa.local_query import DsaLocalQueryLayout, local_indexer_forward_probe
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_indexer import validate_fused_indexer_traces
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.reference import indexer_reference


def _query_meta(lengths, pattern, size, rank):
    meta = _metadata(lengths, pattern, size, rank)
    if pattern == "interior":
        ids = tuple(token for token in meta.q_global_ids
                    if lengths[meta.sequence_position(token)[0]] // 4 <= meta.sequence_position(token)[1]
                    < 3 * lengths[meta.sequence_position(token)[0]] // 4)
        meta = replace(meta, q_global_ids=ids)
    elif pattern == "empty" and size == 1:
        meta = replace(meta, q_global_ids=())
    return meta


def _check_indices(indices, meta, fixture, cpu):
    counts = (indices >= 0).sum(-1)
    expected = torch.tensor([min(2048, meta.sequence_position(token)[1] + 1)
                             for token in meta.q_global_ids], dtype=torch.long)
    torch.testing.assert_close(counts, expected, rtol=0, atol=0)
    for row, token in enumerate(meta.q_global_ids):
        _sequence, position = meta.sequence_position(token)
        active = indices[row][indices[row] >= 0]
        if bool((active > position).any()) or len(active.unique()) != len(active):
            raise RuntimeError("local indexer emitted a future or duplicate key")
    if fixture in ("increasing", "signed_decreasing"):
        rows = []
        for token in meta.q_global_ids:
            position = meta.sequence_position(token)[1]
            count = min(position + 1, 2048)
            start = max(0, position + 1 - 2048) if fixture == "increasing" else 0
            rows.append(list(range(start, start + count)) + [-1] * (2048 - count))
        analytic = torch.tensor(rows, dtype=torch.int32).reshape(-1, 2048)
        torch.testing.assert_close(indices.sort(-1).values, analytic.sort(-1).values, rtol=0, atol=0)
    else:
        oracle_meta = replace(meta, kv_global_ids=tuple(range(meta.global_valid_queries)))
        oracle = indexer_reference(cpu[0][list(meta.q_global_ids)].float(), cpu[1][:, 0].float(),
                                   cpu[2][list(meta.q_global_ids)].float(), oracle_meta, sparse_count=2048)
        starts = torch.tensor([token - meta.sequence_position(token)[1] for token in meta.q_global_ids])[:, None]
        global_indices = torch.where(indices >= 0, indices + starts, -1).to(torch.int32)
        torch.testing.assert_close(global_indices.sort(-1).values, oracle.sort(-1).values, rtol=0, atol=0)


def _run_case(report, states, cpu, stock, lengths, fixture, pattern, groups, size, rank):
    meta = _query_meta(lengths, pattern, size, rank)
    layout = DsaLocalQueryLayout(meta, states[0].device)
    local = (states[0][list(meta.q_global_ids)].contiguous(), states[1],
             states[2][list(meta.q_global_ids)].contiguous())
    schedule = MixedSfaSchedule(groups)
    scratch = None
    forwards = []
    for _repeat in range(2):
        indices, values, traces, scratch = local_indexer_forward_probe(*local, layout, schedule, scratch)
        torch.npu.synchronize()
        snapshot, value_snapshot = indices.cpu(), values.cpu()
        baseline = stock[0][list(meta.q_global_ids)].cpu()
        torch.testing.assert_close(snapshot.sort(-1).values, baseline.sort(-1).values, rtol=0, atol=0)
        torch.testing.assert_close(value_snapshot.view(torch.int16),
                                   stock[1][list(meta.q_global_ids)].cpu().view(torch.int16), rtol=0, atol=0)
        _check_indices(snapshot[:, 0], meta, fixture.removesuffix('_fp32_weights'), cpu)
        evidence = validate_fused_indexer_traces(tuple(trace.cpu() for trace in traces), schedule,
                                                require_ld=bool(meta.q_global_ids and max(lengths) > 2048))
        forwards.append(evidence)
    report['cases'].append({'lengths': lengths, 'fixture': fixture, 'pattern': pattern, 'groups': groups,
                            'weight_dtype': str(local[2].dtype), 'local_queries': len(meta.q_global_ids),
                            'global_queries': sum(lengths), 'gathered_query_rows': 0,
                            'query_cumulative_lengths': layout.cumulative_queries,
                            'strict_causal_bound_rows': sum(exact < conservative for exact, conservative
                                                           in layout.causal_bounds()),
                            'stock_selected_set_exact': True, 'stock_values_exact': True,
                            'independent_position_oracle': True, 'scratch_bytes': scratch.numel(),
                            'forwards': forwards})


def run_validation(report: dict, output_dir: Path, *, smoke: bool = False) -> None:
    """Verify genuine local rows and empty owners concurrently on one, two or four NPUs."""
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.npu.set_device(local_rank)
    dist.init_process_group('hccl', timeout=timedelta(minutes=10))
    size, rank = dist.get_world_size(), dist.get_rank()
    report.update(rank=rank, cp_size=size, cases=[], package_path=hyper_parallel.__file__,
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fixtures = [((3,10), 'random_signed'), ((3,10), 'random_signed_fp32_weights'),
                ((3,2113), 'increasing'), ((2176,2240), 'signed_decreasing'),
                ((3,2113), 'increasing_fp32_weights'), ((2176,2240), 'signed_decreasing_fp32_weights')]
    if smoke:
        fixtures = fixtures[:1]
    for lengths, fixture in fixtures:
        source_fixture = fixture.removesuffix('_fp32_weights') if max(lengths) > 2048 else fixture
        _meta, full_layout, cpu, states = _fixture(lengths, source_fixture, torch.device(f'npu:{local_rank}'))
        if max(lengths) > 2048 and fixture.endswith('_fp32_weights'):
            cpu = (*cpu[:2], cpu[2].float() * 1.003 + .000013)
            states = (*states[:2], cpu[2].to(states[0].device))
        stock = torch.ops.npu.npu_lightning_indexer(
            *states, actual_seq_lengths_query=full_layout.length_tensor,
            actual_seq_lengths_key=full_layout.length_tensor, layout_query='TND', layout_key='TND',
            sparse_count=2048, sparse_mode=3, return_value=True)
        for pattern in ('contiguous', 'strided', 'zigzag', 'interior', 'empty'):
            for groups in ((7,) if max(lengths) > 2048 or smoke else (1,7,19)):
                report['stage'] = {'lengths': lengths, 'fixture': fixture, 'pattern': pattern, 'groups': groups}
                _run_case(report, states, cpu, stock, lengths, fixture, pattern, groups, size, rank)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    loaded = {line.split()[-1] for line in Path('/proc/self/maps').read_text(encoding='utf-8').splitlines()
              if any(name in line for name in
                     ('libcust_opapi.so','libcust_opmaster_rt2.0.so','libhyper_parallel_mega_moe_torch.so'))}
    report.update(status='passed', stage='complete',
                  loaded_library_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in loaded})
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Write per-rank native and independent explicit-position acceptance evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    report = {'status':'running','scope':'local LI only; full canonical K; no SHMEM or training acceptance'}
    try:
        run_validation(report,args.output_dir,smoke=args.smoke)
    except Exception as error:
        report.update(status='error',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/f"rank{os.environ.get('RANK','0')}.json").write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
