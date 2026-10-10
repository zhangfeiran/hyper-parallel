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
"""Worker-only HCCL validation of local selected-KL index-gradient owner return."""

import hashlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.multicore.modules.mega_dsa.cp_reference import DsaCpLayout
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta


def _meta(rank, size):
    partitions = {1: ((6, 4, 0, 2, 1, 5, 3),), 2: ((4, 0), (6, 1, 5, 3, 2)),
                  4: ((4, 0), (6,), (), (2, 5, 1, 3))}
    partition = partitions[size]
    owners, offsets = [0] * 7, [0] * 7
    for owner, ids in enumerate(partition):
        for offset, token in enumerate(ids):
            owners[token], offsets[token] = owner, offset
    return DsaBatchMeta((0, 3, 7), partition[rank], tuple(reversed(partition[rank])),
                        tuple(owners), tuple(offsets), cp_ranks=tuple(range(size)), root_pes=tuple(range(size)),
                        cp_rank=rank)


def _key_partial(rank, size, empty):
    pattern = torch.arange(7 * 128, dtype=torch.float32).reshape(7, 128) / 64
    if empty or size == 4 and rank == 2:
        return torch.zeros_like(pattern)
    base = {1: (0.25,), 2: (256.0, -256.0), 4: (256.0, 0.25, 0.0, -256.0)}[size][rank]
    return torch.full_like(pattern, base) + (pattern if rank == 0 else 0)


def run_validation(report: dict) -> None:
    """Verify actual FP32 collective return, permutations and empty participation on CP1/2/4.

    This exercises only the gradient exchange; it does not execute a local KL
    producer, numerical teacher/predictor kernels, or a complete training step.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size not in (1, 2, 4):
            raise ValueError("local index owner validation requires CP1/2/4")
        device = torch.device("npu", local_rank)
        meta = _meta(rank, size)
        layout = DsaCpLayout(meta, device)
        source = Path(inspect.getfile(DsaCpLayout)).resolve()
        report.update(rank=rank, cp_size=size, torch_version=torch.__version__,
                      torch_npu_version=torch_npu.__version__, cases=[], owner_return_source=str(source),
                      owner_return_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        for empty in (False, True):
            for weight_dtype in (torch.bfloat16, torch.float32):
                rows = () if empty else meta.q_global_ids[::2]
                tile = replace(meta, q_global_ids=rows)
                query = torch.arange(len(rows) * 64 * 128, dtype=torch.float32).reshape(len(rows), 64, 128) / 64
                weight = torch.arange(len(rows) * 64, dtype=torch.float32).reshape(len(rows), 64) / 32
                key = _key_partial(rank, size, empty)
                result = layout.reduce_local_index_gradients(
                    tuple(tensor.to(device) for tensor in (query, key, weight)), tile,
                    (torch.bfloat16, torch.bfloat16, weight_dtype))
                expected_key = sum(_key_partial(peer, size, empty) for peer in range(size))
                expected = (query.bfloat16(), expected_key[list(meta.kv_global_ids)].bfloat16(),
                            weight.to(weight_dtype))
                torch.npu.synchronize()
                for actual, wanted in zip(result, expected):
                    torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
                report["cases"].append({"empty_tile": empty, "weight_dtype": str(weight_dtype),
                                        "query_rows": len(rows), "owner_key_rows": len(meta.kv_global_ids),
                                        "passed": True})
        dist.barrier()
        report.update(status="passed", case_count=len(report["cases"]), full_model_acceptance=False,
                      local_kl_kernel_acceptance=False, sparse_gradient_return=False)
    finally:
        dist.destroy_process_group()


def test_mega_dsa_local_index_owner() -> None:
    """Run HCCL exchange checks and retain a per-rank report when requested."""
    report = {"status": "running", "scope": "local index gradient HCCL owner return"}
    try:
        run_validation(report)
    except Exception as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        output = os.environ.get("HP_DSA_LOCAL_INDEX_OWNER_OUTPUT")
        if output:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"rank{os.environ.get('RANK', '0')}.json"
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    test_mega_dsa_local_index_owner()
