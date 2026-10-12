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
"""Native TP owner-gradient packing, root mapping, FP32 cancellation and one final cast."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import (
    collective_validation_check,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.tp_owner_return import (
    TpOwnerGradientReturnProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumProbe,
    TpTileSumSpec,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _fields(rows, rank, presence):
    result = []
    for width, active in zip((512, 64, 128), presence):
        if not active:
            result.append(None)
            continue
        values = torch.arange(rows*width, dtype=torch.float32).reshape(rows, width)
        small = (values % 11)/4096
        result.append(small+(1.003 if rank == 0 else -1 if rank == 1 else (rank-1)/256))
    return tuple(result)


def _case(report, backend, meta, presence, epoch):
    invocation = backend.prepare(meta, epoch=epoch, presence=presence)
    rank = dist.get_rank(backend.transport.group)
    rows = len(meta.kv_global_ids)
    cpu = _fields(rows, rank, presence)
    tensors = tuple(None if value is None else value.to(backend.transport.arena.device) for value in cpu)
    result = backend.return_owner(tensors, invocation, input_dtypes=(torch.bfloat16, torch.bfloat16, torch.float32))
    torch.npu.synchronize()
    expected = [None if value is None else torch.zeros_like(value) for value in cpu]
    for peer in range(len(backend.transport.members)):
        for total, value in zip(expected, _fields(rows, peer, presence)):
            if total is not None:
                total.add_(value)
    for actual, gradient, wanted, dtype in zip(result.owner_fp32, result.gradients, expected,
                                              (torch.bfloat16, torch.bfloat16, torch.float32)):
        if wanted is not None:
            torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
            torch.testing.assert_close(gradient.cpu(), wanted.to(dtype), rtol=0, atol=0)
    offset = backend.transport.spec.control_offset
    control = backend.transport.arena[offset:offset+backend.transport.spec.slots*640]
    control = control.view(torch.int64).reshape(-1, 80).cpu()
    expected_sequence = epoch*2**32+invocation.transport.rounds
    for slot in control:
        if int(slot[0]) != expected_sequence:
            raise RuntimeError('TP owner return lacks final publication')
        if any(int(slot[(2+peer)*8]) != expected_sequence for peer in range(len(backend.transport.members))):
            raise RuntimeError('TP owner return lacks final reader ACK')
    report['cases'].append({'passed': True, 'rows': rows, 'owner_ids': meta.kv_global_ids,
        'presence': presence, 'epoch': epoch, 'slots': backend.transport.spec.slots,
        'rounds': invocation.transport.rounds, 'exact_ordered_fp32_sum': True, 'exact_final_cast': True,
        'final_readers_acknowledged': True, 'tp_members': backend.transport.members})
    return {'passed': True}


def main() -> None:
    """Validate bounded native return after already completed CP ownership reduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'cases': [], 'scope': 'native TP SUM of already CP-returned owner partials',
              'native_cp_return_integration': False, 'cp_tp_model_acceptance': False, 'performance_acceptance': False}
    try:
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError('TP owner return validation requires four root members')
        rank = dist.get_rank()
        singleton = tuple(dist.new_group([peer]) for peer in range(4))
        pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 2), (1, 3)))
        pair = next(group for members, group in pairs if rank in members)
        for group in (singleton[rank], pair, dist.group.WORLD):
            for slots in (1, 14, 38):
                root = SharedShmemRoot(torch.device('npu', int(os.environ['LOCAL_RANK'])), root_group=dist.group.WORLD)
                transport = TpTileSumProbe(root, 'tp-owner', TpTileSumSpec(slots, 512), tp_group=group)
                backend = TpOwnerGradientReturnProbe(transport)
                try:
                    transport.bind()
                    epoch = 0
                    for rows in (0, 5, 17, 2113):
                        meta = replace(DsaBatchMeta.packed((max(1, rows),)),
                                       kv_global_ids=tuple(reversed(range(rows))), heap_generation=root.generation)
                        for presence in ((True, True, True), (True, True, False),
                                         (False, False, True), (False, False, False)):
                            epoch += 1
                            collective_validation_check(partial(_case, report, backend, meta, presence, epoch))
                finally:
                    transport.close()
                    root.close()
        report.update(status='passed', loaded_library_sha256=_loaded_library_identity())
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
