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
"""Mapped TP native FP32 tile SUM stress, independently of LI/SFA/KL math."""

from __future__ import annotations

import argparse
import json
import os
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
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumProbe,
    TpTileSumSpec,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _input(shape, root_pe, replay, first_pe, last_pe):
    if replay == 1:
        return torch.zeros(shape, dtype=torch.float32)
    values = ((torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.int64).reshape(shape)
               + root_pe * 17 + replay * 11) % 127 - 63).float() / 64
    if root_pe == first_pe:
        values[0, :, 0] = torch.nan
        values[0, :, 1] = torch.inf
        values[0, :, 2] = -torch.inf
        values[0, :, 3] = torch.inf
        values[0, :, 4] = -0.0
    elif root_pe == last_pe:
        values[0, :, 3] = -torch.inf
    return values


def _case(scenario, probe, streams, rounds, replay, epoch):
    if not probe.participates:
        return {'participates': False}
    shape = (rounds, probe.spec.slots, probe.spec.width)
    expected = torch.zeros(shape, dtype=torch.float32)
    for peer in probe.root_pes:
        expected.add_(_input(shape, peer, replay, probe.root_pes[0], probe.root_pes[-1]))
    source = _input(shape, probe.root.rank, replay, probe.root_pes[0], probe.root_pes[-1]).to(probe.root.device)
    invocation = probe.prepare(rounds=rounds, epoch=epoch,
                               identity=(epoch, 0, 0, probe.root.generation, rounds * 100 + replay))
    stream = streams[replay % 2]
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        output = probe.run(source, invocation)
    stream.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if replay == 1 and actual.count_nonzero().item() != 0:
        raise RuntimeError('zero replay retained a preceding nonzero/NaN tile')
    offset = probe.spec.control_offset
    controls = probe.arena[offset:offset + probe.spec.slots * 640].view(torch.int64).reshape(probe.spec.slots, 80).cpu()
    sequence = epoch * 2**32 + rounds
    if not (controls[:, 0] == sequence).all():
        raise RuntimeError('native TP source readiness did not complete every slot')
    for peer in range(len(probe.members)):
        if not (controls[:, (2 + peer) * 8] == sequence).all():
            raise RuntimeError('native TP final ACK is missing')
    scenario['cases'].append({'rounds': rounds, 'replay': replay, 'epoch': epoch, 'slots': probe.spec.slots,
                              'width': probe.spec.width, 'tp_size': len(probe.members), 'passed': True,
                              'exact_fp32_sum': True, 'final_readers_acknowledged': True,
                              'arena_bytes': probe.spec.arena_bytes})
    return {'participates': True}


def _fixture(report, members, group, name, specification):
    root = SharedShmemRoot(torch.device('npu', int(os.environ.get('LOCAL_RANK', '0'))), root_group=dist.group.WORLD)
    probe = TpTileSumProbe(root, 'tp-tiles', specification, tp_group=group)
    scenario = {'name': name, 'members': members, 'participates': probe.participates, 'cases': []}
    report['scenarios'].append(scenario)
    try:
        probe.bind()
        streams = (torch.npu.Stream(), torch.npu.Stream())
        epoch = 0
        for rounds in (1, 17, 257):
            for replay in range(3):
                epoch += 1
                collective_validation_check(partial(_case, scenario, probe, streams, rounds, replay, epoch))
        scenario['status'] = 'passed'
    finally:
        probe.close()
        root.close()


def main() -> None:
    """Validate subgroup mapping, fixed scratch, many rounds and nonzero/zero reuse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'scenarios': [], 'scope': 'native mapped FP32 TP tile transport',
              'native_li_kl_fusion_acceptance': False, 'model_acceptance': False, 'performance_acceptance': False}
    try:
        torch.npu.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError('native TP validator requires four root members')
        report['rank'] = dist.get_rank()
        singleton = tuple(((rank,), dist.new_group([rank])) for rank in range(4))
        paired = tuple((members, dist.new_group(list(members))) for members in ((1, 3), (0, 2)))
        own_single = singleton[dist.get_rank()]
        own_pair = next(item for item in paired if dist.get_rank() in item[0])
        scenarios = ((own_single, 'singleton'), (paired[0], 'one_subgroup'),
                     (own_pair, 'partitioned'), ((tuple(range(4)), dist.group.WORLD), 'world'))
        for (members, group), name in scenarios:
            for slots, width in ((1, 8), (7, 512), (40, 2048)):
                _fixture(report, members, group, name, TpTileSumSpec(slots, width))
        report.update(status='passed', loaded_library_sha256=_loaded_library_identity())
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
