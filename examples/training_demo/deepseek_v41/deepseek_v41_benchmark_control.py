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
"""Keep benchmark coordination on CPU and preserve failures before device exit hooks."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import traceback
from typing import Any

import torch.distributed as dist

from examples.training_demo.deepseek_v41.coordinate_deepseek_v41_cluster import write_json


def record_failure(work: Path, error: BaseException) -> None:
    """Save the original exception before native cleanup can wait for failed peers."""
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'failure.json').exists():
        return
    progress = work / 'progress.json'
    state = json.loads(progress.read_text()) if progress.exists() else None
    write_json(work / 'failure.json', dict(rank=int(os.environ.get('RANK', '0')),
        utc=datetime.now(timezone.utc).isoformat(), progress=state, error_type=type(error).__name__,
        error=str(error), traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__))))


class BenchmarkControl:
    """Coordinate matched steps without inserting timing collectives into HCCL WORLD."""

    def __init__(self, work: Path) -> None:
        """Use a separate namespace in the static torchrun agent's TCPStore."""
        if not dist.is_initialized():
            raise RuntimeError('CPU benchmark control requires an initialized training world')
        self.work = work
        self.timeout = timedelta(seconds=300)
        host, port = os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT'])
        if not host or not 1024 <= port < 65536:
            raise ValueError('Invalid static torchrun endpoint')
        self.store = dist.PrefixStore('dsv41/benchmark/',
                                     dist.TCPStore(host, port, None, False, timeout=self.timeout))
        self.step: int | None = None
        write_json(work / 'benchmark-control.json', dict(backend='tcpstore', timing_device='cpu',
            timeout_seconds=self.timeout.total_seconds(), training_process_group_unchanged=True,
            world_size=dist.get_world_size(), rank=dist.get_rank()))

    def mark(self, phase: str, step: int, **metadata: Any) -> None:
        """Record bounded local progress outside the measured optimizer interval."""
        write_json(self.work / 'progress.json', dict(rank=dist.get_rank(), step=step, phase=phase,
                   utc=datetime.now(timezone.utc).isoformat(), **metadata))

    def barrier(self, step: int) -> None:
        """Identify missing ranks on a CPU control barrier before starting the timer."""
        self.mark('waiting_for_step', step)
        self.step = step
        self.store.set(f'barrier/{step}/{dist.get_rank()}', '1')
        self.store.wait([f'barrier/{step}/{rank}' for rank in range(dist.get_world_size())])

    def maximum(self, seconds: float) -> float:
        """Reduce synchronized local elapsed times on CPU after the timer has stopped."""
        step, rank = self.step, dist.get_rank()
        if step is None:
            raise RuntimeError('Missing CPU timing barrier')
        self.store.set(f'time/{step}/{rank}', seconds.hex())
        if rank == 0:
            values = self.store.multi_get([f'time/{step}/{peer}' for peer in range(dist.get_world_size())])
            self.store.set(f'maximum/{step}', max(float.fromhex(value.decode()) for value in values).hex())
        return float.fromhex(self.store.get(f'maximum/{step}').decode())

    def close(self) -> None:
        """Release only this control store, keeping training and SHMEM groups untouched."""
        self.store = None
