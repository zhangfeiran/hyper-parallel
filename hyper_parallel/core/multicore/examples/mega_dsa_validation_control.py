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
"""Collective agreement for verification failures before another native invocation or teardown."""

from __future__ import annotations

import traceback
from typing import Callable

import torch.distributed as dist


def collective_validation_check(check: Callable[[], dict], *, group: dist.ProcessGroup | None = None) -> dict:
    """Run a rank-local check, then make all members fail together before advancing.

    This helper belongs to validation control only. A local numerical assertion
    must not enter collective teardown while peers enter the next native lease.
    HCCL failures themselves still require launcher/process-group recovery.
    """
    result, failure = None, None
    try:
        result = check()
    except Exception as error:
        failure = {"rank": dist.get_rank(group), "error": repr(error), "traceback": traceback.format_exc()}
    failures = [None]*dist.get_world_size(group)
    dist.all_gather_object(failures, failure, group=group)
    active = tuple(item for item in failures if item is not None)
    if active:
        raise RuntimeError(f"collective validation failed before advancing: {active}")
    return result
