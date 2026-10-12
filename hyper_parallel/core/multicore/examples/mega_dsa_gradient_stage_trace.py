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
"""Owned stage snapshots distinguish input, native backward and shared-return differences."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    gradient_error_metrics,
)


def compare_stage_fields(actual: Mapping[str, torch.Tensor | None],
                         expected: Mapping[str, torch.Tensor | None]) -> dict:
    """Compare matching snapshot schemas without assigning an acceptance budget."""
    if set(actual) != set(expected):
        raise ValueError('gradient stage schemas must match')
    result = {}
    for name, value in actual.items():
        target = expected[name]
        if value is None or target is None:
            result[name] = {'bit_exact': value is target, 'metrics': None}
            continue
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError('gradient stage shape and dtype must match')
        result[name] = {'bit_exact': bool(torch.equal(value, target)),
                        'metrics': (gradient_error_metrics(value.detach().cpu(), target.detach().cpu())
                                    if value.is_floating_point() else None)}
    return result


class GradientStageTrace:
    """Keep detached copies before reusable native buffers can be consumed or overwritten.

    Diagnostics add copies and synchronization and must not be used to report
    performance. Integer fields are compared exactly; only floating fields have
    numerical metrics. An inactive field remains explicitly None.
    """

    def __init__(self) -> None:
        """Create an empty trace for two isolated and two delayed microbatch backwards."""
        self._records = {}

    def record(self, phase: str, microbatch: int, stage: str,
               fields: Mapping[str, torch.Tensor | None]) -> None:
        """Snapshot one unique stage on the submission stream, retaining no source alias."""
        if phase not in ('isolated', 'delayed') or microbatch not in (0, 1) or isinstance(microbatch, bool):
            raise ValueError('gradient trace requires isolated/delayed phase and microbatch 0 or 1')
        if not isinstance(stage, str) or not stage or not fields:
            raise ValueError('gradient trace requires a named nonempty stage')
        key = phase, microbatch, stage
        if key in self._records:
            raise RuntimeError('gradient stage was recorded twice for one backward')
        self._records[key] = {name: None if value is None else value.detach().clone()
                              for name, value in fields.items()}

    def diagnose(self) -> dict:
        """Require complete paired coverage, then compare each stage in microbatch order."""
        stages = {key[2] for key in self._records}
        expected = {(phase, microbatch, stage) for phase in ('isolated', 'delayed')
                    for microbatch in (0, 1) for stage in stages}
        if not stages or set(self._records) != expected:
            raise RuntimeError('gradient trace requires complete isolated/delayed stage coverage')
        return {'full_model_acceptance': False,
                'stages': {stage: [compare_stage_fields(self._records['delayed', microbatch, stage],
                                                        self._records['isolated', microbatch, stage])
                                   for microbatch in (0, 1)] for stage in sorted(stages)}}
