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
"""Schedule selection metadata; device schedule lowering is a separate stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerPipeline:
    """Ordered stages per row-owning worker, pending family-specific lowering."""

    axis: int = 0
    workers: str = "aiv"

    def __post_init__(self):
        if self.axis != 0 or self.workers != "aiv":
            raise ValueError("V0 WorkerPipeline supports row-axis AIV workers only")


@dataclass(frozen=True)
class TaskDAG:
    """An explicit legacy queue policy, pending family-specific lowering."""

    policy: str

    def __post_init__(self):
        if not isinstance(self.policy, str) or not self.policy:
            raise ValueError("TaskDAG requires an explicit policy name")
