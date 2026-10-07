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
"""Restricted Python AST frontend with semantic lowering and CPU references."""

from __future__ import annotations

from hyper_parallel.core.multicore.frontend.diagnostics import FrontendError
from hyper_parallel.core.multicore.frontend.parser import static_range
from hyper_parallel.core.multicore.frontend.program import (
    Program,
    from_source,
    helper,
    program,
)
from hyper_parallel.core.multicore.ir.schedule import (
    HardwareSpec,
    TaskDAG,
    WorkerPipeline,
)

__all__ = [
    "FrontendError",
    "HardwareSpec",
    "Program",
    "TaskDAG",
    "WorkerPipeline",
    "from_source",
    "helper",
    "program",
    "static_range",
]
