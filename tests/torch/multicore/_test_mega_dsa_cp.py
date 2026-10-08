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
"""Worker-only CANN CP reference parity validation."""

import tempfile
from pathlib import Path

from hyper_parallel.core.multicore.examples.mega_dsa_cp_validate import run_validation


def test_mega_dsa_cp_parity() -> None:
    """Validate three packed CP shard patterns, gradient isolation and recomputation."""
    with tempfile.TemporaryDirectory(prefix="mega-dsa-cp-") as directory:
        report = {}
        run_validation(report, Path(directory))
        assert report["status"] == "passed", report
