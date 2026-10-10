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
"""Worker-only local-query fused forward device acceptance."""

import json
import os
import tempfile
import traceback
from pathlib import Path

from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_validate import run_validation


def _run_validation(*, long_history: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="mega-dsa-local-cp-") as directory:
        path = Path(os.environ.get("HP_DSA_LOCAL_CP_OUTPUT", directory))
        report = {"status": "running", "scope": "local-query fused CP forward; no training acceptance"}
        try:
            run_validation(report, path, long_history=long_history)
        except Exception as error:
            report.update(status="error", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            path.mkdir(parents=True, exist_ok=True)
            (path / f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2) + "\n")


def test_mega_dsa_local_cp() -> None:
    """Check true local Q positions, empty tiles and actual selected owner KV publication."""
    _run_validation(long_history=False)


def test_mega_dsa_local_cp_long() -> None:
    """Check native K2048 truncation and cold-key omission in addition to local short cases."""
    _run_validation(long_history=True)
