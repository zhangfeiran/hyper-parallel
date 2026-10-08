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
"""Worker-only fused backward and FP32 owner-return certification."""

import json
import os
import tempfile
import traceback
from pathlib import Path

from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_backward_validate import (
    run_validation,
)


def test_mega_dsa_fused_cp_backward() -> None:
    """Check native gradient phases and exact FP32 owner accumulation over remote inbox stripes."""
    with tempfile.TemporaryDirectory(prefix="mega-dsa-fused-cp-backward-") as directory:
        path = Path(os.environ.get("HP_DSA_FUSED_CP_BACKWARD_OUTPUT", directory))
        report = {"status": "running", "scope": "native CP fused backward", "backward": True}
        try:
            run_validation(report, path, long_history=True, smoke=False)
            if report["status"] != "passed":
                raise RuntimeError(f"native CP fused backward did not pass: {report}")
        except Exception as error:
            report.update(status="error", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            path.mkdir(parents=True, exist_ok=True)
            rank = os.environ.get("RANK", "0")
            (path / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
