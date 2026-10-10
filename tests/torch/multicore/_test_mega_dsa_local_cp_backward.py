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

from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    run_autograd_validation, run_validation,
)


def _run(validator, environment_key, scope, **options):
    with tempfile.TemporaryDirectory(prefix="mega-dsa-local-cp-backward-") as directory:
        path = Path(os.environ.get(environment_key, directory))
        report = {"status": "running", "scope": scope, "backward": True}
        try:
            validator(report, path, smoke=False, **options)
            if report["status"] != "passed":
                raise RuntimeError(f"native local-Q main backward did not pass: {report}")
        except Exception as error:
            report.update(status="error", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            path.mkdir(parents=True, exist_ok=True)
            rank = os.environ.get("RANK", "0")
            (path / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def test_mega_dsa_local_cp_backward() -> None:
    """Check native gradient math, phases and FP32 owner return against the independent oracle gate."""
    profile = os.environ.get("HP_DSA_REFERENCE_PROFILE")
    policy = os.environ.get("HP_DSA_REFERENCE_POLICY_SHA256")
    if bool(profile) != bool(policy):
        raise ValueError("hardware backward requires both a reference profile and its independent policy commitment")
    options = {"reference_profile": Path(profile), "profile_policy_sha256": policy} if profile else {}
    _run(run_validation, "HP_DSA_LOCAL_CP_BACKWARD_OUTPUT", "native local-Q main backward", **options)


def test_mega_dsa_local_cp_autograd() -> None:
    """Check local main autograd lifecycle while retaining the independent numerical acceptance boundary."""
    _run(run_autograd_validation, "HP_DSA_LOCAL_CP_AUTOGRAD_OUTPUT", "local main autograd lifecycle")
