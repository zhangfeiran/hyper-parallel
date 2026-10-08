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
"""Thin in-process launcher for the experimental single-NPU mixed SFA forward/backward probe."""

import importlib
import json

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_dsa_mixed_grad(tmp_path):
    """Verify real SFA partition execution and three-member evidence on one 910B3 NPU."""
    report = {"status": "running", "scope": "CP1 SFA mixed-group forward/backward", "backward": True}
    try:
        worker = importlib.import_module("hyper_parallel.core.multicore.examples.mega_dsa_mixed_grad_validate")
        worker.run_validation(report, long_history=True)
    except Exception as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        (tmp_path / "mixed_sfa_grad_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
