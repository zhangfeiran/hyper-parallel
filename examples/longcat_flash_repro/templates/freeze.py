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
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent
S = Path("__REPO__")
s = importlib.util.spec_from_file_location("helpers_freeze", R / "component/helpers.py")
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=S, text=True).strip()
checkout_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=S, text=True).strip()
files = [
    "optimization_plan.yaml",
    "component/helpers.py",
    "component/controller.py",
    "trainer/controller.py",
    "trainer/controller_prefix.py",
    "trainer/audit_train.py",
    "verify_module.py",
    "audit_raw.py",
    "aggregate.py",
    "pipeline.py",
] + [str(p.relative_to(R)) for p in (R / "trainer").glob("*.yaml")]
(R / "aggregation-freeze.json").write_text(
    json.dumps(
        {
            "created_before_formal_runs": True,
            "sha256": {name: hashlib.sha256((R / name).read_bytes()).hexdigest() for name in files},
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
(R / "frozen-provenance.json").write_text(
    json.dumps(
        {
            "candidate_commit": commit,
            "checkout_status": checkout_status,
            "candidate_source": str(S),
            "source_and_payload_sha256": m.seal(),
            "activation": "fused in both MoE backends",
            "native_payload": "Unchanged V7, CANN8.5.0 ascend910_93",
            "crop": (
                "EP4/E32/Z16 component; EP16/E128/Z64 one-layer trainer; eight real/four identity" + " shares per rank"
            ),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
(R / "fused-default.patch").write_bytes(subprocess.check_output(["git", "show", "--format=fuller", commit], cwd=S))
print(commit, "frozen before formal cases")
