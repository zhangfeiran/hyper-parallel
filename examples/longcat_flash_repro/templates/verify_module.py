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
import json
from pathlib import Path

R = Path(__file__).resolve().parent
reports = []
for folder in ["precision-t2048", "precision-edges"]:
    for rank in range(4):
        p = R / folder / f"rank{rank}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        assert (
            d["rank"] == rank
            and d["shmem_after_close"]["reference_count"] == 0
            and not Path(f"/proc/{d['pid']}").exists()
        )
        for x in d["results"]:
            assert x.get("strict_pass", x.get("pass"))
            for v in x["comparison"].values():
                assert v.get("max_abs", 0) == 0 and v.get("unused_gradient_identical", True)
        reports.append(
            {
                "path": str(p.relative_to(R)),
                "rank": rank,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "cases": len(d["results"]),
            }
        )
(R / "precision-results.json").write_text(
    json.dumps(
        {
            "status": "PASS",
            "full_value_max_abs": 0,
            "geometry": {
                "ep": 4,
                "real": 32,
                "identity": 16,
                "local_real": 8,
                "local_identity": 4,
                "tokens": 2048,
                "hidden": 6144,
                "intermediate": 2048,
                "top_k": 12,
            },
            "reports": reports,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print("All-rank module / empty-route precision PASS, max_abs=0")
