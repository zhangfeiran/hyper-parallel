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
import re
import statistics
import subprocess
from pathlib import Path

from controller_prefix import ROOT, c, compare, run

try:
    raw = subprocess.run(["npu-smi", "info", "-m"], capture_output=True, text=True, check=True).stdout
    mapping = {
        (int(a), int(b)): int(v)
        for a, b, v in re.findall("^\\s*(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+\\d+\\s+Ascend910", raw, re.M)
    }
    source = c.seal()
    c.save(ROOT / "source-seal.json", source)
    run("initialize", ROOT / "initialize.yaml", mapping)
    if not Path("__OUTPUT__/warmstart/global_step_1/.metadata").exists():
        raise RuntimeError("new common warmstart missing")
    numerical = []
    for mode in ["native", "megamoe", "fast-native"]:
        c.save(ROOT / "status.json", {"status": "RUNNING", "case": "numerical-" + mode})
        current = run("numerical-" + mode, ROOT / (mode + ".yaml"), mapping)
        numerical.append(current)
        if c.seal() != source:
            raise RuntimeError("source changed during numerical acceptance")
        if len(numerical) > 1:
            comparison = compare(numerical[0], current)
            comparison["exact_updated_model_and_main_hashes"] = True
            c.save(ROOT / (current["name"] + "-comparison.json"), comparison)
            if comparison["status"] != "PASS" or len(comparison["rows"]) != 160:
                raise RuntimeError("ten-step original-width precision failed")
    pilots = []
    for index in range(3):
        name = f"pilot-{index}-fast-native"
        c.save(ROOT / "status.json", {"status": "RUNNING", "case": name})
        current = run(name, ROOT / "fast-native-performance.yaml", mapping)
        pilots.append(current)
        comparison = compare(numerical[0], current)
        c.save(ROOT / (name + "-comparison.json"), comparison)
        if comparison["status"] != "PASS":
            raise RuntimeError("native pilot precision failed")
    medians = [
        statistics.median([max((rank[i]["local_seconds"] for rank in row["steps"])) for i in range(5, 15)])
        for row in pilots
    ]
    tails = [
        c.percentile([max((rank[i]["local_seconds"] for rank in row["steps"])) for i in range(5, 15)], 0.9)
        for row in pilots
    ]
    span = (max(medians) - min(medians)) / statistics.median(medians)
    tail_span = (max(tails) - min(tails)) / statistics.median(tails)
    c.save(
        ROOT / "pilot-thresholds.json",
        {
            "p50_seconds": medians,
            "p90_seconds": tails,
            "minimum_speedup": 1 + max(0.01, 2 * span),
            "maximum_p90_regression": max(0.03, 2 * tail_span),
        },
    )
    formal = []
    order = ["native", "fast-native", "megamoe", "megamoe", "fast-native", "native"]
    for index, mode in enumerate(order):
        name = f"performance-{index}-{mode}"
        c.save(ROOT / "status.json", {"status": "RUNNING", "case": name})
        current = run(name, ROOT / (mode + "-performance.yaml"), mapping)
        comparison = compare(numerical[0], current)
        c.save(ROOT / (name + "-comparison.json"), comparison)
        if comparison["status"] != "PASS":
            raise RuntimeError("performance process scalar precision failed")
        if c.seal() != source:
            raise RuntimeError("source changed during formal training")
        measured = [max((rank[i]["local_seconds"] for rank in current["steps"])) for i in range(5, 15)]
        formal.append(
            {
                "case": name,
                "implementation": mode,
                "rank_max_seconds": measured,
                "p50_seconds": statistics.median(measured),
                "p90_seconds": c.percentile(measured, 0.9),
                "peak_allocated_bytes": current["peak_allocated_bytes"],
                "peak_reserved_bytes": current["peak_reserved_bytes"],
            }
        )
        c.save(ROOT / "performance-progress.json", formal)
    c.save(
        ROOT / "performance-results.json",
        {
            "cases": formal,
            "order": order,
            "numerical_status": "PASS",
            "scope": (
                "One original-width LongCat logical layer; EP16/denseFSDP16/expertFSDP1; sequence"
                + "2048/globalbatch16; BF16 compute/FP32 main/reduction; same step1 DCP; 10 measure"
                + "d optimizer updates per fresh process"
            ),
            "oracle": (
                "Native GMM with original Torch routing; stronger native control uses identical T"
                + "riton routing to MegaMoE"
            ),
            "norm_policy": "Literal HF RMSNorm in every arm",
            "precision_option": "Fused SwiGLU: torch_npu.npu_swiglu native; embedded CANN swi_glu MegaMoE",
        },
    )
    c.save(ROOT / "status.json", {"status": "COMPLETE"})
except BaseException as error:
    c.save(ROOT / "status.json", {"status": "FAILED", "error": repr(error)})
    raise
