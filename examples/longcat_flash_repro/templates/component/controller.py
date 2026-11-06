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

import helpers as c


def main() -> None:
    "Run this experiment stage and retain its acceptance records."
    raw = subprocess.run(["npu-smi", "info", "-m"], capture_output=True, text=True, check=True).stdout
    mapping = {
        (int(a), int(b)): int(v) for a, b, v in re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+Ascend910", raw, re.M)
    }
    if not set(c.DEVICES).issubset(mapping.values()):
        raise RuntimeError("selected component devices are unavailable")
    source = c.seal()
    c.save(c.ROOT / "source-seal.json", source)
    results = []
    pilots = []
    tokens = 2048
    c.ROUTING_BACKEND = "triton"
    c.DIRECT_GRADIENTS = True
    for index in range(3):
        c.save(c.ROOT / "status.json", {"status": "RUNNING", "phase": "pilot", "index": index})
        current = c.run_case(tokens, "native", index, "pilot", source, mapping)
        if pilots and current["identities"] != pilots[0]["identities"]:
            raise RuntimeError("pilot identity differs")
        current["implementation"] = "native_triton_prefix"
        pilots.append(current)
        results.append(current)
        c.save(c.ROOT / "results-progress.json", results)
    thresholds = {"2048": {}}
    for phase in ["forward", "forward_backward"]:
        medians = [row["phases"][phase]["p50_ms"] for row in pilots]
        tails = [row["phases"][phase]["p90_ms"] for row in pilots]
        span = (max(medians) - min(medians)) / statistics.median(medians)
        tail_span = (max(tails) - min(tails)) / statistics.median(tails)
        thresholds["2048"][phase] = {
            "pilot_p50_ms": medians,
            "pilot_p90_ms": tails,
            "minimum_speedup": 1 + max(0.01, 2 * span),
            "maximum_p90_regression": max(0.03, 2 * tail_span),
        }
    c.save(c.ROOT / "pilot-thresholds.json", thresholds)
    order = [
        "native_torch",
        "native_triton_prefix",
        "push_direct_gradient",
        "pull_direct_gradient",
        "pull_direct_gradient",
        "push_direct_gradient",
        "native_triton_prefix",
        "native_torch",
        "push_direct_gradient",
        "native_torch",
        "pull_direct_gradient",
        "native_triton_prefix",
    ]
    for index, implementation in enumerate(order):
        mode = "native" if implementation.startswith("native") else implementation.split("_")[0]
        c.ROUTING_BACKEND = "torch" if implementation == "native_torch" else "triton"
        c.DIRECT_GRADIENTS = implementation not in ["native_torch", "push_ordinary_gradient"]
        c.save(
            c.ROOT / "status.json",
            {
                "status": "RUNNING",
                "phase": "formal",
                "index": index,
                "implementation": implementation,
                "completed": len(results),
            },
        )
        current = c.run_case(tokens, mode, index, "formal", source, mapping)
        if current["identities"] != pilots[0]["identities"]:
            raise RuntimeError("formal identity differs")
        current["implementation"] = implementation
        results.append(current)
        c.save(c.ROOT / "results-progress.json", results)
    c.save(c.ROOT / "results.json", {"cases": results, "thresholds": thresholds, "source_seal": source})
    c.save(c.ROOT / "status.json", {"status": "COMPLETE", "completed": len(results)})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        c.save(c.ROOT / "status.json", {"status": "FAILED", "error": repr(error)})
        raise
