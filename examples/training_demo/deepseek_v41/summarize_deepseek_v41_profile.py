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
"""Audit packed native/push diagnostics with separate per-case traces."""

import argparse
import json
from pathlib import Path

from examples.training_demo.deepseek_v41.summarize_deepseek_v41_cluster import _compare, _summarize_case


def summarize(run_root: Path, nodes: int, cards_per_node: int, steps: int, warmup_steps: int) -> dict:
    """Retain completed cases and require every selected trace for both cases."""
    cases, fingerprints, traces = {}, {}, {}
    selected_ranks = range(0, nodes * cards_per_node, cards_per_node)
    complete = True
    for label in ("native", "push"):
        try:
            case, signature = _summarize_case(
                run_root / "packed", run_root / "packed/nodes", nodes, cards_per_node, steps, warmup_steps, label,
            )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            cases[f"packed/{label}"] = {"status": "incomplete_or_failed",
                                         "reason": f"{type(error).__name__}: {error}"}
            complete = False
        else:
            cases[f"packed/{label}"] = case
            fingerprints[label] = signature
        traces[label] = {}
        for rank in selected_ranks:
            root = run_root / "profiles" / label / f"rank{rank}"
            files = sorted(
                str(path.relative_to(run_root)) for path in root.rglob("*")
                if path.is_file() and (path.suffix in (".csv", ".db", ".json") or ".trace.json" in path.name)
            )
            traces[label][str(rank)] = files
            if not files:
                complete = False
    comparison = {"available": False, "performance_eligible": False,
                  "reason": "A required case is incomplete"}
    if all(label in fingerprints for label in ("native", "push")):
        comparison = _compare({label: cases[f"packed/{label}"] for label in fingerprints}, fingerprints, "push")
        comparison["performance_eligible"] = False
        comparison["reason"] = "Profiler-on timings are diagnostic only"
    return {"complete": complete, "purpose": "profiler_diagnostic", "speedup_eligible": False,
            "case_order": ["packed/native", "packed/push"], "cases": cases, "traces": traces,
            "profiled_ranks_per_case": list(selected_ranks), "comparisons": {"push_vs_native": comparison}}


def main() -> None:
    """Publish partial diagnostics after each case and audit the final pair."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--cards-per-node", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    args = parser.parse_args()
    if args.nodes < 1 or args.cards_per_node < 1 or not 0 <= args.warmup_steps < args.steps:
        parser.error("Require positive topology and 0 <= warmup-steps < steps")
    report = summarize(args.run_root, args.nodes, args.cards_per_node, args.steps, args.warmup_steps)
    (args.run_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"complete": report["complete"], "case_order": report["case_order"],
                      "profiled_ranks_per_case": report["profiled_ranks_per_case"]}))


if __name__ == "__main__":
    main()
