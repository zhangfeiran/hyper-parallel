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
"""Audit a five-point MoonEP/native B=6 comparison and overlay the prior SVG data."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
import matplotlib.pyplot as plt

from scripts.plot_dirichlet_replica_sweep import _physical

matplotlib.use("Agg")
OLD_VARIANTS = ("native-b0", "native-b1", "megamoe-b0", "megamoe-b1-p2p", "megamoe-b1")
OLD_LABELS = ("Native B=0 (25 points)", "Native B=1 (25 points)", "MegaMoE B=0 (25 points)",
              "MegaMoE B=1 P2P (25 points)", "MegaMoE B=1 adaptive KG (25 points)")
COLORS = ("#666666", "#9467bd", "#0077b6", "#00a878", "#e66b00")
NEW_VARIANTS = ("native-b6", "moonep-b6")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load(directory, keys):
    terminal = _read(directory / "terminal.json")
    if terminal["returncode"] or not terminal["monitor_clean"] or terminal["foreign_pids"]:
        raise ValueError(f"Unclean run: {directory}")
    ranks, identities = [], []
    for rank in range(16):
        complete = _read(directory / f"rank{rank}/complete.json")
        points = {(point["alpha"], point["seed"]): point for point in complete["results"]}
        if complete["points"] != len(keys) or list(points) != keys:
            raise ValueError(f"Incomplete or reordered points: {directory}, rank {rank}")
        ranks.append(points)
        identity = _read(directory / f"rank{rank}/identity.json")
        files = {Path(identity["root"]) / name: digest for name, digest in identity["source_sha256"].items()}
        files.update({Path(item["path"]): item["sha256"] for item in identity["measurement_scripts"].values()})
        files.update({Path(item["path"]): item["sha256"] for item in identity.get("moonep", {}).get("files", [])})
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in files.items()):
            raise ValueError(f"Changed source or MoonEP payload: {directory}")
        identities.append(identity)
    monitors = [_read_line(line) for line in (directory / "npu-monitor.jsonl").read_text().splitlines()]
    if not monitors or any(m["foreign_pids"] or not m["healthy"] or m["returncode"] for m in monitors):
        raise ValueError(f"Invalid monitor: {directory}")
    return ranks, identities, monitors


def _read_line(line):
    return json.loads(line)


def aggregate(root: Path, routes: dict, base: dict) -> tuple[list[dict], dict]:
    """Require two clean fresh runs, matching sources/inputs and all 16 ranks."""
    keys = [(route["alpha"], route["seed"]) for route in routes["routes"]]
    loaded = {(repeat, variant): _load(root / "sweep" / f"repeat{repeat}-{variant}", keys)
              for repeat in (1, 2) for variant in NEW_VARIANTS}
    reference = loaded[1, "native-b6"][1]
    moonep_payload = loaded[1, "moonep-b6"][1][0]["moonep"]
    fields = ("initial_weights", "input_sha256", "gradient_sha256", "sha", "source_manifest_sha256",
              "torch", "torch_npu", "cann", "measurement_scripts")
    configuration = reference[0]["configuration"]
    for (_, variant), (ranks, identities, _) in loaded.items():
        for rank, identity in enumerate(identities):
            if any(identity[field] != reference[rank][field] for field in fields):
                raise ValueError(f"Unmatched provenance: {variant}, rank {rank}")
            if any(identity["configuration"][field] != configuration[field] for field in
                   ("layers", "tokens", "experts", "hidden", "intermediate", "top_k", "warmup", "measured", "pairs")):
                raise ValueError(f"Unmatched workload: {variant}, rank {rank}")
            if identity["configuration"]["budget"] != 6:
                raise ValueError("Five-point overlay expects B=6")
            if variant == "moonep-b6" and identity["moonep"] != moonep_payload:
                raise ValueError("Unmatched MoonEP dependency across ranks or repeats")
            for key, point in ranks[rank].items():
                if not point["finite_validation"] or point["step_ms"]["samples"] != ranks[0][key]["step_ms"]["samples"]:
                    raise ValueError("Finite validation or all-rank timing check failed")
                if len(point["step_ms"]["samples"]) != configuration["measured"]:
                    raise ValueError("Incomplete measured samples")
                if any(after <= before for before, after in zip(point["parameter_versions_before"],
                                                                point["parameter_versions_after"])):
                    raise ValueError("Original parameter did not update")
                if any(point["memory_before"][field] != point["memory"][field] for field in ("capacity", "heap_epoch")):
                    raise ValueError("Measured capacity growth")
                if variant == "moonep-b6":
                    if point["transport"]["actual_transport"] != "moonep_mte":
                        raise ValueError("MoonEP MTE did not execute")
                    if point["transport"]["weight_partial_compute"] != "native_bf16_packed_grouped_gemm":
                        raise ValueError("MoonEP weight-gradient compute differs from native")
                    if point["replica_plan"]["destination_loads"] != [configuration["tokens"] *
                                                                        configuration["top_k"]] * 16:
                        raise ValueError("MoonEP destination load is not balanced")
    acceptance_errors = {}
    for variant in NEW_VARIANTS:
        acceptance = _load(root / "accept" / f"accept-{variant}", keys)
        for rank, identity in enumerate(acceptance[1]):
            if any(identity[field] != reference[rank][field] for field in fields if field != "initial_weights"):
                raise ValueError("Numerical acceptance and timing provenance differ")
            if identity["initial_weights"] != reference[rank]["initial_weights"][:1]:
                raise ValueError("Numerical acceptance and timing initial weights differ")
            if variant == "moonep-b6" and identity["moonep"] != moonep_payload:
                raise ValueError("Numerical acceptance used another MoonEP dependency")
        errors = [metric["relative_l2_rank_max"] for rank in acceptance[0] for point in rank.values()
                  for metric in point["checks"].values()]
        if not errors or any(not math.isfinite(error) or error > .01 for error in errors):
            raise ValueError(f"Numerical acceptance failed: {variant}")
        acceptance_errors[variant] = max(errors)
    old_rows = {(row["alpha"], row["seed"]): row for row in base["rows"]}
    rows, coverage = [], []
    for route in routes["routes"]:
        key = (route["alpha"], route["seed"])
        row = {name: route[name] for name in ("alpha", "seed", "skew", "route_sha256")}
        if row["route_sha256"] != old_rows[key]["route_sha256"]:
            raise ValueError("Selected route does not match old SVG data")
        for variant in NEW_VARIANTS:
            steps, fb, allocated, reserved, physical, process, heaps = [], [], [], [], [], [], []
            for repeat in (1, 2):
                ranks, _, monitors = loaded[repeat, variant]
                points = [rank[key] for rank in ranks]
                if any(p["route_sha256"] != row["route_sha256"] for p in points):
                    raise ValueError("Mismatched route across ranks or variants")
                samples = _physical(points, monitors)
                coverage.append(samples["physical_samples"])
                steps.extend(points[0]["step_ms"]["samples"])
                fb.extend(points[0]["fwd_bwd_ms"]["samples"])
                allocated.append(max(p["memory"]["total_peak_bytes"] for p in points) / 2**30)
                reserved.append(max(p["memory"]["total_reserved_peak_bytes"] for p in points) / 2**30)
                heaps.append(max(p["memory"]["shmem_heap"] for p in points) / 2**30)
                physical.append(samples["physical_card_hbm_gib"])
                process.append(samples["physical_process_hbm_gib"])
                row[f"{variant}_repeat{repeat}_ms"] = points[0]["step_ms"]["median"]
            row.update({f"{variant}_step_ms": statistics.median(steps),
                        f"{variant}_fwd_bwd_ms": statistics.median(fb), f"{variant}_hbm_gib": max(allocated),
                        f"{variant}_reserved_plus_heap_gib": max(reserved), f"{variant}_heap_gib": max(heaps),
                        f"{variant}_physical_card_hbm_gib": max(physical),
                        f"{variant}_physical_process_hbm_gib": max(process)})
        row["moonep_vs_native_b6_percent"] = 100 * (1 - row["moonep-b6_step_ms"] / row["native-b6_step_ms"])
        rows.append(row)
    audit = {"complete": True, "routes": 5, "fresh_runs_per_variant": 2, "ranks": 16,
             "acceptance_max_relative_l2": acceptance_errors, "physical_samples_min": min(coverage),
             "matched_new_variants": list(NEW_VARIANTS), "old_series_reused": list(OLD_VARIANTS),
             "old_and_new_update_and_highwater_histories_differ": True,
             "moonep": loaded[1, "moonep-b6"][1][0]["moonep"]}
    return rows, audit


def render(base: dict, rows: list[dict], output: Path) -> None:
    """Retain the five original curves and add measured five-point B=6 lines."""
    matplotlib.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True,
                               gridspec_kw={"height_ratios": (3.4, 1.5, 2.5, 2.5)})
    metrics = ("step_ms", "hbm_gib", "physical_card_hbm_gib")
    old = base["rows"]
    for variant, label, color in zip(OLD_VARIANTS, OLD_LABELS, COLORS):
        for axis, metric in zip((axes[0], axes[2], axes[3]), metrics):
            line, = axis.plot([r["skew"] for r in old], [r[f"{variant}_{metric}"] for r in old],
                              color=color, marker="o", markersize=3, linewidth=1.6, label=label,
                              linestyle="--" if variant.endswith("p2p") else "-")
            line.set_gid(f"{metric}-{variant}")
    styles = (("native-b6", "Native B=6 P2P (5 points, fresh control)", "#be4b7c", "^"),
              ("moonep-b6", "Native + MoonEP A3 MTE, B=6 (5 points)", "#222222", "D"))
    for variant, label, color, marker in styles:
        for axis, metric in zip((axes[0], axes[2], axes[3]), metrics):
            line, = axis.plot([r["skew"] for r in rows], [r[f"{variant}_{metric}"] for r in rows],
                              color=color, marker=marker, markersize=6, linewidth=2, linestyle="-.", label=label)
            line.set_gid(f"{metric}-{variant}")
    gains = [r["overlap_vs_p2p_percent"] for r in old]
    lower = [min(r["overlap_vs_p2p_repeat1_percent"], r["overlap_vs_p2p_repeat2_percent"]) for r in old]
    upper = [max(r["overlap_vs_p2p_repeat1_percent"], r["overlap_vs_p2p_repeat2_percent"]) for r in old]
    axes[1].fill_between([r["skew"] for r in old], lower, upper, color=COLORS[-1], alpha=.16,
                          label="Original two fresh-process comparisons")
    axes[1].plot([r["skew"] for r in old], gains, color=COLORS[-1], marker="o", markersize=3,
                 label="Original adaptive KG time reduction vs P2P")
    axes[1].axhline(0, color="#666666", linewidth=.8, linestyle="--")
    axes[0].set_ylabel("Median rank-max F+B + AdamW (ms)")
    axes[1].set_ylabel("KG time reduction vs P2P (%)")
    axes[2].set_ylabel("Peak allocator + external pools\n(GiB/rank; max across repeats)")
    axes[3].set_ylabel("Sampled physical HBM (GiB/chip)")
    axes[3].set_xlabel("Routing skew S = max home rows / mean rows, before replication")
    for axis in axes:
        axis.grid(alpha=.25)
    for axis in (axes[0], axes[2], axes[3]):
        axis.legend(loc="upper left", ncol=3, fontsize=8)
        low, high = axis.get_ylim()
        axis.set_ylim(low, high + .12 * (high - low))
    axes[1].legend(loc="upper right", fontsize=8)
    axes[3].set_xlim(.9, 8.1)
    axes[3].set_xticks(range(1, 9))
    figure.suptitle("Native / MegaMoE with MoonEP A3 five-point overlay — EP16 / E96 / TopK8 / BF16\n"
                    "16 checkpointed MoE layers + AdamW; 4096 tokens/rank; H5120 / I1792", fontsize=15)
    figure.text(.5, .015, "Original five lines: 25 routes. New MoonEP and Native B=6 control: 5 routes. "
                "Both use 2 fresh processes/series, 3 warmup + 7 measured steps/point.\n"
                "MoonEP A3 port: online CPU planning; upstream MTE rank dedup, prefetch and FP32 grad reduce. "
                "A5 SIMT / UDMA / server dedup are not executed.\n"
                "B=6 is MoonEP's E/EP default. Native BF16 grouped gradient GEMMs; "
                "FP32 replica reduction. Boundary/home copies and A3 host fences are timed.\n"
                "Old/new update and allocator high-water histories differ; old lines provide trend context. "
                "Physical HBM is sampled. Scope excludes attention and router network.",
                ha="center", fontsize=8.5, color="#555555")
    figure.tight_layout(rect=(0, .085, 1, .945))
    figure.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    figure.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(figure)


def main() -> None:
    """Audit complete artifacts before emitting the overlay and raw data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = _read(args.base)
    if not base["audit"]["all_artifact_checks_passed"]:
        raise ValueError("Original five-series figure was not accepted")
    rows, audit = aggregate(args.input, _read(args.input / "routes5.json"), base)
    audit["base_data_sha256"] = hashlib.sha256(args.base.read_bytes()).hexdigest()
    render(base, rows, args.output)
    additions = {(row["alpha"], row["seed"]): row for row in rows}
    combined = [dict(row, **additions.get((row["alpha"], row["seed"]), {})) for row in base["rows"]]
    audit["all_artifact_checks_passed"] = True
    args.output.with_suffix(".json").write_text(json.dumps({"rows": combined, "five_point_rows": rows,
                                                            "audit": audit, "original": base}, indent=2) + "\n",
                                               encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as target:
        fields = list(dict.fromkeys([*base["rows"][0], *rows[0]]))
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    print(json.dumps({"audit": audit, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
